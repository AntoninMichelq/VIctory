//! Direct title memory (opt-in, `VICTORY_DIRECT_MEMORY=1`): the host GPU
//! reads and writes title buffers in place, as the console's GPU does,
//! instead of through host copies ([`crate::shadow`]) refreshed before
//! each submission and copied back after it.
//!
//! D3D12 can wrap existing system memory as a heap:
//! `ID3D12Device3::OpenExistingHeapFromFileMapping` for the pagefile-backed
//! sections behind the title's physical pages (`xmem`), and
//! `OpenExistingHeapFromAddress` for the title's private `VirtualAlloc`
//! regions. Each wraps the whole section or region (pinned while the heap
//! lives); buffers placed on it (windows of up to 4095 MiB) give every
//! title address in it a host GPU address. The heap is system memory (L0, CPU write-back),
//! as the host copies are, so the GPU reads over the bus either way.
//!
//! What cannot be wrapped (a region only partly committed, a range over
//! two sections or a scatter mapping, a driver that refuses) keeps host
//! copies. Each heap opened is logged once, each failure with its HRESULT.
//!
//! shortcut: a range the title maps other pages over after a list using it
//! was recorded still shows the old pages to that list (see
//! [`crate::shadow::direct_remapped`]); private regions the title frees
//! stay pinned until no copy uses them, and one freed and allocated again
//! at the same address with the same size reuses the old heap.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

use crate::host::host;

/// The section showing a range of title memory (from `xmem`).
pub struct DirectSection {
    /// A duplicate of the section's handle, closed here.
    pub handle: isize,
    /// A number no other section gets.
    pub id: u64,
    /// The section's size in bytes.
    pub size: u64,
    /// Where the range starts in the section.
    pub offset: u64,
}

/// Finds the section behind `[at, at + len)`.
pub type SectionLookup = fn(usize, usize) -> Option<DirectSection>;

static LOOKUP: OnceLock<SectionLookup> = OnceLock::new();

/// The runtime tells how title addresses map to sections (once).
pub fn set_direct_section_lookup(lookup: SectionLookup) {
    let _ = LOOKUP.set(lookup);
}

/// `VICTORY_DIRECT_MEMORY`: 0 (default) off; 1 the title's private
/// `VirtualAlloc` regions; 2 also its physical-page sections (which may
/// commit whole sections, gigabytes, to open them).
fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let level = match std::env::var("VICTORY_DIRECT_MEMORY").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 0,
        };
        if level > 0 {
            tracing::info!(level, "direct title memory on (VICTORY_DIRECT_MEMORY)");
        }
        level
    })
}

/// Whether direct title memory is asked for at all.
pub(crate) fn enabled() -> bool {
    level() > 0
}

/// A heap over title memory, with buffers placed on it as ranges need.
pub(crate) struct Heap {
    device: ID3D12Device3,
    heap: ID3D12Heap,
    what: String,
    size: u64,
    flags: D3D12_HEAP_FLAGS,
    /// Placed buffers by heap offset, with their GPU addresses.
    windows: Mutex<HashMap<u64, (ID3D12Resource, u64)>>,
    /// Whether it counts in [`PINNED`] and [`HEAPS_OPEN`].
    counted: bool,
}

impl Drop for Heap {
    fn drop(&mut self) {
        if self.counted {
            PINNED.fetch_sub(self.size, Ordering::Relaxed);
            HEAPS_OPEN.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Heap {}
// SAFETY: as above.
unsafe impl Sync for Heap {}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Key {
    /// A physical-page section, by number.
    Section(u64),
    /// A private region: allocation base and size.
    Private(usize, usize),
}

enum Slot {
    Open(Weak<Heap>),
    Failed,
}

static HEAPS: Mutex<Option<HashMap<Key, Slot>>> = Mutex::new(None);
/// Bytes of title memory wrapped (pinned) now, and heaps open.
static PINNED: AtomicU64 = AtomicU64::new(0);
static HEAPS_OPEN: AtomicUsize = AtomicUsize::new(0);
/// Failures logged at warn level; later ones go to debug.
static WARNED: AtomicUsize = AtomicUsize::new(0);

fn fail(what: std::fmt::Arguments<'_>) {
    if WARNED.fetch_add(1, Ordering::Relaxed) < 32 {
        tracing::warn!("direct title memory: {what}");
    } else {
        tracing::debug!("direct title memory: {what}");
    }
}

/// A buffer showing title memory `[address, address + len)` in place: its
/// heap (kept alive by whoever uses it), the buffer, its GPU address and
/// where `address` is in it; `None` where it cannot be (the range then
/// keeps a host copy).
pub(crate) fn resolve(address: u64, len: u64) -> Option<(Arc<Heap>, ID3D12Resource, u64, u64)> {
    let (heap, offset) = heap_of(address, len)?;
    match heap.window(offset, len) {
        Ok((buffer, gpu, at)) => Some((heap, buffer, gpu, at)),
        Err(why) => {
            fail(format_args!("{why}"));
            None
        }
    }
}

/// The heap holding `[address, address + len)` and where `address` is in it.
fn heap_of(address: u64, len: u64) -> Option<(Arc<Heap>, u64)> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let (at, len) = (address as usize, len.max(1) as usize);
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 || info.State != MEM_COMMIT {
        return None;
    }
    if info.Type == MEM_MAPPED {
        if level() < 2 {
            return None;
        }
        let section = LOOKUP.get()?(at, len)?;
        let key = Key::Section(section.id);
        let heap = heap(key, || open_section(&section));
        close(section.handle);
        return heap.map(|h| (h, section.offset));
    }
    if info.Type != MEM_PRIVATE {
        return None;
    }
    // The whole region: a heap wraps all of it, so all of it must be
    // committed (pinned pages cannot be reserved ones).
    let base = info.AllocationBase as usize;
    let (mut p, mut reserved) = (base, false);
    loop {
        // SAFETY: as above.
        let mut part: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        let n = unsafe { VirtualQuery(p as *const _, &mut part, size_of_val(&part)) };
        if n == 0 || part.AllocationBase as usize != base {
            break;
        }
        reserved |= part.State != MEM_COMMIT;
        p = part.BaseAddress as usize + part.RegionSize;
    }
    let size = p - base;
    if at + len > base + size {
        return None;
    }
    let key = Key::Private(base, size);
    let heap = heap(key, || {
        if reserved {
            return Err(format!(
                "region {base:#x}+{size:#x} is only partly committed; it keeps host copies"
            ));
        }
        open_private(base, size)
    });
    heap.map(|h| (h, (at - base) as u64))
}

/// The open heap for `key`, opened by `open` if there is none and it did
/// not fail before.
fn heap(key: Key, open: impl FnOnce() -> Result<Arc<Heap>, String>) -> Option<Arc<Heap>> {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let heaps = heaps.get_or_insert_with(HashMap::new);
    match heaps.get(&key) {
        Some(Slot::Failed) => return None,
        Some(Slot::Open(weak)) => {
            if let Some(heap) = weak.upgrade() {
                return Some(heap);
            }
        }
        None => {}
    }
    match open() {
        Ok(heap) => {
            heaps.insert(key, Slot::Open(Arc::downgrade(&heap)));
            Some(heap)
        }
        Err(why) => {
            fail(format_args!("{why}"));
            heaps.insert(key, Slot::Failed);
            None
        }
    }
}

fn close(handle: isize) {
    // SAFETY: a duplicate handle given to us to close.
    unsafe { windows_sys::Win32::Foundation::CloseHandle(handle as _) };
}

fn device3() -> Result<ID3D12Device3, String> {
    let host = host().ok_or("no host device")?;
    static SUPPORT: OnceLock<()> = OnceLock::new();
    SUPPORT.get_or_init(|| {
        let mut data = D3D12_FEATURE_DATA_EXISTING_HEAPS::default();
        // SAFETY: the query writes its own structure of the given size.
        let r = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_EXISTING_HEAPS,
                (&raw mut data).cast(),
                size_of_val(&data) as u32,
            )
        };
        tracing::info!(
            supported = data.Supported.as_bool(),
            query = ?r.err().map(|e| format!("{:#010x}", e.code().0)),
            "direct title memory: existing heaps"
        );
    });
    host.device
        .cast::<ID3D12Device3>()
        .map_err(|e| format!("no ID3D12Device3 (HRESULT {:#010x})", e.code().0))
}

fn open_section(section: &DirectSection) -> Result<Arc<Heap>, String> {
    let device = device3()?;
    let handle = HANDLE(section.handle as *mut _);
    // SAFETY: a live section handle.
    let first = unsafe { device.OpenExistingHeapFromFileMapping::<ID3D12Heap>(handle) };
    let heap = match first {
        Ok(heap) => heap,
        Err(e) => {
            // Title sections are reserved and committed as mapped; pinning
            // may need all of it committed. Commit it whole and try again.
            let committed = commit_section(section);
            tracing::warn!(
                id = section.id,
                size = section.size,
                hresult = format_args!("{:#010x}", e.code().0),
                committed,
                "direct title memory: opening a section heap failed; retrying committed whole"
            );
            // SAFETY: as above.
            unsafe { device.OpenExistingHeapFromFileMapping::<ID3D12Heap>(handle) }.map_err(|e| {
                format!(
                    "section {} ({:#x} bytes): OpenExistingHeapFromFileMapping failed, HRESULT {:#010x}",
                    section.id,
                    section.size,
                    e.code().0
                )
            })?
        }
    };
    finish(&device, heap, format_args!("section {}", section.id))
}

/// Commits all of a section (`SEC_RESERVE`) through a view of our own.
fn commit_section(section: &DirectSection) -> bool {
    use windows_sys::Win32::System::Memory::{
        FILE_MAP_WRITE, MEM_COMMIT, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile, VirtualAlloc,
    };
    let Ok(len) = usize::try_from(section.size) else {
        return false;
    };
    // SAFETY: a view of a live section, unmapped below.
    let view = unsafe { MapViewOfFile(section.handle as _, FILE_MAP_WRITE, 0, 0, len) };
    if view.Value.is_null() {
        return false;
    }
    // SAFETY: commits the section's pages through our view.
    let ok = !unsafe { VirtualAlloc(view.Value, len, MEM_COMMIT, PAGE_READWRITE) }.is_null();
    if !ok {
        tracing::warn!(
            size = section.size,
            error = %std::io::Error::last_os_error(),
            "direct title memory: committing a section failed"
        );
    }
    // SAFETY: our view.
    unsafe { UnmapViewOfFile(view) };
    ok
}

fn open_private(base: usize, size: usize) -> Result<Arc<Heap>, String> {
    let device = device3()?;
    // SAFETY: the base of a live, fully committed region.
    let heap = unsafe { device.OpenExistingHeapFromAddress::<ID3D12Heap>(base as *const _) }.map_err(|e| {
        format!(
            "region {base:#x}+{size:#x}: OpenExistingHeapFromAddress failed, HRESULT {:#010x}",
            e.code().0
        )
    })?;
    finish(&device, heap, format_args!("region {base:#x}"))
}

/// Buffers placed on a heap reach at most this far (buffers of 4 GiB and
/// more are refused by some drivers), starting every [`STRIDE`] bytes so
/// that any range up to `WINDOW - STRIDE` long fits in one.
const WINDOW: u64 = 4095 << 20;
const STRIDE: u64 = 2 << 30;

fn finish(device: &ID3D12Device3, heap: ID3D12Heap, what: std::fmt::Arguments<'_>) -> Result<Arc<Heap>, String> {
    // SAFETY: a live heap.
    let desc = unsafe { heap.GetDesc() };
    let heap = Heap {
        device: device.clone(),
        what: what.to_string(),
        size: desc.SizeInBytes,
        flags: desc.Flags,
        windows: Mutex::new(HashMap::new()),
        counted: false,
        heap,
    };
    // The first window now: a driver that refuses buffers there refuses
    // the heap.
    let (_, gpu, _) = heap.window(0, 1)?;
    let mut heap = heap;
    heap.counted = true;
    let pinned = PINNED.fetch_add(desc.SizeInBytes, Ordering::Relaxed) + desc.SizeInBytes;
    let open = HEAPS_OPEN.fetch_add(1, Ordering::Relaxed) + 1;
    tracing::info!(
        size = desc.SizeInBytes,
        gpu = format_args!("{gpu:#x}"),
        heap_type = desc.Properties.Type.0,
        cpu_page = desc.Properties.CPUPageProperty.0,
        pool = desc.Properties.MemoryPoolPreference.0,
        heap_flags = format_args!("{:#x}", desc.Flags.0),
        open,
        pinned_mib = pinned >> 20,
        "direct title memory heap opened at {what} size {:#x}",
        desc.SizeInBytes
    );
    Ok(Arc::new(heap))
}

impl Heap {
    /// The buffer holding heap bytes `[offset, offset + len)`, its GPU
    /// address, and where `offset` is in it.
    pub(crate) fn window(&self, offset: u64, len: u64) -> Result<(ID3D12Resource, u64, u64), String> {
        let first = offset / STRIDE * STRIDE;
        let end = (first + WINDOW).min(self.size);
        if offset + len.max(1) > end {
            return Err(format!(
                "{}: a range {offset:#x}+{len:#x} fits no buffer window",
                self.what
            ));
        }
        let mut windows = self.windows.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, gpu)) = windows.get(&first) {
            return Ok((buffer.clone(), *gpu, offset - first));
        }
        let mut flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if self.flags.contains(D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER) {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
        }
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: end - first,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: a buffer placed inside a live heap (`first` is a multiple
        // of 64 KiB, `end` within the heap).
        unsafe {
            self.device.CreatePlacedResource(
                &self.heap,
                first,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        }
        .map_err(|e| {
            format!(
                "{}: a buffer at {first:#x}+{:#x} of its heap (heap flags {:#x}) failed, HRESULT {:#010x}",
                self.what,
                end - first,
                self.flags.0,
                e.code().0
            )
        })?;
        let buffer = buffer.ok_or_else(|| format!("{}: no buffer", self.what))?;
        crate::host::name(&buffer, || format!("direct title memory {} at {first:#x}", self.what));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        windows.insert(first, (buffer.clone(), gpu));
        Ok((buffer, gpu, offset - first))
    }
}
