//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! A numbered copy nothing names or finds for `VICTORY_MEMORY_IDLE_S`
//! seconds (30) gives its number back, and the least recently used go while
//! all of them pass `VICTORY_MEMORY_COPIES_MB` (2048); there are at most
//! [`COPIES`] numbers. shortcut: a page given back that shaders still read
//! through pointers alone reads 0 for a frame, until its fault brings the
//! copy back. Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows; the
/// prison with the structure builder on used up 1023 within minutes (its
/// root constants that look like addresses name copies too), after which
/// every new page read zeros and the traced light went dark.
const COPIES: u32 = 4095;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// Frames a number given back waits before it is given again.
const COOL_FRAMES: u64 = 4;
/// A directory page.
const PAGE: u64 = 1 << 16;

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]` (none: a number given back, see [`Memory::sweep`]).
    copies: Vec<Option<Arc<Shadow>>>,
    /// When each number's copy was last named or found for a page.
    touched: Vec<std::time::Instant>,
    /// Numbers given back, with the frame: used again some frames later,
    /// when no list still in flight can hold the old view.
    cooling: Vec<(usize, u64)>,
    /// Numbers given a copy during this `ready`: their copies are brought
    /// up to date by this list.
    fresh: Vec<usize>,
    /// Per host view heap (by address): the first slot of its run, and per
    /// number the copy whose view its slot holds (its id; 0: none).
    heaps: HashMap<usize, (u32, Vec<u64>)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            copies: Vec::new(),
            touched: Vec::new(),
            cooling: Vec::new(),
            fresh: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        let had = unsafe { entry.read() };
        if had != 0 {
            if let Some(t) = self.touched.get_mut(had as usize - 1) {
                *t = std::time::Instant::now();
            }
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return;
        }
        let n = match self.copies.iter().position(|c| c.as_ref().is_some_and(|c| Arc::ptr_eq(c, &shadow))) {
            Some(n) => {
                self.touched[n] = std::time::Instant::now();
                n
            }
            None if !self.cooling.is_empty() || self.copies.len() < COPIES as usize => {
                let frame = self.frame;
                let n = match self.cooling.iter().position(|&(_, f)| f + COOL_FRAMES <= frame) {
                    Some(i) => self.cooling.swap_remove(i).0,
                    None if self.copies.len() < COPIES as usize => {
                        self.copies.push(None);
                        self.touched.push(std::time::Instant::now());
                        self.copies.len() - 1
                    }
                    None => return,
                };
                self.copies[n] = Some(shadow.clone());
                self.touched[n] = std::time::Instant::now();
                self.fresh.push(n);
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
    }

    /// Gives number `n` back: its copy is let go (the shadow module frees it
    /// once idle) and its pages read as without a copy until named again.
    fn retire(&mut self, n: usize) {
        let Some(shadow) = self.copies[n].take() else {
            return;
        };
        let (first, len) = shadow.span();
        let mut at = first & !(PAGE - 1);
        while at < first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
            // SAFETY: inside the directory's mapping.
            let entry = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
            // SAFETY: as above; the number first, so a shader never pairs
            // it with another copy's offset.
            unsafe {
                if entry.read() == n as u32 + 1 {
                    entry.write_volatile(0);
                    entry.add(1).write_volatile(0);
                }
            }
            at += PAGE;
        }
        self.cooling.push((n, self.frame));
    }

    /// Once a frame: copies nothing named or found for a while are given
    /// back, then the least recently used ones while all of them are more
    /// than the cap. Kept for the run, they stayed alive (the shadow module
    /// lets a window go only when nothing else holds it) and every list
    /// refreshed them all: with the structure builder on, 10 GB in four
    /// minutes and ~450 ms of copies a frame. A page given back that a
    /// shader still reads faults and has its copy again a frame later.
    fn sweep(&mut self) {
        static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
        let &(idle_s, cap) = LIMITS.get_or_init(|| {
            let var = |name: &str, default: u64| {
                std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
            };
            (var("VICTORY_MEMORY_IDLE_S", 30), var("VICTORY_MEMORY_COPIES_MB", 2048) << 20)
        });
        for n in 0..self.copies.len() {
            if self.copies[n].is_some() && self.touched[n].elapsed().as_secs() >= idle_s {
                self.retire(n);
            }
        }
        let mut live: Vec<usize> = (0..self.copies.len()).filter(|&n| self.copies[n].is_some()).collect();
        let mut bytes: u64 = live.iter().map(|&n| self.copies[n].as_ref().map_or(0, |c| c.span().1)).sum();
        if bytes > cap {
            live.sort_by_key(|&n| self.touched[n]);
            for n in live {
                if bytes <= cap {
                    break;
                }
                bytes -= self.copies[n].as_ref().map_or(0, |c| c.span().1);
                self.retire(n);
            }
        }
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    m.fresh.clear();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.sweep();
        m.take_faults();
        // What every list running such a shader brings up to date (and, with stores,
        // compares on the CPU): the numbered copies.
        static SAID: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if said.is_none_or(|t| t.elapsed().as_secs() >= 5) {
            *said = Some(std::time::Instant::now());
            let live = m.copies.iter().flatten();
            let (count, bytes) = live.fold((0, 0u64), |(n, b), c| (n + 1, b + c.span().1));
            tracing::info!(
                copies = count,
                numbers = m.copies.len(),
                mb = bytes >> 20,
                "title memory by address: numbered copies, each refreshed by every list that runs such a shader"
            );
        }
    }
    for &address in addresses {
        m.name(address);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8);
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    if !m.heaps.contains_key(&key) {
        // Held for the heap's life: not given back with a list's slots.
        let base = heap.dynamic(COPIES + 1)?;
        heap.null_raw_uav_at(base);
        m.heaps.insert(key, (base, Vec::new()));
    }
    let Memory { heaps, copies, .. } = &mut *m;
    let (base, views) = heaps.get_mut(&key)?;
    let base = *base;
    views.resize(copies.len(), 0);
    // A slot whose number has another copy now gets that copy's view; one
    // given back is made null, so that the slot no longer keeps the copy's
    // buffer (the shadow module frees a copy only when nothing holds it;
    // lists in flight hold theirs through their refreshes).
    for (n, copy) in copies.iter().enumerate() {
        match copy {
            Some(shadow) if views[n] != shadow.id() => {
                let (_, len) = shadow.span();
                heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
                views[n] = shadow.id();
            }
            None if views[n] != 0 => {
                heap.null_raw_uav_at(base + 1 + n as u32);
                views[n] = 0;
            }
            _ => {}
        }
    }
    // Once per list, and for copies numbered since: the builder alone
    // dispatches hundreds of times a frame.
    // shortcut: a copy another list numbers meanwhile is brought up to date
    // by that list's submission, which may run after this one's.
    const READ: usize = usize::MAX - 0x6d;
    const WRITE: usize = usize::MAX - 0x6e;
    let first_read = list.first_use(READ);
    let first_write = writes && list.first_use(WRITE);
    for (n, shadow) in m.copies.iter().enumerate() {
        let Some(shadow) = shadow else {
            continue;
        };
        let (start, len) = shadow.span();
        let fresh = m.fresh.contains(&n);
        if first_read || fresh {
            shadow.refresh(list, start, len);
        }
        if writes && (first_write || fresh) {
            shadow.gpu_writes(list, start, len);
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}
