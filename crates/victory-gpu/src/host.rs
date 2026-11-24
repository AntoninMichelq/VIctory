//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;
/// Maximum number of detached host lists kept for reuse after pending resets.
const LIST_POOL_CAP: usize = 64;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    pub(crate) depth_bounds_supported: bool,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// Detached host lists whose submissions finished, ready to reuse.
    free_lists: Mutex<Vec<HostList>>,
    #[cfg(test)]
    reused_lists: std::sync::atomic::AtomicU64,
    /// Optional CPU-visible heap for placed buffers (the arena is probed lazily).
    unified: OnceLock<Option<UnifiedArena>>,
}

/// A bump-allocated CPU-visible D3D12 heap. Allocations never overlap.
// Fits the 174 MiB title heap called out in #407, with room for alignment.
const UNIFIED_ARENA_SIZE: u64 = 256 << 20;
const UNIFIED_HEAP_ALIGNMENT: u64 = 64 << 10;
const UNIFIED_ALIGNMENT: u64 = 256;

struct UnifiedArena {
    _heap: ID3D12Heap,
    resource: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    next: Mutex<u64>,
}

/// A persistently mapped suballocation in the arena's placed buffer.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct UnifiedBuffer {
    pub(crate) resource: ID3D12Resource,
    pub(crate) data: *mut u8,
    pub(crate) gpu: u64,
    pub(crate) offset: u64,
}

impl UnifiedArena {
    fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: UNIFIED_ARENA_SIZE,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            },
            Alignment: UNIFIED_HEAP_ALIGNMENT,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: the heap description is valid and the output belongs to us.
        unsafe { device.CreateHeap(&heap_desc, &mut heap)? };
        let heap = heap.ok_or_else(windows::core::Error::empty)?;

        // One resource spans the heap. Suballocations therefore share one CPU
        // mapping and one GPU VA; separate placed resources need not have
        // contiguous GPU virtual addresses.
        let resource_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: UNIFIED_ARENA_SIZE,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: the resource occupies the full buffer-only heap; the Map
        // output pointer belongs to this arena.
        unsafe {
            device
                .CreatePlacedResource(
                    &heap,
                    0,
                    &resource_desc,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut resource,
                )
                .and_then(|()| {
                    resource
                        .as_ref()
                        .ok_or_else(windows::core::Error::empty)?
                        .Map(0, None, Some(&mut data))
                })?;
        }
        let resource = resource.ok_or_else(windows::core::Error::empty)?;
        if data.is_null() {
            return Err(windows::core::Error::empty());
        }
        // SAFETY: the resource is live and is a buffer.
        let gpu = unsafe { resource.GetGPUVirtualAddress() };
        if gpu == 0 {
            return Err(windows::core::Error::empty());
        }
        Ok(Self {
            _heap: heap,
            resource,
            data: data.cast(),
            gpu,
            next: Mutex::new(0),
        })
    }

    fn buffer(&self, size: u64) -> Option<UnifiedBuffer> {
        let size = size.max(1);
        let allocation = size.checked_next_multiple_of(UNIFIED_ALIGNMENT)?;
        let mut next = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (*next).checked_add(allocation)?;
        if end > UNIFIED_ARENA_SIZE {
            tracing::warn!(
                requested = size,
                available = UNIFIED_ARENA_SIZE.saturating_sub(*next),
                "CPU/GPU shared buffer arena is full"
            );
            return None;
        }
        let offset = *next;
        let data_offset = usize::try_from(offset).ok()?;
        // SAFETY: the allocation is wholly within the mapped 256 MiB resource.
        let data = unsafe { self.data.add(data_offset) };
        let gpu = self.gpu.checked_add(offset)?;
        *next = end;
        Some(UnifiedBuffer {
            resource: self.resource.clone(),
            data,
            gpu,
            offset,
        })
    }
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission for most of a second in the world: freeing there was
/// a GPU page fault on the freed texture (DXGI_ERROR_DEVICE_HUNG,
/// 2026-11-24, RX 7600).
static RETIRED: Mutex<Vec<Retired>> = Mutex::new(Vec::new());
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Presented host frames a retired object is also kept for. Worker threads
/// record lists ahead of submission and a title frame spans many
/// submissions, so finishing the submissions running when it was retired
/// is not enough on its own. Ages are in frames presented to the host, not
/// the title's 60 Hz clock (which ticks many times per host frame while
/// the host runs at 1-2 fps).
const KEEP_FRAMES: u64 = 4;

/// A retired object, dropped on whatever thread finishes the submission.
struct Retired {
    /// The submission count it waits for.
    submission: u64,
    /// The presented frame it was retired in.
    frame: u64,
    #[allow(dead_code)]
    object: Box<dyn std::any::Any>,
}
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Drops `object` once every submission running now has finished and
/// [`KEEP_FRAMES`] more frames were presented.
pub(crate) fn retire<T: 'static>(object: T) {
    let submission = STARTED.load(Ordering::Acquire);
    let frame = crate::queue::presented();
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push(Retired {
        submission,
        frame,
        object: Box::new(object),
    });
}

/// Drops what waited for submissions up to `finished` and is old enough
/// in presented frames.
fn bury(finished: u64) {
    let now = crate::queue::presented();
    let done: Vec<Retired> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        if retired.is_empty() {
            return;
        }
        let (done, keep) = std::mem::take(&mut *retired).into_iter().partition(|r| {
            r.submission <= finished && now.saturating_sub(r.frame) >= KEEP_FRAMES
        });
        *retired = keep;
        done
    };
    drop(done);
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Since the last frame-rate line: submissions, microseconds in
/// `ExecuteCommandLists` (driver CPU time, part of the GPU time above) and
/// microseconds blocked on the fence.
static SUBMITTING: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Of the preparing: snapshots of ranges shaders write (microseconds,
/// bytes), since the last frame-rate line.
static SNAPSHOTS: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Other parts of the preparing, in microseconds since the last frame-rate
/// line: title-writing and host work run at submission, texture checks,
/// texture upload recording (preludes), the final refresh flush, and
/// directly indexed SRV refreshes.
static PREP_PARTS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

fn prep_part(i: usize, started: std::time::Instant) {
    PREP_PARTS[i].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
static HOST: OnceLock<Option<Host>> = OnceLock::new();

/// The host GPU's memory use as its driver reports it, `[(used, budget)]`
/// in video memory and in system memory; `None` before the device is
/// made (asking does not make it).
pub(crate) fn video_memory() -> Option<[(u64, u64); 2]> {
    static ADAPTER: OnceLock<Option<IDXGIAdapter3>> = OnceLock::new();
    let host = HOST.get()?.as_ref()?;
    let adapter = ADAPTER
        .get_or_init(|| {
            // SAFETY: plain DXGI queries on a live device.
            unsafe {
                let luid = host.device.GetAdapterLuid();
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                factory.EnumAdapterByLuid::<IDXGIAdapter3>(luid).ok()
            }
        })
        .as_ref()?;
    let query = |group| {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: node 0 of the device's own adapter, a sized out structure.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }.ok()?;
        Some((info.CurrentUsage, info.Budget))
    };
    Some([
        query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?,
        query(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?,
    ])
}

pub(crate) fn host() -> Option<&'static Host> {
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug =
            cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_WARP").is_some() {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        log_caps(&device);
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let host = Host::from_device(device, messages)?;
        tracing::info!("host D3D12 device ready");
        Ok(host)
    }
}

/// Logs the host's optional shader capabilities that translated shaders may
/// declare (a pipeline whose shader needs a missing one is refused with
/// E_INVALIDARG and no other reason in release builds).
fn log_caps(device: &ID3D12Device) {
    fn query<T: Default>(device: &ID3D12Device, feature: D3D12_FEATURE) -> Option<T> {
        let mut data = T::default();
        // SAFETY: the query writes its own structure of the given size.
        unsafe {
            device
                .CheckFeatureSupport(feature, (&raw mut data).cast(), size_of::<T>() as u32)
                .ok()
                .map(|()| data)
        }
    }
    let o1: Option<D3D12_FEATURE_DATA_D3D12_OPTIONS1> = query(device, D3D12_FEATURE_D3D12_OPTIONS1);
    let o3: Option<D3D12_FEATURE_DATA_D3D12_OPTIONS3> = query(device, D3D12_FEATURE_D3D12_OPTIONS3);
    let o7: Option<D3D12_FEATURE_DATA_D3D12_OPTIONS7> = query(device, D3D12_FEATURE_D3D12_OPTIONS7);
    let o9: Option<D3D12_FEATURE_DATA_D3D12_OPTIONS9> = query(device, D3D12_FEATURE_D3D12_OPTIONS9);
    let o11: Option<D3D12_FEATURE_DATA_D3D12_OPTIONS11> = query(device, D3D12_FEATURE_D3D12_OPTIONS11);
    let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_7 };
    // SAFETY: as above.
    let sm_ok = unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
            .is_ok()
    };
    tracing::info!(
        shader_model = if sm_ok { format!("{:#x}", sm.HighestShaderModel.0) } else { "?".into() },
        int64_ops = ?o1.map(|o| o.Int64ShaderOps.as_bool()),
        wave_ops = ?o1.map(|o| o.WaveOps.as_bool()),
        barycentrics = ?o3.map(|o| o.BarycentricsSupported.as_bool()),
        mesh_tier = ?o7.map(|o| o.MeshShaderTier.0),
        atomic64_typed = ?o9.map(|o| o.AtomicInt64OnTypedResourceSupported.as_bool()),
        atomic64_group_shared = ?o9.map(|o| o.AtomicInt64OnGroupSharedSupported.as_bool()),
        atomic64_heap = ?o11.map(|o| o.AtomicInt64OnDescriptorHeapResourceSupported.as_bool()),
        "host shader capabilities"
    );
}

impl Host {
    fn from_device(
        device: ID3D12Device,
        messages: Option<ID3D12InfoQueue>,
    ) -> windows::core::Result<Host> {
        // SAFETY: plain D3D12 feature query and queue, fence, event and
        // descriptor heap creation.
        unsafe {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            let depth_bounds_supported = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS2,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
                .is_ok()
                && options.DepthBoundsTestSupported.as_bool();
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let rtvs: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: RTV_RING,
                    ..Default::default()
                })?;
            let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            let dsvs: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: RTV_RING,
                    ..Default::default()
                })?;
            let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
            Ok(Host {
                device,
                depth_bounds_supported,
                queue,
                fence,
                submitted: Mutex::new(0),
                event,
                rtvs,
                rtv_size,
                dsvs,
                dsv_size,
                next_dsv: AtomicU32::new(0),
                messages,
                next_rtv: AtomicU32::new(0),
                presenter: Mutex::new(None),
                free_lists: Mutex::default(),
                #[cfg(test)]
                reused_lists: std::sync::atomic::AtomicU64::new(0),
                unified: OnceLock::new(),
            })
        }
    }

    /// Creates an isolated WARP-backed `Host` for unit tests without touching
    /// the process-global host singleton.
    #[cfg(test)]
    pub(crate) fn new_warp() -> Option<Host> {
        let adapter = warp_adapter().ok()?;
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: `adapter` is a verified software WARP adapter.
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()? };
        Host::from_device(device?, None).ok()
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    /// Checks CPU writes -> GPU UAV reads/writes -> CPU reads through two
    /// suballocations of one persistently mapped placed buffer.
    #[test]
    fn placed_custom_heap_shares_memory_between_cpu_and_gpu() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;

        let Some(host) = host() else {
            eprintln!("no D3D12 device; skipped");
            return;
        };
        const SLOT: u64 = UNIFIED_ALIGNMENT;
        const INPUT: u32 = 0x1234_5678;
        const MASK: u32 = 0xA5A5_5A5A;

        let Some(source) = host.unified_buffer(4) else {
            eprintln!("CPU-visible custom placed heaps are not supported; skipped");
            return;
        };
        let destination = host
            .unified_buffer(4)
            .expect("second suballocation failed after the shared heap was created");
        let world_heap = host
            .unified_buffer(174 << 20)
            .expect("arena cannot fit the 174 MiB title heap from #407");

        assert_eq!(world_heap.offset, 2 * SLOT);
        assert_eq!(
            world_heap.gpu.checked_sub(source.gpu),
            Some(world_heap.offset)
        );
        assert_eq!(source.offset % SLOT, 0);
        assert_eq!(destination.offset.checked_sub(source.offset), Some(SLOT));
        assert_eq!(destination.gpu.checked_sub(source.gpu), Some(SLOT));
        assert_eq!(
            (destination.data as usize).checked_sub(source.data as usize),
            Some(SLOT as usize)
        );
        assert_eq!(
            source.gpu.checked_sub(source.offset),
            destination.gpu.checked_sub(destination.offset)
        );
        // SAFETY: both pointers address four-byte ranges within the mapped arena.
        unsafe {
            source.data.cast::<u32>().write_volatile(INPUT);
            destination.data.cast::<u32>().write_volatile(0);
        }

        const SHADER: &str = r#"
#define RS "UAV(u0), UAV(u1)"
RWByteAddressBuffer source : register(u0);
RWByteAddressBuffer destination : register(u1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    destination.Store(0, source.Load(0) ^ 0xA5A55A5A);
}
"#;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and entry-point strings live through compilation;
        // both output pointers belong to this test.
        let compiled = unsafe {
            D3DCompile(
                SHADER.as_ptr().cast(),
                SHADER.len(),
                s!("unified-memory-probe"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(error) = compiled {
            let detail = errors.as_ref().map_or_else(String::new, |blob| {
                // SAFETY: the compiler blob owns the reported byte range.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    )
                };
                String::from_utf8_lossy(bytes).into_owned()
            });
            panic!("D3DCompile failed: {error}; {detail}");
        }
        let code = code.expect("D3DCompile returned no bytecode");
        // SAFETY: the compiler blob owns the byte range; the root signature
        // is embedded in the compiled shader.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            )
        };
        let root: ID3D12RootSignature =
            unsafe { host.device.CreateRootSignature(0, bytes) }.expect("create root signature");
        // SAFETY: the PSO description borrows the live bytecode and root signature.
        let pso: ID3D12PipelineState = unsafe {
            host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })
        }
        .expect("create compute pipeline");

        let list = host.new_list().expect("create command list");
        let activation = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: std::mem::ManuallyDrop::new(None),
                    // SAFETY: this barrier borrows the live resource; ManuallyDrop
                    // prevents releasing the borrowed COM interface.
                    pResourceAfter: std::mem::ManuallyDrop::new(Some(unsafe {
                        std::mem::transmute_copy(&source.resource)
                    })),
                }),
            },
        };
        // SAFETY: this is the first use of the placed resource; the aliasing
        // barrier activates it before the compute shader accesses the arena.
        unsafe {
            list.list.ResourceBarrier(&[activation]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list.SetComputeRootUnorderedAccessView(0, source.gpu);
            list.list.SetComputeRootUnorderedAccessView(1, destination.gpu);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);

        // execute() waits for the queue fence before the CPU reads the UAV.
        // SAFETY: destination remains mapped and contains a four-byte result.
        let result = unsafe { destination.data.cast::<u32>().read_volatile() };
        assert_eq!(result, INPUT ^ MASK);
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    #[test]
    fn buffer_states_transition_between_writes_and_reads() {
        let mut s = BufferStates::default();
        // First uses: the prelude's.
        assert_eq!(s.update(1, 'a', BUFFER_WRITE), None);
        assert_eq!(s.update(2, 'b', BUFFER_READ), None);
        assert_eq!(s.update(1, 'a', BUFFER_WRITE), None);
        assert_eq!(s.update(1, 'a', BUFFER_READ), Some(BUFFER_WRITE));
        assert_eq!(s.update(1, 'a', BUFFER_READ), None);
        assert_eq!(s.update(1, 'a', BUFFER_WRITE), Some(BUFFER_READ));
        let sorted = |mut v: Vec<(char, u32)>| {
            v.sort_unstable();
            v
        };
        let firsts = sorted(s.firsts().map(|(r, f)| (*r, f)).collect());
        assert_eq!(firsts, [('a', BUFFER_WRITE), ('b', BUFFER_READ)]);
        let lasts = sorted(s.lasts().map(|(r, l)| (*r, l)).collect());
        assert_eq!(lasts, [('a', BUFFER_WRITE), ('b', BUFFER_READ)]);
        s.clear();
        assert_eq!(s.firsts().count(), 0);
    }

    /// Set in the child process `in_child` starts.
    const CHILD: &str = "VICTORY_TEST_DEBUG_LAYER_CHILD";

    /// Runs the test `name` of this module alone in a child process and
    /// checks that it passed; false in that child. The debug layer is
    /// process-wide and must be on before the host device is made.
    fn in_child(name: &str) -> bool {
        if std::env::var_os(CHILD).is_some() {
            return false;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([&format!("host::tests::{name}"), "--exact", "--nocapture"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        eprintln!("{text}");
        assert!(out.status.success(), "the child test failed");
        assert!(text.contains("1 passed"), "the child ran no test");
        true
    }

    fn compile(source: &str, entry: windows::core::PCSTR, target: windows::core::PCSTR) -> Vec<u8> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names live through the call; out pointers
        // ours; the blobs' own bytes are read.
        unsafe {
            let made = D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                None,
                None,
                None,
                entry,
                target,
                0,
                0,
                &mut code,
                Some(&mut errors),
            );
            if let Some(e) = &errors {
                let text = std::slice::from_raw_parts(
                    e.GetBufferPointer().cast::<u8>(),
                    e.GetBufferSize(),
                );
                assert!(made.is_ok(), "{}", String::from_utf8_lossy(text));
            }
            made.unwrap();
            let code = code.unwrap();
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
                .to_vec()
        }
    }

    /// A buffer a shader writes and draws read as vertices, as the world's
    /// 174 MB heap is (#199, #264): the argument clamp writes it and
    /// indirect and plain draws read it, in two lists submitted together.
    /// The debug layer checks every barrier's state before against what
    /// the lists did (id 527, which #264 reported); it does not check
    /// vertex buffer states at draws on every system, so this cannot show
    /// #199's vertex buffer in unordered access.
    #[test]
    fn debug_layer_accepts_buffers_written_then_read_as_vertices() {
        use windows::core::s;
        if in_child("debug_layer_accepts_buffers_written_then_read_as_vertices") {
            return;
        }
        let mut layer: Option<ID3D12Debug> = None;
        // SAFETY: plain queries, before any device exists in this process.
        if unsafe { D3D12GetDebugInterface(&mut layer) }.is_err() {
            eprintln!("no D3D12 debug layer; skipped");
            return;
        }
        // SAFETY: as above.
        unsafe { layer.unwrap().EnableDebugLayer() };
        let Some(host) = host() else {
            eprintln!("no host device; skipped");
            return;
        };
        let messages: ID3D12InfoQueue = host.device.cast().unwrap();

        const SOURCE: &str = r#"
            #define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)"
            [RootSignature(RS)] float4 vs(float4 p : POSITION) : SV_Position { return p; }
            [RootSignature(RS)] float4 ps() : SV_Target { return float4(1, 0, 0, 1); }
        "#;
        let (vs, ps) = (
            compile(SOURCE, s!("vs"), s!("vs_5_1")),
            compile(SOURCE, s!("ps"), s!("ps_5_1")),
        );
        let element = D3D12_INPUT_ELEMENT_DESC {
            SemanticName: s!("POSITION"),
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            ..Default::default()
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let stencil = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        // SAFETY: plain object creation from valid descriptions; the
        // shader bytes outlive the calls.
        let (root, pso, target) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &vs).unwrap();
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.as_ptr().cast(),
                    BytecodeLength: vs.len(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps.as_ptr().cast(),
                    BytecodeLength: ps.len(),
                },
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    FrontFace: stencil,
                    BackFace: stencil,
                    ..Default::default()
                },
                InputLayout: D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: &element,
                    NumElements: 1,
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let pso: ID3D12PipelineState = host.device.CreateGraphicsPipelineState(&desc).unwrap();
            let mut target: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: 4,
                        Height: 4,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut target,
                )
                .unwrap();
            (root, pso, target.unwrap())
        };

        // Three vertices at 0, a draw's arguments (3 vertices, 1 instance)
        // at 256: one buffer, as title memory is.
        let (buffer, data) = host.shared_buffer(1 << 16).unwrap();
        let vertices: [f32; 12] = [
            -1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0,
        ];
        // SAFETY: inside the mapping.
        unsafe {
            std::ptr::copy_nonoverlapping(vertices.as_ptr().cast::<u8>(), data, 48);
            std::ptr::copy_nonoverlapping([3u32, 1, 0, 0].as_ptr().cast::<u8>(), data.add(256), 16);
        }
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let rtv = host.rtv(&target, None);
        // SAFETY: one draw argument, no root arguments.
        let signature: ID3D12CommandSignature = unsafe {
            let argument = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
                ..Default::default()
            };
            let mut signature = None;
            host.device
                .CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 16,
                        NumArgumentDescs: 1,
                        pArgumentDescs: &argument,
                        NodeMask: 0,
                    },
                    None::<&ID3D12RootSignature>,
                    &mut signature,
                )
                .unwrap();
            signature.unwrap()
        };
        // The draw state, with the buffer's vertices bound as `draw.rs`
        // binds them.
        let bind = |list: &HostList| {
            list.buffer_state(&buffer, BUFFER_READ);
            // SAFETY: an open list; our objects, alive until the lists ran.
            unsafe {
                let l = &list.list;
                l.SetGraphicsRootSignature(&root);
                l.SetPipelineState(&pso);
                l.OMSetRenderTargets(1, Some(&rtv), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: 4.0,
                    Height: 4.0,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                l.RSSetScissorRects(&[RECT {
                    right: 4,
                    bottom: 4,
                    ..Default::default()
                }]);
                l.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
                l.IASetVertexBuffers(
                    0,
                    Some(&[D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: gpu,
                        SizeInBytes: 48,
                        StrideInBytes: 16,
                    }]),
                );
            }
        };
        let draw = |list: &HostList| {
            bind(list);
            // SAFETY: an open list with the draw state set.
            unsafe { list.list.DrawInstanced(3, 1, 0, 0) };
        };
        // As `draw.rs` executes indirect draws: a title pass wrote the
        // records (unordered access), the clamp reads them into its scratch
        // buffer, then bind and execute from the scratch.
        let indirect = |list: &HostList| {
            list.buffer_state(&buffer, BUFFER_WRITE);
            list.buffer_state(&buffer, BUFFER_READ);
            let bounded = crate::indirect::bound(list, (&buffer, 256), None, (0, 16, 1))
                .expect("the clamp");
            bind(list);
            // SAFETY: an open list with the draw state set; the scratch is
            // kept by the list.
            unsafe { list.list.ExecuteIndirect(&signature, 1, &bounded, 0, None, 0) };
            crate::indirect::done(list, &bounded);
        };
        // A list that draws, draws indirectly, draws; then one that starts
        // with an indirect draw, submitted together.
        let (a, b) = (host.new_list().unwrap(), host.new_list().unwrap());
        draw(&a);
        indirect(&a);
        draw(&a);
        a.close();
        indirect(&b);
        draw(&b);
        b.close();
        // SAFETY: plain queries.
        unsafe { messages.ClearStoredMessages() };
        host.execute(&[&a, &b]);

        let mut errors = Vec::new();
        // SAFETY: each message is read into a buffer of the size the queue
        // reports.
        unsafe {
            for i in 0..messages.GetNumStoredMessages() {
                let mut size = 0;
                if messages.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut bytes = vec![0u64; size.div_ceil(8)];
                let m = bytes.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if messages.GetMessage(i, Some(m), &mut size).is_ok()
                    && (*m).Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0
                {
                    let text =
                        std::slice::from_raw_parts((*m).pDescription, (*m).DescriptionByteLength);
                    errors.push(format!(
                        "id {}: {}",
                        (*m).ID.0,
                        String::from_utf8_lossy(text)
                    ));
                }
            }
        }
        assert!(errors.is_empty(), "debug layer errors: {errors:#?}");
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) depth_bounds: Option<ID3D12GraphicsCommandList1>,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// The states of the host copies of title memory the list uses.
    buffers: Mutex<BufferStates<ID3D12Resource>>,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// Memory handed out for the list's recording, kept until it is finished
    /// when the title reset or dropped the list while a submission of it was
    /// still to run.
    kept_uploads: Mutex<Vec<Vec<u8>>>,
    /// Dynamic descriptor slots kept alongside `kept_uploads` until the pending
    /// submission completes.
    kept_dynamic: Mutex<Vec<(usize, u32, u32)>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Queries begun in the list and not ended yet: host heap, type, index,
    /// and whether a draw since was not run on the host (`missed_draw`).
    queries: Mutex<Vec<(ID3D12QueryHeap, D3D12_QUERY_TYPE, u32, bool)>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

pub(crate) fn warn_depth_bounds() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!("the host GPU has no depth bounds test; depth bounds are ignored"));
}

/// A list of a slow submission that ran for this long or longer is named
/// with its pipelines (every build).
const SLOW_LIST_MS: u64 = 100;

/// Adds the time since its start to the frame's fence-wait total when
/// dropped.
struct Timed(std::time::Instant);

impl Drop for Timed {
    fn drop(&mut self) {
        SUBMITTING[2].fetch_add(self.0.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// Dynamic descriptor slots, given back when dropped.
struct DynamicSlots(Vec<(usize, u32, u32)>);

impl Drop for DynamicSlots {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.0 {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

impl Drop for HostList {
    fn drop(&mut self) {
        self.retire_held();
        self.release_kept();
    }
}

// SAFETY: D3D12 lists and allocators may be used from any thread, one at
// a time: the title records a list on one thread at a time, and a
// submission runs it after its recording ended (a reset while one is to
// run moves recording to a new list). The rest is locked.
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<usize, TrackedTexture>;

struct TrackedTexture {
    first: u32,
    last: u32,
    copy: std::sync::Arc<crate::resource::HostTexture>,
    /// Subresources temporarily in a different state from `last`.
    mips: std::collections::BTreeMap<u32, u32>,
}

impl TrackedTexture {
    fn unify(&mut self, mut record: impl FnMut(&ID3D12Resource, u32, u32, u32)) {
        for (mip, before) in std::mem::take(&mut self.mips) {
            record(&self.copy.texture, mip, before, self.last);
        }
    }
}

/// Every read of a host copy of title memory (vertex, index, constant and
/// shader-resource data, indirect arguments).
pub(crate) const BUFFER_READ: u32 = D3D12_RESOURCE_STATE_GENERIC_READ.0 as u32;
/// Shader writes to a host copy of title memory.
pub(crate) const BUFFER_WRITE: u32 = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;

/// The states of the host copies of title memory (buffers) one list uses,
/// by copy: the state it starts in and the one it is in now. A buffer is in
/// the common state between lists: the list's prelude moves it to its first
/// state and the list moves it back before it closes. A buffer a shader
/// writes and a later draw reads as vertices (#199, #264) is transitioned in
/// between; the implicit promotion from common that it relied on allows
/// either use in a list, not both.
///
/// shortcut: buffers read only through directly indexed heaps are not
/// tracked, and a command that both writes a buffer and reads it as
/// vertices, indices or arguments leaves it in the read state.
pub(crate) struct BufferStates<R> {
    by_copy: std::collections::HashMap<usize, (u32, u32, R)>,
}

impl<R> Default for BufferStates<R> {
    fn default() -> Self {
        BufferStates {
            by_copy: std::collections::HashMap::new(),
        }
    }
}

impl<R> BufferStates<R> {
    /// The list uses `copy` (`resource`) in `wanted`: the state it leaves,
    /// if a transition must be recorded now. A first use records nothing:
    /// the prelude moves the buffer there.
    pub(crate) fn update(&mut self, copy: usize, resource: R, wanted: u32) -> Option<u32> {
        match self.by_copy.entry(copy) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, resource));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// The prelude's transitions: each buffer from common to its first state.
    pub(crate) fn firsts(&self) -> impl Iterator<Item = (&R, u32)> {
        self.by_copy.values().map(|(first, _, r)| (r, *first))
    }

    /// The list's closing transitions: each buffer from its last state back
    /// to common.
    pub(crate) fn lasts(&self) -> impl Iterator<Item = (&R, u32)> {
        self.by_copy.values().map(|(_, last, r)| (r, *last))
    }

    pub(crate) fn clear(&mut self) {
        self.by_copy.clear();
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some()
    })
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_TIMING").is_some())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date and, when dirty, copies
    /// it into the host texture on the list's prelude: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    ),
}

/// A texture-to-buffer readback copied from a mapped readback buffer into
/// title memory once the GPU submission completes.
struct Readback {
    dst: usize,
    src: usize,
    dst_pitch: usize,
    src_pitch: usize,
    lines: usize,
    row_bytes: usize,
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload/readback buffers its GPU commands name, kept until reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    /// Post-submission readbacks from GPU textures into title buffers.
    readbacks: Vec<Readback>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Bound view heaps, including the slots mirrored after recording.
    view_heaps: Vec<std::sync::Arc<crate::descriptor::host::HostHeap>>,
    /// Textures named by aliasing and UAV barriers without a state transition.
    barrier_textures: std::collections::HashMap<usize, std::sync::Arc<crate::resource::HostTexture>>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Per host copy, the longest range refreshed since the last title copy
    /// (address, end): a range inside it is not refreshed again. Root SRVs
    /// and UAVs have no size and refresh to their buffer's end, each from
    /// its own address; without this every such bind queried the write
    /// watch of the rest of the buffer again (the world's heap is 174 MB).
    refreshed_spans: std::collections::HashMap<usize, (usize, usize)>,
    /// Ranges shaders write declared since the last title copy, as
    /// `refreshed`: a repeat would only flush and find the range taken.
    gpu_written: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// A recycled host list (reset and open), or a new one when the pool is
    /// empty.
    pub(crate) fn acquire_list(&self) -> windows::core::Result<HostList> {
        let pooled = self
            .free_lists
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop();
        if let Some(list) = pooled {
            #[cfg(test)]
            self.reused_lists.fetch_add(1, Ordering::Relaxed);
            list.reset();
            return Ok(list);
        }
        self.new_list()
    }

    #[cfg(test)]
    pub(crate) fn reused_lists(&self) -> u64 {
        self.reused_lists.load(Ordering::Relaxed)
    }

    /// Returns a finished, unreferenced host list to the pool up to
    /// [`LIST_POOL_CAP`], releasing any kept upload blocks and dynamic slots
    /// immediately.
    pub(crate) fn recycle_list(&self, list: HostList) {
        // What its submission needed (copies of title memory, texture
        // staging, textures, buffers it named) is not held while the list
        // waits in the pool: up to 64 idle lists kept gigabytes of loose
        // memory copies from being released (2026-11-24). It is retired, not
        // dropped: the submission may still be running (a page fault inside
        // a freed 256x256 texture after pooled lists cleared their holds).
        list.retire_held();
        list.release_kept();
        let mut pool = self.free_lists.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < LIST_POOL_CAP {
            pool.push(list);
        }
    }

    /// The wave sizes the host runs (D3D12 options 1), which translated
    /// compute shaders that declare the console's wave size must fit;
    /// logged once. Unknown (the query fails): any.
    pub(crate) fn target(&self) -> victory_shader::emit::Target {
        static TARGET: OnceLock<victory_shader::emit::Target> = OnceLock::new();
        *TARGET.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the query writes the structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let target = if ok && options.WaveOps.as_bool() && options.WaveLaneCountMin > 0 {
                victory_shader::emit::Target {
                    lanes_min: options.WaveLaneCountMin,
                    lanes_max: options.WaveLaneCountMax,
                }
            } else {
                victory_shader::emit::Target::ANY
            };
            tracing::info!(
                lanes_min = target.lanes_min,
                lanes_max = target.lanes_max,
                "host wave sizes"
            );
            if !target.runs(64) {
                tracing::warn!(
                    "the host has no 64-lane waves: console wave64 compute programs that read                      other lanes do not translate, and their dispatches are dropped"
                );
            }
            target
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            Ok(HostList {
                allocator,
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                buffers: Mutex::default(),
                prelude: (prelude_allocator, prelude),
                kept_uploads: Mutex::default(),
                kept_dynamic: Mutex::default(),
                pipelines: Mutex::default(),
                queries: Mutex::default(),
                budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// Waits until the queue has signalled the fence with `value` (the
    /// caller holds `submitted`, under which the event is waited on).
    fn wait_for(&self, value: u64) {
        let waiting = std::time::Instant::now();
        let _timed = Timed(waiting);
        // SAFETY: the fence and event are ours.
        unsafe {
            match self.fence.SetEventOnCompletion(value, self.event) {
                Ok(()) => {
                    WaitForSingleObject(self.event, INFINITE);
                }
                Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
            }
        }
    }

    /// Runs closed lists and waits for them.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let (back, srvs) = {
            let mut batch = Batch::default();
            // Asks of the write watch in epochs (see `watch::begin_epoch`);
            // our own title writes below end one and start the next.
            crate::watch::begin_epoch();
            for list in runs {
                let uploads = list.run_deferred(&mut batch);
                firsts.push(lists.len());
                let preluding = std::time::Instant::now();
                let prelude = list.prelude(&uploads);
                prep_part(2, preluding);
                if let Some(prelude) = prelude {
                    lists.push(Some(prelude));
                }
                lists.push(list.list.cast().ok());
            }
            let flushing = std::time::Instant::now();
            batch.flush();
            prep_part(3, flushing);
            (batch.copy_back, std::time::Instant::now())
        };
        let _direct_srvs = refresh_direct_srvs(runs);
        prep_part(4, srvs);
        crate::watch::end_epoch();
        let prepared = std::time::Instant::now();
        // Each run (prelude and list) is a submission of its own: the host
        // resets the device when one submission runs for about 2 s, and
        // the world's frames (a second or more over dozens of lists) fit
        // that only list by list. Each run signals the fence; the waits
        // come after the last one, in order, so the time each run ended is
        // known in every build and a slow submission names its slow lists.
        // With `VICTORY_GPU_TIMING` (diagnostics) each run is waited for
        // and timed on its own.
        let groups: Vec<&[Option<ID3D12CommandList>]> = firsts
            .iter()
            .zip(firsts.iter().skip(1).chain([&lists.len()]))
            .map(|(&a, &b)| &lists[a..b])
            .collect();
        #[cfg(feature = "diagnostics")]
        let each = timing();
        #[cfg(not(feature = "diagnostics"))]
        let each = false;
        // (run, fence value) still to wait for.
        let mut pending: Vec<(usize, u64)> = Vec::with_capacity(groups.len());
        for (i, group) in groups.iter().enumerate() {
            let started = std::time::Instant::now();
            // SAFETY: closed lists.
            unsafe { self.queue.ExecuteCommandLists(group) };
            SUBMITTING[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            *submitted += 1;
            // SAFETY: the fence is ours.
            if let Err(e) = unsafe { self.queue.Signal(&self.fence, *submitted) } {
                tracing::warn!(error = %e, "host submission not waited for");
                continue;
            }
            pending.push((i, *submitted));
            if !each {
                continue;
            }
            self.wait_for(*submitted);
            pending.clear();
            #[cfg(feature = "diagnostics")]
            if started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        // When each run ended (a fence wakes us within microseconds), so
        // its GPU time is the gap to the run before it: one queue runs them
        // in order.
        let mut ended: Vec<(usize, std::time::Duration)> = Vec::with_capacity(pending.len());
        let mut last = prepared;
        for (i, value) in pending {
            self.wait_for(value);
            let now = std::time::Instant::now();
            ended.push((i, now - last));
            last = now;
        }
        let _ = &firsts;
        FINISHED.store(this_submission, Ordering::Release);
        bury(this_submission);
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        SUBMITTING[0].fetch_add(1, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
            // The lists that took the time, slowest first, with their
            // pipelines: a list is a run of its own on the queue.
            ended.sort_by(|a, b| b.1.cmp(&a.1));
            for &(i, took) in ended.iter().take(4) {
                if took < std::time::Duration::from_millis(SLOW_LIST_MS) {
                    break;
                }
                let pipelines = runs[i].pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone();
                tracing::warn!(
                    ms = took.as_millis(),
                    list = i,
                    ?pipelines,
                    "slow host list in that submission"
                );
            }
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done.
        let copying = std::time::Instant::now();
        let mut copied_back = 0u64;
        for (shadow, address, len) in &back {
            let (address, len) = (*address, *len);
            copied_back += shadow.copy_back(address, len) as u64;
        }
        COPY_BACK[1].fetch_add(copied_back, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::CopyBackBytes, copied_back);
            add(C::CopyBackUs, copying.elapsed().as_micros() as u64);
        }
        for list in runs {
            for rb in &list.deferred().readbacks {
                let written = if rb.dst_pitch == rb.row_bytes && rb.src_pitch == rb.row_bytes {
                    let bytes = rb.row_bytes * rb.lines;
                    crate::watch::writing(rb.dst, bytes);
                    write_mapped(rb.dst as *mut u8, rb.src as *const u8, bytes);
                    bytes
                } else {
                    let mut n = 0;
                    for line in 0..rb.lines {
                        let to = rb.dst + line * rb.dst_pitch;
                        let from = rb.src + line * rb.src_pitch;
                        crate::watch::writing(to, rb.row_bytes);
                        write_mapped(to as *mut u8, from as *const u8, rb.row_bytes);
                        n += rb.row_bytes;
                    }
                    n
                };
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        // Queues these writes release run their work once no lock is held:
        // it submits lists itself (this lock, the lists' `deferred`).
        let mut woken = crate::fence::Woken::default();
        // Host timestamps the lists resolved, on the title's GPU clock.
        let mut clock = None;
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                let what = match what {
                    crate::queue::Written::HostTimestamp(raw) => match clock.get_or_insert_with(|| self.clock()) {
                        // SAFETY: the list's mapped read-back memory, which
                        // the submission that ran resolved into.
                        Some(c) => crate::queue::Written::Value(c.convert(unsafe { (raw as *const u64).read_volatile() })),
                        None => crate::queue::Written::GpuTimestamp,
                    },
                    what => what,
                };
                woken.add(what.write(at as *mut u8, bytes));
            }
        }
        self.log_messages();
        drop(submitted);
        woken.run();
    }

    /// How host GPU timestamps map onto the title's GPU clock
    /// ([`crate::queue::Written::GpuTimestamp`]): the queue's calibration
    /// of its timestamps against the CPU's performance counter, which
    /// `Instant` reads too.
    fn clock(&self) -> Option<Clock> {
        use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
        let (mut gpu, mut cpu, mut now, mut cpu_frequency) = (0, 0, 0, 0);
        // SAFETY: plain queries into our locals.
        let frequency = unsafe {
            let frequency = self.queue.GetTimestampFrequency().ok()?;
            self.queue.GetClockCalibration(&mut gpu, &mut cpu).ok()?;
            QueryPerformanceCounter(&mut now);
            QueryPerformanceFrequency(&mut cpu_frequency);
            frequency
        };
        let title_now = crate::queue::gpu_clock();
        let ticks = i128::from(crate::queue::TIMESTAMP_FREQUENCY);
        // The title's clock when the host GPU's read `gpu`.
        let since = (i128::from(now) - i128::from(cpu)) * ticks / i128::from(cpu_frequency.max(1));
        (frequency > 0).then_some(Clock { gpu, at: i128::from(title_now) - since, frequency, now: title_now })
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

/// A host GPU timestamp `gpu` taken when the title's GPU clock read `at`,
/// the host's ticks per second, and the title's clock when taken.
struct Clock {
    gpu: u64,
    at: i128,
    frequency: u64,
    now: u64,
}

impl Clock {
    /// A host GPU timestamp on the title's clock: never past the time it
    /// was read, so bottom-of-pipe timestamps written after stay later.
    fn convert(&self, raw: u64) -> u64 {
        let ticks = i128::from(crate::queue::TIMESTAMP_FREQUENCY);
        let since = (i128::from(raw) - i128::from(self.gpu)) * ticks / i128::from(self.frequency);
        (self.at + since).clamp(0, i128::from(self.now)) as u64
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Keeps `blocks` (memory handed out for the recording) until the list
    /// finishes executing and is recycled or dropped.
    pub(crate) fn keep_uploads(&self, blocks: Vec<Vec<u8>>) {
        if !blocks.is_empty() {
            self.kept_uploads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(blocks);
        }
    }

    /// Keeps dynamic descriptor slots alive until this host list finishes
    /// executing and is recycled or dropped.
    pub(crate) fn keep_dynamic(&self, slots: Vec<(usize, u32, u32)>) {
        if !slots.is_empty() {
            self.kept_dynamic
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(slots);
        }
    }

    /// Frees kept upload blocks and returns kept dynamic descriptor slots.
    /// Through [`retire`]: a submission still running may read them.
    fn release_kept(&self) {
        let uploads = std::mem::take(&mut *self.kept_uploads.lock().unwrap_or_else(|e| e.into_inner()));
        if !uploads.is_empty() {
            retire(uploads);
        }
        let dynamic =
            std::mem::take(&mut *self.kept_dynamic.lock().unwrap_or_else(|e| e.into_inner()));
        if !dynamic.is_empty() {
            retire(DynamicSlots(dynamic));
        }
    }

    pub(crate) fn close(&self) {
        // The host closes no list with queries running; their title ends,
        // in a later list, are dropped (`end_query`).
        for (heap, kind, index, _) in self.queries.lock().unwrap_or_else(|e| e.into_inner()).drain(..) {
            // SAFETY: an open list and a query begun in it.
            unsafe { self.list.EndQuery(&heap, kind, index) };
        }
        // The submission prelude and presentation use one state per texture.
        let mut barriers = Vec::new();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        for state in states.values_mut() {
            state.unify(|texture, mip, before, after| {
                barriers.push(transition_subresource(
                    texture,
                    mip,
                    D3D12_RESOURCE_STATES(before as i32),
                    D3D12_RESOURCE_STATES(after as i32),
                ));
            });
        }
        drop(states);
        if !barriers.is_empty() {
            // SAFETY: an open list and resources held by the tracker.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // The buffers it used back to common, where the next list's prelude
        // expects them.
        let back: Vec<D3D12_RESOURCE_BARRIER> = self
            .buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .lasts()
            .map(|(b, last)| {
                transition(
                    b,
                    D3D12_RESOURCE_STATES(last as i32),
                    D3D12_RESOURCE_STATE_COMMON,
                )
            })
            .collect();
        if !back.is_empty() {
            // SAFETY: an open list and buffers the list holds.
            unsafe { self.list.ResourceBarrier(&back) };
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    /// Hands the objects the list holds for its commands (buffers it
    /// named, textures it moved) to [`retire`]: a submission still running
    /// may read them (a page fault inside a freed rectangle records buffer,
    /// 2026-11-24).
    fn retire_held(&self) {
        // The whole recording's holds: buffers it named, texture staging
        // and buffer copies its copies read.
        let deferred = std::mem::take(&mut *self.deferred());
        retire(deferred);
        let buffers = std::mem::take(&mut *self.buffers.lock().unwrap_or_else(|e| e.into_inner()));
        retire(buffers);
        let states = std::mem::take(&mut *self.states.lock().unwrap_or_else(|e| e.into_inner()));
        if !states.is_empty() {
            retire(states);
        }
    }

    pub(crate) fn reset(&self) {
        self.retire_held();
        self.release_kept();
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.buffers.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.queries.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
    }

    /// Records the state changes required by a texture use. The first use
    /// sets the uniform state chosen by the submission prelude.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
        mips: Option<std::ops::Range<u32>>,
        record: impl FnMut(&ID3D12Resource, u32, u32, u32),
    ) {
        let Some(copy) = crate::resource::host_copy(resource) else {
            return;
        };
        self.transition_tracked(&copy, wanted, mips, record);
    }

    /// [`HostList::transition`] for a host copy in hand.
    fn transition_tracked(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
        mips: Option<std::ops::Range<u32>>,
        mut record: impl FnMut(&ID3D12Resource, u32, u32, u32),
    ) {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let state = states
            .entry(std::sync::Arc::as_ptr(copy) as usize)
            .or_insert_with(|| TrackedTexture {
                first: wanted,
                last: wanted,
                copy: copy.clone(),
                mips: Default::default(),
            });
        if let Some(mips) = mips {
            for mip in mips {
                let before = if wanted == state.last {
                    state.mips.remove(&mip)
                } else {
                    state.mips.insert(mip, wanted)
                }
                .unwrap_or(state.last);
                if before != wanted {
                    record(&state.copy.texture, mip, before, wanted);
                }
            }
        } else {
            state.unify(&mut record);
            let before = std::mem::replace(&mut state.last, wanted);
            if before != wanted {
                record(
                    &state.copy.texture,
                    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    before,
                    wanted,
                );
            }
        }
    }

    /// As [`HostList::transition`], for a host copy named directly (one a
    /// title texture reads through another's memory, see
    /// `resource::written_alias`).
    pub(crate) fn transition_copy(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Vec<D3D12_RESOURCE_BARRIER> {
        let mut barriers = Vec::new();
        self.transition_tracked(copy, wanted, None, |texture, mip, before, after| {
            barriers.push(transition_subresource(
                texture,
                mip,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(after as i32),
            ));
        });
        barriers
    }

    /// Records that this list starts and ends with `copy` in `wanted` state
    /// (test helper matching `transition`'s first-use tracking).
    #[cfg(test)]
    pub(crate) fn expect_texture_state(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states
            .entry(std::sync::Arc::as_ptr(copy) as usize)
            .or_insert_with(|| TrackedTexture {
                first: wanted,
                last: wanted,
                copy: copy.clone(),
                mips: Default::default(),
            });
    }

    #[cfg(test)]
    pub(crate) fn has_deferred_work(&self) -> bool {
        !self.deferred().copies.is_empty()
    }

    /// Moves a host copy of title memory to `wanted` ([`BUFFER_READ`] or
    /// [`BUFFER_WRITE`]) in this list: see [`BufferStates`].
    pub(crate) fn buffer_state(&self, buffer: &ID3D12Resource, wanted: u32) {
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(before) = buffers.update(buffer.as_raw() as usize, buffer.clone(), wanted) {
            let barrier = transition(
                buffer,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a buffer the list holds.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// At submission: the list that copies any newly dirtied textures from
    /// their staging buffers into their host textures and moves each texture
    /// this list uses from its real state to the one the list starts with;
    /// the textures' real states become those the list ends with.
    fn prelude(
        &self,
        uploads: &[(
            std::sync::Arc<crate::texture::Staging>,
            std::sync::Arc<crate::resource::HostTexture>,
        )],
    ) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
        let mut pre_barriers = Vec::new();
        for (_, copy) in uploads {
            let real = copy.state.swap(COPY_DEST, Ordering::Relaxed);
            if real != COPY_DEST {
                pre_barriers.push(transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(real as i32),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
            }
        }
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let post_barriers: Vec<D3D12_RESOURCE_BARRIER> = states
            .values()
            .filter_map(|state| {
                let real = state.copy.state.swap(state.last, Ordering::Relaxed);
                (real != state.first).then(|| {
                    transition(
                        &state.copy.texture,
                        D3D12_RESOURCE_STATES(real as i32),
                        D3D12_RESOURCE_STATES(state.first as i32),
                    )
                })
            })
            .chain(buffers.firsts().map(|(b, first)| {
                transition(
                    b,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATES(first as i32),
                )
            }))
            .collect();
        if uploads.is_empty() && post_barriers.is_empty() {
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures and staging buffers are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !pre_barriers.is_empty() {
                    list.ResourceBarrier(&pre_barriers);
                }
                for (staging, copy) in uploads {
                    staging.record_copies(list, &copy.texture);
                }
                if !post_barriers.is_empty() {
                    list.ResourceBarrier(&post_barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Begins query `index` of `heap` (occlusion or statistics, never a
    /// timestamp) unless it is running in the list already. Whether it
    /// began.
    pub(crate) fn begin_query(&self, heap: &ID3D12QueryHeap, kind: u32, index: u32) -> bool {
        let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
        if kind == crate::query::TIMESTAMP || queries.iter().any(|(h, _, i, _)| h == heap && *i == index) {
            return false;
        }
        let kind = D3D12_QUERY_TYPE(kind as i32);
        // SAFETY: an open list and a live heap, kept until the list is reset.
        unsafe { self.list.BeginQuery(heap, kind, index) };
        queries.push((heap.clone(), kind, index, false));
        drop(queries);
        self.keep(heap);
        true
    }

    /// Ends query `index` of `heap`: a timestamp is taken; another query
    /// ends only if it began in this list, as the type it began as. The
    /// type it ended as, if it did and its result is the host's to give: an
    /// occlusion query with a draw inside that the host did not run counts
    /// too few samples, and a title would cull by it what the draw shows.
    pub(crate) fn end_query(&self, heap: &ID3D12QueryHeap, kind: u32, index: u32) -> Option<u32> {
        let (kind, missed) = if kind == crate::query::TIMESTAMP {
            (D3D12_QUERY_TYPE(kind as i32), false)
        } else {
            let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
            let at = queries.iter().position(|(h, _, i, _)| h == heap && *i == index)?;
            let (_, kind, _, missed) = queries.swap_remove(at);
            (kind, missed)
        };
        // SAFETY: an open list and a live heap, kept until the list is reset.
        unsafe { self.list.EndQuery(heap, kind, index) };
        self.keep(heap);
        let occlusion = matches!(kind.0 as u32, crate::query::OCCLUSION | crate::query::BINARY_OCCLUSION);
        (!(missed && occlusion)).then_some(kind.0 as u32)
    }

    /// A draw the title recorded was not run on the host (no pipeline yet,
    /// a shader that does not translate, ...): the queries running around
    /// it did not see it.
    pub(crate) fn missed_draw(&self) {
        for q in self.queries.lock().unwrap_or_else(|e| e.into_inner()).iter_mut() {
            q.3 = true;
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
        deferred.refreshed_spans.clear();
        deferred.gpu_written.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let copy = std::sync::Arc::as_ptr(shadow) as usize;
        let end = address.saturating_add(len);
        match deferred.refreshed_spans.entry(copy) {
            std::collections::hash_map::Entry::Occupied(mut span) => {
                let (from, to) = *span.get();
                if from <= address && end <= to {
                    return;
                }
                if len > to - from {
                    span.insert((address, end));
                }
            }
            std::collections::hash_map::Entry::Vacant(span) => {
                span.insert((address, end));
            }
        }
        if deferred.refreshed.insert((copy, address, len)) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        // Once per range until a title copy: at submission a repeat flushes
        // pending refreshes (done at the next step that needs them anyway)
        // and finds the range already snapshotted and pending.
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.gpu_written.insert(key) {
            deferred
                .copies
                .push(Step::GpuWrite(shadow.clone(), address, len));
        }
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// Copies a GPU texture readback from `src` into title memory at `dst`
    /// once the list has finished running on the host GPU.
    pub(crate) fn defer_readback(
        &self,
        dst: *mut u8,
        src: *const u8,
        dst_pitch: usize,
        src_pitch: usize,
        lines: usize,
        row_bytes: usize,
    ) {
        self.deferred().readbacks.push(Readback {
            dst: dst as usize,
            src: src as usize,
            dst_pitch,
            src_pitch,
            lines,
            row_bytes,
        });
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// A mapped readback buffer of `size` bytes, alive until the list is
    /// reset.
    pub(crate) fn readback(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let (buffer, data) = host.readback_buffer(size)?;
        name(&buffer, || format!("list readback {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data));
        }
        if s.used + len > SCRATCH_CHUNK && s.current < s.chunks.len() {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let (buffer, data) = host.upload_buffer(SCRATCH_CHUNK)?;
            name(&buffer, || "list scratch".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            s.chunks.push((buffer, data as usize, gpu));
            s.used = 0;
        }
        let (_, data, gpu) = s.chunks[s.current];
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    pub(crate) fn use_view_heap(&self, heap: &std::sync::Arc<crate::descriptor::host::HostHeap>) {
        let mut deferred = self.deferred();
        if !deferred
            .view_heaps
            .iter()
            .any(|other| std::sync::Arc::ptr_eq(other, heap))
        {
            deferred.view_heaps.push(heap.clone());
        }
    }

    /// Retains the host texture named by an aliasing or UAV barrier.
    pub(crate) fn keep_barrier_texture(&self, texture: std::sync::Arc<crate::resource::HostTexture>) {
        self.deferred()
            .barrier_textures
            .entry(std::sync::Arc::as_ptr(&texture) as usize)
            .or_insert(texture);
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.refreshed.clear();
        deferred.refreshed_spans.clear();
        deferred.gpu_written.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted (and copies it to `texture` on the prelude only if dirty),
    /// keeping both alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        texture: std::sync::Arc<crate::resource::HostTexture>,
    ) {
        self.deferred()
            .copies
            .push(Step::Texture(staging, texture));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges). Returns the textures
    /// whose staging buffers changed and must be copied onto the prelude.
    fn run_deferred(
        &self,
        batch: &mut Batch,
    ) -> Vec<(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    )> {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        let mut uploads = Vec::new();
        for step in &self.deferred().copies {
            let started = std::time::Instant::now();
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_written();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    // Only the pages written are asked again.
                    crate::watch::uncover(dst, len);
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    let run = matches!(step, Step::Run(_));
                    if run {
                        batch.title_written();
                        crate::watch::end_epoch();
                    }
                    work();
                    if run {
                        crate::watch::begin_epoch();
                    }
                    prep_part(0, started);
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    batch.gpu_write(shadow, *address, *len);
                    SNAPSHOTS[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                    SNAPSHOTS[1].fetch_add(*len as u64, Ordering::Relaxed);
                    continue;
                }
                Step::Texture(staging, copy) => {
                    if batch.texture_due(staging) && staging.refresh() {
                        uploads.push((staging.clone(), copy.clone()));
                    }
                    prep_part(1, started);
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
        uploads
    }
}

#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn test_submission_counters() -> (u64, u64) {
    (
        SUBMISSION_US[0].load(Ordering::Relaxed),
        COPY_BACK[1].load(Ordering::Relaxed),
    )
}

#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn test_submission_copy_bytes() -> u64 {
    PREPARING[1].load(Ordering::Relaxed)
}

/// Refresh live mirrored SRVs after deferred title writes and before
/// GPU work. Retain the resources until the submission has finished.
fn refresh_direct_srvs(
    runs: &[&HostList],
) -> Vec<std::sync::Arc<Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>>> {
    use std::sync::Arc;

    let started = std::time::Instant::now();
    let mut heaps = std::collections::HashSet::new();
    let mut held = Vec::new();
    let mut copied = 0usize;
    let mut merged_count = 0u64;
    #[cfg(feature = "diagnostics")]
    let mut requested = 0usize;
    for list in runs {
        for heap in &list.deferred().view_heaps {
            if !heaps.insert(Arc::as_ptr(heap)) {
                continue;
            }
            let (slots, ranges) = heap.merged_srvs();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::DirectSlots, slots as u64);
            let _ = slots;
            for (shadow, start, end) in ranges.iter() {
                copied += shadow.update(*start, end - start);
                merged_count += 1;
                #[cfg(feature = "diagnostics")]
                {
                    requested += end - start;
                }
            }
            held.push(ranges);
        }
    }
    let elapsed = started.elapsed().as_micros() as u64;
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::DirectRanges, merged_count);
        add(C::DirectRequested, requested as u64);
        add(C::DirectBytes, copied as u64);
        add(C::DirectUs, elapsed);
    }
    let _ = merged_count;
    PREPARING[0].fetch_add(elapsed, Ordering::Relaxed);
    PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
    held
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
    /// Write ranges already declared in this submission (shadow identity, address, bytes).
    write_ranges: std::collections::HashSet<(usize, usize, usize)>,
    /// The same ranges, holding the shadows in first-declaration order for copy-back.
    copy_back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        match self
            .refreshes
            .iter_mut()
            .find(|(s, ..)| std::sync::Arc::ptr_eq(s, shadow))
        {
            Some((_, from, to)) => (*from, *to) = ((*from).min(address), (*to).max(end)),
            None => self.refreshes.push((shadow.clone(), address, end)),
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    fn gpu_write(
        &mut self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        // Even a repeated declaration comes after the preceding refreshes.
        self.flush();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if self.write_ranges.insert(key) {
            shadow.writing(address, len);
            self.copy_back.push((shadow.clone(), address, len));
        }
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
    }

    fn flush(&mut self) {
        for (shadow, from, to) in self.refreshes.drain(..) {
            let started = std::time::Instant::now();
            let copied = shadow.update(from, to - from);
            let us = started.elapsed().as_micros() as u64;
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add, range};
                range(
                    std::sync::Arc::as_ptr(&shadow) as usize,
                    from,
                    to - from,
                    copied,
                    us,
                );
                add(C::ConventionalRanges, 1);
                add(C::ConventionalRequested, (to - from) as u64);
                add(C::ConventionalBytes, copied as u64);
                add(C::ConventionalUs, us);
            }
            PREPARING[0].fetch_add(us, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// The copy runs in the kernel (`ReadProcessMemory`), which reports an
/// unmapped page of the source, or of the destination it writes, instead of
/// faulting. Asking `VirtualQuery` first cost about 2 ms per copy in the
/// title's big pools (it walks the page tables to the region's end), which
/// made copies run at 100-200 MB/s; a title destination is asked only once
/// its copy has failed (asked before each, the world's 80 copies per frame
/// cost 60 ms).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Through [`read`]: the kernel refuses an unmapped destination page as it
/// does a source page. `WriteProcessMemory` makes the same copy, but asks
/// for the destination's region first, at `VirtualQuery`'s cost (see
/// [`copy_mapped`]): 0.35 ms for each piece of the world's copy-back, 39 ms
/// per frame.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let write = |at: usize, n: usize| read(src.wrapping_add(at - dst as usize), at as *mut u8, n);
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        write_mapped(to, from.as_ptr(), 3 * BLOCK);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        write_mapped(unsafe { to.add(100) }, from.as_ptr(), BLOCK - 300);
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        write_mapped(unsafe { to.add(2 * BLOCK) }, from.as_ptr(), BLOCK);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }

    /// A buffer in the readback heap, mapped for reading.
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a readback buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no readback buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    drawn: u64,
    logged: std::time::Instant,
    /// Frame denominator for diagnostics retained between log lines.
    #[cfg(feature = "diagnostics")]
    diagnostic_frames: u64,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            #[cfg(feature = "diagnostics")]
            diagnostic_frames: 0,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        #[cfg(feature = "diagnostics")]
        {
            self.diagnostic_frames += u64::from(self.frames);
        }
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        // Where a frame's CPU time goes, for slow frames: recording
        // (title threads, summed), and of the submissions (one at a
        // time, on the queue's thread) preparing, the driver's submit,
        // the fence wait and copying back.
        let per = |n: u64, scale: f64| n as f64 / scale / frames;
        let [submissions, submit_us, wait_us] =
            [0, 1, 2].map(|i| SUBMITTING[i].swap(0, Ordering::Relaxed));
        let [snapshot_us, snapshot_b] = [0, 1].map(|i| SNAPSHOTS[i].swap(0, Ordering::Relaxed));
        let [asks, ask_us, kernel_asks] = crate::watch::take_asks();
        let recording_us = crate::draw::take_recording_us();
        let [runs_us, texture_check_us, prelude_us, flush_us, srv_us] =
            [0, 1, 2, 3, 4].map(|i| PREP_PARTS[i].swap(0, Ordering::Relaxed));
        let [pso_wait_us, buffer_lock_us, list_lock_us] = crate::draw::take_blocked_us();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect",
                making = crate::pipeline::making()
            );
            tracing::info!(
                "frame CPU split: recording {:.1} ms (blocked: pipeline waits {:.1} ms, \
                 buffer lock {:.1} ms, list lock {:.1} ms); {:.1} submissions: preparing {cpu:.1} ms \
                 (snapshots {:.1} ms for {:.1} MB declared, {:.0} write-watch asks ({:.0} of the kernel) {:.1} ms; \
                 copies {copy_ms:.1} ms {copy_mb:.1} MB, run work {:.1} ms, texture checks {:.1} ms \
                 ({texture_mb:.1} MB), texture uploads recorded {:.1} ms, last refresh {:.1} ms, \
                 indexed SRVs {:.1} ms), submitting \
                 {:.1} ms, waiting {:.1} ms, copying back {back_ms:.1} ms",
                per(recording_us, 1e3),
                per(pso_wait_us, 1e3),
                per(buffer_lock_us, 1e3),
                per(list_lock_us, 1e3),
                per(submissions, 1.0),
                per(snapshot_us, 1e3),
                per(snapshot_b, 1e6),
                per(asks, 1.0),
                per(kernel_asks, 1.0),
                per(ask_us, 1e3),
                per(runs_us, 1e3),
                per(texture_check_us, 1e3),
                per(prelude_us, 1e3),
                per(flush_us, 1e3),
                per(srv_us, 1e3),
                per(submit_us, 1e3),
                per(wait_us, 1e3),
            );
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::log(std::mem::take(&mut self.diagnostic_frames) as f64);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A suballocation in a lazily probed CPU-visible buffer shared with the GPU.
    /// Unsupported adapters or exhausted capacity return None; title allocations
    /// do not use this arena yet.
    // A probe: only its test calls it until title allocations use it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn unified_buffer(&self, size: u64) -> Option<UnifiedBuffer> {
        let arena = self.unified.get_or_init(|| match UnifiedArena::new(&self.device) {
            Ok(arena) => {
                tracing::info!(bytes = UNIFIED_ARENA_SIZE, "CPU/GPU shared buffer arena available");
                Some(arena)
            }
            Err(error) => {
                tracing::info!(error = %error, "CPU/GPU shared buffer arena unavailable");
                None
            }
        });
        arena.as_ref()?.buffer(size)
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    transition_subresource(
        resource,
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        before,
        after,
    )
}

/// Mip indices directly address the first plane of these color targets.
pub(crate) fn color_mip_count(texture: &ID3D12Resource) -> Option<u32> {
    // SAFETY: the host texture is live.
    let desc = unsafe { texture.GetDesc() };
    (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.DepthOrArraySize == 1
        && desc.MipLevels > 1
        && desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET))
    .then_some(u32::from(desc.MipLevels))
}

pub(crate) fn transition_subresource(
    resource: &ID3D12Resource,
    subresource: u32,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: subresource,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }
}
