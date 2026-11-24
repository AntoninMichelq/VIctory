//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, depth bounds, topology, targets) is set on the host
//! list as the title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<(usize, Option<std::ops::Range<u32>>)>,
    depth: usize,
    /// None leaves the host list's default bounds (0..1) in place.
    depth_bounds: Option<(f32, f32)>,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// Root shader views made tables in this list: (title heap, host
    /// buffer, title address, length) → dynamic slot.
    root_srv_slots: std::collections::HashMap<(usize, usize, u64, u64), u32>,
    /// Whether the last [`State::bind`] left a used unordered access view
    /// with nothing behind it (a null view).
    null_uav: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// CPU time (microseconds, summed over the recording threads) spent in
/// draw, dispatch and indirect execution calls since the last
/// [`take_recording_us`]: what recording costs per frame.
static RECORDING_US: AtomicU64 = AtomicU64::new(0);

/// Adds the time since `started` to the recording time.
pub(crate) fn recorded(started: std::time::Instant) {
    RECORDING_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Microseconds spent recording draws and dispatches since the last call.
pub(crate) fn take_recording_us() -> u64 {
    RECORDING_US.swap(0, Ordering::Relaxed)
}

/// Parts of the recording time spent blocked, in microseconds since the
/// last [`take_blocked_us`]: waiting for a pipeline (the per-frame budget),
/// for the title buffers' lock, and for a list's draw state lock.
pub(crate) static BLOCKED_US: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Pipeline waits, buffer-lock waits and list-lock waits since the last
/// call, in microseconds.
pub(crate) fn take_blocked_us() -> [u64; 3] {
    [0, 1, 2].map(|i| BLOCKED_US[i].swap(0, Ordering::Relaxed))
}

/// Locks `m`, adding the time spent waiting for it to [`BLOCKED_US`]`[i]`
/// (nothing when it was free).
pub(crate) fn lock_timed<T>(m: &std::sync::Mutex<T>, i: usize) -> std::sync::MutexGuard<'_, T> {
    match m.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            let started = std::time::Instant::now();
            let g = m.lock().unwrap_or_else(|e| e.into_inner());
            BLOCKED_US[i].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            g
        }
    }
}

/// Groups the host runs per dimension of a dispatch (and of a mesh
/// dispatch). A count past it is undefined behaviour on the host; drivers
/// may run it for seconds and reset the display driver.
const MAX_GROUPS: u32 = 65_535;

/// Whether dispatches that write through a null unordered access view are
/// dropped (the default; `VICTORY_NULL_UAV_DISPATCH=1` runs them). Reads of
/// a null view give zeros and its atomics return zero, so a compute pass
/// that loops on a counter or flag kept there (work queues, spin waits)
/// never sees it move and runs to the translator's loop limit in every
/// lane: seconds of GPU time, and the host resets the display driver
/// (DXGI_ERROR_DEVICE_HUNG, 2026-11-24, in the world on an RX 7600; the
/// device-removed data named a dispatch with two null views in its table).
/// shortcut: what such a pass writes to its other views is lost too.
fn skip_null_uav_dispatches() -> bool {
    static SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SKIP.get_or_init(|| std::env::var_os("VICTORY_NULL_UAV_DISPATCH").is_none_or(|v| v != "1"))
}

/// Logs, once per pipeline, a dispatch dropped for a null unordered access
/// view.
fn null_uav_dropped(pipeline: &str) {
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() < 256 && !seen.iter().any(|s| s == pipeline) {
        seen.push(pipeline.to_owned());
        tracing::warn!(
            pipeline,
            "a dispatch writing through a null unordered access view dropped"
        );
    }
}

/// Whether the host can run a dispatch of `x * y * z` groups.
fn dispatch_fits(x: u32, y: u32, z: u32) -> bool {
    x <= MAX_GROUPS && y <= MAX_GROUPS && z <= MAX_GROUPS
}

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootView {
    Cbv,
    Srv,
    Uav,
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory. Constant buffers refresh up to the D3D12 64 KiB
/// maximum; SRVs and UAVs refresh the full remaining extent of the backing
/// buffer (dirty-page watched) and UAVs also schedule write-back.
/// `None` for a null address or one outside title buffers.
fn root_view(list: &HostList, address: u64, kind: RootView) -> Option<u64> {
    if address == 0 {
        return None;
    }
    let shadow = crate::shadow::find(address, 1)?;
    let gpu = shadow.gpu(address);
    match kind {
        RootView::Cbv => {
            let len = shadow.left(address).min(1 << 16);
            if gpu.is_multiple_of(256) {
                shadow.refresh(list, address, len);
                list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
                return Some(gpu);
            }
            let (gpu, to) = list.scratch(len)?;
            list.defer_fill(to, address as *const u8, len as usize);
            Some(gpu)
        }
        RootView::Srv => {
            if !gpu.is_multiple_of(4) {
                return None;
            }
            let len = shadow.left(address);
            shadow.refresh(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            Some(gpu)
        }
        RootView::Uav => {
            if !gpu.is_multiple_of(4) {
                return None;
            }
            let len = shadow.left(address);
            shadow.refresh(list, address, len);
            shadow.gpu_writes(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
            Some(gpu)
        }
    }
}

/// How far past a mesh program's root SRV it reads. The console's mesh
/// programs (`MS_VS_Transform*`) keep a table of vertex streams at the
/// view's start: dwords whose bits 7-24 are a stream's byte offset from
/// the view (up to 32 MiB), read at that offset plus vertex index times
/// stride (at most 60 bytes, so under 4 MiB more). The largest offset in
/// the first 32 dwords plus 4 MiB covers every stream; dwords that are not
/// offsets only make the copy longer. 0 (the copy ends with the title
/// buffer, as before) when the view is short enough anyway or
/// `VICTORY_MESH_VIEW_REACH=0`.
fn mesh_view_reach(address: u64) -> u64 {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| {
        let on = std::env::var("VICTORY_MESH_VIEW_REACH").map_or(true, |v| v.trim() != "0");
        if !on {
            tracing::info!(
                "mesh root views end with their title buffer (VICTORY_MESH_VIEW_REACH=0)"
            );
        }
        on
    }) {
        return 0;
    }
    const WORDS: usize = 32;
    if address == 0
        || !address.is_multiple_of(4)
        || !crate::texture::committed(address as *const u8, WORDS * 4)
    {
        return 0;
    }
    let far = (0..WORDS)
        .map(|i| {
            // SAFETY: committed title memory, checked just above; the title
            // may write it meanwhile, which only changes the value read.
            let d = unsafe { (address as *const u32).add(i).read_volatile() };
            u64::from((d >> 7) & 0x01ff_fffc)
        })
        .max()
        .unwrap_or(0);
    let reach = far + (4 << 20);
    static SEEN: AtomicU64 = AtomicU64::new(0);
    if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
        tracing::info!(
            address = format_args!("{address:#x}"),
            reach_mb = reach >> 20,
            "mesh root view: stream table reach"
        );
    }
    reach
}

fn emit_uav_barrier(list: &HostList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: an open command list and a global UAV barrier.
    unsafe { list.list.ResourceBarrier(&[barrier]) };
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// Takes the dynamic descriptor slots held by this recording, so a
    /// pending host list can keep them alive until its submission finishes.
    pub(crate) fn take_dynamic(&mut self) -> Vec<(usize, u32, u32)> {
        // What was made in those slots is no longer ours to reuse.
        self.fresh_tables.clear();
        self.root_srv_slots.clear();
        std::mem::take(&mut self.dynamic)
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self
            .vertex
            .iter_mut()
            .skip(start as usize)
            .take(count as usize)
        {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            let desc = record.desc();
            self.targets.push((
                record.resource as usize,
                crate::descriptor::rtv_mips(texture, desc.as_ref()),
            ));
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(texture, desc.as_ref()));
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bound heap holding `handle`.
    /// A title root shader view (`uav`: unordered access view) as a
    /// one-descriptor table in the bound view heap: a raw view to the end
    /// of the title buffer, so reads past it return zero and writes past it
    /// are dropped (a root view reaches whatever host memory is there).
    /// `reach`: how far past `address` the program reads (0: to the end
    /// of the title buffer holding it, as before).
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: u64,
        uav: bool,
        reach: u64,
    ) -> Option<D3D12_GPU_DESCRIPTOR_HANDLE> {
        if address == 0 {
            return None;
        }
        let shadow = if reach > 0 {
            crate::shadow::find_reaching(address, reach)?
        } else {
            crate::shadow::find(address, 1)?
        };
        let gpu = shadow.gpu(address);
        if !gpu.is_multiple_of(4) {
            return None;
        }
        // SAFETY: a live buffer.
        let base = unsafe { shadow.buffer.GetGPUVirtualAddress() };
        // A raw UAV must start 16 bytes into its buffer (the debug layer:
        // "the offset of the first element ... must be a multiple of 16
        // bytes"); the host removes the device for one that does not.
        if uav && !(gpu - base).is_multiple_of(16) {
            once!(
                address = format_args!("{address:#x}"),
                "a root UAV not 16-byte aligned in its copy: no bounded table"
            );
            return None;
        }
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let len = shadow.left(address);
        let elements = u32::try_from(len / 4).ok()?;
        // The top bit of the length tells a UAV from an SRV of the same range.
        let key = (
            std::ptr::from_ref(heap) as usize,
            shadow.buffer.as_raw() as usize,
            address,
            len | u64::from(uav) << 63,
        );
        let first = match self.root_srv_slots.get(&key) {
            Some(&first) => first,
            None => {
                let first = host.dynamic(1)?;
                self.dynamic.push((key.0, first, 1));
                if uav {
                    host.raw_uav_at(first, &shadow.buffer, (gpu - base) / 4, elements);
                } else {
                    host.raw_srv_at(first, &shadow.buffer, (gpu - base) / 4, elements);
                }
                self.root_srv_slots.insert(key, first);
                first
            }
        };
        // A view reaching far needs only what its program reads fresh.
        shadow.refresh(list, address, if reach > 0 { len.min(reach) } else { len });
        if uav {
            shadow.gpu_writes(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
        } else {
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
        }
        Some(host.gpu_handle(first))
    }

    /// A one-descriptor table holding a null raw SRV (reads return 0), for
    /// pointer views whose address is null.
    fn null_srv_table(&mut self, uav: bool) -> Option<D3D12_GPU_DESCRIPTOR_HANDLE> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let key = (
            std::ptr::from_ref(heap) as usize,
            0,
            0,
            u64::from(uav) << 63,
        );
        let first = match self.root_srv_slots.get(&key) {
            Some(&first) => first,
            None => {
                let first = host.dynamic(1)?;
                self.dynamic.push((key.0, first, 1));
                if uav {
                    host.null_raw_uav_at(first);
                } else {
                    host.null_raw_srv_at(first);
                }
                self.root_srv_slots.insert(key, first);
                first
            }
        };
        Some(host.gpu_handle(first))
    }

    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some((min, max));
        if list.depth_bounds.is_none() {
            crate::host::warn_depth_bounds();
        }
        self.apply_depth_bounds(list);
    }

    fn apply_depth_bounds(&self, list: &HostList) {
        if let (Some(list), Some((min, max))) = (&list.depth_bounds, self.depth_bounds) {
            // SAFETY: the host supports depth bounds and the list is open.
            unsafe { list.OMSetDepthBounds(min, max) };
        }
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(
                mesh,
                "a draw or mesh dispatch with another kind of pipeline"
            );
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            // After `bind`: a shader may have written the buffer before.
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for (resource, mips) in &self.targets {
            crate::list::ensure_mips(list, *resource as This, RENDER_TARGET, mips.clone());
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        self.null_uav = false;
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| {
                    let title = crate::descriptor::heap(h as This);
                    let heap = title.host()?;
                    if !title.samplers() {
                        list.use_view_heap(heap);
                    }
                    Some(Some(heap.heap.clone()))
                })
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            // Also reached after the indirect clamp invalidates the pipeline.
            if !compute {
                self.apply_depth_bounds(list);
            }
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        let mut bound_params = 0u64;
        for stage in &p.stages {
            let t = stage.translated();
            let Some(made) = self.tables(list, &t.used, compute) else {
                return false;
            };
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, kind) in [
                (&stage.root_cbvs, RootView::Cbv),
                (&stage.root_srvs, RootView::Srv),
                (&stage.root_uavs, RootView::Uav),
            ] {
                for &(slot, param) in views {
                    // A root view two stages share has one parameter.
                    if param < 64 {
                        if bound_params & (1u64 << param) != 0 {
                            continue;
                        }
                        bound_params |= 1u64 << param;
                    }
                    // A pointer the program writes through is a root UAV.
                    let pointer = slot >= victory_shader::emit::POINTER_SLOT
                        && matches!(kind, RootView::Srv | RootView::Uav);
                    let address = if pointer {
                        // An address the program keeps in root constants.
                        let index = u32::from(slot - victory_shader::emit::POINTER_SLOT);
                        let word = |i: u32| {
                            t.host
                                .user_data
                                .iter()
                                .find(|u| u.index == i)
                                .map(|u| self.user_value(compute, u.slot, u.root, u.dword))
                        };
                        if index >= 64 {
                            // Kept shifted right by 8 in one dword.
                            let Some(x) = word(index - 64) else {
                                once!(slot, "a pointer view without its user data");
                                return false;
                            };
                            u64::from(x) << 8
                        } else {
                            let (Some(lo), Some(hi)) = (word(index), word(index + 1)) else {
                                once!(slot, "a pointer view without its user data");
                                return false;
                            };
                            u64::from(hi) << 32 | u64::from(lo)
                        }
                    } else {
                        let Some(Root::View(address)) =
                            self.roots_ref(compute).get(usize::from(slot))
                        else {
                            once!(
                                slot,
                                ?kind,
                                "a shader reads a root view the title did not set"
                            );
                            return false;
                        };
                        *address
                    };
                    if kind != RootView::Cbv && crate::pipeline::host::root_srv_tables() {
                        let table = if pointer && address == 0 {
                            // A null address the program tests before reading.
                            self.null_srv_table(kind == RootView::Uav)
                        } else {
                            let reach = if kind == RootView::Srv && p.mesh && !pointer {
                                mesh_view_reach(address)
                            } else {
                                0
                            };
                            self.root_srv_table(list, address, kind == RootView::Uav, reach)
                        };
                        let Some(table) = table else {
                            once!(
                                slot,
                                address = format_args!("{address:#x}"),
                                "a root shader view could not be made a bounded table; draw dropped"
                            );
                            return false;
                        };
                        // SAFETY: an open list; the slot views live memory
                        // and is held until the list is done.
                        unsafe {
                            if compute {
                                l.SetComputeRootDescriptorTable(param, table);
                            } else {
                                l.SetGraphicsRootDescriptorTable(param, table);
                            }
                        }
                        continue;
                    }
                    // A null address the program tests before reading: any
                    // buffer will do.
                    let null = (pointer && address == 0)
                        .then(|| list.scratch(256).map(|(gpu, _)| gpu))
                        .flatten();
                    let Some(gpu) = null.or_else(|| root_view(list, address, kind)) else {
                        once!(
                            slot,
                            ?kind,
                            address = format_args!("{address:#x}"),
                            "a shader reads a root view outside title buffers"
                        );
                        return false;
                    };
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, kind) {
                            (false, RootView::Cbv) => {
                                l.SetGraphicsRootConstantBufferView(param, gpu)
                            }
                            (false, RootView::Srv) => {
                                l.SetGraphicsRootShaderResourceView(param, gpu)
                            }
                            (false, RootView::Uav) => {
                                l.SetGraphicsRootUnorderedAccessView(param, gpu)
                            }
                            (true, RootView::Cbv) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, RootView::Srv) => l.SetComputeRootShaderResourceView(param, gpu),
                            (true, RootView::Uav) => {
                                l.SetComputeRootUnorderedAccessView(param, gpu)
                            }
                        }
                    }
                }
            }
            if let Some(param) = stage.stride_table {
                let gpu = (self.heaps[0] != 0)
                    .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                    .flatten()
                    .and_then(|h| h.stride_table_gpu());
                if let Some(gpu) = gpu {
                    // SAFETY: an open list; the stride table buffer lives with the host heap.
                    unsafe {
                        if compute {
                            l.SetComputeRootShaderResourceView(param, gpu);
                        } else {
                            l.SetGraphicsRootShaderResourceView(param, gpu);
                        }
                    }
                } else {
                    once!("a shader reads the runtime stride table without a host view heap");
                }
            }
        }
        true
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        let traced = self.trace_dispatch(x, y, z);
        #[cfg(feature = "diagnostics")]
        let pipe = self.pipeline;
        #[cfg(feature = "diagnostics")]
        let outcome = |why: &str| {
            if traced {
                tracing::debug!(
                    pipeline = format_args!("{pipe:#x}"),
                    "dispatch {x}x{y}x{z} {why}"
                );
            }
        };
        #[cfg(not(feature = "diagnostics"))]
        let outcome = |_: &str| {};
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            outcome("DROPPED (pipeline skipped)");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            outcome("DROPPED (no host pipeline)");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            outcome("DROPPED (graphics pipeline)");
            return;
        }
        if !dispatch_fits(x, y, z) {
            once!(x, y, z, "a dispatch past the host's group limits dropped");
            outcome("DROPPED (past group limits)");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if !self.bind(list, p, true) {
            outcome("DROPPED (bind failed)");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if self.null_uav && skip_null_uav_dispatches() {
            outcome("DROPPED (null UAV)");
            null_uav_dropped(&p.name);
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        if p.has_root_uavs() {
            emit_uav_barrier(list);
        }
        #[cfg(feature = "diagnostics")]
        if traced {
            outcome(&format!("RUN reads {:?}", self.reads(true)));
        }
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether this dispatch is in a listed (captured) frame and debug
    /// logging is on; logs its pipeline and shader hashes (diagnostics).
    #[cfg(feature = "diagnostics")]
    fn trace_dispatch(&self, x: u32, y: u32, z: u32) -> bool {
        if !tracing::enabled!(tracing::Level::DEBUG)
            || !crate::capture::listed(crate::queue::presented() + 1)
        {
            return false;
        }
        let hashes: Vec<String> = crate::pipeline::host_pipeline(self.pipeline as This)
            .map(|p| {
                p.stages
                    .iter()
                    .map(|s| format!("{}", s.shader.hash))
                    .collect()
            })
            .unwrap_or_default();
        tracing::debug!(
            "dispatch {x}x{y}x{z} pipeline {:#x} shaders {hashes:?}",
            self.pipeline
        );
        true
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage
                .root_cbvs
                .iter()
                .chain(&stage.root_srvs)
                .chain(&stage.root_uavs)
            {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        if x > MAX_GROUPS {
            once!(x, "a mesh dispatch past the host's group limit dropped");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    if crate::pipeline::host_pipeline(self.pipeline as This)
                        .is_some_and(|p| p.has_root_uavs())
                    {
                        emit_uav_barrier(list);
                    }
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            list.missed_draw();
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&mut self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        match host.mirror(index, handle, true) {
            Some(view) => readable(list, u, &view),
            None => self.null_uav |= u.kind == ResourceType::Uav,
        }
    }

    /// Gives tables with shader resources or constants slots owned by the
    /// command list. Later title descriptor writes cannot change what a
    /// recorded draw samples. Constants also get 256-byte-aligned storage,
    /// filled at submission (the host wants constant buffers 256-byte
    /// aligned, the console only 16). Identical descriptors share slots
    /// within a list. Returns the fresh tables' new byte offsets by root
    /// parameter, or `None` when the draw must be dropped (a constant table
    /// found no dynamic slots).
    fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        compute: bool,
    ) -> Option<Vec<(u8, u32)>> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| matches!(u.kind, ResourceType::Cbv | ResourceType::Srv))
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA, 2,048 samplers anywhere)
        // are made afresh too: the title's slots there have no host slot of
        // their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            let last = table + u64::from(n - 1) * 32;
            if heap.index(last).is_none() {
                for u in entries {
                    self.table_entry(list, u, compute);
                }
                continue;
            }
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    match crate::descriptor::view(handle) {
                        Some(view) => readable(list, u, &view),
                        None => self.null_uav |= u.kind == ResourceType::Uav,
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // A table holding constants made from the title's slots
                // would view them misaligned: the draw is dropped. Other
                // tables fall back to the title's slots.
                if entries.iter().any(|u| u.kind == ResourceType::Cbv) {
                    once!(
                        n,
                        "a constant table did not fit in command-list slots; draw dropped"
                    );
                    return None;
                }
                once!(n, "descriptor table did not fit in command-list slots");
                for u in entries {
                    self.table_entry(list, u, compute);
                }
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null_kind(index, ResourceType::Cbv, 4);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        if let Some(u) = used {
                            readable(list, u, &view);
                        }
                    }
                    None => {
                        if let Some(u) = used {
                            host.make_null_kind(index, u.kind, u.dwords);
                            once!(slot, "a table entry without a view");
                            self.null_uav |= u.kind == ResourceType::Uav;
                        } else {
                            host.make_null(index);
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        Some(made)
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&(resource, _)| crate::resource::host_resource(resource as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let rects = self.topology == RECTANGLES;
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers (their
            // count checked to fit by `rectangles`).
            let (count, from) = if rects {
                (expanded_vertices(vertices).unwrap_or(0), 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            if crate::pipeline::host_pipeline(self.pipeline as This)
                .is_some_and(|p| p.has_root_uavs())
            {
                emit_uav_barrier(list);
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            list.missed_draw();
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        // The draw runs `count / 3 * 6` vertices, a count the host takes in
        // 32 bits.
        if expanded_vertices(count).is_none() {
            once!(count, "too many rectangle vertices");
            return false;
        }
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let source = rect_source(first, count, v.stride, v.size)
                .filter(|_| v.address != 0)
                .map(|offset| v.address + offset)
                .filter(|&from| crate::shadow::find(from, rects * 3 * stride).is_some());
            let Some(from) = source else {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            };
            // The expanded vertices' view takes its size in 32 bits.
            let Ok(span) = u32::try_from(rects * 6 * stride) else {
                once!(slot, "rectangle vertices too many to expand");
                return false;
            };
            let Some((gpu, to)) = list.scratch(u64::from(span)) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: span,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        if self.topology == RECTANGLES {
            once!("indexed rectangles not drawn");
        }
        let ran = self.topology != RECTANGLES && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            if crate::pipeline::host_pipeline(self.pipeline as This)
                .is_some_and(|p| p.has_root_uavs())
            {
                emit_uav_barrier(list);
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            list.missed_draw();
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// For an indirect draw of rectangles: every per-vertex buffer the
    /// pipeline reads expanded into triangles, and the records rewritten
    /// for them, both on the GPU ([`crate::rects`]). Records before the
    /// pipeline is bound (it replaces it). `None` if this draw cannot be
    /// expanded, with the reason (indexed, unaligned, too many elements...).
    #[allow(clippy::type_complexity)]
    fn indirect_rectangles(
        &self,
        list: &HostList,
        kind: u32,
        bounded: &ID3D12Resource,
        stride: u32,
        max: u32,
    ) -> Result<(ID3D12Resource, Vec<(u32, D3D12_VERTEX_BUFFER_VIEW)>), &'static str> {
        if kind != 0 {
            return Err("indexed");
        }
        let host = crate::host::host().ok_or("no host")?;
        let p = crate::pipeline::host_pipeline(self.pipeline as This).ok_or("no host pipeline")?;
        if p.compute || p.mesh {
            return Err("not a graphics pipeline");
        }
        let mut views = Vec::new();
        for slot in 0..32u32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            if v.address == 0 || v.size == 0 || v.stride == 0 {
                return Err("no vertex buffer bound");
            }
            let shadow = crate::shadow::find(v.address, u64::from(v.size))
                .ok_or("vertices outside title buffers")?;
            shadow.refresh(list, v.address, u64::from(v.size));
            // The expansion binds the source as a root unordered-access view.
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
            let (_, gpu, size) = crate::rects::vertices(
                list,
                host,
                shadow.gpu(v.address),
                u64::from(v.size),
                v.stride,
                &elements,
            )?;
            views.push((
                slot,
                D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: gpu,
                    SizeInBytes: size,
                    StrideInBytes: v.stride,
                },
            ));
        }
        // The clamped records (`crate::indirect::bound`'s scratch, from 0).
        // SAFETY: a live buffer.
        let from = unsafe { bounded.GetGPUVirtualAddress() };
        let records = crate::rects::records(list, host, bounded, from, stride, max)?;
        Ok((records, views))
    }

    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // In a captured frame, indirect draws and dispatches are traced with
        // their outcome like direct dispatches (compute passes such as tiled
        // lighting often run indirect, with arguments a classification pass
        // wrote).
        #[cfg(feature = "diagnostics")]
        let traced = tracing::enabled!(tracing::Level::DEBUG)
            && crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(feature = "diagnostics")]
        if traced {
            let hashes: Vec<String> = crate::pipeline::host_pipeline(self.pipeline as This)
                .map(|p| {
                    p.stages
                        .iter()
                        .map(|s| format!("{}", s.shader.hash))
                        .collect()
                })
                .unwrap_or_default();
            tracing::debug!(
                "indirect kind {kind} max {max} stride {stride} count {} at {arguments:#x} pipeline {:#x} shaders {hashes:?}",
                if count.is_some() { "buffer" } else { "none" },
                self.pipeline
            );
        }
        let ran = self.execute_indirect_inner(list, (kind, stride, max), arguments, count);
        #[cfg(feature = "diagnostics")]
        if traced {
            let outcome = if ran {
                "RUN".to_owned()
            } else if crate::pipeline::host_pipeline(self.pipeline as This).is_none() {
                format!(
                    "DROPPED ({})",
                    crate::pipeline::drop_reason(self.pipeline as This)
                )
            } else {
                "DROPPED (see the why line above)".to_owned()
            };
            tracing::debug!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "indirect kind {kind} {outcome}"
            );
        }
        ran
    }

    fn execute_indirect_inner(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            if tracing::enabled!(tracing::Level::DEBUG)
                && crate::capture::listed(crate::queue::presented() + 1)
            {
                tracing::debug!("indirect kind {kind} why: pipeline skipped");
            }
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| {
            if tracing::enabled!(tracing::Level::DEBUG)
                && crate::capture::listed(crate::queue::presented() + 1)
            {
                tracing::debug!("indirect kind {kind} why: {what}");
            }
            crate::pipeline::note_why(format!("indirect kind {kind}: {what}"))
        };
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            why(&format!(
                "no host pipeline: {}",
                crate::pipeline::drop_reason(self.pipeline as This)
            ));
            #[cfg(not(feature = "diagnostics"))]
            why("no host pipeline (not translated, or still compiling)");
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        let matches_kind = match kind {
            DRAW | INDEXED => !p.compute && !p.mesh,
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // Bounded first: the clamp binds a compute pipeline of its own and
        // copies clamped active records into a scratch buffer at offset 0.
        let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
        // The clamp reads the records and their count: after whatever shader
        // wrote them earlier in the list.
        list.buffer_state(&args.buffer, crate::host::BUFFER_READ);
        if let Some((s, _)) = &counter {
            list.buffer_state(&s.buffer, crate::host::BUFFER_READ);
        }
        let bounded = match crate::indirect::bound_why(
            list,
            (&args.buffer, args.offset(arguments)),
            count_arg,
            (kind, stride, max),
        ) {
            Ok(b) => b,
            Err(_reason) => {
                why(&format!("not bounded: {_reason}"));
                return false;
            }
        };
        // Rectangle lists: their vertices expanded and the records rewritten
        // on the GPU, before the draw's own pipeline is bound.
        let mut rect_args: Option<ID3D12Resource> = None;
        let mut rect_vertices: Vec<(u32, D3D12_VERTEX_BUFFER_VIEW)> = Vec::new();
        if matches!(kind, DRAW | INDEXED) && self.topology == RECTANGLES {
            match self.indirect_rectangles(list, kind, &bounded, stride, max) {
                Ok((a, views)) => {
                    rect_args = Some(a);
                    rect_vertices = views;
                }
                Err(_reason) => {
                    crate::indirect::done(list, &bounded);
                    why(&format!("a rectangle list not expanded: {_reason}"));
                    return false;
                }
            }
        }
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            DRAW if rect_args.is_some() => {
                let ok = self.prepare(list, false);
                if ok {
                    for (slot, view) in &rect_vertices {
                        // SAFETY: an open list; the view names a buffer the list keeps.
                        unsafe { list.list.IASetVertexBuffers(*slot, Some(&[*view])) };
                    }
                }
                ok
            }
            DRAW | INDEXED => self.topology != RECTANGLES && self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => match crate::pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => {
                    if !self.bind(list, p, true) {
                        false
                    } else {
                        let skip = self.null_uav && skip_null_uav_dispatches();
                        if skip {
                            null_uav_dropped(&p.name);
                        }
                        !skip
                    }
                }
                _ => false,
            },
            _ => false,
        };
        if !ran {
            crate::indirect::done(list, &bounded);
            why(&format!(
                "prepare failed (kind {kind}, topology {}, compute {}, mesh {})",
                self.topology, p.compute, p.mesh
            ));
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect kind {kind} max {max} at {arguments:#x}"),
                kind == DISPATCH,
            )
        });
        let arg_buffer = rect_args.as_ref().unwrap_or(&bounded);
        // SAFETY: an open list with everything set; the scratch buffers and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                arg_buffer,
                0,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        crate::indirect::done(list, &bounded);
        if crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.has_root_uavs())
        {
            emit_uav_barrier(list);
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn readable(list: &HostList, u: &UsedResource, view: &View) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_WRITE);
        } else {
            list.buffer_state(&shadow.buffer, crate::host::BUFFER_READ);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, desc), ResourceType::Srv) => {
            let Some(t) = crate::resource::host_resource(*resource as This) else {
                return;
            };
            if crate::texture::title_written(t) {
                // What the GPU wrote at its address in another format, or
                // else the title's memory.
                let alias = crate::resource::written_alias(*resource as This);
                if !alias.is_some_and(|from| {
                    crate::texture::copy_alias(list, &from, *resource as This, t)
                }) {
                    crate::texture::upload(list, *resource as This, t);
                }
            }
            let mips = desc
                .as_ref()
                .and_then(|d| crate::descriptor::host::srv_mips(t, d));
            crate::list::ensure_mips(list, *resource as This, SHADER_RESOURCE, mips);
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!(
                "{}x{} f{} flags {:#x}",
                d.Width, d.Height, d.Format.0, d.Flags.0
            )
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav {t}"),
            None => match crate::descriptor::host::buffer_range(v) {
                Some(range) => format!("uav buffer {range:?}"),
                None => "uav without host resource".into(),
            },
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// The host vertices `vertices` rectangle-list vertices become (six per
/// rectangle), if the count fits in 32 bits.
fn expanded_vertices(vertices: u32) -> Option<u32> {
    (vertices / 3).checked_mul(6)
}

/// The byte offset of vertex `first` in a buffer of `size` bytes holding
/// `stride`-byte vertices, if vertices `first..first + count` all lie in
/// it. Computed wide: `first + count` and the byte offsets overflow 32
/// bits with the values a title may pass.
fn rect_source(first: u32, count: u32, stride: u32, size: u32) -> Option<u64> {
    let stride = u64::from(stride);
    let end = (u64::from(first) + u64::from(count)).checked_mul(stride)?;
    (stride != 0 && end <= u64::from(size)).then(|| u64::from(first) * stride)
}

#[cfg(test)]
mod rect_tests {
    use super::{expanded_vertices, rect_source};

    #[test]
    fn rectangle_vertices_inside_their_buffer() {
        assert_eq!(rect_source(0, 3, 16, 48), Some(0));
        assert_eq!(rect_source(3, 3, 16, 96), Some(48));
        assert_eq!(rect_source(3, 3, 16, 95), None);
        assert_eq!(rect_source(0, 3, 0, 48), None);
    }

    /// `first + count` past 32 bits wrapped to a small count before, which
    /// passed the bounds check and read far outside the buffer.
    #[test]
    fn rectangle_bounds_do_not_wrap() {
        assert_eq!(rect_source(u32::MAX, 3, 16, u32::MAX), None);
        assert_eq!(rect_source(u32::MAX - 1, 3, 1, u32::MAX), None);
        assert_eq!(rect_source(u32::MAX, u32::MAX, u32::MAX, u32::MAX), None);
    }

    #[test]
    fn expanded_counts_fit_32_bits() {
        assert_eq!(expanded_vertices(3), Some(6));
        assert_eq!(expanded_vertices(7), Some(12));
        assert_eq!(expanded_vertices(u32::MAX / 6 * 3), Some(u32::MAX / 6 * 6));
        assert_eq!(expanded_vertices(u32::MAX), None);
    }
}

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
mod group_limit_tests {
    #[test]
    fn dispatches_past_the_host_limit_do_not_fit() {
        assert!(super::dispatch_fits(65_535, 65_535, 65_535));
        assert!(super::dispatch_fits(0, 0, 0));
        assert!(!super::dispatch_fits(65_536, 1, 1));
        assert!(!super::dispatch_fits(1, u32::MAX, 1));
        assert!(!super::dispatch_fits(1, 1, 65_536));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::native::{ComputeIo, StageIo};
    use victory_shader::{
        Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource,
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
        VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    struct Allocation {
        ptr: *mut u8,
    }

    impl Allocation {
        fn watched(len: usize) -> Self {
            // SAFETY: allocates a fresh write-watched virtual range freed in Drop.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!ptr.is_null());
            Self { ptr }
        }

        fn reserved_unmapped(len: usize) -> Self {
            // SAFETY: reserves an uncommitted VA range so accesses are unmapped.
            let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_NOACCESS) }
                .cast::<u8>();
            assert!(!ptr.is_null());
            Self { ptr }
        }

        fn addr(&self) -> u64 {
            self.ptr as u64
        }
    }

    impl Drop for Allocation {
        fn drop(&mut self) {
            // SAFETY: `self.ptr` came from `VirtualAlloc`.
            unsafe {
                VirtualFree(self.ptr.cast(), 0, MEM_RELEASE);
            }
        }
    }

    /// Synthetic GFX10 compute shader reading Root CBV (slot 0, dword 0) and
    /// Root SRV (slot 1, byte offset `+1.5 MiB`), and storing `cbv ^ srv ^
    /// 0xA5A5_5A5A` to Root UAV (slot 2, byte offset `+1.5 MiB`).
    fn build_root_view_cs() -> (victory_shader::emit::HostShader, Vec<UsedResource>) {
        let s_mov_b32 =
            |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32;
        let s_load_dword = |dst: u8, base: u8, byte_off: u32| -> [u32; 2] {
            [
                0x3D << 26 | (dst as u32) << 6 | (base as u32 >> 1),
                125 << 25 | (byte_off & 0x1F_FFFF),
            ]
        };
        let vop1 = |op: u16, vdst: u8, src0: u16| -> u32 {
            0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
        };
        let buffer_load_dword_offen = |vdst: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x0C << 18 | (1 << 12),
                0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdst as u32) << 8 | vaddr as u32,
            ]
        };
        let buffer_store_dword_offen = |vdata: u8, vaddr: u8, srsrc: u8| -> [u32; 2] {
            [
                0x38 << 26 | 0x1C << 18 | (1 << 12),
                0x80 << 24
                    | ((srsrc as u32 >> 2) & 0x1F) << 16
                    | (vdata as u32) << 8
                    | vaddr as u32,
            ]
        };

        let mut words = Vec::new();
        // s[0:1] = CBV 0, s[2:3] = SRV 1, s[4:5] = UAV 2.
        words.extend_from_slice(&s_load_dword(6, 0, 0));
        // SRV descriptor in s[8:11], UAV descriptor in s[12:15].
        words.push(s_mov_b32(8, 2));
        words.push(s_mov_b32(9, 3));
        words.push(s_mov_b32(10, 0xC1));
        words.extend_from_slice(&[s_mov_b32(11, 0xFF), 0x3101_4FAC]);
        words.push(s_mov_b32(12, 4));
        words.push(s_mov_b32(13, 5));
        words.push(s_mov_b32(14, 0xC1));
        words.extend_from_slice(&[s_mov_b32(15, 0xFF), 0x3101_4FAC]);
        let vop2 = |op: u8, vdst: u8, src0: u16, vsrc1: u8| -> u32 {
            (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
        };
        // v0 = 0x18_0000 (+1.5 MiB); v1 = srv[0x18_0000]; v2 = s6 ^ v1 ^ 0xA5A5_5A5A.
        words.extend_from_slice(&[vop1(0x001, 0, 0xFF), 0x0018_0000]);
        words.extend_from_slice(&buffer_load_dword_offen(1, 0, 8));
        words.push(vop2(0x1D, 2, 6, 1)); // v_xor_b32 v2, s6, v1
        words.extend_from_slice(&[vop2(0x1D, 2, 0xFF, 2), 0xA5A5_5A5A]); // v_xor_b32 v2, literal, v2
        words.extend_from_slice(&buffer_store_dword_offen(2, 0, 12));
        words.push(0xBF81_0000); // s_endpgm

        let used = vec![
            UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 4,
                slot: 2,
            },
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 1,
                    user_data: 2,
                    slot: 1,
                },
                RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 2,
                    user_data: 4,
                    slot: 2,
                },
            ],
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 6 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = victory_shader::ir::build(&native.code, false, 8).unwrap();
        let res = victory_shader::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = victory_shader::io::raise(&p, &native).unwrap();
        let st = victory_shader::structure::structure(&p).unwrap();
        let c = Container {
            digest: [0; 16],
            version: (1, 0),
            parts: Vec::new(),
        };
        let shader = victory_shader::emit::emit(
            &c,
            &native,
            &p,
            &st,
            &io,
            &res,
            victory_shader::emit::Target::ANY,
        )
        .unwrap();
        (shader, used)
    }

    #[test]
    fn null_and_unmapped_root_views_are_refused() {
        assert!(crate::shadow::find(0, 1).is_none());
        assert!(crate::shadow::find(0x1000, 64).is_none());
        assert!(crate::shadow::find(u64::MAX - 16, 1).is_none());
        assert!(crate::shadow::find(u64::MAX - (1 << 20), 1).is_none());

        let unmapped = Allocation::reserved_unmapped(8 << 20);
        let mid = unmapped.addr() + (4 << 20);
        assert!(crate::shadow::find(mid, 64).is_none());

        let Some(host) = crate::host::host() else {
            return;
        };
        let list = host.new_list().expect("new_list");
        assert!(root_view(&list, 0, RootView::Cbv).is_none());
        assert!(root_view(&list, 0, RootView::Srv).is_none());
        assert!(root_view(&list, 0, RootView::Uav).is_none());
        assert!(root_view(&list, mid, RootView::Srv).is_none());
        assert!(root_view(&list, mid, RootView::Uav).is_none());
        assert!(root_view(&list, u64::MAX - 16, RootView::Srv).is_none());
        assert!(root_view(&list, u64::MAX - 16, RootView::Uav).is_none());

        list.close();
    }

    #[test]
    fn root_srv_refreshes_dirty_data_past_one_mib() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let len = 2 << 20;
        let alloc = Allocation::watched(len);
        let base = alloc.addr();
        let resource_id = alloc.ptr as usize;
        crate::shadow::register(resource_id, base, len as u64);

        let off_a = 0x18_0000usize; // +1.5 MiB
        let off_b = 0x1C_0000usize; // +1.75 MiB
        // SAFETY: `off_a` and `off_b` lie within the 2 MiB committed allocation.
        unsafe {
            alloc.ptr.add(off_a).cast::<u32>().write(0xCAFE_1234);
        }

        let list = host.new_list().expect("new_list");
        let shadow = crate::shadow::find(base, 4).expect("shadow");
        assert_eq!(
            root_view(&list, base, RootView::Srv),
            Some(shadow.gpu(base))
        );
        list.close();
        host.execute(&[&list]);
        assert_eq!(shadow.read_u32(off_a), 0xCAFE_1234);

        // Second submission mutates another page past +1 MiB and verifies
        // incremental dirty-page refresh updates it too.
        list.reset();
        // SAFETY: within the 2 MiB allocation.
        unsafe {
            alloc.ptr.add(off_b).cast::<u32>().write(0xBEEF_5678);
        }
        assert_eq!(
            root_view(&list, base, RootView::Srv),
            Some(shadow.gpu(base))
        );
        list.close();
        host.execute(&[&list]);
        assert_eq!(shadow.read_u32(off_a), 0xCAFE_1234);
        assert_eq!(shadow.read_u32(off_b), 0xBEEF_5678);

        crate::shadow::unregister(resource_id, base);
    }

    #[test]
    fn dispatch_drops_unset_root_views_and_copies_root_uav_writes_back() {
        let Some(host) = crate::host::host() else {
            return;
        };
        victory_shader::emit::enable_root_uavs_for_tests();
        let (shader, used) = build_root_view_cs();
        let pso = crate::pipeline::test_compute_pipeline(shader, used);

        let unmapped = Allocation::reserved_unmapped(8 << 20);
        let mid_unmapped = unmapped.addr() + (4 << 20);

        let len = 2 << 20;
        let srv_buf = Allocation::watched(len);
        let uav_buf = Allocation::watched(len);
        let cbv_buf = Allocation::watched(64 << 10);
        crate::shadow::register(srv_buf.ptr as usize, srv_buf.addr(), len as u64);
        crate::shadow::register(uav_buf.ptr as usize, uav_buf.addr(), len as u64);
        crate::shadow::register(cbv_buf.ptr as usize, cbv_buf.addr(), 64 << 10);

        let off = 0x18_0000usize; // +1.5 MiB
        // SAFETY: offsets are inside the committed allocations.
        unsafe {
            cbv_buf.ptr.cast::<u32>().write(0x1357_2468);
            srv_buf.ptr.add(off).cast::<u32>().write(0xDEAD_BEEF);
        }

        // A dispatch whose root views the title did not set, set to null or
        // set outside mapped memory is dropped (no fallback buffers: a
        // compute pass spinning on zeros hangs the GPU), and writes nothing.
        let list = host.new_list().expect("new_list");
        let mut state = State::default();
        state.set_pipeline(pso);
        state.set_view(true, 1, 0);
        state.set_view(true, 2, mid_unmapped);
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading the u32 at `+1.5 MiB` in title memory.
        assert_eq!(unsafe { uav_buf.ptr.add(off).cast::<u32>().read() }, 0);

        // Dispatch with all three root views bound (SRV reading +1.5 MiB and
        //    UAV writing +1.5 MiB) and verify full extent refresh + write-back.
        list.reset();
        state.reset(pso);
        state.set_view(true, 0, cbv_buf.addr());
        state.set_view(true, 1, srv_buf.addr());
        state.set_view(true, 2, uav_buf.addr());
        state.dispatch(&list, 1, 1, 1);
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let got = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(got, 0x1357_2468 ^ 0xDEAD_BEEF ^ 0xA5A5_5A5A);

        // Indirect dispatch (`execute_indirect`) with Root UAVs bound also
        //    emits its UAV barrier cleanly and copies updated writes back.
        let args_buf = Allocation::watched(64 << 10);
        crate::shadow::register(args_buf.ptr as usize, args_buf.addr(), 64 << 10);
        // SAFETY: updating CBV[0] and writing [1, 1, 1] dispatch args inside committed allocations.
        unsafe {
            cbv_buf.ptr.cast::<u32>().write(0x2468_ACE0);
            args_buf.ptr.cast::<[u32; 3]>().write([1, 1, 1]);
        }
        list.reset();
        state.reset(pso);
        state.set_view(true, 0, cbv_buf.addr());
        state.set_view(true, 1, srv_buf.addr());
        state.set_view(true, 2, uav_buf.addr());
        assert!(state.execute_indirect(&list, (2, 12, 1), args_buf.addr(), None));
        list.close();
        host.execute(&[&list]);
        // SAFETY: reading back the u32 at `+1.5 MiB` in title memory.
        let indirect_got = unsafe { uav_buf.ptr.add(off).cast::<u32>().read() };
        assert_eq!(indirect_got, 0x2468_ACE0 ^ 0xDEAD_BEEF ^ 0xA5A5_5A5A);
        crate::shadow::unregister(args_buf.ptr as usize, args_buf.addr());

        crate::shadow::unregister(srv_buf.ptr as usize, srv_buf.addr());
        crate::shadow::unregister(uav_buf.ptr as usize, uav_buf.addr());
        crate::shadow::unregister(cbv_buf.ptr as usize, cbv_buf.addr());
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(pso, 0x10);
        assert_eq!(release(pso), 0);
    }
}
