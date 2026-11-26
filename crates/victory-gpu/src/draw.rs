//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
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

/// Where the pass over textures picked by index records its uploads: into
/// a list of the title's, or into lists of the thread that runs it ahead.
pub(crate) enum Into<'a> {
    List(&'a HostList),
    Ahead(&'a mut crate::heap_worker::Batch),
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// While `State::early_textures` runs the texture pass: the
    /// milliseconds and the bytes that run may take, in place of the
    /// pass's own per frame, and whether it may make textures past the
    /// card's room (a bake draw).
    static PASS_BUDGET: std::cell::Cell<Option<(u64, u64, bool)>> = const { std::cell::Cell::new(None) };
}

/// The milliseconds of every frame the pass that uploads textures picked
/// by index takes before a shader that picks by index has asked for it
/// (`VICTORY_EARLY_TEXTURES_MS`, 12; 0: never), and the milliseconds it
/// may take just before a draw of the title's decoration bake
/// (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0: nothing special).
///
/// The title bakes its characters' skin and decorations into textures of
/// its own, once, with a pixel shader that picks its sources by index:
/// whatever was not on the card then was baked as nothing for the run (the
/// player's face, neck and arms flat black in full light, while other
/// pedestrians, which are not baked, had their skin). Ported from
/// fxceless/victory-fork c96b4f2b (by hand, the bake part only); here a
/// bake's pass also makes textures when the card has no room.
fn early_budgets() -> (u64, u64) {
    static MS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        (ms("VICTORY_EARLY_TEXTURES_MS", 12), ms("VICTORY_BAKE_TEXTURES_MS", 1500))
    })
}

/// The pixel shaders of the title's decoration bake (`PS_BakeDecorations_*`),
/// by the first 16 hex digits of their hash.
const BAKE_PIXEL_SHADERS: [&str; 4] =
    ["c0a7fe77b1862e10", "0d1575a396dfe513", "b7cfd8dda91ebb6b", "280967ad158cee0b"];

/// Since the start: early runs of the texture pass, and those of them
/// made for a bake draw.
pub(crate) static EARLY_TEXTURE_PASSES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// The title binds buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes: 3.97
/// GB a frame in 31 submissions, 0.49 s of snapshots and 0.28 s of
/// comparing (run 28). On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW=1`; off
/// by default).
///
/// The title issues its mesh draws one record per call, each from the next
/// slot of the same argument buffer: 3,460 a frame in gameplay, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// the timing run of the lit scene (run 79) a shadow draw of next to
/// nothing cost 14-16 microseconds, 4,126 records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours, until one of these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the title is seen to take, not the stride its
/// command signature names: it draws one record a call, for which the
/// stride means nothing, and the title names 12 bytes (the three
/// words of a mesh dispatch) for records it keeps 32 bytes apart (2,173 of
/// a frame's 3,349 mesh records in run 87's listed frame; 283 are 16
/// apart, which ends a window). Walked by the named stride the window
/// never opened: run 87, with the switch, still clamped 5,068 records a
/// frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// hold garbage in this title (slots of all ones: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_some_and(|v| v == "1"))
}

/// A render target's or depth target's format as a pipeline names it:
/// typed. A target bound without a view description has its texture's
/// format, which may be typeless (a texture also read as another type), and
/// the host refuses a pipeline with a typeless target format. The usual
/// typed member of each family; depth families by their depth format.
fn typed_target(format: u32, depth: bool) -> u32 {
    // `VICTORY_TYPED_TARGETS=0`: the formats as bound (before walls12).
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_TYPED_TARGETS").is_none_or(|v| v != "0")) {
        return format;
    }
    match (format, depth) {
        (1, _) => 2,     // R32G32B32A32 -> FLOAT
        (5, _) => 6,     // R32G32B32 -> FLOAT
        (9, _) => 10,    // R16G16B16A16 -> FLOAT
        (15, _) => 16,   // R32G32 -> FLOAT
        (19, _) => 20,   // R32G8X24 -> D32_FLOAT_S8X24_UINT
        (23, _) => 24,   // R10G10B10A2 -> UNORM
        (27, _) => 28,   // R8G8B8A8 -> UNORM
        (33, _) => 34,   // R16G16 -> FLOAT
        (39, true) => 40, // R32 -> D32_FLOAT
        (39, false) => 41, // R32 -> R32_FLOAT
        (44, _) => 45,   // R24G8 -> D24_UNORM_S8_UINT
        (48, _) => 49,   // R8G8 -> UNORM
        (53, true) => 55, // R16 -> D16_UNORM
        (53, false) => 54, // R16 -> R16_FLOAT
        (60, _) => 61,   // R8 -> UNORM
        (87 | 90, _) => 87, // B8G8R8A8 -> UNORM
        (91, _) => 88,   // B8G8R8X8 -> UNORM
        (f, _) => f,
    }
}

/// What one title list has set, and what its host list was last given.
/// Equality key for render-target/depth bind deduplication (`VICTORY_BIND_CACHE=1`).
type TargetKey = (usize, u32, [u32; 6]);

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
    targets: Vec<usize>,
    depth: usize,
    /// The bound targets' host formats, count and samples: what a
    /// precompiled graphics pipeline is made for
    /// ([`crate::precompiled::Targets`]).
    target_key: crate::precompiled::Targets,
    /// The last answer of [`State::host`]: pipeline, key, host pipeline.
    host_cache: std::cell::Cell<Option<(usize, crate::precompiled::Targets, usize)>>,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth handles, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_handles: Vec<u64>,
    depth_handle: u64,
    targets_set: bool,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    /// The depth bounds set on the host list (`VICTORY_DEPTH_BOUNDS=1`).
    bounds: Option<(f32, f32)>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    /// What the host list has: pipeline, root signature, heaps, VRS rate.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Current `D3D12_SHADING_RATE` set on the host list (`0` = `1X1`).
    host_vrs: u32,
    /// Last render targets and depth bound on the host list (`VICTORY_BIND_CACHE=1`).
    host_targets: Option<(Vec<Option<TargetKey>>, Option<TargetKey>)>,
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list has run, is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

impl State {
    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of the opening cutscene
    /// asks for 12,500 tables; held until the resets they ran the 131,072
    /// slots out, and up to 4,900 draws a frame went without their tables.
    /// shortcut: a list the title runs a second time without a reset then
    /// reads slots others may have taken (logged once).
    pub(crate) fn tables_done(&mut self) {
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a list ran again after its descriptor tables were given back"));
        }
        for (heap, first, n) in self.dynamic.drain(..) {
            crate::descriptor::release_dynamic(heap, first, n);
        }
        self.fresh_tables.clear();
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.tables_gone = true;
    }
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. The title's room at
/// the start of the prologue lost its walls in five runs of seven, and in
/// those runs the player's body and the world outside were cut by one
/// rectangle around the window (run 85, frame 2046): one run with this on
/// says whether the walls are drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(p: Option<&crate::pipeline::host::HostPipeline>) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((parts.next()?.to_owned(), parts.next().unwrap_or("").to_owned()))
    }) else {
        return true;
    };
    let Some(p) = p else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15]) && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);
/// Draws the title asked for, run or dropped (a pipeline not made yet).
static ASKED: AtomicU64 = AtomicU64::new(0);

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
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Whether [`State::heap_buffers`] reuses its per-shadow span plan across
/// frames and avoids periodic 80k-slot full rescans while no buffer resource
/// was released (`VICTORY_PLAN_REUSE=1`, off by default; read per frame so it
/// can be toggled live).
/// Live (`crate::live`).
pub(crate) fn plan_reuse() -> bool {
    static ON: crate::live::Switch = crate::live::Switch::new();
    ON.get("VICTORY_PLAN_REUSE", |v, _| u64::from(v.is_some_and(|v| v.trim() == "1"))) == 1
}

/// Since the last frame-rate line (with `VICTORY_PLAN_REUSE=1`):
/// [0] passes reusing the plan, [1] span compactions from `ranges`,
/// [2] full `buffer_slots` rescans, [3] slots walked,
/// [4] unresolved slots pending a host copy (gauge),
/// [5] merged shadow spans in the plan (gauge).
pub(crate) static PLAN_REUSE_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Compaction interval (frames) when removed/changed slots touched a span
/// boundary so the cached span may be slightly wider than active slots.
const PLAN_LOOSE_FRAMES: u64 = 8;

/// Minimum frames between full `buffer_slots()` rescans when buffer resources
/// are released (`descriptor::buffer_forgotten_epoch`).
const PLAN_FULL_RESCAN_MIN_FRAMES: u64 = 64;

/// Per-heap cached bindless buffer plan for [`State::heap_buffers`].
#[derive(Default)]
struct HeapPlan {
    passes: u64,
    ranges: std::collections::HashMap<u32, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)>,
    /// Buffer slots seen with no host copy yet: retried incrementally instead
    /// of rescanning all ~80k slots every 16 frames.
    unresolved: std::collections::HashSet<u32>,
    /// Merged `(shadow, min_from, max_to)` per `Shadow` pointer, kept across
    /// frames under `VICTORY_PLAN_REUSE=1`.
    spans: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)>,
    /// Whether `spans` may cover slightly more than `ranges` due to a removed
    /// or shrunk boundary slot.
    loose: bool,
    /// Whether `ranges`/`spans` have been seeded by an initial full scan.
    initialized: bool,
    last_compact: u64,
    last_full: u64,
    forgotten_seen: u64,
}

impl HeapPlan {
    fn rebuild_spans(&mut self) {
        self.spans.clear();
        for (shadow, from, to) in self.ranges.values() {
            let key = std::sync::Arc::as_ptr(shadow) as usize;
            let span = self
                .spans
                .entry(key)
                .or_insert_with(|| (shadow.clone(), *from, *to));
            span.1 = span.1.min(*from);
            span.2 = span.2.max(*to);
        }
        self.loose = false;
    }

    fn note_removed_boundary(&mut self, old: &(std::sync::Arc<crate::shadow::Shadow>, u64, u64)) {
        let key = std::sync::Arc::as_ptr(&old.0) as usize;
        if let Some((_, cur_from, cur_to)) = self.spans.get(&key)
            && (old.1 <= *cur_from || old.2 >= *cur_to)
        {
            self.loose = true;
        }
    }

    fn update_slot(
        &mut self,
        index: u32,
        new: Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>,
    ) {
        let old = self.ranges.remove(&index);
        match (old, new) {
            (Some((os, of, ot)), Some((ns, nf, nt)))
                if std::sync::Arc::ptr_eq(&os, &ns) && of == nf && ot == nt =>
            {
                self.ranges.insert(index, (ns, nf, nt));
            }
            (old, Some((ns, nf, nt))) => {
                if let Some(old) = old {
                    self.note_removed_boundary(&old);
                }
                let key = std::sync::Arc::as_ptr(&ns) as usize;
                let span = self
                    .spans
                    .entry(key)
                    .or_insert_with(|| (ns.clone(), nf, nt));
                span.1 = span.1.min(nf);
                span.2 = span.2.max(nt);
                self.ranges.insert(index, (ns, nf, nt));
            }
            (Some(old), None) => {
                self.note_removed_boundary(&old);
            }
            (None, None) => {}
        }
    }
}

/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);
/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);
/// Slots given up for lack of room, since the start.
pub(crate) static HEAP_TEXTURES_EVICTED: AtomicU64 = AtomicU64::new(0);
/// Slots the pass made with a small host copy while the card was full
/// (`VICTORY_EVICT_LESS=1`), since the start.
pub(crate) static HEAP_TEXTURES_SMALL: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_EVICT_LESS` (`=1` on; off by default, as before): while the card
/// is full, the texture pass gives up slots by the bytes it must free, at
/// most once every `VICTORY_EVICT_LESS_MS` (6000) ms, only where that frees a
/// host copy, least lately used first
/// (`crate::descriptor::host::HostHeap::texture_slots_evict_less`); and the
/// slots it would have left waiting get a small host copy of the texture
/// (its levels from `VICTORY_EVICT_LESS_SIZE`, 256, texels down:
/// `crate::resource::shrinking`) while the card is under
/// `VICTORY_EVICT_LESS_MAX` (95) percent of its budget. Before, every pass
/// that found the card full gave up the 256 oldest made slots, whether that
/// freed anything or not, while their copies took 5 s idle to go: 33,000
/// to 65,000 slots grey or black in five minutes of the common room.
/// Live (`crate::live`): `VICTORY_EVICT_LESS` (its `_MS` too) is read by
/// each pass; `_SIZE` and `_MAX` once.
fn evict_less() -> Option<u64> {
    static ON: crate::live::Switch = crate::live::Switch::new();
    static MS: crate::live::Switch = crate::live::Switch::new();
    let on = ON.get("VICTORY_EVICT_LESS", |v, _| u64::from(v.is_some_and(|v| v.trim() == "1")));
    if on == 0 {
        return None;
    }
    Some(MS.get("VICTORY_EVICT_LESS_MS", |v, _| {
        let ms = v.and_then(|v| v.trim().parse().ok()).unwrap_or(6000u64).max(500);
        tracing::info!(every_ms = ms, "VICTORY_EVICT_LESS=1: slots given up by bytes, least used first; small copies while the card is full");
        ms
    }))
}
/// Slots made again with a lowest level of detail (a texture uploaded in
/// part) or without it again, since the start ([`partly_clamp`]).
pub(crate) static HEAP_TEXTURES_CLAMPED: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_PARTLY_CLAMP` (on; `=0` off): a texture streaming in that was
/// uploaded in part is sampled only from its first level all there
/// (`crate::texture::first_whole_level`), not from levels uploaded from
/// unmapped memory (zeros). About 1,700 slots stayed uploaded in part in
/// the common room, where characters and posters drew black and chrome.
fn partly_clamp() -> bool {
    // Live (`crate::live`): for the slots made again after a change.
    static ON: crate::live::Switch = crate::live::Switch::new();
    ON.get("VICTORY_PARTLY_CLAMP", |v, _| u64::from(v.is_none_or(|v| v.trim() != "0"))) == 1
}
/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Why the draw or dispatch being prepared on this thread cannot run,
    /// for the dropped draws' report ([`crate::pipeline::tally_drop`]);
    /// empty when the pipeline itself is missing.
    static DROP_WHY: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Notes why the draw being prepared is dropped; always `false`.
fn drop_why(why: &'static str) -> bool {
    DROP_WHY.with(|w| w.set(why));
    false
}

/// Counts a dropped draw or dispatch of `pipeline` by its reason, the one
/// [`drop_why`] noted, else the pipeline's own state.
fn tally_drop(pipeline: usize) {
    let why = DROP_WHY.with(|w| w.replace(""));
    crate::pipeline::tally_drop(pipeline as This, why);
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

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB
/// (cut at their buffer's end), other views as all that is left of their
/// buffer. (They were taken as 1 MiB: 1,452 of the 2,021 mesh vertex
/// streams of a cutscene frame reach further, and what lay past the first
/// megabyte was never brought up to date - meshes without bodies, exploded
/// triangles. `VICTORY_ROOT_VIEW_MB` sets a limit again.)
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv { limit } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some(gpu);
    }
    list.fill_scratch(address as *const u8, len)
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

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
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
        // (A pipeline let go and another made at its address: not this.)
        self.host_cache.set(None);
    }

    /// The bound pipeline's host pipeline, for the targets bound and the
    /// topology set ([`crate::pipeline::host_pipeline`]). The last answer
    /// is kept: a draw asks two or three times, and a precompiled
    /// pipeline's lookup takes a lock.
    fn host<'a>(&self) -> Option<&'a crate::pipeline::host::HostPipeline> {
        let targets = self.target_key.drawing(self.topology);
        // `VICTORY_HOST_CACHE=0`: every ask goes to the pipeline (before walls12).
        static CACHE_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let cache_on = *CACHE_ON.get_or_init(|| std::env::var_os("VICTORY_HOST_CACHE").is_none_or(|v| v != "0"));
        if !cache_on {
            return crate::pipeline::host_pipeline(self.pipeline as This, &targets);
        }
        if let Some((pipeline, key, made)) = self.host_cache.get()
            && pipeline == self.pipeline
            && key == targets
        {
            // SAFETY: host pipelines live as long as the process (made
            // ones are never let go).
            return Some(unsafe { &*(made as *const crate::pipeline::host::HostPipeline) });
        }
        let made = crate::pipeline::host_pipeline(self.pipeline as This, &targets)?;
        self.host_cache.set(Some((self.pipeline, targets, std::ptr::from_ref(made) as usize)));
        Some(made)
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
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
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
        self.target_handles.clear();
        self.target_handles.extend_from_slice(handles);
        self.depth_handle = depth;
        self.targets_set = true;
        let cache = crate::host::bind_cache().then(|| {
            let rtv_keys = handles
                .iter()
                .map(|&h| crate::descriptor::rtv_record(h).map(|r| r.key()))
                .collect();
            let dsv_key = crate::descriptor::dsv_record(depth).map(|r| r.key());
            (rtv_keys, dsv_key)
        });
        if cache.is_some() && self.host_targets == cache {
            return;
        }
        self.targets.clear();
        let mut rtvs = Vec::new();
        // Formats of the views bound and the samples of their textures.
        let mut formats = Vec::new();
        let mut samples = 1;
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    formats.push(desc.Format.0 as u32);
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            let desc = record.desc();
            // SAFETY: a live host texture.
            let texture_desc = unsafe { texture.GetDesc() };
            samples = texture_desc.SampleDesc.Count;
            formats.push(typed_target(
                desc.map(|d| d.Format.0).filter(|&f| f != 0).unwrap_or(texture_desc.Format.0) as u32,
                false,
            ));
            rtvs.push(host.rtv(Some(texture), desc.as_ref()));
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let mut depth_format = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            let desc = record.depth_desc();
            // SAFETY: a live host texture.
            let texture_desc = unsafe { texture.GetDesc() };
            samples = texture_desc.SampleDesc.Count;
            depth_format = typed_target(
                desc.map(|d| d.Format.0).filter(|&f| f != 0).unwrap_or(texture_desc.Format.0) as u32,
                true,
            );
            self.depth = record.resource as usize;
            self.depth_state = depth_state(texture, record.depth_desc().as_ref());
            if crate::resource::depth_and_unordered(record.resource) {
                crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, desc.as_ref()))
        });
        self.target_key = crate::precompiled::Targets::new(&formats, depth_format, samples);
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
        self.host_targets = cache;
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend = factor.copied();
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        crate::draw::stencil_ref(list, reference);
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        let bounds = depth_bounds_for(min, max);
        self.bounds = Some(bounds);
        depth_bounds(list, bounds);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    /// shortcut: the render target views are made again from the handles'
    /// descriptors as they are now, not as they were when the title set
    /// the targets (a descriptor rewritten in between names another view).
    /// Others recorded into the host list (`crate::external`): its heaps,
    /// root signatures and pipeline are set again before the next draw or
    /// dispatch.
    pub(crate) fn host_state_lost(&mut self) {
        // No title heaps (0) differ from what the host list has, too.
        // `u32::MAX` for VRS: a compute dispatch (e.g. `ambient`) does not reset
        // the graphics shading rate on the open command list, so the next draw
        // must re-emit `RSSetShadingRate` even when it wants `1x1`.
        (self.host_pipeline, self.host_roots, self.host_heaps, self.host_vrs) =
            (0, [0; 2], [usize::MAX; 2], u32::MAX);
        self.host_targets = None;
    }

    pub(crate) fn replay(&mut self, list: &HostList) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        self.host_vrs = 0;
        self.host_targets = None;
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        if self.targets_set {
            let handles = std::mem::take(&mut self.target_handles);
            self.set_targets(list, &handles, self.depth_handle);
        }
        if !self.viewports.is_empty() {
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            crate::draw::stencil_ref(list, reference);
        }
        if let Some(bounds) = self.bounds {
            depth_bounds(list, bounds);
        }
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
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
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

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{
            USER_MEMORY_BASE, USER_TABLE_ADDRESS, USER_VIEW_ADDRESS, USER_VIEW_BIAS,
        };
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        if kind == USER_TABLE_ADDRESS || kind == USER_TABLE_ADDRESS + 1 {
            // The title address of the descriptor table the shader reads
            // dwords of by a computed offset, through title memory.
            let table = if dword == 1 {
                match self.heaps[0] {
                    0 => 0,
                    h => crate::descriptor::heap(h as This).start,
                }
            } else {
                match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::Table(table)) => table,
                    _ => 0,
                }
            };
            return if kind == USER_TABLE_ADDRESS { table as u32 } else { (table >> 32) as u32 };
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    ((s.gpu(address) - base) & 15) as u32
                }),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        // The title address of the buffer view at that dword of the table.
        let address = match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4)
                .and_then(|view| crate::descriptor::host::buffer_range(&view))
                .map_or(0, |range| range.0),
            _ => 0,
        };
        if address == 0 {
            once!(slot, dword, "a shader takes the address of a buffer view that is not there: it reads 0");
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages of the root views the list has set made ready.
    fn memory_ready(&self, list: &HostList, compute: bool, writes: bool) -> Option<crate::memory::Ready> {
        let heap = match self.heaps[0] {
            0 => None,
            h => crate::descriptor::heap(h as This).host(),
        }?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        for root in self.roots_ref(compute) {
            match root {
                Root::View(address) => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        crate::memory::ready(list, heap, &addresses, &constants, writes)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches,
    /// timed (`VICTORY_PREP_STATS`).
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        let _timed = crate::prepstats::Timed::new(crate::prepstats::P::DrawPrepare);
        self.prepare_untimed(list, indexed, mesh)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_untimed(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = self.host() else {
            return false;
        };
        // `VICTORY_QUALITY_STATS`: the shaders the textures read next are for.
        crate::quality::draw_shaders(|| p.stages.last().map(|s| s.shader.hash.clone()));
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return drop_why("a draw or mesh dispatch with another kind of pipeline");
        }
        if p.rects && self.topology != RECTANGLES {
            // Its geometry stage would turn each triangle into a rectangle.
            once!("a rectangle pipeline drawn with another topology");
        }
        if !self.bind(list, p, false) {
            return false;
        }
        self.apply_vrs(list, &p.name);
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        // A vertex buffer the input layout reads that is not set or has no
        // host copy: the draw still runs, reading zeros there (reported
        // with the dropped draws, `crate::pipeline::tally_drop`).
        let mut unbound = false;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                unbound = true;
                continue;
            }
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                unbound = true;
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
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
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                return drop_why("an index buffer without a host copy");
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        if unbound {
            crate::pipeline::tally_drop(self.pipeline as This, crate::pipeline::DRAWN_UNBOUND);
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(self.depth as This, false, true);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a root SRV it reads
    /// could not be given a view.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let l = &list.list;
        // Textures picked by index, before anything asks for them.
        self.early_textures(list, p, compute);
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        let root = p.root.as_raw() as usize;
        let root_changed = self.host_roots[usize::from(compute)] != root;
        if root_changed {
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
        let pso = p.pso.as_raw() as usize;
        let pipeline_changed = self.host_pipeline != pso;
        if pipeline_changed {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = pso;
        }
        if pipeline_changed
            || ((crate::host::tex_upload_pool() || crate::host::bind_cache()) && root_changed)
        {
            list.keep_pipeline(p);
        }
        let _aliases = crate::texture::HoldAliases::new(list);
        for stage in &p.stages {
            let t = stage.translated();
            STORES.set(t.stores);
            let made = self.tables(list, &t.used, &t.host.root_indexed_views, compute);
            STORES.set(true);
            if made.iter().any(|m| m.1 == u32::MAX) {
                once!("a draw whose descriptor table found no free slots is dropped");
                return drop_why("a descriptor table without free slots");
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards.
            let writes = (t.stores || !write_skip())
                && !crate::pipeline::builder_quarantined(self.pipeline as This);
            // Title memory by address: what the shader finds it by.
            let memory = if t.host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY) {
                let Some(ready) = self.memory_ready(list, compute, writes) else {
                    once!("a shader that reads title memory by address without a view heap to hold its views: the draw is dropped");
                    return drop_why("title memory by address without a view heap");
                };
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
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
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address.
                        if let Some(m) = &memory {
                            // SAFETY: an open list; the directory lives as
                            // long as the process and the list holds it.
                            unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, m.directory);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, m.directory);
                                }
                            }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        match self.heap_buffers(list) {
                            // SAFETY: an open list; the table lives as long
                            // as the heap the list holds.
                            Some(gpu) => unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, gpu);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, gpu);
                                }
                            },
                            None => once!("a shader picks buffers from a view heap that is not set"),
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return false;
                        }
                        continue;
                    };
                    #[cfg(feature = "diagnostics")]
                    crate::trace::view(slot, if srv { "srv" } else { "cbv" }, address);
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv
                        && t.host.user_data.iter().any(|u| {
                            u.slot == slot
                                && u.root == RootType::Other(victory_shader::emit::USER_VIEW_BIAS)
                        });
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return false;
                        }
                        continue;
                    }
                    let Some(mut gpu) = root_view(list, address, srv) else {
                        continue;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where shaders note the pages they found empty.
                    if let Some(m) = &memory {
                        // SAFETY: an open list; the buffer lives as long as
                        // the process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, m.faults);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, m.faults);
                            }
                        }
                    }
                    continue;
                }
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes through a root view the title did not set");
                        None
                    }
                };
                #[cfg(feature = "diagnostics")]
                if let Some(a) = address {
                    crate::trace::view(slot, "uav", a);
                }
                if !self.root_uav(list, address, param, compute, writes) {
                    return false;
                }
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        true
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        let ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            let len = s.left(a).min(limit);
            s.refresh(list, a, len);
            if writes {
                s.gpu_writes(list, a, len);
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                return drop_why("a root UAV with nothing to bind");
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            // SAFETY: a live buffer.
            let base = unsafe { s.buffer.GetGPUVirtualAddress() };
            let gpu = base + ((s.gpu(a) - base) & !15);
            // SAFETY: an open list; the address is in a live buffer.
            unsafe {
                if compute {
                    list.list.SetComputeRootUnorderedAccessView(param, gpu);
                } else {
                    list.list.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return drop_why("a root UAV without a view heap");
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return drop_why("a root UAV without a free slot");
                };
                self.dynamic.push((key.0, index, 1));
                if let Some((a, s)) = &shadow {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    let offset = s.gpu(*a) - base;
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(index, &s.buffer, offset - bias, s.left(*a).min(limit) + bias);
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            ready(*a, s);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return drop_why("a root SRV without a view heap");
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return drop_why("a root SRV without a free slot");
                };
                self.dynamic.push((key.0, index, 1));
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            s.refresh(list, *a, s.left(*a));
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// shortcut: the buffers are read, not written: what such a shader
    /// writes through a picked unordered access view stays in the host
    /// copy. A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&self, list: &HostList) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // One key per list, as textures have: no title resource is there.
        const ONCE: usize = usize::MAX - 0x5e;
        if !list.first_use(ONCE) {
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // The pass itself once per frame: per list it ran up to 51 times a
        // frame over 81,000 views and 4,100 host copies, 18 s of every
        // frame (2026-11-24, run 11).
        // shortcut: what the title writes into such buffers between lists
        // of one frame is seen a frame late.
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) == frame {
            return Some(gpu);
        }
        let started = std::time::Instant::now();
        let reuse = plan_reuse();
        // Per host heap: the host copy and range each slot's buffer view
        // reads, kept from pass to pass. Without `VICTORY_PLAN_REUSE=1`, all
        // slots are rescanned every 16 passes and `spans` is rebuilt from all
        // `ranges` every pass. With `VICTORY_PLAN_REUSE=1`, `spans` is kept
        // incrementally across frames, unresolved slots are retried on their
        // own, and full rescans are gated by `descriptor::buffer_forgotten_epoch`.
        static RANGES: std::sync::Mutex<Option<std::collections::HashMap<usize, HeapPlan>>> =
            std::sync::Mutex::new(None);
        let mut kept = RANGES.lock().unwrap_or_else(|e| e.into_inner());
        let plan = kept
            .get_or_insert_with(Default::default)
            .entry(std::ptr::from_ref(host) as usize)
            .or_default();
        let changed = host.buffer_changes();
        let forgotten = crate::descriptor::buffer_forgotten_epoch();
        let full_rescan = if reuse {
            !plan.initialized
                || (forgotten != plan.forgotten_seen
                    && frame >= plan.last_full.saturating_add(PLAN_FULL_RESCAN_MIN_FRAMES))
        } else {
            plan.initialized = false;
            plan.passes.is_multiple_of(16)
        };
        let slots: Vec<u32> = if full_rescan {
            plan.ranges.clear();
            plan.unresolved.clear();
            plan.spans.clear();
            plan.loose = false;
            plan.last_full = frame;
            plan.forgotten_seen = forgotten;
            host.buffer_slots()
        } else if reuse && !plan.unresolved.is_empty() {
            let mut merged = changed;
            for &idx in &plan.unresolved {
                if !merged.contains(&idx) {
                    merged.push(idx);
                }
            }
            merged
        } else {
            changed
        };
        plan.passes += 1;
        // Whether the device was there before the pass: a view that removes
        // it is named below.
        static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut alive = !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
        for (n, &index) in slots.iter().enumerate() {
            let Some(handle) = heap.handle(index) else {
                plan.unresolved.remove(&index);
                if reuse && !full_rescan {
                    plan.update_slot(index, None);
                } else {
                    plan.ranges.remove(&index);
                }
                continue;
            };
            let Some(view) = host.mirror(index, handle, true) else {
                plan.unresolved.remove(&index);
                if reuse && !full_rescan {
                    plan.update_slot(index, None);
                } else {
                    plan.ranges.remove(&index);
                }
                continue;
            };
            // Every 64 views while the device lives (a cheap query).
            if alive
                && n % 64 == 63
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                alive = false;
                GONE.store(true, Ordering::Relaxed);
                let last: Vec<String> = slots[n - 63..=n]
                    .iter()
                    .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                    .map(|s| s.chars().take(200).collect())
                    .collect();
                tracing::error!(why, slot = index, "the host device went while buffer views of the heap were made; one of: {last:#?}");
                crate::host::log_removal_context();
            }
            // The copy the slot shows (`buffer_copy`), not the one the
            // view's range alone would find.
            let unordered = matches!(view, crate::descriptor::View::Unordered(..));
            let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(&view) else {
                if reuse {
                    plan.unresolved.insert(index);
                }
                if reuse && !full_rescan {
                    plan.update_slot(index, None);
                } else {
                    plan.ranges.remove(&index);
                }
                continue;
            };
            // Shaders that pick it may write it, and nothing copies that
            // back (see above): `shadow::recopy_tick` must leave it alone.
            if unordered {
                shadow.mark_gpu_written();
            }
            #[cfg(feature = "diagnostics")]
            if host.shows(index, &shadow.buffer) == Some(false) {
                HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
            }
            plan.unresolved.remove(&index);
            if reuse && !full_rescan {
                plan.update_slot(index, Some((shadow, address, address + len)));
            } else {
                plan.ranges.insert(index, (shadow, address, address + len));
            }
        }
        let slots_done = std::time::Instant::now();
        HEAP_SPLIT_US[0].fetch_add((slots_done - started).as_micros() as u64, Ordering::Relaxed);
        let views = plan.ranges.len() as u32;
        // Per host copy, the span its picked views cover.
        let spans_list: Vec<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> = if reuse {
            if full_rescan || !plan.initialized {
                plan.rebuild_spans();
                plan.initialized = true;
                plan.last_compact = frame;
                PLAN_REUSE_STATS[2].fetch_add(1, Ordering::Relaxed);
            } else if plan.loose && frame >= plan.last_compact.saturating_add(PLAN_LOOSE_FRAMES) {
                plan.rebuild_spans();
                plan.last_compact = frame;
                PLAN_REUSE_STATS[1].fetch_add(1, Ordering::Relaxed);
            } else {
                PLAN_REUSE_STATS[0].fetch_add(1, Ordering::Relaxed);
            }
            PLAN_REUSE_STATS[3].fetch_add(slots.len() as u64, Ordering::Relaxed);
            PLAN_REUSE_STATS[4].store(plan.unresolved.len() as u64, Ordering::Relaxed);
            PLAN_REUSE_STATS[5].store(plan.spans.len() as u64, Ordering::Relaxed);
            plan.spans.values().cloned().collect()
        } else {
            let mut spans: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)> =
                std::collections::HashMap::new();
            for (shadow, from, to) in plan.ranges.values() {
                let key = std::sync::Arc::as_ptr(shadow) as usize;
                let span = spans.entry(key).or_insert_with(|| (shadow.clone(), *from, *to));
                span.1 = span.1.min(*from);
                span.2 = span.2.max(*to);
            }
            spans.into_values().collect()
        };
        drop(kept);
        let bytes: u64 = spans_list.iter().map(|s| s.2 - s.1).sum();
        // Memory that is not write-watched is copied whole by every refresh.
        // It was once refreshed only every 120 frames (unmapped pages read
        // one by one cost 14 GB a frame in run 11, before they were
        // remembered): streamed geometry then drew from stale buffers, and
        // the garbage ended runs (the card stopped answering 14-90 frames
        // into the opening cutscene; with every frame it ran on). The pass
        // costs some 30 ms a frame this way.
        static WHOLE: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> = std::sync::Mutex::new(None);
        let mut whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
        let whole = whole.get_or_insert_with(Default::default);
        for (shadow, from, to) in &spans_list {
            if shadow.unwatched() {
                // By the copy's number: a copy made where one let go was
                // (`shadow::let_go_idle`) has never been refreshed.
                let key = shadow.id() as usize;
                // `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame).
                static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let every = *EVERY.get_or_init(|| {
                    std::env::var("VICTORY_UNWATCHED_FRAMES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1)
                });
                match whole.get(&key) {
                    Some(&last) if frame < last + every => continue,
                    _ => {
                        whole.insert(key, frame);
                    }
                }
            }
            shadow.refresh(list, *from, to - from);
        }
        let textures_began = std::time::Instant::now();
        HEAP_SPLIT_US[1].fetch_add((textures_began - slots_done).as_micros() as u64, Ordering::Relaxed);
        if !crate::heap_worker::kick(heap) {
            Self::heap_textures(Into::List(list), heap, host);
        }
        HEAP_SPLIT_US[2].fetch_add(textures_began.elapsed().as_micros() as u64, Ordering::Relaxed);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        // The first pass and every growth of the set by half.
        let seen = LOGGED.load(Ordering::Relaxed);
        if u64::from(views) > seen + seen / 2 {
            LOGGED.store(u64::from(views), Ordering::Relaxed);
            tracing::info!(
                slots = slots.len(),
                views,
                copies = spans_list.len(),
                mb = bytes >> 20,
                ms = started.elapsed().as_millis(),
                "buffer views of the heap made ready for shaders that pick them by index"
            );
        }
        Some(gpu)
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    /// Runs the pass that uploads textures picked by index
    /// ([`State::heap_textures`]) where nothing has asked for it yet
    /// ([`early_budgets`]): once a frame for a few milliseconds, at the
    /// frame's first draw, and with a larger budget of its own before every
    /// draw whose pixel shader is one of the title's bake programs. The
    /// uploads are recorded into `list` ahead of the draw being bound.
    fn early_textures(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        let (early_ms, bake_ms) = early_budgets();
        if compute || (early_ms == 0 && bake_ms == 0) {
            return;
        }
        let bake = bake_ms > 0
            && p.stages.iter().any(|s| BAKE_PIXEL_SHADERS.iter().any(|h| s.shader.hash.starts_with(h)));
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        // The milliseconds this frame's bake draws have taken so far: a
        // frame has several, and together they get `bake_ms`.
        static BAKED_MS: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        let first = FRAME.swap(frame, Ordering::Relaxed) != frame;
        if first {
            BAKED_MS.store(0, Ordering::Relaxed);
        }
        let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
        let bake = bake && left > 0;
        let budget = if bake {
            // What a bake may read must be there now: up to 2 GiB.
            (left, 2 << 30, true)
        } else if first && early_ms > 0 {
            (early_ms, 64 << 20, false)
        } else {
            return;
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        // What can wait for a later list is the worker's.
        if !bake && crate::heap_worker::kick(heap) {
            return;
        }
        let _pass = crate::heap_worker::on().then(crate::heap_worker::urgent);
        let n = EARLY_TEXTURE_PASSES[usize::from(bake)].fetch_add(1, Ordering::Relaxed) + 1;
        if bake && n.is_power_of_two() {
            let [slots, due, ..] = HEAP_TEXTURES_QUEUE.each_ref().map(|c| c.load(Ordering::Relaxed));
            tracing::info!(
                frame,
                bakes = n,
                queued_at_the_last_pass = slots,
                due_at_the_last_pass = due,
                uploaded_so_far = HEAP_TEXTURES.load(Ordering::Relaxed),
                "a draw of the title's decoration bake: textures picked by index are uploaded first (VICTORY_BAKE_TEXTURES_MS=0: not)"
            );
        }
        let began = std::time::Instant::now();
        PASS_BUDGET.set(Some(budget));
        Self::heap_textures(Into::List(list), heap, host);
        PASS_BUDGET.set(None);
        if bake {
            BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
    }

    pub(crate) fn heap_textures(
        mut into: Into<'_>,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return;
        }
        let _timed = crate::prepstats::Timed::new(crate::prepstats::P::HeapTextures);
        let bake = PASS_BUDGET.get().is_some_and(|b| b.2);
        if !crate::residency::admit_texture_pass(bake) {
            return;
        }
        let memory = crate::host::host().and_then(|h| h.video_memory());
        let mut per_frame = staging_budget_per_frame(memory.map(|[budget, ..]| budget));
        if crate::residency::enabled() && !bake {
            per_frame = crate::residency::prefetch_limit(
                per_frame,
                crate::residency::pressured(),
                crate::residency::limited_host_ram(),
                crate::texture::staged_now()[1],
            );
        }
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        // With `VICTORY_WIDE_TEXTURES=1`: the bytes are a budget per 150 ms
        // of frame, up to four times the frame's, so that the world's first
        // frames (a second each while it streams in) take more of the queue
        // (18,000 slots due then, in the run of 2026-11-25 02:18).
        static FRAME_AT: AtomicU64 = AtomicU64::new(0);
        static FRAME_MS: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            SPENT.store(0, Ordering::Relaxed);
            let now = crate::host::now_ms();
            let at = FRAME_AT.swap(now, Ordering::Relaxed);
            FRAME_MS.store(if at == 0 { 0 } else { now - at }, Ordering::Relaxed);
        }
        if crate::host::wide_textures() {
            per_frame = per_frame * FRAME_MS.load(Ordering::Relaxed).clamp(150, 600) / 150;
        }
        // While the title shows little (a loading screen, a black frame
        // with text: under `VICTORY_PRELOAD_DRAWS`, 400, draws in the last
        // frame), the queue takes eight times the bytes and up to a second:
        // what the world will show is uploaded before the world is.
        let loading = loading_frame(frame);
        if loading {
            per_frame = per_frame.saturating_mul(8);
            HEAP_TEXTURES_PRELOAD.fetch_add(1, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        // A bake's run makes what it reads whatever the room: the bake is
        // done once, and what it misses stays black for the run.
        let room = crate::resource::room(memory) || PASS_BUDGET.get().is_some_and(|b| b.2);
        if !room {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        // `VICTORY_EVICT_LESS=1`: what to free, and whether small copies
        // may still be made, while the card is full.
        let less = evict_less();
        let (need, small_room) = match less {
            Some(_) if !room => crate::resource::room_need(memory),
            _ => (0, false),
        };
        let mut evicted = false;
        let mut refused = 0;
        // Small copies made while the card was full are made whole again
        // once it has room (at most once a second, up to the low mark).
        if less.is_some() && room && PASS_BUDGET.get().is_none() {
            static LAST_UP: AtomicU64 = AtomicU64::new(0);
            let (now, last) = (crate::host::now_ms(), LAST_UP.load(Ordering::Relaxed));
            if now.saturating_sub(last) >= 1000
                && LAST_UP.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_ok()
            {
                // At most `VICTORY_EVICT_LESS_UP_MB` (64) a second: each is
                // uploaded again whole.
                static UP_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let cap = *UP_MB.get_or_init(|| {
                    std::env::var("VICTORY_EVICT_LESS_UP_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64u64)
                }) << 20;
                let up = crate::resource::room_up(memory).min(cap);
                if up > 0 {
                    host.texture_slots_upgrade(up, 256);
                }
            }
        }
        // Slots whose texture's memory the runtime mapped, unmapped or
        // wrote since the last pass are due again.
        host.texture_slots_remapped(frame);
        let (due, census) = host.texture_slots_due(frame, 4096);
        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
            count.store(now, Ordering::Relaxed);
        }
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the opening cutscene's
        // half a frame per second, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms();
        let took = now.saturating_sub(PASSED.swap(now, Ordering::Relaxed)).max(1);
        // The worker may run two passes in a frame: no shorter than that.
        let took = if matches!(into, Into::Ahead(_)) { took.max(100) } else { took };
        let most = (4000 / took).clamp(16, 256);
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        // Also within a time per pass: the pass runs on a thread that
        // records the frame, and making and staging 192 MB of textures took
        // 1.6-2.2 s of each of the world's first twenty frames (run 20).
        // The rest waits for the next frames, as it does past the bytes.
        let span = std::time::Duration::from_millis(if loading { 1000 } else { 250 });
        // A run made early (`State::early_textures`) has a budget of its
        // own: its bytes on top of what the frame has spent so far.
        let (span, bytes_at_most) = match PASS_BUDGET.get() {
            Some((ms, bytes, _)) => {
                (std::time::Duration::from_millis(ms), SPENT.load(Ordering::Relaxed).saturating_add(bytes))
            }
            None => (span, per_frame),
        };
        let began = std::time::Instant::now();
        // The barriers leaving uploaded textures readable: together at the
        // end (`VICTORY_UPLOAD_BARRIERS`).
        let list_ref = match &into {
            Into::List(list) => Some(*list),
            Into::Ahead(_) => None,
        };
        let _held = list_ref.map(crate::list::HoldReads::new);
        let _aliases = list_ref.map(crate::texture::HoldAliases::new);
        // Host copies are made small only where the loop says so, and
        // never past it (`resource::shrinking`).
        struct Unshrink;
        impl Drop for Unshrink {
            fn drop(&mut self) {
                crate::resource::shrinking(false);
            }
        }
        let _unshrink = Unshrink;
        for (index, write) in due {
            crate::resource::shrinking(false);
            if SPENT.load(Ordering::Relaxed) >= bytes_at_most
                || began.elapsed() > span
                || crate::residency::upload_refused()
            {
                break;
            }
            let ahead = matches!(into, Into::Ahead(_));
            if ahead && crate::heap_worker::make_way() {
                break;
            }
            // The record the slot has now, and the texture it names held
            // while a thread of ours works on it. (Released meanwhile, or
            // not a texture: nothing to upload.)
            let record = heap.handle(index).and_then(crate::descriptor::view_and_key);
            let _held_tex = match &record {
                Some((View::Shader(object, _), _)) if ahead => {
                    let held = crate::resource::hold(*object as This);
                    if held.is_none() {
                        host.texture_slot_done(index, write, None);
                        continue;
                    }
                    held
                }
                _ => None,
            };
            // A placed texture whose memory is not mapped (not streamed in
            // yet, or streamed out again: the heap keeps tens of thousands
            // of such views) gets no host texture; looked at again soon (a
            // title writes a texture's view before its pages are mapped,
            // and the pass runs early in a frame). One of which some is
            // mapped (a texture streaming in: its smaller mips lie at the
            // start of its memory and arrive first) is uploaded with what
            // is there, and again when more is. (It was uploaded again
            // every 120 frames, changed or not; and asked by its first
            // page alone, one whose first page is the one missing counted
            // as not there.)
            let mut partly = None;
            if let Some((View::Placed(false, resource, desc), _)) = &record {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let len = crate::resource::placed_bytes(resource).unwrap_or(1);
                let mapped = crate::texture::mapped_bytes(address as *const u8, len as usize);
                if mapped == 0 {
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    host.texture_slot_later(index, write, frame, most);
                    continue;
                }
                if mapped < len {
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some((View::Shader(object, _), _)) = &record
                && crate::resource::buffer_bytes(*object as This).is_none()
                && !crate::resource::title_fills(*object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            // From here on a host texture is made: not without room. The
            // slot stays due.
            if !room {
                if !evicted {
                    evicted = true;
                    // Slots keep the copies of what they showed, so the
                    // card stayed full for the rest of the run (uploads
                    // stopped at 14 GB with 4,500 slots due): the oldest
                    // made give theirs up, but not the last seconds'.
                    match less {
                        None => {
                            HEAP_TEXTURES_EVICTED.fetch_add(
                                host.texture_slots_evict(now.saturating_sub(4000), 256) as u64,
                                Ordering::Relaxed,
                            );
                        }
                        // By bytes, at most once in `every` ms: what is
                        // given up is let go some seconds later
                        // (`resource::let_go_idle`), and the card shows it
                        // later still.
                        Some(every) => {
                            static LAST: AtomicU64 = AtomicU64::new(0);
                            let last = LAST.load(Ordering::Relaxed);
                            if (last == 0 || now.saturating_sub(last) >= every)
                                && LAST.compare_exchange(last, now.max(1), Ordering::Relaxed, Ordering::Relaxed).is_ok()
                            {
                                // Queued again by when their copies went.
                                let frame_ms = FRAME_MS.load(Ordering::Relaxed).max(16);
                                let later = ((every + 1500) / frame_ms).clamp(30, 1000);
                                // At most `VICTORY_EVICT_LESS_ROUND_MB` (128) a
                                // round: 320-480 MB a round came back as as
                                // many uploads soon after (run cumul 22:46).
                                static ROUND_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                                let round = *ROUND_MB.get_or_init(|| {
                                    std::env::var("VICTORY_EVICT_LESS_ROUND_MB")
                                        .ok()
                                        .and_then(|v| v.trim().parse().ok())
                                        .unwrap_or(128u64)
                                }) << 20;
                                let (n, _) =
                                    host.texture_slots_evict_less(now.saturating_sub(4000), need.min(round), 4096, later);
                                HEAP_TEXTURES_EVICTED.fetch_add(n as u64, Ordering::Relaxed);
                            }
                        }
                    }
                }
                if !small_room {
                    refused += 1;
                    if refused > 256 {
                        break;
                    }
                    continue;
                }
            }
            // A small copy while the card is full (`VICTORY_EVICT_LESS=1`).
            let small = !room;
            crate::resource::shrinking(small);
            let shrunk_before = crate::resource::SHRUNK.load(Ordering::Relaxed);
            let mirroring = std::time::Instant::now();
            // On the worker the slot shows the view when the upload is in
            // the queue ([`crate::heap_worker::Batch::publish`]).
            let view = match &into {
                Into::List(_) => heap.handle(index).and_then(|handle| host.mirror(index, handle, true)),
                Into::Ahead(_) => record.as_ref().map(|(view, _)| view.clone()),
            };
            HEAP_SPLIT_US[3].fetch_add(mirroring.elapsed().as_micros() as u64, Ordering::Relaxed);
            let placed = view.as_ref().and_then(crate::descriptor::host::placed_texture);
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            if small && crate::resource::SHRUNK.load(Ordering::Relaxed) > shrunk_before {
                let n = HEAP_TEXTURES_SMALL.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!(
                        small = n,
                        "texture slots made with a small host copy while the card is full (VICTORY_EVICT_LESS=1)"
                    );
                }
            }
            let title_fills = crate::resource::host_resource(resource).is_some_and(crate::texture::title_written);
            if !title_fills {
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            let tabled = crate::texture::is_staged(resource);
            if let Some(view) = &view {
                let staging = std::time::Instant::now();
                match &mut into {
                    Into::List(list) => readable(list, sampled.kind, view, false),
                    Into::Ahead(batch) => {
                        let Some(list) = batch.list() else {
                            return;
                        };
                        readable(list, sampled.kind, view, false);
                        if let (Some(handle), Some((_, key))) = (heap.handle(index), &record) {
                            batch.publish(index, handle, *key);
                        }
                    }
                }
                crate::resource::shrinking(false);
                HEAP_SPLIT_US[4].fetch_add(staging.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            // A texture view the host refuses removes the device: name it.
            static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !GONE.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                GONE.store(true, Ordering::Relaxed);
                let text: String = format!("{view:?}").chars().take(300).collect();
                tracing::error!(why, slot = index, "the host device went while textures picked by index were made ready; last: {text}");
                crate::host::log_removal_context();
                return;
            }
            let (bytes, whole) = crate::texture::staged(resource).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            // Uploaded in part: the levels not there yet are not sampled
            // (they read zeros), none at all if no level is all there (the
            // grey stand-in); all of them again once it is whole. Before
            // the tabled case below, which the clamp concerns as much (it
            // stayed on for good there once the texture was whole). The
            // level is kept with the slot and its record
            // (`HostHeap::remake_from_level`): a slot given up and made
            // again keeps it, one written since drops it.
            if !ahead
                && partly_clamp()
                && let Some(handle) = heap.handle(index)
            {
                let least = if partly.is_some() { crate::texture::first_whole_level(resource).unwrap_or(0) } else { 0 };
                if least != host.slot_least(index, handle) {
                    host.remake_from_level(index, handle, least);
                    let n = HEAP_TEXTURES_CLAMPED.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_power_of_two() {
                        tracing::info!(
                            remade = n,
                            level = least,
                            "texture slots uploaded in part remade to sample only the levels there (VICTORY_PARTLY_CLAMP=0 turns this off)"
                        );
                    }
                }
            }
            if tabled {
                // Nothing was staged here, so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                host.texture_slot_done(index, write, again);
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            crate::texture::unstage(resource);
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            if let Into::Ahead(batch) = &mut into {
                batch.uploaded(bytes);
            }
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            host.texture_slot_made(index, now, resource as usize);
            host.texture_slot_done(index, write, again);
        }
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        let before = DISPATCHED.load(Ordering::Relaxed);
        peek_dispatch(self.pipeline, [x, y, z]);
        self.dispatch_inner(list, x, y, z);
        #[cfg(feature = "diagnostics")]
        self.trace_op("dispatch", &[x.into(), y.into(), z.into()], DISPATCHED.load(Ordering::Relaxed) != before);
    }

    fn dispatch_inner(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        // It may write the arguments a window holds clamped.
        self.window = None;
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::tally_drop(self.pipeline as This, "skipped (VICTORY_SKIP_PIPELINES)");
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::tally_drop(self.pipeline as This, crate::pipeline::BUILDER_OFF);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = self.host() else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            tally_drop(self.pipeline);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.list_dropped_roots([x, y, z]);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            tally_drop(self.pipeline);
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_off("one of its dispatches could not be bound");
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        list.count_write();
        list.count_dispatch();
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
        // After the program that resolves the indirect light: the ambient
        // light in its place (`crate::ambient`); the title's bindings go
        // back on the list before its next command.
        if self.ambient(list, p) {
            self.host_state_lost();
        }
    }

    /// After a dispatch of `p`: the least indirect light, if it is the
    /// program that resolves that. Whether our bindings replaced the title's.
    fn ambient(&self, list: &HostList, p: &crate::pipeline::host::HostPipeline) -> bool {
        let Some(stage) = p.stages.first().filter(|s| s.shader.resolves_light) else {
            return false;
        };
        let view = |kind: ResourceType, (slot, offset): (u8, u32)| {
            let u = stage.translated().used.iter().find(|u| {
                u.kind == kind && u.root == RootType::Table && (u.slot, u.table_offset_dwords) == (slot, offset)
            })?;
            let Some(Root::Table(table)) = self.roots_ref(true).get(usize::from(u.slot)) else {
                return None;
            };
            crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4)
        };
        let light = match view(ResourceType::Uav, crate::ambient::LIGHT) {
            Some(crate::descriptor::View::Unordered(r, ..)) => r,
            other => {
                once!(
                    view = ?other.map(|v| std::mem::discriminant(&v)),
                    "no ambient light: the light resolve's output is not a plain image view here"
                );
                return false;
            }
        };
        if light == 0 {
            return false;
        }
        let depth = match view(ResourceType::Srv, crate::ambient::DEPTH) {
            Some(crate::descriptor::View::Shader(r, _)) if r != 0 => Some(r as This),
            _ => None,
        };
        crate::ambient::raise(list, light as This, depth)
    }

    /// Counts a draw; one whose shaders store to memory may have written
    /// indirect arguments (`HostList::count_write`).
    fn drawn(&self, list: &HostList) {
        DRAWN.fetch_add(1, Ordering::Relaxed);
        if self.host().is_some_and(|p| p.stages.iter().any(|stage| stage.translated().stores)) {
            list.count_write();
        }
    }

    /// `VICTORY_ARGS_PEEK=<hash prefix>,...`: the arguments (clamped, as
    /// the card reads them) of the indirect commands whose pipeline names
    /// one of those shaders, read back once the list has run and logged
    /// with the command (its targets' sizes too), at most 40 times in all.
    #[cfg(feature = "diagnostics")]
    fn peek_args(&self, list: &HostList, bounded: &ID3D12Resource, at: u64, stride: u32, max: u32, kind: u32) {
        static WANTED: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        let wanted = WANTED.get_or_init(|| {
            std::env::var("VICTORY_ARGS_PEEK")
                .unwrap_or_default()
                .split(',')
                .map(|h| h.trim().to_lowercase())
                .filter(|h| !h.is_empty())
                .collect()
        });
        if wanted.is_empty() {
            return;
        }
        let Some(p) = self.host() else {
            return;
        };
        if !wanted.iter().any(|w| p.name.contains(w.as_str())) {
            return;
        }
        static PEEKS: AtomicU64 = AtomicU64::new(0);
        if PEEKS.fetch_add(1, Ordering::Relaxed) >= 40 {
            return;
        }
        let records = max.clamp(1, 4) as usize;
        let row = stride as usize * records;
        let Some(host) = crate::host::host() else {
            return;
        };
        let Some(readback) = host.read_back_buffer(row.next_multiple_of(256) as u64) else {
            return;
        };
        // Where the bytes land: ours, never freed (40 peeks at most).
        let dst: &'static mut [u8] = Box::leak(vec![0u8; row].into_boxed_slice());
        let dst_at = dst.as_mut_ptr();
        // SAFETY: an open list; the scratch buffer (COMMON after `done`,
        // promoted to a copy source) and the readback buffer are live.
        unsafe { list.list.CopyBufferRegion(&readback, 0, bounded, at, row as u64) };
        list.defer_read_back(readback, dst_at, [row, 1, row]);
        // The topology, and the index buffer's first indices (title memory).
        let IndexView { address, size, format } = self.index;
        let first: Vec<u32> = if address != 0 && size >= 16 && crate::texture::committed(address as *const u8, 16) {
            // SAFETY: 16 committed bytes of title memory.
            let raw = unsafe { (address as *const [u8; 16]).read_unaligned() };
            if format == 42 {
                raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            } else {
                raw.chunks_exact(2).map(|c| u32::from(u16::from_le_bytes([c[0], c[1]]))).collect()
            }
        } else {
            Vec::new()
        };
        let text = format!(
            "topology {} index buffer {address:#x} size {size} format {format} first {first:?}; {}",
            self.topology,
            self.op_text(format!("indirect kind {kind} max {max}"), kind == 2)
        );
        let address = dst_at as usize;
        std::thread::spawn(move || {
            // The list runs within a few frames: the bytes are there by then.
            std::thread::sleep(std::time::Duration::from_secs(3));
            // SAFETY: the leaked buffer above, `row` bytes.
            let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, row) };
            let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            tracing::warn!("indirect arguments peeked (VICTORY_ARGS_PEEK), stride {stride}: {words:?} for {text}");
        });
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = self.host() else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(text, " targets {targets:?} depth {:?} state [{}]", described(self.depth), p.state);
            // The title's scissor and viewport at this draw (the first of
            // each): the title draws what it sees through a
            // portal (a window, from inside a room) under the portal's
            // rectangle, and the listing did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(text, " scissor [{} {} {} {}]", r.left, r.top, r.right, r.bottom);
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
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

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    /// An indexed indirect draw of small instances (four vertices each,
    /// the title's index buffer starting with four indices below 4) drawn
    /// as one instance: six indices of ours per instance into vertex
    /// buffers of ours that repeat the title's four vertices, with the
    /// pipeline's vertex shader translated to take the instance from the
    /// vertex number (`crate::pipeline::flat_pipeline`). The title's
    /// tile-grid composite (VS e732a8b4) draws ~8,000 instances of a quad
    /// in one draw; the card ran them one at a time, ~9 ms a frame with
    /// one pixel drawn as with all of them (2026-11-25). `None`: not such a
    /// draw, the caller draws it as before.
    fn draw_flat(
        &mut self,
        list: &HostList,
        p: &crate::pipeline::host::HostPipeline,
        args: &std::sync::Arc<crate::shadow::Shadow>,
        arguments: u64,
        stride: u32,
        signature: &ID3D12CommandSignature,
    ) -> Option<bool> {
        if p.inputs.iter().any(|e| e.per_instance) {
            return None;
        }
        // The title's four indices, in title memory.
        let IndexView { address, size, format } = self.index;
        let wide = match format {
            42 => true,
            57 => false,
            _ => return None,
        };
        let bytes = if wide { 16 } else { 8 };
        if address == 0 || size < bytes || !crate::texture::committed(address as *const u8, bytes as usize) {
            return None;
        }
        let mut first = [0u32; 4];
        for (k, i) in first.iter_mut().enumerate() {
            // SAFETY: committed title memory, checked above.
            *i = unsafe {
                if wide {
                    (address as *const u32).add(k).read_unaligned()
                } else {
                    u32::from((address as *const u16).add(k).read_unaligned())
                }
            };
        }
        if first.iter().any(|&i| i >= 4) {
            return None;
        }
        let [a, b, c, d] = first;
        // Rectangles: the host draws each instance's first three indices as
        // one triangle that the pipeline's geometry stage completes.
        let pattern = match self.topology {
            TRIANGLE_STRIP => vec![a, b, c, c, b, d],
            QUADS => vec![a, b, c, a, c, d],
            RECTANGLES => vec![a, b, c],
            _ => return None,
        };
        // The record as the card will read it (refreshed from title memory
        // below): drawn flat only as one instance of four (three) indices
        // from index, vertex and instance 0, which is what our buffers
        // repeat; anything else (offsets, an emptied record, more instances
        // than a flat draw holds) is drawn as before.
        if !flat_record_fits(args, arguments, if self.topology == RECTANGLES { 3 } else { 4 }) {
            return None;
        }
        // The four vertices of each buffer the layout reads.
        let mut vertices = Vec::new();
        for (slot, v) in self.vertex.iter().enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            let four = 4 * v.stride;
            if v.address == 0 || v.stride == 0 || v.size < four || !crate::texture::committed(v.address as *const u8, four as usize) {
                return None;
            }
            // SAFETY: committed title memory, checked above.
            let bytes = unsafe { std::slice::from_raw_parts(v.address as *const u8, four as usize) }.to_vec();
            vertices.push((slot as u32, v.stride, bytes));
        }
        // Whether the title's vertex bytes may still be on the card only
        // (written there and not copied back): ours would be stale.
        let stale = self.vertex.iter().enumerate().any(|(slot, v)| {
            p.slots & 1 << slot != 0
                && crate::shadow::find_any(v.address, u64::from(v.size))
                    .is_some_and(|s| s.writes_pending() || s.stale_under_writes(v.address as usize, (4 * v.stride) as usize))
        });
        if stale {
            // Our copy would be of bytes from before the card's writes, kept
            // for later draws too: the title's own draw instead.
            once!(true, "flat instances: title vertex bytes possibly stale on the CPU, drawn as before (logged once)");
            return None;
        }
        let len = u64::from(stride);
        // `VICTORY_FLAT_CHECK=vb`: the title's own pipeline, indices and
        // instanced arguments over our vertex buffers (their first four
        // vertices are the copy): an image as without the switch says the
        // copy is right, and a wrong one the flat vertex shader.
        if flat_check_vb() {
            let buffers = flat_buffers(&pattern, &vertices)?;
            args.refresh(list, arguments, len);
            let bounded = crate::indirect::bound(list, (&args.buffer, args.offset(arguments)), None, (1, stride, 1))?;
            if !self.prepare(list, true) {
                crate::indirect::done(list, &bounded);
                return Some(false);
            }
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indirect flat check (title draw, our vertices), stride {stride}"), false));
            // SAFETY: an open list with the pipeline bound; our buffers live
            // for the run, the scratch buffer until the list is reset.
            unsafe {
                for (slot, view) in &buffers.vertices {
                    list.list.IASetVertexBuffers(*slot, Some(&[*view]));
                }
                list.list.ExecuteIndirect(signature, 1, &bounded, 0, None, 0);
            }
            crate::indirect::done(list, &bounded);
            self.drawn(list);
            return Some(true);
        }
        let flat = crate::pipeline::flat_pipeline(self.pipeline as This)?;
        let buffers = flat_buffers(&pattern, &vertices)?;
        args.refresh(list, arguments, len);
        let bounded = crate::indirect::bound_flat(list, (&args.buffer, args.offset(arguments)), stride, pattern.len() as u32)?;
        if !self.prepare(list, true) {
            crate::indirect::done(list, &bounded);
            return Some(false);
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("indirect flat instances, stride {stride}"), false));
        // SAFETY: an open list with the pipeline's roots bound; our buffers
        // live for the run, the scratch buffer until the list is reset.
        unsafe {
            let l = &list.list;
            l.SetPipelineState(&flat.pso);
            l.IASetIndexBuffer(Some(&buffers.index));
            for (slot, view) in &buffers.vertices {
                l.IASetVertexBuffers(*slot, Some(&[*view]));
            }
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            l.ExecuteIndirect(signature, 1, &bounded, 0, None, 0);
            // The title's topology again (its buffers and pipeline are bound
            // again by the next draw's preparation).
            let host = match self.topology {
                RECTANGLES | TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => 4,
                t => t,
            };
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32));
        }
        self.host_pipeline = 0;
        crate::indirect::done(list, &bounded);
        FLAT_DRAWS.fetch_add(1, Ordering::Relaxed);
        self.drawn(list);
        Some(true)
    }

    /// `VICTORY_SCISSOR_PROBE=<hash prefix>,...`: the draws whose pipeline
    /// names one of those shaders rasterise one pixel (their vertices and
    /// instances all run). With the timing on, what is left of such a
    /// draw's time is its geometry and instances, not its pixels. Whether
    /// it applied (`on`); with `on` false, the title's rectangles again.
    #[cfg(feature = "diagnostics")]
    fn scissor_probe(&self, list: &HostList, on: bool) -> bool {
        static WANTED: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        let wanted = WANTED.get_or_init(|| {
            std::env::var("VICTORY_SCISSOR_PROBE")
                .unwrap_or_default()
                .split(',')
                .map(|h| h.trim().to_lowercase())
                .filter(|h| !h.is_empty())
                .collect()
        });
        if wanted.is_empty() {
            return false;
        }
        if !on {
            if !self.scissors.is_empty() {
                crate::draw::scissors(list, &self.scissors);
            }
            return false;
        }
        let Some(p) = self.host() else {
            return false;
        };
        if !wanted.iter().any(|w| p.name.contains(w.as_str())) {
            return false;
        }
        once!(pipeline = %p.name, "VICTORY_SCISSOR_PROBE: this pipeline's draws rasterise one pixel (logged once)");
        let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
        crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        true
    }

    fn mesh_scissor(&self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        if on {
            let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        let before = DRAWN.load(Ordering::Relaxed);
        ASKED.fetch_add(1, Ordering::Relaxed);
        self.dispatch_mesh_inner(list, x);
        #[cfg(feature = "diagnostics")]
        self.trace_op("mesh", &[x.into()], DRAWN.load(Ordering::Relaxed) != before);
    }

    fn dispatch_mesh_inner(&mut self, list: &HostList, x: u32) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = mesh_test() != 1
            && mesh_kept(self.host())
            && self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    self.mesh_scissor(list, true);
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    self.mesh_scissor(list, false);
                    self.drawn(list);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            tally_drop(self.pipeline);
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

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
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
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u.kind, &view, true);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            }
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        indexed: &[RootIndexedView],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
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
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            selected.sort_unstable();
            selected.dedup();
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8)
                .chain(selected.last().copied())
                .max()
                .unwrap_or(0)
                + 1;
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
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
                // Host copies made again at another size since
                // (`crate::resource::loading_full_frame`): the table is
                // made again, with the new copies.
                #[cfg(windows)]
                crate::resource::LOADING_FULL_CHANGES.load(Ordering::Relaxed).hash(&mut h);
                (table, n, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u.kind, &view, true);
                    }
                }
                for k in selected {
                    if entries.iter().any(|u| u.table_offset_dwords / 8 == k) {
                        continue;
                    }
                    let handle = table + u64::from(k) * 32;
                    if let Some(view) = crate::descriptor::view(handle)
                        && let Some(kind) = view_kind(&view)
                    {
                        readable(list, kind, &view, true);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it.
                made.push((slot, u32::MAX));
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null(index);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            readable(list, kind, &view, true);
                        }
                        // The title's slot is a table's: out of the queue
                        // of textures picked by index if the draw read it
                        // (its texture is up to date now), and made in
                        // place as the texture pass would have made it,
                        // for a shader that picks it by index all the same.
                        if let Some(title) = title_first
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            host.texture_slot_tabled(title + k, kind.is_some());
                            if kind.is_some() {
                                host.mirror(title + k, table + u64::from(k) * 32, true);
                            }
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    /// A line of the frame trace ([`crate::trace`]) for the op just
    /// recorded or refused.
    #[cfg(feature = "diagnostics")]
    fn trace_op(&self, kind: &str, args: &[u64], ran: bool) {
        if crate::trace::active().is_none() {
            crate::trace::clear();
            return;
        }
        let stages: Vec<String> = self
            .host()
            .map(|p| p.stages.iter().map(|s| s.shader.hash.to_string()).collect())
            .unwrap_or_default();
        let targets: Vec<(usize, u64, u32, i32)> = self
            .targets
            .iter()
            .map(|&resource| {
                crate::resource::host_resource(resource as This).map_or((resource, 0, 0, 0), |t| {
                    // SAFETY: a live resource.
                    let d = unsafe { t.GetDesc() };
                    (resource, d.Width, d.Height, d.Format.0)
                })
            })
            .collect();
        crate::trace::op(kind, args, ran, self.pipeline, &stages, &targets);
    }

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
            .filter_map(|&t| crate::resource::host_made(t as This))
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
        let Some(p) = self.host() else {
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
        let before = DRAWN.load(Ordering::Relaxed);
        ASKED.fetch_add(1, Ordering::Relaxed);
        self.draw_inner(list, vertices, instances, first, first_instance);
        #[cfg(feature = "diagnostics")]
        self.trace_op(
            "draw",
            &[vertices.into(), instances.into(), first.into(), first_instance.into()],
            DRAWN.load(Ordering::Relaxed) != before,
        );
    }

    fn draw_inner(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with the rectangle geometry stage draws the title's
        // vertices as they are; any other gets its rectangles expanded.
        let rects = self.topology == RECTANGLES && !self.completes_rects();
        let ran = ready && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
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
            self.drawn(list);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            tally_drop(self.pipeline);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Whether the bound pipeline completes rectangles itself, in the
    /// geometry stage rectangle pipelines get (`HostPipeline::rects`).
    fn completes_rects(&self) -> bool {
        self.host().is_some_and(|p| p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = self.host() else {
            return false;
        };
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
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // The vertices are read in place, at submission: no change
                // of mapping under the read (see
                // `crate::watch::title_memory_reading`).
                let _mapped = crate::watch::title_memory_reading();
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
                SizeInBytes: (rects * 6 * stride as u64) as u32,
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
        let before = DRAWN.load(Ordering::Relaxed);
        ASKED.fetch_add(1, Ordering::Relaxed);
        self.draw_indexed_inner(list, indices, instances, first, base, first_instance);
        #[cfg(feature = "diagnostics")]
        self.trace_op(
            "indexed",
            &[indices.into(), instances.into(), first.into(), base as u64, first_instance.into()],
            DRAWN.load(Ordering::Relaxed) != before,
        );
    }

    fn draw_indexed_inner(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles are drawn by the rectangle geometry stage
        // alone (nothing expands them on the CPU).
        let lost = self.topology == RECTANGLES && !self.completes_rects();
        if lost {
            once!("indexed rectangles not drawn");
        }
        let ran = !lost && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            self.drawn(list);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            tally_drop(self.pipeline);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
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
        clamps: Option<&HostList>,
    ) -> bool {
        let ran = self.execute_indirect_logged(list, (kind, stride, max), arguments, count, clamps);
        #[cfg(feature = "diagnostics")]
        self.trace_op(
            "indirect",
            &[kind.into(), stride.into(), max.into(), arguments, count.unwrap_or(0)],
            ran,
        );
        ran
    }

    fn execute_indirect_logged(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
        clamps: Option<&HostList>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = self.host() else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        // Indirect rectangles (the title's tiled lighting, 256
        // indirect draws a frame, makes their corners from the vertex
        // number) are completed by the pipeline's geometry stage: the
        // arguments stay the title's, three vertices a rectangle. They
        // cannot be expanded on the CPU, which does not know their count.
        // shortcut: in a pipeline without that stage, a rectangle with no
        // per-vertex inputs is drawn as its first triangle (three vertices,
        // half of it), and one with vertex buffers is not drawn.
        let made_rects = !p.rects
            && kind == DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
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
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
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
        // Zero records execute nothing on the GPU (`VICTORY_SCOPED_UAV=1`,
        // PR #2168): skip refresh, clamp dispatch, barriers, and state invalidation.
        if max == 0 && crate::indirect::scoped_uav() {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| {
                self.op_text(format!("indirect kind {kind} max 0 at {arguments:#x}"), kind == DISPATCH)
            });
            if kind == DISPATCH {
                DISPATCHED.fetch_add(1, Ordering::Relaxed);
            } else {
                DRAWN.fetch_add(1, Ordering::Relaxed);
            }
            return true;
        }
        if kind == MESH && (mesh_test() == 1 || !mesh_kept(self.host())) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // Many small instances, drawn flat (`VICTORY_FLAT_INSTANCES`).
        if kind == INDEXED
            && max == 1
            && count.is_none()
            && flat_wanted(&p.name)
            && let Some(ran) = self.draw_flat(list, p, &args, arguments, stride, &signature)
        {
            return ran;
        }
        // A window of records clamped in one pass (`windows`): for single
        // records without a count buffer, drawn by shaders that store to no
        // buffer. Anything else ends the window: it uses the scratch buffer
        // or may write the arguments.
        const WINDOW_USE: usize = usize::MAX - 0x77;
        let offset = args.offset(arguments);
        // A draw of a run whose clamps go to the list before this one
        // (`crate::indirect::hoisting`, `crate::list`): nothing of ours comes
        // between the draws. Dispatches may write arguments and rectangles
        // are rewritten on the GPU: theirs stay before each.
        let hoisted = match clamps {
            Some(clamps) if kind != DISPATCH && self.topology != RECTANGLES => {
                args.refresh(clamps, arguments, len);
                if let Some((s, at)) = &counter {
                    s.refresh(clamps, *at, 4);
                }
                #[cfg(feature = "diagnostics")]
                clamps.note_op(|| format!("hoisted clamp of kind {kind}"));
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                crate::indirect::bound_before(clamps, (&args.buffer, offset), count_arg, (kind, stride, max))
            }
            _ => None,
        };
        let inline = hoisted.is_none();
        let windowed = inline
            && windows()
            && kind != DISPATCH
            && max == 1
            && count.is_none()
            && !p.stages.iter().any(|stage| stage.translated().stores);
        // Another host list (a new one after a cut, or this one after its
        // reset) has another scratch buffer, or this one's with old bytes.
        if !windowed || list.first_use(WINDOW_USE) {
            self.window = None;
        }
        let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
        let hit = self
            .window
            .as_ref()
            .filter(|w| same(w) && offset >= w.base)
            .filter(|w| (offset - w.base) % u64::from(w.step) == 0 && (offset - w.base) / u64::from(w.step) < u64::from(w.records))
            .map(|w| (w.scratch.clone(), offset - w.base));
        // With a ledger, the hoisted clamp says how many records run.
        let mut ledger_count = None;
        let (bounded, at) = match (hoisted, hit) {
            (Some((buffer, at, count)), _) => {
                ledger_count = count;
                (buffer, at)
            }
            (None, Some(hit)) => hit,
            (None, None) => {
                // A record a short step after the last window's last one
                // (the same step as that window's, if it had one): the
                // title walks the buffer, so the ones that step apart
                // after it are clamped with it.
                let (step, records) = match &self.window {
                    Some(w) if windowed && same(w) => {
                        let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                        let step = offset.saturating_sub(last);
                        let walks = step >= u64::from(stride)
                            && step <= 256
                            && step % 4 == 0
                            && (w.records == 1 || step == u64::from(w.step));
                        if walks {
                            let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                            (step as u32, (if w.records == 1 { 16 } else { w.records * 2 }).min(fit).max(1))
                        } else {
                            (stride, max)
                        }
                    }
                    _ => (stride, max),
                };
                args.refresh(list, arguments, len.max(u64::from(step) * u64::from(records)));
                if let Some((s, at)) = &counter {
                    s.refresh(list, *at, 4);
                }
                // Bounded first: the clamp binds a compute pipeline of its
                // own and copies clamped active records into a scratch
                // buffer at offset 0.
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                // More than the record asked for: a window clamped ahead,
                // whose unfilled slots must not spend the list's budget.
                let (bounded, args_at, compact_count) = if windowed && records > 1 {
                    match crate::indirect::bound_ahead(list, (&args.buffer, offset), (kind, step, records)) {
                        Some(b) => (b, 0, None),
                        None => {
                            self.window = None;
                            why("not bounded (no clamp, or arguments not 4-byte aligned)");
                            return false;
                        }
                    }
                } else if count_arg.is_none()
                    && crate::indirect::indirect_compact()
                    && records >= crate::indirect::COMPACT_MIN_RECORDS
                {
                    match crate::indirect::bound_compact(list, (&args.buffer, offset), (kind, step, records)) {
                        Some((b, at, cnt)) => (b, at, Some(cnt)),
                        None => {
                            self.window = None;
                            why("not bounded (no compact clamp, or arguments not 4-byte aligned)");
                            return false;
                        }
                    }
                } else {
                    match crate::indirect::bound(list, (&args.buffer, offset), count_arg, (kind, step, records)) {
                        Some(b) => (b, 0, None),
                        None => {
                            self.window = None;
                            why("not bounded (no clamp, or arguments not 4-byte aligned)");
                            return false;
                        }
                    }
                };
                if let Some(cnt) = compact_count {
                    ledger_count = Some(cnt);
                }
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
                self.window = (windowed && args_at == 0).then(|| Window {
                    copy: args.id(),
                    kind,
                    stride,
                    step,
                    base: offset,
                    records,
                    scratch: bounded.clone(),
                });
                (bounded, args_at)
            }
        };
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            if inline {
                crate::indirect::done(list, &bounded);
            }
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        if kind == MESH {
            self.mesh_scissor(list, true);
        }
        #[cfg(feature = "diagnostics")]
        let probed = kind != DISPATCH && self.scissor_probe(list, true);
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            match &ledger_count {
                Some((ledger, entry)) => {
                    list.list.ExecuteIndirect(&signature, max, &bounded, at, ledger, *entry)
                }
                None => list.list.ExecuteIndirect(
                    &signature,
                    max,
                    &bounded,
                    at,
                    counter.as_ref().map(|(s, _)| &s.buffer),
                    counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
                ),
            }
        };
        if kind == MESH {
            self.mesh_scissor(list, false);
        }
        #[cfg(feature = "diagnostics")]
        if probed {
            self.scissor_probe(list, false);
        }
        if inline {
            crate::indirect::done(list, &bounded);
            // (Hoisted arguments stay readable as arguments only.)
            #[cfg(feature = "diagnostics")]
            self.peek_args(list, &bounded, at, stride, max, kind);
        }
        if kind == DISPATCH {
            list.count_write();
            list.count_dispatch();
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            self.drawn(list);
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
fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

/// `drawn`: a draw reads it (not the pass over textures picked by index):
/// noted for `crate::resource::loading_full_frame`.
fn readable(list: &HostList, kind: ResourceType, view: &View, drawn: bool) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            if STORES.get() || !write_skip() {
                shadow.gpu_writes(list, address, len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if drawn {
                crate::resource::note_drawn(*resource as This);
            }
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let alias = crate::resource::alias_of(*resource as This)
                .map_or(crate::texture::Alias::None, |source| {
                    crate::texture::alias(list, *resource as This, source, state)
                });
            if matches!(alias, crate::texture::Alias::None)
                && let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            if drawn {
                crate::quality::texture_read(*resource as This);
            }
            if !matches!(alias, crate::texture::Alias::Deferred) {
                crate::texture::flush_aliases_if(list, *resource as This);
                crate::list::ensure_state(list, *resource as This, state);
            }
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
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
        (View::Unordered(resource, _, words), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            // Some compute-written textures are also read by the title CPU.
            list.texture_written(*resource as This);
            // Unordered access to a depth texture is its colour twin's.
            crate::texture::flush_aliases_if(list, *resource as This);
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            } else if crate::descriptor::host::uav_writes_depth_plane(words.as_ref()) {
                list.twin_plane_written(*resource as This);
            }
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x} of {r:?}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t}"),
            None => format!(
                "uav resource {r:#x} buffer {:?}",
                crate::descriptor::host::buffer_range(v)
            ),
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

/// Draws made flat since the start (`State::draw_flat`).
pub(crate) static FLAT_DRAWS: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_FLAT_INSTANCES=<shader hash prefix>,...`: the indexed indirect
/// draws of pipelines naming one of those shaders have their small
/// instances drawn flat (`State::draw_flat`). Unset: none.
/// `VICTORY_FLAT_CHECK=vb` (see `State::draw_flat`).
fn flat_check_vb() -> bool {
    static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHECK.get_or_init(|| std::env::var("VICTORY_FLAT_CHECK").is_ok_and(|v| v.trim() == "vb"))
}

/// `VICTORY_DISPATCH_PEEK=<hash prefix>,...`: the group counts of the
/// title's direct dispatches with such a shader, logged (the first 40; any
/// build). What a compute shader costs a frame is its groups times what one
/// group does (2026-11-25: CS 39f0e9ae, one dispatch, 8.8 ms).
fn peek_dispatch(pipeline: usize, groups: [u32; 3]) {
    static WANTED: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let wanted = WANTED.get_or_init(|| {
        std::env::var("VICTORY_DISPATCH_PEEK")
            .unwrap_or_default()
            .split(',')
            .map(|h| h.trim().to_lowercase())
            .filter(|h| !h.is_empty())
            .collect()
    });
    if wanted.is_empty() {
        return;
    }
    // A dispatch: no targets (only precompiled graphics pipelines need them).
    let targets = crate::precompiled::Targets::default();
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This, &targets) else {
        return;
    };
    if !wanted.iter().any(|w| p.name.contains(w.as_str())) {
        return;
    }
    static PEEKS: AtomicU64 = AtomicU64::new(0);
    if PEEKS.fetch_add(1, Ordering::Relaxed) >= 40 {
        return;
    }
    let [x, y, z] = groups;
    tracing::info!(pipeline = %p.name, x, y, z, groups = u64::from(x) * u64::from(y) * u64::from(z), "VICTORY_DISPATCH_PEEK: a direct dispatch");
}

/// Live (`crate::live`): each draw reads it.
fn flat_wanted(name: &str) -> bool {
    static WANTED: crate::live::Words = crate::live::Words::new();
    WANTED.any_in("VICTORY_FLAT_INSTANCES", name)
}

/// Our index and vertex buffers for flat draws: the index buffer's views
/// and each vertex slot's.
struct FlatBuffers {
    index: D3D12_INDEX_BUFFER_VIEW,
    vertices: Vec<(u32, D3D12_VERTEX_BUFFER_VIEW)>,
    /// What the views point into.
    _kept: Vec<ID3D12Resource>,
}

// SAFETY: views (addresses) and free-threaded D3D12 buffers.
unsafe impl Send for FlatBuffers {}
// SAFETY: as above.
unsafe impl Sync for FlatBuffers {}

/// `VICTORY_FLAT_RECT4=1`: see [`flat_record_fits`]. Off by default.
/// Live (`crate::live`): each record looked at reads it, as
/// `VICTORY_FLAT_INSTANCES`.
fn flat_rect4() -> bool {
    static ON: crate::live::Switch = crate::live::Switch::new();
    ON.get("VICTORY_FLAT_RECT4", |v, _| u64::from(v.is_some_and(|v| v.trim() == "1"))) == 1
}

/// Whether the indexed record at `arguments` (title memory, not written on
/// the card since: what the refresh copies is what runs) is one a flat draw
/// stands for: `indices` indices per instance, from index, vertex and
/// instance 0, with 1 to [`crate::indirect::FLAT_MOST`] instances.
fn flat_record_fits(args: &crate::shadow::Shadow, arguments: u64, indices: u32) -> bool {
    if args.writes_pending() || args.stale_under_writes(arguments as usize, 20) {
        return false;
    }
    if arguments == 0 || !crate::texture::committed(arguments as *const u8, 20) {
        return false;
    }
    // SAFETY: committed title memory, checked above.
    let r: [u32; 5] =
        std::array::from_fn(|k| unsafe { (arguments as *const u32).add(k).read_unaligned() });
    let [count, instances, start, base, first] = r;
    // A rectangle list of four indices is one rectangle (the fourth index
    // starts no primitive): drawn from its first three, as the flat draw's
    // pattern does (`VICTORY_FLAT_RECT4=1`). The title's tile grid
    // (`e732a8b4`) has such records, which were drawn as before: one host
    // instance at a time, ~8.6 ms of the card a frame.
    let fits = (count == indices || indices == 3 && count == 4 && flat_rect4())
        && (1..=crate::indirect::FLAT_MOST).contains(&instances)
        && start == 0
        && base == 0
        && first == 0;
    static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if !fits && SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
        tracing::info!(
            count,
            instances,
            start,
            base,
            first,
            "flat instances: a record drawn as before"
        );
    }
    fits
}

/// The buffers for a flat draw of `pattern` (`per` of the title's indices:
/// a quad's two triangles, or a rectangle's three corners) over `vertices`
/// (slot, stride, the four vertices' bytes), made once per distinct set and
/// kept for the run: index `per * instance + k` is
/// `4 * instance + pattern[k]`, and each vertex
/// buffer holds its four vertices once per instance.
fn flat_buffers(pattern: &[u32], vertices: &[(u32, u32, Vec<u8>)]) -> Option<std::sync::Arc<FlatBuffers>> {
    use std::collections::HashMap;
    static MADE: std::sync::Mutex<Option<HashMap<[u8; 32], std::sync::Arc<FlatBuffers>>>> = std::sync::Mutex::new(None);
    let mut key = blake3::Hasher::new();
    for i in pattern {
        key.update(&i.to_le_bytes());
    }
    for (slot, stride, bytes) in vertices {
        key.update(&slot.to_le_bytes());
        key.update(&stride.to_le_bytes());
        key.update(bytes);
    }
    let key = *key.finalize().as_bytes();
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    let made = made.get_or_insert_with(HashMap::new);
    if let Some(found) = made.get(&key) {
        return Some(found.clone());
    }
    // Up to a few distinct sets (a quad's vertices rarely change).
    if made.len() >= 16 {
        return None;
    }
    let host = crate::host::host()?;
    let most = crate::indirect::FLAT_MOST as usize;
    let per = pattern.len();
    let index_bytes = (most * per * 4) as u64;
    let (index, data) = host.upload_buffer(index_bytes)?;
    crate::host::name(&index, || "flat instances: indices".to_owned());
    // SAFETY: the mapping of `index_bytes` bytes, ours.
    let indices = unsafe { std::slice::from_raw_parts_mut(data.cast::<u32>(), most * per) };
    for (n, i) in indices.iter_mut().enumerate() {
        *i = 4 * (n / per) as u32 + pattern[n % per];
    }
    // SAFETY: a live buffer.
    let index_view = D3D12_INDEX_BUFFER_VIEW {
        BufferLocation: unsafe { index.GetGPUVirtualAddress() },
        SizeInBytes: index_bytes as u32,
        Format: DXGI_FORMAT(42),
    };
    let mut kept = vec![index];
    let mut views = Vec::new();
    for (slot, stride, bytes) in vertices {
        let total = (bytes.len() * most) as u64;
        let (buffer, data) = host.upload_buffer(total)?;
        crate::host::name(&buffer, || format!("flat instances: vertices of slot {slot}"));
        for n in 0..most {
            // SAFETY: inside the mapping of `total` bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.add(n * bytes.len()), bytes.len()) };
        }
        views.push((
            *slot,
            D3D12_VERTEX_BUFFER_VIEW {
                // SAFETY: a live buffer.
                BufferLocation: unsafe { buffer.GetGPUVirtualAddress() },
                SizeInBytes: total as u32,
                StrideInBytes: *stride,
            },
        ));
        kept.push(buffer);
    }
    // The copied vertices, as words and as floats (what a check compares).
    let words: Vec<String> = vertices
        .iter()
        .map(|(slot, stride, bytes)| {
            let w: Vec<String> = bytes
                .chunks_exact(4)
                .map(|c| {
                    let u = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    format!("{u:#x}={}", f32::from_bits(u))
                })
                .collect();
            format!("slot {slot} stride {stride}: [{}]", w.join(", "))
        })
        .collect();
    tracing::info!(?pattern, slots = vertices.len(), vertices = %words.join("; "), "flat instances: buffers made");
    let buffers = std::sync::Arc::new(FlatBuffers { index: index_view, vertices: views, _kept: kept });
    made.insert(key, buffers.clone());
    Some(buffers)
}

/// The console's own primitive topologies (and the strip both have).
const TRIANGLE_STRIP: u32 = 5;
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

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

/// The title's depth bounds as the host takes them: within 0..=1 (D3D12
/// wants both there). Bounds that are no range (min above max, NaN) pass
/// every depth, as with the test dropped, rather than none: what the
/// console does with them is not known. Reversed depth changes nothing:
/// the bounds are depth values, compared with the stored depth as is.
fn depth_bounds_for(min: f32, max: f32) -> (f32, f32) {
    let (lo, hi) = (min.clamp(0.0, 1.0), max.clamp(0.0, 1.0));
    if min.is_nan() || max.is_nan() || lo > hi {
        static ODD: AtomicU64 = AtomicU64::new(0);
        if ODD.fetch_add(1, Ordering::Relaxed) < 4 {
            tracing::info!(min, max, "depth bounds that are no range: every depth passes");
        }
        return (0.0, 1.0);
    }
    (lo, hi)
}

/// Sets depth bounds on the host list (`ID3D12GraphicsCommandList1`).
fn depth_bounds(list: &HostList, (min, max): (f32, f32)) {
    if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
        // SAFETY: an open list.
        unsafe { list1.OMSetDepthBounds(min, max) };
    }
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || format!("STENCIL REF {reference:#x}"));
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

/// Passes over textures picked by index that took the loading budget
/// ([`loading_frame`]).
pub(crate) static HEAP_TEXTURES_PRELOAD: AtomicU64 = AtomicU64::new(0);
/// Of the heap passes' microseconds, since the last frame-rate line: the
/// slots looked at, the buffer views' refreshes, the textures picked by
/// index; and of those, making the slots' host views (`mirror`: host
/// textures made) and making them readable (staging buffers made, uploads
/// recorded).
pub(crate) static HEAP_SPLIT_US: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Whether the frame before `frame` drew under `VICTORY_PRELOAD_DRAWS`
/// (400) draws: a loading screen or a frame with little on it, whose time
/// goes to loading what comes next (`VICTORY_PRELOAD_DRAWS=0`: never).
/// Whether the title is loading now (its last frame drew little).
pub(crate) fn loading_now() -> bool {
    loading_frame(crate::frame::current())
}

fn loading_frame(frame: u64) -> bool {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    static SEEN: AtomicU64 = AtomicU64::new(u64::MAX);
    static AT_START: AtomicU64 = AtomicU64::new(0);
    static LAST: AtomicU64 = AtomicU64::new(u64::MAX);
    let most = *MOST.get_or_init(|| {
        std::env::var("VICTORY_PRELOAD_DRAWS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(400)
    });
    let ended = SEEN.swap(frame, Ordering::Relaxed);
    if ended != frame {
        // Draws asked for, not run: a world frame whose pipelines are
        // still being made drops most of its draws and is no loading one
        // (cold run bd5b339 A kept starting every pipeline in the world).
        let now = ASKED.load(Ordering::Relaxed);
        let before = AT_START.swap(now, Ordering::Relaxed);
        let drew = now.saturating_sub(before);
        LAST.store(drew, Ordering::Relaxed);
        // The pictures the frame that ended drew go full size if it was a
        // loading screen's.
        #[cfg(windows)]
        crate::resource::loading_full_frame(ended, most > 0 && drew < most);
    }
    let last = LAST.load(Ordering::Relaxed);
    most > 0 && last < most
}

/// D3D12 `D3D12_SHADING_RATE` raw values supported on all VRS Tier 1/2 GPUs.
const VRS_1X1: u32 = 0; // D3D12_SHADING_RATE_1X1
const VRS_1X2: u32 = 1; // D3D12_SHADING_RATE_1X2
const VRS_2X1: u32 = 4; // D3D12_SHADING_RATE_2X1
const VRS_2X2: u32 = 5; // D3D12_SHADING_RATE_2X2

/// Heavy fullscreen / tiled-lighting pixel shader passes measured in the
/// `GPU_TIMING` run (`run-20261126-035934-essai-gputime.log`):
/// - `456f02b5` / `e732a8b4`: 8.54 ms/frame (TiledLighting)
/// - `6318f669` / `7affbd5b`: 1.86 ms/frame (fullscreen lighting/post pass)
/// - `25ebc920`: 1.38 ms/frame (`48e74d16` 0.55 + `1cd9bc63` 0.50 + `e5a6ec76` 0.33)
/// - `7f99b1cc` / `3c0e9845`: 0.26 ms/frame
// Pixel shader hash prefixes for the heavy fullscreen / tiled-lighting passes
// measured in `gputime` (`456f02b5` 8.54 ms, `6318f669` 1.86 ms,
// `48e74d16`/`1cd9bc63`/`e5a6ec76` 1.38 ms, `3c0e9845` 0.26 ms).
const VRS_TILES_PIXEL_HASHES: [&str; 6] = [
    "456f02b5",
    "6318f669",
    "48e74d16",
    "1cd9bc63",
    "e5a6ec76",
    "3c0e9845",
];

#[derive(Clone, Default)]
struct VrsConfig {
    raw: String,
    /// `0` = off, else `VRS_1X2` / `VRS_2X1` / `VRS_2X2`.
    rate: u32,
    /// Apply to all single-RT pixel draws (excluding skin bake and MRT G-buffer).
    all: bool,
    /// When true, match `hashes` only against the pixel shader portion (`pixel ...`).
    pixel_only: bool,
    /// Hash prefixes to shade at `rate`.
    hashes: Vec<String>,
}

impl VrsConfig {
    fn rate_for(&self, pipeline_name: &str, targets_len: usize) -> u32 {
        if self.rate == VRS_1X1 || targets_len == 0 {
            return VRS_1X1;
        }
        if BAKE_PIXEL_SHADERS.iter().any(|h| pipeline_name.contains(h)) {
            return VRS_1X1;
        }
        if self.all {
            // Exclude MRT passes (G-buffer: normals, roughness, motion vectors).
            return if targets_len == 1 { self.rate } else { VRS_1X1 };
        }
        let haystack = if self.pixel_only {
            match pipeline_name.split_once("pixel ") {
                Some((_, ps)) => ps,
                None => return VRS_1X1,
            }
        } else {
            pipeline_name
        };
        if self.hashes.iter().any(|h| haystack.contains(h.as_str())) {
            self.rate
        } else {
            VRS_1X1
        }
    }
}

fn parse_vrs_rate(token: &str) -> Option<u32> {
    match token.trim().to_ascii_lowercase().as_str() {
        "1x1" => Some(VRS_1X1),
        "1x2" => Some(VRS_1X2),
        "2x1" => Some(VRS_2X1),
        "2x2" => Some(VRS_2X2),
        _ => None,
    }
}

fn parse_vrs(raw: &str) -> VrsConfig {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("0")
        || trimmed.eq_ignore_ascii_case("off")
        || trimmed.eq_ignore_ascii_case("none")
    {
        return VrsConfig::default();
    }
    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "1" | "tiles" | "lights" => VrsConfig {
            raw: trimmed.to_owned(),
            rate: VRS_2X2,
            all: false,
            pixel_only: true,
            hashes: VRS_TILES_PIXEL_HASHES.iter().map(|&s| s.to_owned()).collect(),
        },
        "tiles2x1" | "lights2x1" => VrsConfig {
            raw: trimmed.to_owned(),
            rate: VRS_2X1,
            all: false,
            pixel_only: true,
            hashes: VRS_TILES_PIXEL_HASHES.iter().map(|&s| s.to_owned()).collect(),
        },
        "all" | "2x2" => VrsConfig {
            raw: trimmed.to_owned(),
            rate: VRS_2X2,
            all: true,
            pixel_only: false,
            hashes: Vec::new(),
        },
        "2x1" => VrsConfig {
            raw: trimmed.to_owned(),
            rate: VRS_2X1,
            all: true,
            pixel_only: false,
            hashes: Vec::new(),
        },
        "1x2" => VrsConfig {
            raw: trimmed.to_owned(),
            rate: VRS_1X2,
            all: true,
            pixel_only: false,
            hashes: Vec::new(),
        },
        _ => {
            let (rate, list) = match trimmed.split_once(':') {
                Some((r, rest)) if parse_vrs_rate(r).is_some() => {
                    (parse_vrs_rate(r).unwrap_or(VRS_2X2), rest)
                }
                _ => (VRS_2X2, trimmed),
            };
            let hashes: Vec<String> = list
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect();
            if hashes.is_empty() || rate == VRS_1X1 {
                VrsConfig::default()
            } else {
                VrsConfig {
                    raw: trimmed.to_owned(),
                    rate,
                    all: false,
                    pixel_only: false,
                    hashes,
                }
            }
        }
    }
}

/// Live per-frame cache of `VICTORY_VRS` (refreshed once per frame so it is
/// toggleable live via `VICTORY_LIVE` without per-draw env syscalls, locks, or
/// heap allocations).
static VRS_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
/// Whether `VICTORY_VRS` asked for a coarse rate at least once in this run:
/// until then [`State::apply_vrs`] records nothing (no `RSSetShadingRate`
/// at all, as before VRS was added; with VRS off it used to set 1x1 again
/// after every `host_state_lost`).
static VRS_EVER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static VRS_ACTIVE_RATE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(VRS_1X1);
static VRS_CACHE: std::sync::RwLock<Option<VrsConfig>> = std::sync::RwLock::new(None);
/// `[0]` draws shaded at a coarse VRS rate, `[1]` `RSSetShadingRate` calls.
static VRS_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

fn vrs_refresh() {
    let frame = crate::frame::current();
    if VRS_FRAME.load(Ordering::Relaxed) == frame {
        return;
    }
    // Through `crate::live`: a VICTORY_LIVE file change applies (read once a frame).
    let raw = crate::live::var("VICTORY_VRS").unwrap_or_default();
    let cfg = parse_vrs(&raw);
    let rate = cfg.rate;
    if rate != VRS_1X1 {
        VRS_EVER.store(true, Ordering::Release);
    }
    *VRS_CACHE.write().unwrap_or_else(|e| e.into_inner()) = Some(cfg);
    VRS_ACTIVE_RATE.store(rate, Ordering::Release);
    VRS_FRAME.store(frame, Ordering::Release);
}

pub(crate) fn vrs_mode_label() -> Option<String> {
    vrs_refresh();
    if VRS_ACTIVE_RATE.load(Ordering::Acquire) == VRS_1X1 {
        return None;
    }
    let guard = VRS_CACHE.read().unwrap_or_else(|e| e.into_inner());
    guard
        .as_ref()
        .filter(|cfg| cfg.rate != VRS_1X1)
        .map(|cfg| cfg.raw.clone())
}

pub(crate) fn take_vrs_stats() -> [u64; 2] {
    [
        VRS_STATS[0].swap(0, Ordering::Relaxed),
        VRS_STATS[1].swap(0, Ordering::Relaxed),
    ]
}

impl State {
    fn apply_vrs(&mut self, list: &HostList, pipeline_name: &str) {
        if crate::host::vrs_tier() == 0 {
            return;
        }
        vrs_refresh();
        if !VRS_EVER.load(Ordering::Acquire) {
            return;
        }
        let wanted = if self.targets.is_empty() || VRS_ACTIVE_RATE.load(Ordering::Acquire) == VRS_1X1 {
            VRS_1X1
        } else {
            let guard = VRS_CACHE.read().unwrap_or_else(|e| e.into_inner());
            guard
                .as_ref()
                .map_or(VRS_1X1, |cfg| cfg.rate_for(pipeline_name, self.targets.len()))
        };
        if wanted != VRS_1X1 {
            VRS_STATS[0].fetch_add(1, Ordering::Relaxed);
        }
        if self.host_vrs == wanted {
            return;
        }
        if let Ok(l5) = list.list.cast::<ID3D12GraphicsCommandList5>() {
            // SAFETY: open direct command list; Tier 1/2 verified via OPTIONS6;
            // `None` combiners defaults to PASSTHROUGH.
            unsafe {
                l5.RSSetShadingRate(D3D12_SHADING_RATE(wanted as i32), None);
            }
            self.host_vrs = wanted;
            VRS_STATS[1].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Per-frame texture staging byte budget scaled from the adapter's local VRAM
/// budget (`None` keeps the 192 MiB 12 GiB baseline).
fn staging_budget_per_frame(vram_budget: Option<u64>) -> u64 {
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    match vram_budget {
        Some(budget) if budget <= 6 * GIB => 96 * MIB,
        Some(budget) if budget >= 20 * GIB => 384 * MIB,
        Some(budget) if budget > 12 * GIB => 256 * MIB,
        _ => 192 * MIB,
    }
}

#[cfg(test)]
mod tests {
    use super::{VRS_1X1, VRS_1X2, VRS_2X1, VRS_2X2, parse_vrs, staging_budget_per_frame};

    #[test]
    fn vrs_config_parses_modes_and_protects_skin_bake() {
        assert_eq!(parse_vrs("").rate, VRS_1X1);
        assert_eq!(parse_vrs("off").rate, VRS_1X1);
        let tiles = parse_vrs("tiles");
        assert_eq!(tiles.rate_for("[vertex e732a8b4bea4c0dd, pixel 456f02b52910aabc]", 1), VRS_2X2);
        assert_eq!(tiles.rate_for("[vertex 25ebc92075b62822, pixel 48e74d16cbfa45e3]", 1), VRS_2X2);
        // Unrelated pixel shader sharing the same VS must stay 1x1.
        assert_eq!(tiles.rate_for("[vertex 25ebc92075b62822, pixel deadbeef12345678]", 1), VRS_1X1);
        // Skin bake and depth-only stay 1x1.
        assert_eq!(tiles.rate_for("[mesh f0c7efd2aceb2aaa, pixel 7d85c9cab3c80721]", 1), VRS_1X1);
        assert_eq!(tiles.rate_for("[vertex e732a8b4bea4c0dd, pixel 456f02b52910aabc]", 0), VRS_1X1);
        // Global mode protects MRT G-buffer (targets_len > 1).
        let all = parse_vrs("2x2");
        assert_eq!(all.rate_for("[vertex aaaa, pixel bbbb]", 1), VRS_2X2);
        assert_eq!(all.rate_for("[vertex aaaa, pixel bbbb]", 4), VRS_1X1);
        let custom = parse_vrs("2x1:456f02b5, 6318f669");
        assert_eq!(custom.rate_for("[vertex e732a8b4, pixel 456f02b52910aabc]", 1), VRS_2X1);
        assert_eq!(custom.rate_for("[vertex 7affbd5b, pixel 6318f669e4f17d65]", 1), VRS_2X1);
        let all = parse_vrs("1x2");
        assert_eq!(all.rate_for("[mesh f0c7efd2, pixel 7d85c9ca]", 1), VRS_1X2);
        // Skin bake pixel shaders always stay at 1x1.
        assert_eq!(all.rate_for("[vertex 1234, pixel c0a7fe77b1862e10]", 1), VRS_1X1);
    }

    #[test]
    fn staging_budget_scales_with_vram_budget() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        assert_eq!(staging_budget_per_frame(None), 192 * MIB);
        assert_eq!(staging_budget_per_frame(Some(4 * GIB)), 96 * MIB);
        assert_eq!(staging_budget_per_frame(Some(6 * GIB)), 96 * MIB);
        assert_eq!(staging_budget_per_frame(Some(8 * GIB)), 192 * MIB);
        assert_eq!(staging_budget_per_frame(Some(12 * GIB)), 192 * MIB);
        assert_eq!(staging_budget_per_frame(Some(13 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_per_frame(Some(14 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_per_frame(Some(16 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_per_frame(Some(20 * GIB)), 384 * MIB);
        assert_eq!(staging_budget_per_frame(Some(24 * GIB)), 384 * MIB);
        assert_eq!(staging_budget_per_frame(Some(32 * GIB)), 384 * MIB);
    }
}
