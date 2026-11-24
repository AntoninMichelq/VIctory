//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each, and a title's first world
//! frame can request over 1,200 new pipelines: making them synchronously on
//! recording threads stalls the frame for tens of seconds and can trigger a
//! GPU watchdog timeout (decision 0007, "Pipelines made off the title's
//! threads").
//!
//! A first use queues the pipeline and waits for it within an atomic per-frame
//! budget shared across all recording threads, so cheap or one-off passes
//! during startup/warmup are ready immediately while steady-state frames stay
//! responsive without multi-hundred-millisecond freezes. Past the budget, a
//! draw or dispatch whose pipeline is not ready is dropped while the worker
//! finishes compiling it in the background.
//!
//! By default the budget is adaptive: [`WARMUP_BUDGET_MS`] (500 ms) during
//! early warmup frames (`frame < 60`) so boot, menu, and one-shot init passes
//! compile reliably, then [`STEADY_BUDGET_MS`] (8 ms) once presenting steadily
//! so live gameplay avoids large compile hitches. Setting
//! `VICTORY_PSO_WAIT_MS` (`0..=5000`) overrides the per-frame budget for every
//! frame (`0` disables synchronous waiting completely).
//!
//! Workers take the pipelines the latest title frame asked for first: a
//! backlog of hundreds is served newest frame first, so what is on screen
//! now is made before what an earlier frame (or the last run's early list)
//! wanted.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// Per-pipeline wake pair so unrelated background compiles do not wake
    /// threads waiting on a different pipeline.
    wait_lock: Mutex<()>,
    wait_done: Condvar,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
    /// One more than the last title frame a draw asked for it (0: only
    /// started early): the queue serves the most recent frame first.
    wanted: AtomicU64,
    /// A worker has started making it.
    taken: AtomicBool,
    /// Started early and not made yet (counted in [`EARLY_LEFT`]).
    early: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(
            graphics
                .as_ref()
                .map(|g| g.describe())
                .unwrap_or_default()
                .as_bytes(),
        );
        Pending {
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            made: OnceLock::new(),
            wait_lock: Mutex::new(()),
            wait_done: Condvar::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            wanted: AtomicU64::new(0),
            taken: AtomicBool::new(false),
            early: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        let started = Instant::now();
        let made = self
            .made
            .get_or_init(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
        let us = started.elapsed().as_micros() as u64;
        STATS.us.fetch_add(us, Ordering::Relaxed);
        STATS.max_us.fetch_max(us, Ordering::Relaxed);
        let counter = if made.is_some() { &STATS.ok } else { &STATS.failed };
        counter.fetch_add(1, Ordering::Relaxed);
        if self.early.swap(false, Ordering::Relaxed) {
            EARLY_LEFT.fetch_sub(1, Ordering::Relaxed);
        }
        let _held = self.wait_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.wait_done.notify_all();
    }
}

/// Pipelines started early (the last run drew with them) not made yet.
static EARLY_LEFT: AtomicU64 = AtomicU64::new(0);

/// Pipelines started early not made yet.
pub(crate) fn early_left() -> u64 {
    EARLY_LEFT.load(Ordering::Relaxed)
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Number of initial presented frames that receive [`WARMUP_BUDGET_MS`] so
/// boot, menus, and one-shot initialization passes compile synchronously.
const WARMUP_FRAMES: u64 = 60;
/// Per-frame wait budget during initial warmup frames.
const WARMUP_BUDGET_MS: u32 = 500;
/// Per-frame wait budget during steady-state presentation (keeps cached/fast
/// PSOs immediate without causing multi-frame hitches on slow compiles).
const STEADY_BUDGET_MS: u32 = 8;
/// Upper bound on `VICTORY_PSO_WAIT_MS` (5 seconds).
const MAX_WAIT_MS: u32 = 5_000;

/// What the workers did since the last progress line.
struct Stats {
    ok: AtomicU64,
    failed: AtomicU64,
    /// Time making them (translation, root signature, library load or
    /// host compile), total and longest, in microseconds.
    us: AtomicU64,
    max_us: AtomicU64,
}

static STATS: Stats = Stats {
    ok: AtomicU64::new(0),
    failed: AtomicU64::new(0),
    us: AtomicU64::new(0),
    max_us: AtomicU64::new(0),
};

/// Every 5 s while there is pipeline work: one line on the backlog and
/// what the workers made since the last one.
fn progress() {
    use super::library::{COMPILE_US, LOADED, MADE, SHARED};
    let (mut made, mut loaded, mut shared, mut compile_us) = (0, 0, 0, 0);
    let (mut hits, mut misses) = crate::shader_cache::counts();
    loop {
        std::thread::sleep(Duration::from_secs(5));
        let ok = STATS.ok.swap(0, Ordering::Relaxed);
        let failed = STATS.failed.swap(0, Ordering::Relaxed);
        let us = STATS.us.swap(0, Ordering::Relaxed);
        let max_us = STATS.max_us.swap(0, Ordering::Relaxed);
        let queued = WAITING.load(Ordering::Relaxed);
        let delta = |total: &AtomicU64, last: &mut u64| {
            let now = total.load(Ordering::Relaxed);
            now.saturating_sub(std::mem::replace(last, now))
        };
        let compiled = delta(&MADE, &mut made);
        let from_library = delta(&LOADED, &mut loaded);
        let deduped = delta(&SHARED, &mut shared);
        let compile_ms = delta(&COMPILE_US, &mut compile_us) / 1000;
        let (h, m) = crate::shader_cache::counts();
        let shader_hits = h.saturating_sub(std::mem::replace(&mut hits, h));
        let shader_translated = m.saturating_sub(std::mem::replace(&mut misses, m));
        if queued == 0 && ok == 0 && failed == 0 {
            continue;
        }
        let done = ok + failed;
        tracing::info!(
            queued,
            made = ok,
            failed,
            compiled,
            from_library,
            deduped,
            shader_hits,
            shader_translated,
            avg_ms = us.checked_div(done).unwrap_or(0) / 1000,
            max_ms = max_us / 1000,
            host_compile_avg_ms = compile_ms.checked_div(compiled).unwrap_or(0),
            queue_entries = QUEUE.lock().unwrap_or_else(|e| e.into_inner()).heap.len(),
            "pipeline workers (last 5 s)"
        );
    }
}

/// Bit packing for [`FrameBudget`]: upper 36 bits hold the frame tag and lower
/// 28 bits hold reserved microseconds (up to ~268 s per frame).
const RESERVED_BITS: u32 = 28;
const RESERVED_MASK: u64 = (1u64 << RESERVED_BITS) - 1;
const FRAME_TAG_MASK: u64 = (1u64 << (64 - RESERVED_BITS)) - 1;
const MAX_RESERVED_US: u32 = RESERVED_MASK as u32;
/// Sentinel frame tag before any frame has waited.
const NO_FRAME_TAG: u64 = FRAME_TAG_MASK;

/// Lock-free per-frame wait-budget tracker shared across recording threads.
///
/// Packs `(frame_tag, reserved_us)` into a single `AtomicU64` so concurrent
/// callers can never observe a torn frame reset or collectively reserve more
/// than the frame's wait budget. Callers reserve remaining budget upfront and
/// refund unused microseconds if the pipeline finishes early (only if the same
/// frame is still active).
struct FrameBudget {
    state: AtomicU64,
}

impl FrameBudget {
    const fn new() -> Self {
        Self {
            state: AtomicU64::new(NO_FRAME_TAG << RESERVED_BITS),
        }
    }

    fn frame_tag(frame: u64) -> u64 {
        // Modulo `NO_FRAME_TAG` (`(1 << 36) - 1`) keeps the tag strictly in
        // `0..NO_FRAME_TAG` without collisions at wraparound.
        frame % NO_FRAME_TAG
    }

    fn is_newer_tag(tag: u64, cur_tag: u64) -> bool {
        if cur_tag == NO_FRAME_TAG {
            return true;
        }
        let diff = if tag >= cur_tag {
            tag - cur_tag
        } else {
            NO_FRAME_TAG - cur_tag + tag
        };
        diff > 0 && diff < (NO_FRAME_TAG / 2)
    }

    fn pack(tag: u64, reserved_us: u32) -> u64 {
        (tag << RESERVED_BITS) | u64::from(reserved_us)
    }

    fn unpack(raw: u64) -> (u64, u32) {
        (raw >> RESERVED_BITS, (raw & RESERVED_MASK) as u32)
    }

    /// Atomically reserves up to the remaining budget (in microseconds) for
    /// `frame`, resetting the counter only when `frame` advances past the
    /// active frame tag (stale tags from earlier frames receive 0 without
    /// altering active state).
    fn reserve(&self, frame: u64, limit_us: u32) -> u32 {
        let limit = limit_us.min(MAX_RESERVED_US);
        if limit == 0 {
            return 0;
        }
        let tag = Self::frame_tag(frame);
        let mut granted = 0u32;
        let _ = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                let (cur_tag, cur_used) = Self::unpack(cur);
                let used = if cur_tag == tag {
                    cur_used
                } else if Self::is_newer_tag(tag, cur_tag) {
                    0
                } else {
                    granted = 0;
                    return None;
                };
                let avail = limit.saturating_sub(used);
                granted = avail;
                if avail == 0 && cur_tag == tag {
                    return None;
                }
                Some(Self::pack(tag, used + avail))
            });
        granted
    }

    /// Returns `unused_us` back to `frame`'s budget if `frame` is still active.
    fn refund(&self, frame: u64, unused_us: u32) {
        if unused_us == 0 {
            return;
        }
        let tag = Self::frame_tag(frame);
        let _ = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                let (cur_tag, cur_used) = Self::unpack(cur);
                if cur_tag != tag {
                    return None;
                }
                Some(Self::pack(tag, cur_used.saturating_sub(unused_us)))
            });
    }
}

static FRAME_BUDGET: FrameBudget = FrameBudget::new();

fn parse_wait_ms(value: Option<&str>) -> Option<u32> {
    value?
        .trim()
        .parse::<u32>()
        .ok()
        .map(|ms| ms.min(MAX_WAIT_MS))
}

fn configured_wait_ms() -> Option<u32> {
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        let raw = std::env::var("VICTORY_PSO_WAIT_MS").ok();
        let parsed = parse_wait_ms(raw.as_deref());
        if let Some(ms) = parsed {
            tracing::info!(ms, "VICTORY_PSO_WAIT_MS: per-frame pipeline wait budget");
        } else if let Some(raw) = raw.filter(|s| !s.trim().is_empty()) {
            tracing::warn!(
                value = raw,
                "ignoring invalid VICTORY_PSO_WAIT_MS (expected integer milliseconds)"
            );
        }
        parsed
    })
}

fn frame_budget_us(frame: u64, override_ms: Option<u32>) -> u32 {
    let ms = override_ms.unwrap_or(if frame < WARMUP_FRAMES {
        WARMUP_BUDGET_MS
    } else {
        STEADY_BUDGET_MS
    });
    ms.saturating_mul(1_000)
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made.get().is_none() {
        let frame = crate::frame::current();
        start(p, frame.saturating_add(1));
        if p.graphics.is_none() && dispatches_wait() {
            wait_made(p, Duration::from_secs(5));
        } else {
            wait(p, frame);
        }
    }
    let made = p.made.get()?.as_ref();
    if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    made
}

/// Queues `p` for the workers as wanted by `wanted` (one more than a
/// title frame, 0 for an early start): again each time a newer frame asks
/// for it while no worker has started on it, so it moves to the front.
fn start(p: &Arc<Pending>, wanted: u64) {
    if !p.queued.swap(true, Ordering::Relaxed) {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        p.wanted.fetch_max(wanted, Ordering::Relaxed);
        queue(p.clone(), wanted);
    } else if !p.taken.load(Ordering::Relaxed) && p.wanted.fetch_max(wanted, Ordering::Relaxed) < wanted {
        queue(p.clone(), wanted);
    }
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        if p.made.get().is_none() && !p.early.swap(true, Ordering::Relaxed) {
            EARLY_LEFT.fetch_add(1, Ordering::Relaxed);
            // Made meanwhile: counted back once, by whoever clears the flag.
            if p.made.get().is_some() && p.early.swap(false, Ordering::Relaxed) {
                EARLY_LEFT.fetch_sub(1, Ordering::Relaxed);
            }
        }
        start(p, 0);
    }
}

/// `VICTORY_DISPATCH_WAIT=1` (opt-in): a compute dispatch whose pipeline is
/// not made yet waits for it (up to 5 s: the title ends itself after
/// ~30 s in one frame) instead of being dropped. A
/// dropped dispatch leaves data later passes read wrong (culling, light
/// lists, indirect arguments), so geometry built on it goes missing for
/// as long as it compiles; draws keep the frame budget.
fn dispatches_wait() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_DISPATCH_WAIT").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!("VICTORY_DISPATCH_WAIT: compute dispatches wait for their pipeline");
        }
        on
    })
}

/// Waits for `p` until it is made or `limit` passes, outside the frame
/// budget (counted as a pipeline wait in the frame CPU split).
fn wait_made(p: &Pending, limit: Duration) {
    // At most 10 s of such waits per title frame (summed over dispatches
    // and threads): the title ends itself after ~30 s in one frame.
    const PER_FRAME_US: u64 = 10_000_000;
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static SPENT_US: AtomicU64 = AtomicU64::new(0);
    let frame = crate::frame::current();
    if FRAME.swap(frame, Ordering::Relaxed) != frame {
        SPENT_US.store(0, Ordering::Relaxed);
    }
    let spent = SPENT_US.load(Ordering::Relaxed);
    if spent >= PER_FRAME_US {
        return;
    }
    let limit = limit.min(Duration::from_micros(PER_FRAME_US - spent));
    let started = Instant::now();
    let mut held = p.wait_lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = limit.saturating_sub(started.elapsed());
        if rest.is_zero() {
            tracing::warn!(key = %p.key, "a dispatch's pipeline not made within its wait; dropped");
            break;
        }
        held = p.wait_done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let us = started.elapsed().as_micros() as u64;
    SPENT_US.fetch_add(us, Ordering::Relaxed);
    crate::draw::BLOCKED_US[0].fetch_add(us, Ordering::Relaxed);
}

/// Waits for `p` while frame `frame` has budget left, reserving budget
/// atomically so concurrent recording threads cannot collectively overshoot it.
fn wait(p: &Pending, frame: u64) {
    let limit_us = frame_budget_us(frame, configured_wait_ms());
    let reserved_us = FRAME_BUDGET.reserve(frame, limit_us);
    if reserved_us == 0 {
        return;
    }
    let left = Duration::from_micros(u64::from(reserved_us));
    let started = Instant::now();
    let mut held = p.wait_lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = p
            .wait_done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    drop(held);
    crate::draw::BLOCKED_US[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    let elapsed_us = started.elapsed().as_micros().min(u128::from(reserved_us)) as u32;
    FRAME_BUDGET.refund(frame, reserved_us - elapsed_us);
}

/// How many workers make pipelines: `VICTORY_PIPELINE_THREADS` if set,
/// else all logical processors but two (2 to 16). They run below the
/// title's priority, so they mostly take time the title leaves idle; half
/// the processors (at most 8) left a backlog of hundreds of pipelines for
/// seconds in the world's first frames.
fn workers() -> usize {
    if let Some(n) = std::env::var("VICTORY_PIPELINE_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n.min(64);
    }
    std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(2).clamp(2, 16))
}

/// A queued pipeline, ordered by the title frame that last wanted it
/// (newest first), then by when it was queued (oldest first): the
/// pipelines the current frame draws with come before those of frames
/// gone by and before early starts (wanted 0), which are only guesses
/// from the last run.
struct Entry {
    wanted: u64,
    seq: u64,
    p: Arc<Pending>,
}

impl Entry {
    /// Superseded by a newer entry, or already made or being made.
    fn stale(&self) -> bool {
        self.p.taken.load(Ordering::Relaxed)
            || self.p.made.get().is_some()
            || self.p.wanted.load(Ordering::Relaxed) != self.wanted
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.wanted.cmp(&other.wanted).then(other.seq.cmp(&self.seq))
    }
}

struct Queue {
    heap: std::collections::BinaryHeap<Entry>,
    seq: u64,
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue {
    heap: std::collections::BinaryHeap::new(),
    seq: 0,
});
/// Signalled when an entry is queued.
static WORK: Condvar = Condvar::new();

/// Takes the next pipeline to make, waiting for one.
fn next() -> Arc<Pending> {
    let mut q = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        while let Some(e) = q.heap.pop() {
            if !e.stale() && !e.p.taken.swap(true, Ordering::Relaxed) {
                return e.p;
            }
        }
        q = WORK.wait(q).unwrap_or_else(|e| e.into_inner());
    }
}

/// Makes `p`, taken from the queue, and does the bookkeeping.
fn work(p: &Pending) {
    p.make();
    if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
        drained();
    } else {
        super::library::save(false);
    }
}

/// Hands `p` to the workers (see [`workers`]), started on first use.
fn queue(p: Arc<Pending>, wanted: u64) {
    static STARTED: OnceLock<usize> = OnceLock::new();
    let started = *STARTED.get_or_init(|| {
        let workers = workers();
        tracing::info!(workers, "pipeline workers");
        let _ = std::thread::Builder::new()
            .name("victory-pipelines-progress".into())
            .spawn(progress);
        let mut started = 0;
        for i in 0..workers {
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(move || {
                    // Below the title's own threads (loading, recording).
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    loop {
                        work(&next());
                    }
                });
            match spawned {
                Ok(_) => started += 1,
                Err(e) => tracing::warn!(error = %e, "no pipeline worker"),
            }
        }
        started
    });
    if started == 0 {
        // No worker: made here instead.
        if !p.taken.swap(true, Ordering::Relaxed) {
            work(&p);
        }
        return;
    }
    let mut q = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    // Entries superseded by a newer frame's pile up while a backlog lasts:
    // drop them once they outnumber the live ones.
    if q.heap.len() >= 4096 && q.heap.len() as u64 > 4 * WAITING.load(Ordering::Relaxed) {
        q.heap.retain(|e| !e.stale());
    }
    q.seq += 1;
    let seq = q.seq;
    q.heap.push(Entry { wanted, seq, p });
    drop(q);
    WORK.notify_one();
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn concurrent_budget_reservation_never_overshoots() {
        use std::sync::Barrier;

        let budget = FrameBudget::new();
        let limit_us = 8_000u32;
        let threads = 8;
        let barrier = Barrier::new(threads);
        let total_granted = AtomicU64::new(0);

        std::thread::scope(|s| {
            for _ in 0..threads {
                let budget = &budget;
                let barrier = &barrier;
                let total_granted = &total_granted;
                s.spawn(move || {
                    barrier.wait();
                    let g = budget.reserve(100, limit_us);
                    total_granted.fetch_add(u64::from(g), Ordering::SeqCst);
                });
            }
        });
        assert_eq!(
            total_granted.load(Ordering::SeqCst),
            u64::from(limit_us),
            "total reserved across all concurrent threads never exceeds the frame limit"
        );
        // No more budget remains on frame 100 until a caller refunds unused time.
        assert_eq!(budget.reserve(100, limit_us), 0);

        // Refunding unused microseconds makes them available to subsequent waits
        // on the same frame.
        budget.refund(100, 3_500);
        assert_eq!(budget.reserve(100, limit_us), 3_500);

        // Advancing to a new frame resets the budget atomically.
        assert_eq!(budget.reserve(101, limit_us), limit_us);

        // A stale refund from frame 100 must not alter frame 101's budget.
        budget.refund(100, 5_000);
        assert_eq!(budget.reserve(101, limit_us), 0);
    }

    #[test]
    fn stale_reservation_never_rewinds_active_frame_or_wraparound() {
        let budget = FrameBudget::new();
        let limit_us = 8_000u32;

        // Deterministic descheduled-thread interleaving: frame 101 reserves
        // first, then a delayed thread from frame 100 attempts to reserve.
        assert_eq!(budget.reserve(101, limit_us), limit_us);
        assert_eq!(
            (budget.reserve(100, limit_us), budget.reserve(101, limit_us)),
            (0, 0),
        );

        // Partial budget on frame 102 stays intact when stale frame 101 arrives.
        assert_eq!(budget.reserve(102, limit_us), limit_us);
        budget.refund(102, 2_500);
        assert_eq!(budget.reserve(101, limit_us), 0);
        assert_eq!(budget.reserve(102, limit_us), 2_500);

        // Wraparound across `NO_FRAME_TAG` (`(1 << 36) - 1`) boundary.
        let wrap_budget = FrameBudget::new();
        assert_eq!(wrap_budget.reserve(NO_FRAME_TAG - 1, limit_us), limit_us);
        assert_eq!(wrap_budget.reserve(NO_FRAME_TAG, limit_us), limit_us);
        assert_eq!(
            (
                wrap_budget.reserve(NO_FRAME_TAG - 1, limit_us),
                wrap_budget.reserve(NO_FRAME_TAG, limit_us),
            ),
            (0, 0),
        );
        assert_eq!(wrap_budget.reserve(NO_FRAME_TAG + 1, limit_us), limit_us);
        assert_eq!(wrap_budget.reserve(NO_FRAME_TAG, limit_us), 0);
    }

    #[test]
    fn wait_ms_parsing_clamping_and_adaptive_budget() {
        assert_eq!(parse_wait_ms(None), None);
        assert_eq!(parse_wait_ms(Some("")), None);
        assert_eq!(parse_wait_ms(Some("abc")), None);
        assert_eq!(parse_wait_ms(Some("-5")), None);
        assert_eq!(parse_wait_ms(Some("0")), Some(0));
        assert_eq!(parse_wait_ms(Some(" 16 ")), Some(16));
        assert_eq!(parse_wait_ms(Some("999999")), Some(MAX_WAIT_MS));

        // Adaptive defaults: generous warmup, tight steady-state.
        assert_eq!(frame_budget_us(0, None), WARMUP_BUDGET_MS * 1_000);
        assert_eq!(
            frame_budget_us(WARMUP_FRAMES - 1, None),
            WARMUP_BUDGET_MS * 1_000
        );
        assert_eq!(
            frame_budget_us(WARMUP_FRAMES, None),
            STEADY_BUDGET_MS * 1_000
        );
        assert_eq!(frame_budget_us(10_000, None), STEADY_BUDGET_MS * 1_000);

        // Explicit override applies to every frame (including 0 = never block).
        assert_eq!(frame_budget_us(0, Some(0)), 0);
        assert_eq!(frame_budget_us(100, Some(25)), 25_000);
    }
}
