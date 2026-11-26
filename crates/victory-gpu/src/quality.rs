//! `VICTORY_QUALITY_STATS=1`: a count of what the draws' textures show, to
//! tell without looking at shots whether textures are all there.
//!
//! Every five seconds one frame is sampled: each texture a draw reads
//! through its tables or roots (`crate::draw` `readable`) is classed once
//! for that frame (its host copy made or not, made small, cut, its console
//! memory mapped in whole, in part or not at all), and the reads are
//! counted by class. Once the frame is over one INFO line gives the
//! counts, the title slots of texture views in the shader-visible heaps by
//! what they show now (host copy, grey stand-in, null view: the textures
//! shaders pick by index, which the CPU does not see read), the refreshes
//! of buffer copies a flight's shaders were still writing, and the five
//! shaders with the most reads of textures that show zeros or only part of
//! their levels. Off: nothing is counted (one relaxed load per read).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::object::This;

/// Whether the counts are on (`VICTORY_QUALITY_STATS=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_QUALITY_STATS").is_some_and(|v| v == "1");
        if on {
            tracing::info!("VICTORY_QUALITY_STATS=1: what the draws' textures show, one frame every 5 s");
        }
        on
    })
}

/// The frame being sampled (`u64::MAX`: none).
static SAMPLED: AtomicU64 = AtomicU64::new(u64::MAX);
/// When the last sample started (`crate::host::now_ms`).
static LAST_MS: AtomicU64 = AtomicU64::new(0);

/// A texture's class in the frame sampled.
#[derive(Clone, Copy, Default)]
struct Class {
    /// No host copy (none could be made: refused, or no room).
    waiting: bool,
    /// Its console memory is not mapped at all: it reads zeros.
    zero: bool,
    /// Mapped in part (levels streaming in): the rest reads zeros.
    partly: bool,
    /// A small copy (`VICTORY_EVICT_LESS`).
    small: bool,
    /// Cut (`VICTORY_TEXTURE_HALF`, or small).
    cut: bool,
}

impl Class {
    fn degraded(self) -> bool {
        self.waiting || self.zero || self.partly
    }
}

#[derive(Default)]
struct Sample {
    frame: u64,
    reads: u64,
    /// Reads by class: waiting, zero, partly, small, cut.
    by: [u64; 5],
    seen: HashMap<usize, Class>,
    /// Degraded reads by shader.
    shaders: HashMap<String, u64>,
}

static SAMPLE: Mutex<Option<Sample>> = Mutex::new(None);

/// Refreshes of buffer copies whose title memory a flight's shaders were
/// still writing (`crate::host`): read before the GPU's bytes landed.
pub(crate) static BUFFERS_EARLY: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The shader of the draw being prepared on this thread.
    static SHADER: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Whether frame `frame` is sampled; starts a sample every 5 s, and logs
/// the last one once its frame is over.
fn sampling(frame: u64) -> bool {
    let at = SAMPLED.load(Ordering::Relaxed);
    if at == frame {
        return true;
    }
    let now = crate::host::now_ms();
    let last = LAST_MS.load(Ordering::Relaxed);
    if at != u64::MAX && at < frame {
        // The sampled frame is over.
        if SAMPLED.compare_exchange(at, u64::MAX, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            report();
        }
    }
    if now.saturating_sub(last) < 5000 {
        return false;
    }
    if LAST_MS.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return SAMPLED.load(Ordering::Relaxed) == frame;
    }
    *SAMPLE.lock().unwrap_or_else(|e| e.into_inner()) = Some(Sample { frame, ..Sample::default() });
    SAMPLED.store(frame, Ordering::Relaxed);
    true
}

/// A draw is prepared on this thread with these shaders (`hash`: the last
/// stage's, made only while a frame is sampled).
pub(crate) fn draw_shaders(hash: impl FnOnce() -> Option<String>) {
    if !on() || SAMPLED.load(Ordering::Relaxed) != crate::frame::current() {
        return;
    }
    let h = hash();
    SHADER.with(|s| *s.borrow_mut() = h);
}

/// A draw reads title texture `this` (`crate::draw` `readable`).
pub(crate) fn texture_read(this: This) {
    if !on() || this.is_null() {
        return;
    }
    let frame = crate::frame::current();
    if !sampling(frame) {
        return;
    }
    let mut sample = SAMPLE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = sample.as_mut().filter(|s| s.frame == frame) else {
        return;
    };
    let class = *s.seen.entry(this as usize).or_insert_with(|| classify(this));
    s.reads += 1;
    for (n, yes) in s.by.iter_mut().zip([class.waiting, class.zero, class.partly, class.small, class.cut]) {
        *n += u64::from(yes);
    }
    if class.degraded() {
        let shader = SHADER.with(|h| h.borrow().clone()).unwrap_or_else(|| "?".to_owned());
        *s.shaders.entry(shader).or_default() += 1;
    }
}

fn classify(this: This) -> Class {
    let Some((made, small, cut, memory, bytes)) = crate::resource::quality_of(this) else {
        return Class::default();
    };
    let mapped = if bytes == 0 { bytes } else { crate::texture::mapped_bytes(memory, bytes as usize) };
    Class {
        waiting: !made,
        zero: bytes > 0 && mapped == 0,
        partly: mapped > 0 && mapped < bytes,
        small,
        cut,
    }
}

/// The INFO line of the sample just over.
fn report() {
    let Some(s) = SAMPLE.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    let distinct = s.seen.len();
    let mut d = [0usize; 5];
    for c in s.seen.values() {
        for (n, yes) in d.iter_mut().zip([c.waiting, c.zero, c.partly, c.small, c.cut]) {
            *n += usize::from(yes);
        }
    }
    let [made, grey, null] = crate::descriptor::shown_census();
    let mut worst: Vec<(&String, &u64)> = s.shaders.iter().collect();
    worst.sort_unstable_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let worst: Vec<String> = worst.iter().take(5).map(|(h, n)| format!("{h} {n}")).collect();
    let [w, z, p, sm, c] = s.by;
    tracing::info!(
        "quality (frame {}): texture reads by draws {}: {z} zero (memory not mapped), {p} partly uploaded (levels missing), \
         {w} waiting (no host copy), {sm} small copies (EVICT_LESS), {c} cut (TEXTURE_HALF/small); distinct textures {distinct}: \
         {} zero, {} partly, {} waiting, {} small, {} cut; heap slots of texture views: {made} host copy, {grey} grey stand-in, \
         {null} null; buffers read before their copy landed {} (in all); worst shaders by zero/partly/waiting reads: [{}]",
        s.frame,
        s.reads,
        d[1],
        d[2],
        d[0],
        d[3],
        d[4],
        BUFFERS_EARLY.load(Ordering::Relaxed),
        worst.join(", "),
    );
}
