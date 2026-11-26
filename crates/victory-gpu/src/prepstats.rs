//! `VICTORY_PREP_STATS=1`: where the CPU's preparing of a frame goes.
//!
//! The frame line's "preparing" (`crate::host` `SUBMISSION_US[0]`) is the
//! time from a submission's start to its lists going to the card, on the
//! submitting thread. Here it is cut into its pieces (the asking of the
//! write-watch and the checks against flights, the async compute
//! bookkeeping, the copies between lists, each kind of deferred step, the
//! preludes, the flush of refreshes and textures, the snapshots, the card
//! copies, the settles of flights inside it), and the work done for each
//! draw on the title's recording threads (`crate::draw`: state, tables,
//! barriers, bindings; the pass over the textures picked by index) is
//! timed too. One INFO line per frame-rate line: the ten largest in ms a
//! frame, with how many times each ran, and the microseconds a draw takes.
//! Off: one relaxed load per piece.

use std::sync::atomic::{AtomicU64, Ordering};

/// The pieces timed.
#[derive(Clone, Copy)]
pub(crate) enum P {
    /// The write-watch's asking for the submission, and the checks of its
    /// refreshes against flights (without the settles).
    AskAndFlights,
    /// The textures touched and their transitions for the compute queue.
    AsyncCompute,
    /// Card copies between host copies of the same title memory
    /// (`cross_list_copies`).
    CrossCopies,
    /// `Step::Copy`: title copies (CPU).
    TitleCopies,
    /// `Step::Fill`.
    Fills,
    /// `Step::Run`: the title's own work at submission.
    TitleRuns,
    /// `Step::Host`.
    HostRuns,
    /// `Step::GpuWrite`: the copy's refresh first, then its snapshot.
    GpuWrites,
    /// `Step::Texture` (refreshed here, or queued for the flush).
    Textures,
    /// A list's prelude (twins, copies between aliases, starting states).
    Preludes,
    /// The flush of the batched refreshes and textures.
    BatchFlush,
    /// `crate::shadow::take_snapshots`.
    Snapshots,
    /// `card_before`: the card's copies of what changed.
    CardBefore,
    /// Settles of flights inside the preparing (waits for the card).
    Settles,
    /// The whole preparing (as the frame line's, settles included).
    Whole,
    /// A draw's preparation on a title thread (`State::prepare_for`).
    DrawPrepare,
    /// The pass over textures picked by index (`State::heap_textures`).
    HeapTextures,
}

const N: usize = 17;
const NAMES: [&str; N] = [
    "write-watch asking + flight checks",
    "async compute bookkeeping",
    "card copies between aliases",
    "title copies (Copy steps)",
    "fills (Fill steps)",
    "title work (Run steps)",
    "host work (Host steps)",
    "GPU-write steps (refresh + snapshot)",
    "texture steps",
    "list preludes",
    "batch flush (refreshes, textures)",
    "snapshots",
    "card copies before (card_before)",
    "settles of flights",
    "whole preparing",
    "draw preparation (title threads, summed)",
    "textures picked by index pass (title threads, summed)",
];

static US: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static COUNT: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];

/// Whether the pieces are timed (`VICTORY_PREP_STATS=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PREP_STATS").is_some_and(|v| v == "1"))
}

/// The start of a timed piece (`None` while off).
#[inline]
pub(crate) fn start() -> Option<std::time::Instant> {
    on().then(std::time::Instant::now)
}

/// A piece started at `at` is done.
#[inline]
pub(crate) fn done(p: P, at: Option<std::time::Instant>) {
    if let Some(at) = at {
        add(p, at.elapsed());
    }
}

/// Adds `took` to piece `p` (while on).
pub(crate) fn add(p: P, took: std::time::Duration) {
    if !on() {
        return;
    }
    US[p as usize].fetch_add(took.as_micros() as u64, Ordering::Relaxed);
    COUNT[p as usize].fetch_add(1, Ordering::Relaxed);
}

/// The INFO line for the frame-rate line, over `frames` frames.
pub(crate) fn line(frames: f64) {
    if !on() || frames <= 0.0 {
        return;
    }
    let taken: Vec<(usize, u64, u64)> =
        (0..N).map(|i| (i, US[i].swap(0, Ordering::Relaxed), COUNT[i].swap(0, Ordering::Relaxed))).collect();
    let draws = taken[P::DrawPrepare as usize];
    let per_draw = if draws.2 > 0 { draws.1 as f64 / draws.2 as f64 } else { 0.0 };
    let mut top: Vec<&(usize, u64, u64)> = taken.iter().filter(|t| t.1 > 0).collect();
    top.sort_unstable_by_key(|t| std::cmp::Reverse(t.1));
    let text: Vec<String> = top
        .iter()
        .take(10)
        .map(|&&(i, us, n)| format!("{} {:.1} ms ({:.0}x)", NAMES[i], us as f64 / 1000.0 / frames, n as f64 / frames))
        .collect();
    tracing::info!(
        "prep (ms a frame, times a frame): {}; draws prepared {:.0} a frame at {per_draw:.1} us each",
        text.join("; "),
        draws.2 as f64 / frames,
    );
}

/// Times piece `.0` until dropped (while on).
pub(crate) struct Timed(P, Option<std::time::Instant>);

impl Timed {
    #[inline]
    pub(crate) fn new(p: P) -> Timed {
        Timed(p, start())
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        done(self.0, self.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_cover_every_piece() {
        assert_eq!(NAMES.len(), P::HeapTextures as usize + 1);
    }
}
