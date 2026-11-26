//! `VICTORY_COPY_STATS=1` (off by default): what goes between title memory
//! and the host copies, by copy, in an INFO table every frame line (5 s):
//! the 20 copies that move the most bytes a frame, each with its title
//! address and size, where its host copy lives (the PC's memory or the
//! card's), and its bytes a frame:
//! - `up`: read from title memory into the copy, because the title's CPU
//!   wrote the pages since the last refresh (write watch, or the guards of
//!   physical views; our own copy-back writes count too: the next refresh
//!   reads them again, an "echo");
//! - `sent`: of a copy in the card's memory, bytes sent from its CPU side
//!   to its card side;
//! - `declared`: ranges shaders were declared to write (unordered access);
//! - `snap`: of those, bytes snapshotted on the CPU before the submission
//!   (copies in the PC's memory) to find what the shaders wrote;
//! - `back`: bytes written back into title memory after the submission
//!   (found by the CPU compare, or fetched from the card).
//!
//! Also why big buffers were not given a card copy (`VICTORY_CARD_BUFFERS`).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// What a count is of (see the module's text).
pub(crate) const UP: usize = 0;
pub(crate) const SENT: usize = 1;
pub(crate) const DECLARED: usize = 2;
pub(crate) const SNAP: usize = 3;
pub(crate) const BACK: usize = 4;
/// Of `up`, for a copy in the card's memory with
/// `VICTORY_CARD_SEND_CHANGED=1`: bytes read that equalled the copy and
/// were not sent (mostly the echo of `back`).
pub(crate) const SAVED: usize = 5;
const KINDS: usize = 6;

/// One copy's counts since the last line.
struct Row {
    start: u64,
    len: u64,
    card: bool,
    window: bool,
    bytes: [u64; KINDS],
    times: [u64; KINDS],
}

static ROWS: Mutex<Option<HashMap<u64, Row>>> = Mutex::new(None);

/// Big title buffers (`VICTORY_CARD_BUFFERS_MIN_MB` or more) refused a card
/// copy since the last line: the card was not ready, the bytes of card
/// copies were at `VICTORY_CARD_BUFFERS_MB`, the card's memory in use was
/// past four fifths of its budget, the host refused the buffer.
pub(crate) static CARD_REFUSED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Their bytes, by the same causes.
pub(crate) static CARD_REFUSED_BYTES: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Whether the counts are kept (`VICTORY_COPY_STATS=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPY_STATS").is_some_and(|v| v == "1"))
}

/// Counts `bytes` of kind `kind` for the copy numbered `id` (title
/// memory from `start`, `len` bytes; in the card's memory: `card`; a
/// window of memory outside buffers: `window`).
pub(crate) fn add(id: u64, start: u64, len: u64, card: bool, window: bool, kind: usize, bytes: u64) {
    if bytes == 0 || kind >= KINDS {
        return;
    }
    let mut rows = ROWS.lock().unwrap_or_else(|e| e.into_inner());
    let row = rows.get_or_insert_with(HashMap::new).entry(id).or_insert_with(|| Row {
        start,
        len,
        card,
        window,
        bytes: [0; KINDS],
        times: [0; KINDS],
    });
    row.bytes[kind] += bytes;
    row.times[kind] += 1;
}

/// A big title buffer was refused a card copy for cause `cause` (see
/// [`CARD_REFUSED`]).
pub(crate) fn card_refused(cause: usize, bytes: u64) {
    if on() && cause < 4 {
        CARD_REFUSED[cause].fetch_add(1, Ordering::Relaxed);
        CARD_REFUSED_BYTES[cause].fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Microseconds spent merging what shaders wrote into title memory (the
/// CPU's part of "copying back": compare, read, merge, write), all threads.
static MERGE_US: AtomicU64 = AtomicU64::new(0);

/// A merge begun at `started` (with the counts on) has ended.
pub(crate) fn merged(started: Option<std::time::Instant>) {
    if let Some(t) = started {
        MERGE_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// Where the submitting thread's preparing goes, each counted alone (not
/// inside another), microseconds since the last table: see [`TIMES`].
static TIME_US: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];
/// Names of [`TIME_US`]'s counts, for the line.
const TIMES: [&str; 13] = [
    "all deferred steps (wall)",
    "flushes of refreshes (wall)",
    "write-watch asking ahead",
    "refreshes, write-watch asking (summed over threads)",
    "refreshes, copying stale pages (summed over threads)",
    "title-to-title copies",
    "fills of upload memory",
    "snapshots of shader-written ranges",
    "texture refreshes on the submitting thread",
    "deferred runs of ours",
    "the submission's one write-watch asking",
    "checks against flights (stale, aliased; merges under writes included)",
    "refreshes before write steps, their part only (VICTORY_FLUSH_PART)",
];
pub(crate) const T_ALL: usize = 0;
pub(crate) const T_FLUSH: usize = 1;
pub(crate) const T_AHEAD: usize = 2;
pub(crate) const T_ASK: usize = 3;
pub(crate) const T_COPY: usize = 4;
pub(crate) const T_TITLE: usize = 5;
pub(crate) const T_FILL: usize = 6;
pub(crate) const T_SNAP: usize = 7;
pub(crate) const T_TEXTURE: usize = 8;
pub(crate) const T_RUN: usize = 9;
pub(crate) const T_ONCE: usize = 10;
pub(crate) const T_CHECKS: usize = 11;
pub(crate) const T_PART: usize = 12;

/// A start for [`time`], if the counts are on.
pub(crate) fn start() -> Option<std::time::Instant> {
    on().then(std::time::Instant::now)
}

/// Counts the time since `started` (from [`start`]) as `kind`.
pub(crate) fn time(kind: usize, started: Option<std::time::Instant>) {
    if let (Some(t), Some(c)) = (started, TIME_US.get(kind)) {
        c.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// Why flights were settled before a submission (title memory stale under
/// their shader writes: `stale_under_flights`, `merge_under_writes`):
/// counts since the last table, by [`SETTLE_NAMES`].
static SETTLES: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
const SETTLE_NAMES: [&str; 8] = [
    "title memory stale under shader writes",
    "written by an earlier list of copies the flights alias",
    "of the first: a copy of guarded (not write-watched) memory",
    "the write watch gave no answer",
    "more than 512 changed runs for the card",
    "no snapshot to merge into",
    "waits for some flights only (VICTORY_SETTLE_PARTIAL=1)",
    "aliased: waits until no flight writes the ranges (VICTORY_ALIAS_PARTIAL=1)",
];
pub(crate) const S_STALE: usize = 0;
pub(crate) const S_ALIASED: usize = 1;
pub(crate) const S_UNWATCHED: usize = 2;
pub(crate) const S_NO_ANSWER: usize = 3;
pub(crate) const S_RUNS: usize = 4;
pub(crate) const S_NO_SNAPSHOT: usize = 5;
pub(crate) const S_PARTIAL: usize = 6;
pub(crate) const S_ALIAS_PARTIAL: usize = 7;

/// At presents since the last table: their count, the submissions in the
/// air summed and at most, the frames ahead summed and at most
/// (`crate::host::in_air`).
static AIR: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// A present found `n` submissions in the air, the oldest `frames` frames back.
pub(crate) fn in_air(n: u64, frames: u64) {
    AIR[0].fetch_add(1, Ordering::Relaxed);
    AIR[1].fetch_add(n, Ordering::Relaxed);
    AIR[2].fetch_max(n, Ordering::Relaxed);
    AIR[3].fetch_add(frames, Ordering::Relaxed);
    AIR[4].fetch_max(frames, Ordering::Relaxed);
}

/// A settle before a submission had cause `cause` ([`SETTLE_NAMES`]).
pub(crate) fn settle_cause(cause: usize) {
    if let Some(c) = SETTLES.get(cause) {
        c.fetch_add(1, Ordering::Relaxed);
    }
}

/// Frames since the last table.
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// `n` more frames went by (with the frame counts).
pub(crate) fn frames(n: u32) {
    if on() {
        FRAMES.fetch_add(u64::from(n), Ordering::Relaxed);
    }
}

/// The table, per frame since the last one (with the frame line).
pub(crate) fn line() {
    if !on() {
        return;
    }
    let rows = ROWS.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    let frames = (FRAMES.swap(0, Ordering::Relaxed) as f64).max(1.0);
    let mb = |b: u64| b as f64 / 1e6 / frames;
    let mut all: Vec<(u64, Row)> = rows.into_iter().collect();
    let total = |r: &Row| r.bytes[UP] + r.bytes[SENT] + r.bytes[SNAP] + r.bytes[BACK];
    let mut sums = [0u64; KINDS];
    for (_, r) in &all {
        for (s, b) in sums.iter_mut().zip(r.bytes) {
            *s += b;
        }
    }
    all.sort_unstable_by_key(|(_, r)| std::cmp::Reverse(total(r)));
    let refused = |i: usize| {
        (
            CARD_REFUSED[i].swap(0, Ordering::Relaxed),
            CARD_REFUSED_BYTES[i].swap(0, Ordering::Relaxed) >> 20,
        )
    };
    let [not_ready, cap, full, host] = [0, 1, 2, 3].map(refused);
    tracing::info!(
        "copy stats per frame (VICTORY_COPY_STATS=1), {} copies moved bytes: up {:.1} MB, sent to the card {:.1} MB, \
         declared shader writes {:.1} MB, snapshotted {:.1} MB, back {:.1} MB ({:.1} ms merging on the CPU), not sent as unchanged {:.1} MB; big buffers refused a card copy since the \
         last line: card not ready {} ({} MB), at VICTORY_CARD_BUFFERS_MB {} ({} MB), card past 80% of its budget {} ({} MB), \
         host refused {} ({} MB); the top {}:",
        all.len(),
        mb(sums[UP]),
        mb(sums[SENT]),
        mb(sums[DECLARED]),
        mb(sums[SNAP]),
        mb(sums[BACK]),
        MERGE_US.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames,
        mb(sums[SAVED]),
        not_ready.0,
        not_ready.1,
        cap.0,
        cap.1,
        full.0,
        full.1,
        host.0,
        host.1,
        all.len().min(20)
    );
    let times: Vec<String> = TIMES
        .iter()
        .zip(&TIME_US)
        .map(|(name, us)| format!("{name} {:.1} ms", us.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames))
        .collect();
    tracing::info!("copy stats, preparing per frame (VICTORY_COPY_STATS=1): {}", times.join(", "));
    let settles: Vec<String> = SETTLE_NAMES
        .iter()
        .zip(&SETTLES)
        .map(|(name, n)| format!("{name} {:.1}", n.swap(0, Ordering::Relaxed) as f64 / frames))
        .collect();
    tracing::info!("copy stats, settles before a submission per frame, by cause: {}", settles.join(", "));
    let air = AIR.each_ref().map(|a| a.swap(0, Ordering::Relaxed));
    let presents = air[0].max(1) as f64;
    tracing::info!(
        "copy stats, at a present: {:.1} submissions in the air (at most {}), the oldest {:.1} frames back (at most {}): how far the title runs ahead of the card and the copy-back",
        air[1] as f64 / presents,
        air[2],
        air[3] as f64 / presents,
        air[4]
    );
    for (rank, (id, r)) in all.iter().take(20).enumerate() {
        let per = |k: usize| r.times[k] as f64 / frames;
        tracing::info!(
            "copy stats #{:<2} copy {id} title {:#x}+{:.1} MB {}{}: up {:.2} MB ({:.0}x), sent {:.2} MB ({:.0}x), declared {:.2} MB ({:.0}x), \
             snap {:.2} MB, back {:.2} MB ({:.0}x), not sent {:.2} MB{}",
            rank + 1,
            r.start,
            r.len as f64 / 1e6,
            if r.card { "card" } else { "pc" },
            if r.window { " window" } else { " buffer" },
            mb(r.bytes[UP]),
            per(UP),
            mb(r.bytes[SENT]),
            per(SENT),
            mb(r.bytes[DECLARED]),
            per(DECLARED),
            mb(r.bytes[SNAP]),
            mb(r.bytes[BACK]),
            per(BACK),
            mb(r.bytes[SAVED]),
            // Echo: bytes read up again after we wrote them back.
            if r.bytes[BACK] > 0 && r.bytes[UP] > 0 { " (up includes the echo of back)" } else { "" }
        );
    }
}
