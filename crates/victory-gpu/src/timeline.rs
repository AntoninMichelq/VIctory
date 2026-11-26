//! `VICTORY_TIMELINE=<frames>` (off by default): the timeline of that many
//! frames in a row, from `VICTORY_TIMELINE_AT_S` seconds after the start
//! (240 unless set), one INFO line per frame at its present: each event
//! with its time in milliseconds since the frame's first event and the
//! thread it was on. Events: a submission's preparing begins (`prep#n`),
//! it is sent to the card (`sent#n`), a wait of a thread until every
//! flight in the air has landed begins and ends (`settle:<why>` /
//! `settled`), the thread that copies back lands flights (`land` /
//! `landed`, with the submission it landed up to), the title presents
//! (`present`). It says what waits for what: a frame whose preparing of
//! the next submission waits on `settle` behind `land` is serial.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Frames still to be written out, once the start is reached.
static LEFT: AtomicU32 = AtomicU32::new(0);
/// Whether the frames are being taken now.
static TAKING: AtomicBool = AtomicBool::new(false);
/// The events of the frame under way: ms since the start, thread, what, a number.
static EVENTS: Mutex<Vec<(f64, u32, &'static str, u64)>> = Mutex::new(Vec::new());

fn settings() -> Option<(u32, u64)> {
    static S: std::sync::OnceLock<Option<(u32, u64)>> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let n = std::env::var("VICTORY_TIMELINE").ok()?.trim().parse::<u32>().ok().filter(|&n| n > 0)?;
        let at = std::env::var("VICTORY_TIMELINE_AT_S")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(240);
        LEFT.store(n.min(1000), Ordering::Relaxed);
        Some((n, at))
    })
}

/// A small number for this thread, for the lines.
fn thread() -> u32 {
    static NEXT: AtomicU32 = AtomicU32::new(1);
    std::thread_local! {
        static ID: u32 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    ID.with(|i| *i)
}

/// Notes event `what` (with number `n`) now, if frames are being taken.
pub(crate) fn mark(what: &'static str, n: u64) {
    if !TAKING.load(Ordering::Relaxed) {
        return;
    }
    let ms = crate::host::now_ms_f64();
    EVENTS.lock().unwrap_or_else(|e| e.into_inner()).push((ms, thread(), what, n));
}

/// The title presents: the frame's events are written out (while frames
/// are being taken), and the taking starts once the start is reached.
pub(crate) fn present() {
    let Some((_, at)) = settings() else {
        return;
    };
    if !TAKING.load(Ordering::Relaxed) {
        if LEFT.load(Ordering::Relaxed) > 0 && crate::host::now_ms() >= at * 1000 {
            EVENTS.lock().unwrap_or_else(|e| e.into_inner()).clear();
            TAKING.store(true, Ordering::Relaxed);
            tracing::info!("timeline (VICTORY_TIMELINE): frames from now, ms since each frame's first event, t<thread>");
        }
        return;
    }
    mark("present", 0);
    let events = std::mem::take(&mut *EVENTS.lock().unwrap_or_else(|e| e.into_inner()));
    let first = events.first().map_or(0.0, |e| e.0);
    let text: Vec<String> = events
        .iter()
        .map(|&(ms, t, what, n)| {
            if n > 0 {
                format!("{:.1} t{t} {what}#{n}", ms - first)
            } else {
                format!("{:.1} t{t} {what}", ms - first)
            }
        })
        .collect();
    tracing::info!("timeline: {}", text.join(" | "));
    if LEFT.fetch_sub(1, Ordering::Relaxed) <= 1 {
        TAKING.store(false, Ordering::Relaxed);
    }
}
