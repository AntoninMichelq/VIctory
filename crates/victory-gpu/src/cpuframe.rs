//! The CPU's work per frame around a submission: switches that change how
//! it is done (never what is copied), and counters that say where it goes.
//!
//! - `VICTORY_WIDE_THREADS=n` (1-32): threads kept for the work spread over
//!   several at once (`crate::shadow::wide`: snapshots of shader-written
//!   ranges, their compares after the card, refreshes of host copies,
//!   texture refreshes, merges). Unset: a quarter of the logical cores,
//!   2 to 6, as before (4 on an 8-core, 16-thread CPU). Refreshes read title
//!   memory through the kernel (`ReadProcessMemory`), which one thread does
//!   at a few GB/s, far below what memory gives: more threads go faster.
//! - `VICTORY_SNAP_STREAM=1`: snapshots are written with AVX2 streaming
//!   stores (no read of the destination's old bytes into the cache, which
//!   an ordinary copy does first, and no eviction of what the submitting
//!   thread is working on). A snapshot is read only after the card has
//!   run, tens of milliseconds later. Without AVX2 the copy is as before.
//! - `VICTORY_CPU_STATS=1`: one line per frame-rate line, the milliseconds
//!   per frame of each of these pieces of work (time it took, not the sum
//!   over threads).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Threads kept for `crate::shadow::wide` (`VICTORY_WIDE_THREADS`), or
/// `None` for the default.
pub(crate) fn wide_threads() -> Option<usize> {
    static N: OnceLock<Option<usize>> = OnceLock::new();
    *N.get_or_init(|| parse_threads(std::env::var("VICTORY_WIDE_THREADS").ok().as_deref()))
}

/// The switch's value. Pure, for testing.
fn parse_threads(v: Option<&str>) -> Option<usize> {
    v?.trim().parse::<usize>().ok().filter(|&n| n >= 1).map(|n| n.min(32))
}

/// Whether snapshots are written with streaming stores
/// (`VICTORY_SNAP_STREAM=1` and AVX2 present).
pub(crate) fn stream() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_SNAP_STREAM").is_some_and(|v| v == "1") && avx2()
    })
}

fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Copies `len` bytes for a snapshot: with [`stream`], by streaming
/// stores; else `copy_nonoverlapping`.
///
/// # Safety
/// As `std::ptr::copy_nonoverlapping`.
pub(crate) unsafe fn snapshot_copy(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    if stream() && len >= 256 {
        // SAFETY: AVX2 is present (`stream`); the caller's ranges.
        unsafe { stream_copy(src, dst, len) };
        return;
    }
    // SAFETY: the caller's.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
}

/// [`snapshot_copy`] by AVX2 streaming stores: the destination's head up
/// to a 32-byte boundary and its tail are copied plainly.
///
/// # Safety
/// AVX2 is present; as `std::ptr::copy_nonoverlapping`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn stream_copy(src: *const u8, dst: *mut u8, len: usize) {
    use std::arch::x86_64::{__m256i, _mm_sfence, _mm256_loadu_si256, _mm256_stream_si256};
    let head = (dst as usize).wrapping_neg() & 31;
    let head = head.min(len);
    // SAFETY: the caller's ranges, in parts.
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, head);
        let (mut s, mut d) = (src.add(head), dst.add(head));
        let body = (len - head) / 128 * 128;
        let end = d.add(body);
        while d < end {
            let a = _mm256_loadu_si256(s.cast::<__m256i>());
            let b = _mm256_loadu_si256(s.add(32).cast::<__m256i>());
            let c = _mm256_loadu_si256(s.add(64).cast::<__m256i>());
            let e = _mm256_loadu_si256(s.add(96).cast::<__m256i>());
            _mm256_stream_si256(d.cast::<__m256i>(), a);
            _mm256_stream_si256(d.add(32).cast::<__m256i>(), b);
            _mm256_stream_si256(d.add(64).cast::<__m256i>(), c);
            _mm256_stream_si256(d.add(96).cast::<__m256i>(), e);
            s = s.add(128);
            d = d.add(128);
        }
        std::ptr::copy_nonoverlapping(s, d, len - head - body);
        // Streaming stores are weakly ordered: done before anyone is told.
        _mm_sfence();
    }
}

/// `VICTORY_REFRESH_SPLIT_MB=n` (1-256): a refresh of a host copy over
/// more than n MiB is cut into pieces of n MiB (at 64 KiB of title memory)
/// that the kept threads bring up to date at once, instead of one thread
/// for the whole copy; the title's big heap (174 MB) was one thread's
/// work, and on the submitting thread when shaders were about to be given
/// a range of it. Unset: not cut, as before.
pub(crate) fn refresh_split() -> Option<usize> {
    static N: OnceLock<Option<usize>> = OnceLock::new();
    *N.get_or_init(|| {
        let mb = std::env::var("VICTORY_REFRESH_SPLIT_MB").ok()?.trim().parse::<usize>().ok()?;
        (mb >= 1).then_some(mb.min(256) << 20)
    })
}

/// `[from, to)` cut into pieces of `step` bytes at multiples of `step`
/// (of the addresses), in order.
pub(crate) fn cut(from: usize, to: usize, step: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut at = from;
    while at < to {
        let next = ((at / step + 1) * step).min(to);
        out.push((at, next));
        at = next;
    }
    out
}

/// What the counters count.
#[derive(Clone, Copy)]
pub(crate) enum Work {
    /// `crate::shadow::take_snapshots`.
    Snapshots,
    /// `crate::shadow::find_all`: compares after the card.
    Compares,
    /// Refreshes of host copies spread over threads (a flush).
    WideRefresh,
    /// Refreshes of one copy on the submitting thread (before a range is
    /// given to shaders, or a flush of fewer than four).
    OneRefresh,
    /// Copies the title asked for (`Step::Copy`, `Step::Fill`).
    TitleCopies,
    /// The write-watch asking ahead of a flush.
    AskAhead,
}

const WORKS: usize = 6;
const NAMES: [&str; WORKS] =
    ["snapshots", "compares after the card", "refreshes spread", "refreshes one by one", "title copies", "write-watch asking"];

/// Microseconds and bytes per [`Work`] since the last line.
static US: [AtomicU64; WORKS] = [const { AtomicU64::new(0) }; WORKS];
static BYTES: [AtomicU64; WORKS] = [const { AtomicU64::new(0) }; WORKS];

/// Whether the counters are kept (`VICTORY_CPU_STATS=1`).
pub(crate) fn stats() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CPU_STATS").is_some_and(|v| v == "1"))
}

/// A start time if the counters are kept.
pub(crate) fn start() -> Option<std::time::Instant> {
    stats().then(std::time::Instant::now)
}

/// `work` took from `started` until now and moved `bytes`.
pub(crate) fn done(work: Work, started: Option<std::time::Instant>, bytes: usize) {
    if let Some(at) = started {
        US[work as usize].fetch_add(at.elapsed().as_micros() as u64, Ordering::Relaxed);
        BYTES[work as usize].fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// The counters' line, per frame over `frames` frames, and a new start.
pub(crate) fn log(frames: f64) {
    if !stats() || frames <= 0.0 {
        return;
    }
    let mut text = String::new();
    for (i, name) in NAMES.iter().enumerate() {
        let ms = US[i].swap(0, Ordering::Relaxed) as f64 / 1e3 / frames;
        let mb = BYTES[i].swap(0, Ordering::Relaxed) as f64 / 1e6 / frames;
        if !text.is_empty() {
            text.push_str(", ");
        }
        text.push_str(&format!("{name} {ms:.1} ms {mb:.1} MB"));
    }
    tracing::info!(
        "CPU work per frame (VICTORY_CPU_STATS=1; wide threads {}, streaming snapshots {}): {text}",
        crate::shadow::wide_count(),
        stream()
    );
}

#[cfg(test)]
mod tests {
    use super::{cut, parse_threads, snapshot_copy};

    #[test]
    fn cuts_at_multiples() {
        assert_eq!(cut(5, 5, 4), []);
        assert_eq!(cut(1, 3, 4), [(1, 3)]);
        assert_eq!(cut(3, 13, 4), [(3, 4), (4, 8), (8, 12), (12, 13)]);
        assert_eq!(cut(4, 8, 4), [(4, 8)]);
    }

    #[test]
    fn reads_the_thread_count() {
        assert_eq!(parse_threads(None), None);
        assert_eq!(parse_threads(Some("x")), None);
        assert_eq!(parse_threads(Some("0")), None);
        assert_eq!(parse_threads(Some(" 12 ")), Some(12));
        assert_eq!(parse_threads(Some("99")), Some(32));
    }

    /// The streaming copy gives the same bytes at every alignment and
    /// length (heads, bodies and tails of all sizes).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn streaming_copy_copies() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let src: Vec<u8> = (0..2000u32).map(|i| (i * 31 + 7) as u8).collect();
        for off in [0usize, 1, 5, 31, 32, 33] {
            for len in [0usize, 1, 31, 127, 128, 255, 256, 257, 1000, 1900] {
                let mut dst = vec![0u8; 2000 + 64];
                // SAFETY: inside both buffers, which do not overlap.
                unsafe { super::stream_copy(src.as_ptr().add(3), dst.as_mut_ptr().add(off), len) };
                assert_eq!(&dst[off..off + len], &src[3..3 + len], "{off} {len}");
                assert!(dst[..off].iter().all(|&b| b == 0));
                assert!(dst[off + len..].iter().all(|&b| b == 0));
                let mut plain = vec![0u8; len];
                // SAFETY: as above.
                unsafe { snapshot_copy(src.as_ptr(), plain.as_mut_ptr(), len) };
                assert_eq!(plain, &src[..len]);
            }
        }
    }
}
