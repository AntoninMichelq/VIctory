//! Host memory guardrails for texture uploads and staging (`VICTORY_MEM_GUARD=1`).
//!
//! Adapted from cool-fox-codec `ab75ac3` (500 ms backoff after `E_OUTOFMEMORY`
//! and cached `GlobalMemoryStatusEx` commit-headroom gate) and `18bdf79`
//! (bounding per-frame speculative texture staging on <= 32 GiB RAM hosts or
//! under memory pressure).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// HRESULT `E_OUTOFMEMORY` (`0x8007000E`).
const E_OUTOFMEMORY: i32 = 0x8007_000E_u32 as i32;

/// Backoff after a host upload/texture allocation fails with `E_OUTOFMEMORY`.
const RETRY_MS: u64 = 500;
/// Minimum interval between `GlobalMemoryStatusEx` polls.
const POLL_MS: u64 = 32;

/// Available commit headroom required to run speculative texture passes.
const LOOKAHEAD_COMMIT_FLOOR: u64 = 4 * GIB;
/// Lower commit headroom required for one-shot bake passes.
const BAKE_COMMIT_FLOOR: u64 = 2 * GIB;
/// Available physical RAM below which host memory is considered pressured.
const PRESSURE_PHYS_FLOOR: u64 = 2 * GIB;

/// Total physical RAM ceiling defining a limited-RAM host (32 GiB).
const LIMITED_RAM_CEILING: u64 = 32 * GIB;
/// Live texture staging threshold above which a limited-RAM host throttles
/// speculative uploads.
const STAGING_CAP: u64 = GIB;
/// Speculative texture staging cap per frame under memory pressure.
const PRESSURE_PREFETCH_LIMIT: u64 = 16 * MIB;
/// Speculative texture staging cap per frame on limited-RAM hosts with large
/// live staging.
const LIMITED_PREFETCH_LIMIT: u64 = 32 * MIB;

static RETRY_UPLOAD_AT: AtomicU64 = AtomicU64::new(0);
static LAST_POLL_AT: AtomicU64 = AtomicU64::new(0);
static LAST_AVAIL_COMMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static LAST_AVAIL_PHYS: AtomicU64 = AtomicU64::new(u64::MAX);
static REFUSED_LOGGED: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_MEM_GUARD=1` enables the host-memory guardrails (off by default).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MEM_GUARD").is_some_and(|v| v == "1"))
}

/// Records an upload or texture allocation error. On `E_OUTOFMEMORY`, backs
/// off speculative uploads for 500 ms so the runtime can drain in-flight work.
pub(crate) fn allocation_error(error: &windows::core::Error) {
    if !enabled() || error.code().0 != E_OUTOFMEMORY {
        return;
    }
    let until = crate::host::now_ms().saturating_add(RETRY_MS);
    RETRY_UPLOAD_AT.fetch_max(until, Ordering::Relaxed);
    let n = REFUSED_LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::warn!(
            occurrences = n,
            retry_ms = RETRY_MS,
            "host allocation hit E_OUTOFMEMORY; pausing speculative texture uploads (VICTORY_MEM_GUARD=1)"
        );
    }
}

/// Whether speculative uploads are currently paused after an `E_OUTOFMEMORY`.
pub(crate) fn upload_refused() -> bool {
    enabled() && crate::host::now_ms() < RETRY_UPLOAD_AT.load(Ordering::Relaxed)
}

/// Queries `GlobalMemoryStatusEx`, returning `(avail_commit, avail_phys, total_phys)`.
fn query_memory_status() -> Option<(u64, u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: `dwLength` is initialized to the exact struct size.
    let ok = unsafe { GlobalMemoryStatusEx(&raw mut status) };
    (ok != 0).then_some((status.ullAvailPageFile, status.ullAvailPhys, status.ullTotalPhys))
}

/// Cached `(avail_commit, avail_phys)` refreshed at most once every 32 ms.
fn poll_memory() -> (u64, u64) {
    let now = crate::host::now_ms();
    let last = LAST_POLL_AT.load(Ordering::Relaxed);
    if now.saturating_sub(last) >= POLL_MS
        && LAST_POLL_AT
            .compare_exchange(last, now.max(1), Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        && let Some((commit, phys, _)) = query_memory_status()
    {
        LAST_AVAIL_COMMIT.store(commit, Ordering::Relaxed);
        LAST_AVAIL_PHYS.store(phys, Ordering::Relaxed);
        return (commit, phys);
    }
    (
        LAST_AVAIL_COMMIT.load(Ordering::Relaxed),
        LAST_AVAIL_PHYS.load(Ordering::Relaxed),
    )
}

/// Whether the host has 32 GiB or less of physical RAM.
pub(crate) fn limited_host_ram() -> bool {
    static LIMITED: OnceLock<bool> = OnceLock::new();
    *LIMITED.get_or_init(|| {
        query_memory_status().is_some_and(|(_, _, total)| total <= LIMITED_RAM_CEILING)
    })
}

/// Whether host commit or physical memory is currently pressured.
pub(crate) fn pressured() -> bool {
    if !enabled() {
        return false;
    }
    let (avail_commit, avail_phys) = poll_memory();
    is_pressured(avail_commit, avail_phys)
}

fn is_pressured(avail_commit: u64, avail_phys: u64) -> bool {
    avail_commit < LOOKAHEAD_COMMIT_FLOOR || avail_phys < PRESSURE_PHYS_FLOOR
}

fn commit_admits(avail_commit: u64, bake: bool) -> bool {
    let floor = if bake {
        BAKE_COMMIT_FLOOR
    } else {
        LOOKAHEAD_COMMIT_FLOOR
    };
    avail_commit >= floor
}

/// Whether a texture upload pass may proceed under current host memory
/// headroom (`bake` passes keep a lower 2 GiB floor vs 4 GiB for lookahead).
pub(crate) fn admit_texture_pass(bake: bool) -> bool {
    if !enabled() {
        return true;
    }
    if upload_refused() {
        return false;
    }
    let (avail_commit, _) = poll_memory();
    commit_admits(avail_commit, bake)
}

/// Caps the per-frame speculative texture staging budget under memory pressure
/// (16 MiB) or on limited-RAM hosts with >= 1 GiB live staging (32 MiB).
pub(crate) fn prefetch_limit(
    bytes: u64,
    pressure: bool,
    limited: bool,
    staging_bytes: u64,
) -> u64 {
    if pressure {
        bytes.min(PRESSURE_PREFETCH_LIMIT)
    } else if limited && staging_bytes >= STAGING_CAP {
        bytes.min(LIMITED_PREFETCH_LIMIT)
    } else {
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_limit_caps_under_pressure_and_limited_ram() {
        assert_eq!(prefetch_limit(192 * MIB, false, false, 0), 192 * MIB);
        assert_eq!(prefetch_limit(192 * MIB, false, true, 512 * MIB), 192 * MIB);
        assert_eq!(prefetch_limit(192 * MIB, false, true, GIB), 32 * MIB);
        assert_eq!(prefetch_limit(192 * MIB, true, false, 0), 16 * MIB);
        assert_eq!(prefetch_limit(8 * MIB, true, true, 2 * GIB), 8 * MIB);
    }

    #[test]
    fn commit_floor_distinguishes_bake_and_lookahead() {
        assert!(!commit_admits(GIB, true));
        assert!(commit_admits(2 * GIB, true));
        assert!(!commit_admits(2 * GIB, false));
        assert!(!commit_admits(3 * GIB, false));
        assert!(commit_admits(4 * GIB, false));
        assert!(is_pressured(3 * GIB, 8 * GIB));
        assert!(is_pressured(8 * GIB, GIB));
        assert!(!is_pressured(4 * GIB, 2 * GIB));
    }
}
