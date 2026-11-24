//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! Mirrored buffer SRVs in bound heaps also refresh at submission, since
//! a shader may select their index dynamically. Directly indexed UAVs
//! still need read/write tracking independent of conventional table use.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Before>,
    /// Numbers the copy for [`KEPT`].
    id: u64,
    /// Bit per page (as `copied`): refreshed since the snapshot last looked
    /// (its bytes there may be older than the copy's).
    refreshed: Mutex<Vec<u64>>,
}

#[derive(Default)]
struct Before {
    snapshot: Option<Snapshot>,
    unchanged: UnchangedPages,
    /// Title pages copy-back made equal to title memory (the copy's bytes
    /// changed) in this submission.
    synced: Vec<usize>,
}

/// The copy's bytes before a submission, over whole title pages (cut to
/// the buffer) from offset `at` on.
struct Snapshot {
    at: usize,
    bytes: Vec<u8>,
    /// Per title page from `at`'s on: the submission for which it holds the
    /// copy's bytes (taken in it, or equal to the copy after the previous
    /// one's copy-back).
    ok: Vec<u64>,
}

/// Snapshots kept after a submission's copy-back for the next submission,
/// by copy: where the copy did not change since, the next snapshot takes
/// their bytes instead of the copy's (the world's ~40 submissions per
/// frame snapshot the same big ranges, 1.9 GB per frame). Only the
/// previous submission's are kept: a submission in between may have let
/// the GPU write the copy.
/// Entries: (copy's id, the submission it is for, snapshot).
static KEPT: Mutex<Vec<(u64, u64, Snapshot)>> = Mutex::new(Vec::new());

/// Before submission `now`: kept snapshots not for it go back to the pool.
pub(crate) fn prune_kept(now: u64) {
    let old: Vec<Snapshot> = {
        let mut kept = KEPT.lock().unwrap_or_else(|e| e.into_inner());
        if kept.iter().all(|&(_, of, _)| of == now) {
            return;
        }
        let (keep, old): (Vec<_>, Vec<_>) =
            std::mem::take(&mut *kept).into_iter().partition(|&(_, of, _)| of == now);
        *kept = keep;
        old.into_iter().map(|(_, _, s)| s).collect()
    };
    for s in old {
        give_back(s.bytes);
    }
}

/// Whether snapshots are kept across submissions; `VICTORY_KEEP_SNAPSHOTS=0`
/// takes each submission's afresh.
fn keep_snapshots() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_KEEP_SNAPSHOTS").map_or(true, |v| v.trim() != "0"))
}

/// Since the last [`take_snapshot_stats`]: snapshot bytes taken from a
/// kept snapshot instead of the copy, and bytes copied into kept snapshots
/// after copy-back (pages it changed).
static SNAPSHOT_STATS: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

pub(crate) fn take_snapshot_stats() -> [u64; 2] {
    [0, 1].map(|i| SNAPSHOT_STATS[i].swap(0, Ordering::Relaxed))
}

static IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Pages already compared in full against this submission's snapshot.
#[derive(Default)]
struct UnchangedPages {
    start: usize,
    end: usize,
    pages: Vec<bool>,
}

impl UnchangedPages {
    fn begin(&mut self, start: usize, end: usize) {
        if self.pages.is_empty() {
            self.start = start;
            self.end = end;
            self.pages.resize(end.div_ceil(PAGE) - start / PAGE, false);
        }
    }

    fn differs(&mut self, address: usize, current: &[u8], before: &[u8]) -> bool {
        let page = address / PAGE;
        let (start, end) = (self.start, self.end);
        let unchanged = &mut self.pages[page - start / PAGE];
        page_differs(unchanged, (start, end), address, current, before)
    }

    fn clear(&mut self) {
        self.pages.clear();
    }
}

/// Whether `current` (the copy at `address`, inside one page) differs
/// from `before`; a page found equal over all its compared bytes is marked
/// `unchanged` so later ranges skip it. `bounds`: the snapshot's span.
fn page_differs(
    unchanged: &mut bool,
    (start, end): (usize, usize),
    address: usize,
    current: &[u8],
    before: &[u8],
) -> bool {
    if *unchanged {
        return false;
    }
    if current != before {
        return true;
    }
    // A later range can touch bytes a partial comparison missed. At
    // the edges, only bytes covered by the snapshot can be compared.
    let page = address / PAGE;
    *unchanged = address == (page * PAGE).max(start) && address + current.len() == ((page + 1) * PAGE).min(end);
    false
}

/// Merges what the GPU wrote into title bytes: where `host` differs from
/// `old` (its bytes before the submission) the GPU's byte goes into
/// `title`; elsewhere the title's byte stays. `host` then takes `title`'s
/// bytes. Whether `title` changed. Word by word, without branches per
/// byte (the world copies back hundreds of MB per frame).
fn merge_gpu_writes(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    const LO: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    const HI: u64 = 0x8080_8080_8080_8080;
    let n = title.len().min(host.len()).min(old.len());
    let words = n / 8;
    let mut changed = 0u64;
    let rd = |b: &[u8], i: usize| u64::from_ne_bytes(b[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]));
    for i in 0..words {
        let (t, h, o) = (rd(title, i), rd(host, i), rd(old, i));
        // 0xff in each byte where h != o, 0 elsewhere.
        let x = h ^ o;
        let nz = (((x & LO).wrapping_add(LO)) | x) & HI;
        let mask = (nz >> 7).wrapping_mul(0xff);
        let merged = (h & mask) | (t & !mask);
        changed |= merged ^ t;
        title[i * 8..i * 8 + 8].copy_from_slice(&merged.to_ne_bytes());
        host[i * 8..i * 8 + 8].copy_from_slice(&merged.to_ne_bytes());
    }
    let mut tail_changed = false;
    for j in words * 8..n {
        let (t, h, o) = (title[j], host[j], old[j]);
        if h != o && t != h {
            title[j] = h;
            tail_changed = true;
        }
        host[j] = title[j];
    }
    changed != 0 || tail_changed
}

/// Threads for copying back `len` bytes: one below 8 MiB, else up to
/// half the logical processors (at most 8), `VICTORY_COPY_BACK_THREADS`
/// overriding (1 turns it off).
fn copy_back_threads(len: usize) -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let n = *N.get_or_init(|| {
        std::env::var("VICTORY_COPY_BACK_THREADS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).clamp(1, 8)))
            .clamp(1, 32)
    });
    if len < 8 << 20 { 1 } else { n.min(len / (1 << 20)).max(1) }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Bytes of the snapshot memory kept for reuse.
pub(crate) fn pool_bytes() -> u64 {
    let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.iter().map(|v| v.capacity() as u64).sum()
}

/// Bytes the pool keeps at most: snapshots of the world's 174 MB heap
/// would otherwise keep 8 of that size (1.4 GB of host memory) for good.
const POOL_BYTES: usize = 512 << 20;

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest,
/// and no more than [`POOL_BYTES`] in all).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let mut total = 0;
    pool.retain(|v| {
        total += v.capacity();
        total <= POOL_BYTES
    });
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Drop for Shadow {
    fn drop(&mut self) {
        // SAFETY: a live resource.
        let bytes = unsafe { self.buffer.GetDesc() }.Width;
        crate::memstat::sub(&crate::memstat::SHADOWS, bytes);
        crate::memstat::sub(&crate::memstat::SHADOW_COUNT, 1);
    }
}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// The copy's bytes at `[address, address + len)` (inside the buffer).
    #[cfg(test)]
    pub(crate) fn host_bytes(&self, address: u64, len: usize) -> Vec<u8> {
        // SAFETY: inside the mapping; no submission runs while tests read.
        unsafe { std::slice::from_raw_parts(self.data.add(self.offset(address) as usize), len) }.to_vec()
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        // SAFETY: a live resource.
        let bytes = unsafe { buffer.GetDesc() }.Width;
        crate::memstat::add(&crate::memstat::SHADOWS, bytes);
        crate::memstat::add(&crate::memstat::SHADOW_COUNT, 1);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            id: IDS.fetch_add(1, Ordering::Relaxed),
            refreshed: Mutex::new(vec![0; (pages as usize).div_ceil(64)]),
        }
    }

    /// The copy's offsets of title page `page` (cut to the buffer).
    fn page_span(&self, page: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        ((page * PAGE).max(start) - start, ((page + 1) * PAGE).min(end) - start)
    }

    /// Notes copy offsets `[from, to)` refreshed: snapshots no longer hold
    /// the copy's bytes there.
    fn refreshed(&self, from: usize, to: usize) {
        let base = self.start as usize / PAGE;
        let start = self.start as usize;
        let mut bits = self.refreshed.lock().unwrap_or_else(|e| e.into_inner());
        for page in (start + from) / PAGE..(start + to).div_ceil(PAGE) {
            let i = page - base;
            bits[i / 64] |= 1 << (i % 64);
        }
    }

    /// Takes the pages refreshed since the last call: `snapshot`'s bytes
    /// there are no longer the copy's.
    fn forget_refreshed(&self, mut snapshot: Option<&mut Snapshot>) {
        let base = self.start as usize / PAGE;
        let start = self.start as usize;
        let mut bits = self.refreshed.lock().unwrap_or_else(|e| e.into_inner());
        for (w, word) in bits.iter_mut().enumerate() {
            let mut set = std::mem::take(word);
            let Some(snapshot) = snapshot.as_mut() else {
                continue;
            };
            let first = (start + snapshot.at) / PAGE;
            while set != 0 {
                let page = base + w * 64 + set.trailing_zeros() as usize;
                set &= set - 1;
                if let Some(ok) = page.checked_sub(first).and_then(|i| snapshot.ok.get_mut(i)) {
                    *ok = 0;
                }
            }
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame). Whole title pages are
    /// taken; pages the previous submission's snapshot still holds (see
    /// [`KEPT`]) are not taken again, nor pages taken already in this
    /// submission and not refreshed since.
    /// Returns the bytes copied into the snapshot (0 if it held the range
    /// already).
    pub(crate) fn writing(&self, address: usize, len: usize) -> usize {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        if from >= to {
            return 0;
        }
        let start = self.start as usize;
        // Whole title pages, cut to the buffer.
        let lo = ((start + from) / PAGE * PAGE).max(start) - start;
        let hi = (start + to).next_multiple_of(PAGE).min(start + self.len as usize) - start;
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let pages = |a: usize, b: usize| {
            if a < b { (start + b).div_ceil(PAGE) - (start + a) / PAGE } else { 0 }
        };
        let now = crate::host::submission();
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let before = &mut before.snapshot;
        let mut from_kept = false;
        if before.is_none() && keep_snapshots() {
            // The previous submission's, if it is for this one.
            let mut kept = KEPT.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(i) = kept.iter().position(|&(id, of, _)| id == self.id && of == now) {
                *before = Some(kept.swap_remove(i).2);
                from_kept = true;
            }
        }
        self.forget_refreshed(before.as_mut());
        if from_kept && let Some(snapshot) = before.as_ref() {
            // What the previous submission's snapshot still holds (for logs).
            let held = snapshot.ok.iter().filter(|&&ok| ok == now).count();
            SNAPSHOT_STATS[0].fetch_add((held * PAGE) as u64, Ordering::Relaxed);
        }
        let mut taken = 0;
        let snapshot = match before {
            Some(snapshot) => {
                let (at, end) = (snapshot.at, snapshot.at + snapshot.bytes.len());
                if lo < at || hi > end {
                    // Parts newly covered from the copy, the rest as it was.
                    let (lo, hi) = (lo.min(at), hi.max(end));
                    let mut grown = pooled(hi - lo);
                    grown.extend_from_slice(host(lo, at));
                    grown.extend_from_slice(&snapshot.bytes);
                    grown.extend_from_slice(host(end, hi));
                    let mut ok = Vec::with_capacity(pages(lo, hi));
                    ok.resize(pages(lo, at), now);
                    ok.extend_from_slice(&snapshot.ok);
                    ok.resize(ok.len() + pages(end, hi), now);
                    give_back(std::mem::replace(&mut snapshot.bytes, grown));
                    (snapshot.at, snapshot.ok) = (lo, ok);
                    taken += hi - lo;
                }
                snapshot
            }
            None => {
                let mut bytes = pooled(hi - lo);
                bytes.extend_from_slice(host(lo, hi));
                taken += hi - lo;
                before.insert(Snapshot {
                    at: lo,
                    bytes,
                    ok: vec![now; pages(lo, hi)],
                })
            }
        };
        // Pages the snapshot does not hold for this submission: from the copy.
        let first = (start + snapshot.at) / PAGE;
        for page in (start + lo) / PAGE..(start + hi).div_ceil(PAGE) {
            let (a, b) = self.page_span(page);
            let ok = &mut snapshot.ok[page - first];
            if *ok == now {
                continue;
            }
            *ok = now;
            let at = snapshot.at;
            snapshot.bytes[a - at..b - at].copy_from_slice(host(a, b));
            taken += b - a;
        }
        taken
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Before {
            snapshot,
            unchanged,
            synced: synced_all,
        } = &mut *before;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some(Snapshot {
            at: taken,
            bytes: before,
            ..
        }) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            unchanged.begin(start + taken, start + taken + before.len());
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            // An unchanged page is never modified by copy-back, so later
            // overlapping ranges can skip it for the rest of the submission.
            // Page-aligned parts, each on its own thread when the range is
            // big: pages are independent (own unchanged flag, own bytes).
            let bounds = (unchanged.start, unchanged.end);
            let first_page = unchanged.start / PAGE;
            let data = self.data as usize;
            let part = |lo: usize, hi: usize, flags: &mut [bool], flags_first: usize| {
                let mut title: Option<Vec<u8>> = None;
                let mut synced = Vec::new();
                let mut written = 0;
                let mut at = lo;
                while at < hi {
                    let n = (PAGE - (start + at) % PAGE).min(hi - at);
                    // SAFETY: inside the mapping; only the submitting
                    // thread's copy-back writes it, each page from one part.
                    let host = unsafe { std::slice::from_raw_parts_mut((data as *mut u8).add(at), n) };
                    let old = &before[at - from..at - from + n];
                    let flag = &mut flags[(start + at) / PAGE - flags_first];
                    if page_differs(flag, bounds, start + at, host, old)
                        && let title = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n]
                        && crate::host::read_into((start + at) as *const u8, title)
                    {
                        // Title memory with the GPU's bytes put in; the copy
                        // takes the title's bytes in turn.
                        if merge_gpu_writes(title, host, old) {
                            crate::watch::writing(start + at, n);
                            crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                            written += n;
                        }
                        synced.push((start + at) / PAGE);
                    }
                    at += n;
                }
                (written, synced)
            };
            let threads = copy_back_threads(to - from);
            let mut synced = Vec::new();
            if threads <= 1 {
                let (w, s) = part(from, to, &mut unchanged.pages, first_page);
                written += w;
                synced = s;
            } else {
                // Cut at page boundaries of title memory.
                let pages_total = (start + to).div_ceil(PAGE) - (start + from) / PAGE;
                let per = pages_total.div_ceil(threads);
                let mut cuts = vec![from];
                let mut page = (start + from) / PAGE;
                loop {
                    page += per;
                    let cut = page * PAGE - start;
                    if cut >= to {
                        break;
                    }
                    cuts.push(cut);
                }
                cuts.push(to);
                // Split the flags the same way (by title page).
                let mut rest: &mut [bool] = &mut unchanged.pages;
                let mut rest_first = first_page;
                let mut jobs = Vec::new();
                for w in cuts.windows(2) {
                    let (lo, hi) = (w[0], w[1]);
                    let end_page = (start + hi).div_ceil(PAGE).max((start + lo) / PAGE + 1);
                    let take = end_page.saturating_sub(rest_first).min(rest.len());
                    let (mine, more) = std::mem::take(&mut rest).split_at_mut(take);
                    jobs.push((lo, hi, mine, rest_first));
                    rest = more;
                    rest_first += take;
                }
                let results: Vec<(usize, Vec<usize>)> = std::thread::scope(|scope| {
                    let handles: Vec<_> = jobs
                        .into_iter()
                        .map(|(lo, hi, mine, mine_first)| {
                            let part = &part;
                            scope.spawn(move || part(lo, hi, mine, mine_first))
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
                });
                for (w, s) in results {
                    written += w;
                    synced.extend(s);
                }
            }
            synced_all.extend_from_slice(&synced);
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty()
            && let Some(snapshot) = snapshot.take()
        {
            if keep_snapshots() {
                self.keep(snapshot, unchanged, synced_all);
            } else {
                give_back(snapshot.bytes);
            }
            unchanged.clear();
            synced_all.clear();
        }
        written
    }

    /// After the submission's last copy-back: keeps `snapshot` for the
    /// next submission where it holds the copy's bytes now, i.e. pages
    /// copy-back compared in full and found unchanged (it leaves those
    /// alone), and pages it changed (`synced`), copied into it again.
    /// Elsewhere (pages not compared in full) the GPU may have written the
    /// copy unseen; those are taken again.
    fn keep(&self, mut snapshot: Snapshot, unchanged: &UnchangedPages, synced: &mut Vec<usize>) {
        let next = crate::host::submission() + 1;
        let start = self.start as usize;
        let first = (start + snapshot.at) / PAGE;
        if !unchanged.pages.is_empty() {
            let from = unchanged.start / PAGE;
            for (k, &same) in unchanged.pages.iter().enumerate() {
                if same
                    && let Some(ok) = (from + k).checked_sub(first).and_then(|i| snapshot.ok.get_mut(i))
                {
                    *ok = next;
                }
            }
        }
        synced.sort_unstable();
        synced.dedup();
        let mut copied = 0;
        for &page in synced.iter() {
            let Some(i) = page.checked_sub(first).filter(|&i| i < snapshot.ok.len()) else {
                continue;
            };
            let (a, b) = self.page_span(page);
            let at = snapshot.at;
            // SAFETY: inside the mapping; the submission has run.
            let host = unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
            snapshot.bytes[a - at..b - at].copy_from_slice(host);
            snapshot.ok[i] = next;
            copied += b - a;
        }
        SNAPSHOT_STATS[1].fetch_add(copied as u64, Ordering::Relaxed);
        // Refreshes before now are in the copy the pages above were
        // compared with.
        self.forget_refreshed(None);
        KEPT.lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((self.id, next, snapshot));
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        self.refreshed(from - start, to - start);
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of a title buffer holding `[address, address + len)`: of one
/// whose copy is made already if there is one (a buffer inside another,
/// say in the world's 174 MB heap, then shares its copy instead of paying
/// for its bytes twice, and the GPU's writes land in one copy), else of
/// the innermost, made now.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    // Called for every buffer a draw reads: one pass, no allocation.
    let (mut innermost, mut made) = (None, None);
    for (k, e) in map.range(..=(address, usize::MAX)).rev().take(64) {
        if e.end < end {
            continue;
        }
        if e.shadow.is_some() {
            made = Some(*k);
            break;
        }
        innermost.get_or_insert(*k);
    }
    let key = made.or(innermost);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    // Any window copied already that holds the range: a range crossing a
    // window's end used to get a second copy of the same bytes.
    if let Some(shadow) = map
        .range(..=(start, u64::MAX))
        .rev()
        .take(64)
        .find(|((s, l), _)| *s <= address && s + l >= end)
        .map(|(_, shadow)| shadow.clone())
    {
        return Some(shadow);
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_merge_matches_byte_merge() {
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 4) as u8
        };
        for len in [0usize, 1, 7, 8, 9, 63, 64, 65, 1000] {
            for _ in 0..100 {
                let old: Vec<u8> = (0..len).map(|_| next()).collect();
                let host: Vec<u8> = old.iter().map(|&o| if next() == 0 { next() } else { o }).collect();
                let title: Vec<u8> = old.iter().map(|&o| if next() == 0 { next() } else { o }).collect();
                let (mut t1, mut h1) = (title.clone(), host.clone());
                let (mut t2, mut h2) = (title.clone(), host.clone());
                let fast = super::merge_gpu_writes(&mut t1, &mut h1, &old);
                let mut slow = false;
                for ((t, h), o) in t2.iter_mut().zip(h2.iter_mut()).zip(&old) {
                    if *h != *o && *t != *h {
                        *t = *h;
                        slow = true;
                    }
                    *h = *t;
                }
                assert_eq!((fast, &t1, &h1), (slow, &t2, &h2), "len {len}");
            }
        }
    }
    use std::ops::Range;

    #[test]
    fn partial_match_leaves_unchecked_bytes_visible() {
        let mut unchanged = UnchangedPages::default();
        let before = vec![0; PAGE];
        let mut host = before.clone();
        host[PAGE - 1] = 1;
        unchanged.begin(PAGE, 2 * PAGE);

        assert!(!unchanged.differs(PAGE, &host[..4], &before[..4]));
        assert!(!unchanged.pages[0]);
        assert!(unchanged.differs(PAGE, &host, &before));
        assert!(!unchanged.pages[0]);
    }

    #[test]
    fn snapshot_edges_are_cacheable_and_survive_later_ranges() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 3, &[1; 3], &[1; 3]));
        assert!(!unchanged.differs(PAGE, &[2; 4], &[2; 4]));
        assert_eq!(unchanged.pages, [true, true]);

        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 2, &[1; 2], &[1; 2]));
        assert!(!unchanged.differs(PAGE + 1, &[2; 2], &[2; 2]));
        assert_eq!(unchanged.pages, [true, true]);
    }

    #[test]
    fn next_submission_checks_the_same_page_again() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, 4);
        assert!(!unchanged.differs(0, &[0; 4], &[0; 4]));
        unchanged.clear();
        unchanged.begin(0, 4);
        assert!(unchanged.differs(0, &[0, 0, 0, 1], &[0; 4]));
    }

    // Both versions walk exactly the same declared ranges, in their
    // original order. Only the page comparison differs.
    fn scan(
        host: &[u8],
        before: &[u8],
        ranges: &[Range<usize>],
        mut differs: impl FnMut(usize, &[u8], &[u8]) -> bool,
    ) -> usize {
        let mut changed = 0;
        for range in ranges {
            let mut at = range.start;
            while at < range.end {
                let n = (PAGE - at % PAGE).min(range.end - at);
                changed += usize::from(differs(at, &host[at..at + n], &before[at..at + n]));
                at += n;
            }
        }
        changed
    }

    #[test]
    fn overlapping_ranges_match_uncached_comparisons() {
        let before = vec![0; 4 * PAGE];
        let mut host = before.clone();
        for at in [2, PAGE - 1, PAGE + 5, 3 * PAGE + 7] {
            host[at] = 1;
        }
        let ranges = [
            0..4,
            2..2 * PAGE + 3,
            PAGE - 2..4 * PAGE,
            2 * PAGE..3 * PAGE,
        ];
        let uncached = scan(&host, &before, &ranges, |_, host, before| host != before);
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, before.len());
        let cached = scan(&host, &before, &ranges, |at, host, before| {
            unchanged.differs(at, host, before)
        });
        assert_eq!(uncached, 6);
        assert_eq!(cached, uncached);
        assert_eq!(unchanged.pages, [false, false, true, false]);
    }

    // cargo test --release -p victory-gpu benchmark_overlapping_copy_back -- --ignored --nocapture
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn benchmark_overlapping_copy_back() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const SIZE: usize = 128 << 20;
        let before = vec![0xA5; SIZE];
        let mut host = before.clone();
        for page in (0..SIZE / PAGE).step_by(64) {
            host[page * PAGE + PAGE / 2] ^= 0xFF;
        }
        let ranges: Vec<_> = (0..24).map(|i| i * PAGE + i..SIZE - i * 13).collect();
        let mut baseline = [Duration::ZERO; 5];
        let mut cached = [Duration::ZERO; 5];
        let mut unchanged = UnchangedPages::default();
        for (baseline, cached) in baseline.iter_mut().zip(&mut cached) {
            let started = Instant::now();
            let expected = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |_, host, before| host != before,
            );
            *baseline = started.elapsed();

            unchanged.clear();
            let started = Instant::now();
            unchanged.begin(0, before.len());
            let actual = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |at, host, before| unchanged.differs(at, host, before),
            );
            *cached = started.elapsed();
            assert_eq!(actual, expected);
        }
        baseline.sort_unstable();
        cached.sort_unstable();
        println!(
            "median of five runs, 24 overlapping ranges across 128 MiB: {:?} -> {:?} ({:.1}x)",
            baseline[2],
            cached[2],
            baseline[2].as_secs_f64() / cached[2].as_secs_f64(),
        );
    }
}
