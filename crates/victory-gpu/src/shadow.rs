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

#[cfg(feature = "diagnostics")]
use crate::copystat::{C, add};
use crate::host::{HostList, host};
use crate::watch::PAGE;

#[cfg(test)]
mod submission_tests;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    /// The [`LOOSE_USES`] count at its last lookup (loose windows only):
    /// the least recently used go first past [`loose_budget`].
    used: std::sync::atomic::AtomicU64,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, or wholly unwatched.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// Ranges refreshed whole in one watch generation (see
    /// [`crate::watch::generation`]): asked again in it, nothing changed.
    refreshed: Mutex<(u64, Vec<(usize, usize)>)>,
    /// The copy's bytes before the submission ran, by title page, covering
    /// the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Before>,
}

/// A copy's bytes before a submission ran.
#[derive(Default)]
struct Before {
    /// By title page.
    pages: BTreeMap<usize, Snap>,
    /// The ranges declared (offsets in the copy, merged where they meet): a
    /// range declared again (the world's are, dozens of times) is inside
    /// one and needs no walk over its pages, and the parts of a range
    /// outside them are fresh (see [`Shadow::writing`]).
    spans: Vec<(usize, usize)>,
    unchanged: UnchangedPages,
}

/// A title page's bytes in the copy before the submission ran: `[lo, hi)`
/// of `bytes` (offsets in the page; a page's buffer is a page long).
struct Snap {
    lo: usize,
    hi: usize,
    bytes: Box<[u8]>,
}

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
        let unchanged = &mut self.pages[page - self.start / PAGE];
        if *unchanged {
            return false;
        }
        if current != before {
            return true;
        }
        // A later range can touch bytes a partial comparison missed. At
        // the edges, only bytes covered by the snapshot can be compared.
        *unchanged = address == (page * PAGE).max(self.start)
            && address + current.len() == ((page + 1) * PAGE).min(self.end);
        false
    }

    fn clear(&mut self) {
        self.pages.clear();
    }
}

/// Page buffers kept for reuse (already touched, so no page faults). A
/// snapshot as one buffer grown to the submission's bounding range took a
/// fresh buffer whenever the few kept were in use or too small, and copied
/// all of the old one into it at each growth.
static PAGES: Mutex<Vec<Box<[u8]>>> = Mutex::new(Vec::new());
/// At most this many page buffers kept (256 MiB): snapshots of the world's
/// 174 MB heap would otherwise keep all of theirs for good.
const KEPT: usize = 4096;

/// Merges what the GPU wrote into title bytes: where `host` differs from
/// `old` (its bytes before the submission) the GPU's byte goes into
/// `title`; elsewhere the title's byte stays. `host` and `old` then take
/// `title`'s bytes. Whether `title` changed. Word by word, without branches per
/// byte (the world copies back hundreds of MB per frame).
fn merge_gpu_writes(title: &mut [u8], host: &mut [u8], old: &mut [u8]) -> bool {
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
        old[i * 8..i * 8 + 8].copy_from_slice(&merged.to_ne_bytes());
    }
    let mut tail_changed = false;
    for j in words * 8..n {
        let (t, h, o) = (title[j], host[j], old[j]);
        if h != o && t != h {
            title[j] = h;
            tail_changed = true;
        }
        host[j] = title[j];
        old[j] = title[j];
    }
    changed != 0 || tail_changed
}

/// A page buffer, from the pool if it has one.
fn page_buffer() -> Box<[u8]> {
    let kept = PAGES.lock().unwrap_or_else(|e| e.into_inner()).pop();
    kept.unwrap_or_else(|| vec![0; PAGE].into_boxed_slice())
}

/// Bytes of the snapshot memory kept for reuse.
pub(crate) fn pool_bytes() -> u64 {
    let pool = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    (pool.len() * PAGE) as u64
}

/// Returns page buffers to the pool (at most [`KEPT`]).
fn give_back(buffers: impl IntoIterator<Item = Box<[u8]>>) {
    let mut pool = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    for b in buffers {
        if pool.len() >= KEPT {
            break;
        }
        pool.push(b);
    }
}

/// A page's `copied` entry while its title memory is uncommitted and its
/// copy holds zeros for it.
const ZEROED: u64 = u64::MAX;

const WATCH_UNKNOWN: u8 = 0;
const WATCHED: u8 = 1;
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

    /// The title range the copy holds: start and length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
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

    #[cfg(test)]
    pub(crate) fn read_u32(&self, offset: usize) -> u32 {
        assert!(offset + 4 <= self.len as usize);
        // SAFETY: `offset..offset + 4` lies within the mapped host buffer.
        unsafe { self.data.add(offset).cast::<u32>().read_unaligned() }
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
            watched: AtomicU8::new(WATCH_UNKNOWN),
            pending: Mutex::default(),
            refreshed: Mutex::default(),
            before: Mutex::default(),
            used: std::sync::atomic::AtomicU64::new(0),
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
    /// per title page and submission, each page's part grown to cover every
    /// range the submission declares in it (the world declares overlapping
    /// views of its 174 MB heap dozens of times), parts newly covered taken
    /// from the copy as it is now, in reused page buffers (fresh snapshots
    /// cost ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        // In-page offsets `[a, b)` of `page`, from the copy.
        // SAFETY: inside the mapping (offsets in the buffer).
        let host = |page: usize, a: usize, b: usize| unsafe {
            std::slice::from_raw_parts(self.data.add(page * PAGE + a - start), b - a)
        };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let before = &mut *before;
        if from >= to || before.spans.iter().any(|&(a, b)| a <= from && to <= b) {
            return;
        }
        // The parts no earlier range of the submission declared: their
        // refresh was free to bring the title's bytes into the copy (a
        // refresh leaves declared ranges alone), and may just have.
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let fresh = outside(from, to, before.spans.clone());
        let (mut a, mut b) = (from, to);
        before.spans.retain(|&(x, y)| {
            let meets = x <= b && a <= y;
            if meets {
                (a, b) = (a.min(x), b.max(y));
            }
            !meets
        });
        before.spans.push((a, b));
        let mut at = from;
        while at < to {
            let (page, lo) = ((start + at) / PAGE, (start + at) % PAGE);
            let hi = lo + (PAGE - lo).min(to - at);
            let snap = before.pages.entry(page).or_insert_with(|| Snap {
                lo,
                hi: lo,
                bytes: page_buffer(),
            });
            let old = (snap.lo, snap.hi);
            if lo < snap.lo {
                snap.bytes[lo..snap.lo].copy_from_slice(host(page, lo, snap.lo));
                snap.lo = lo;
            }
            if hi > snap.hi {
                let from = snap.hi.max(snap.lo);
                snap.bytes[from..hi].copy_from_slice(host(page, from, hi));
                snap.hi = hi;
            }
            // What the page's part already held of the fresh parts (it
            // spans the gaps between the ranges declared so far) was taken
            // before their refresh: taken again, or the refresh would pass
            // for a shader's write and go back over what the title writes
            // meanwhile.
            let base = page * PAGE;
            for &(x, y) in &fresh {
                let (x, y) = ((start + x).max(base + old.0), (start + y).min(base + old.1));
                if x < y {
                    let (x, y) = (x - base, y - base);
                    snap.bytes[x..y].copy_from_slice(host(page, x, y));
                }
            }
            at += hi - lo;
        }
        #[cfg(feature = "diagnostics")]
        {
            add(C::SnapshotBytes, (to - from) as u64);
            add(C::SnapshotUs, started.elapsed().as_micros() as u64);
        }
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
            pages,
            spans,
            unchanged,
        } = &mut *before;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        let extent = spans
            .iter()
            .map(|s| s.0)
            .min()
            .zip(spans.iter().map(|s| s.1).max());
        if from < to
            && let Some((taken, end)) = extent
        {
            unchanged.begin(start + taken, start + end);
            #[cfg(feature = "diagnostics")]
            add(C::CopyBackCompared, (to - from) as u64);
            // By title page: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            // An unchanged page is never modified by copy-back, so later
            // overlapping ranges can skip it for the rest of the submission.
            let mut title = None;
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let (page, lo) = ((start + at) / PAGE, (start + at) % PAGE);
                let n = (PAGE - lo).min(to - at);
                let Some(snap) = pages
                    .get_mut(&page)
                    .filter(|s| s.lo <= lo && lo + n <= s.hi)
                else {
                    at += n;
                    continue;
                };
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &mut snap.bytes[lo..lo + n];
                if unchanged.differs(start + at, host, old)
                    && let title = &mut title.get_or_insert_with(page_buffer)[..n]
                    && crate::host::read_into((start + at) as *const u8, title)
                {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn, and so does the
                    // snapshot: to a later range that overlaps this one,
                    // they are not a shader's write to put back.
                    // Word by word (the world copies back hundreds of MB
                    // per frame).
                    if merge_gpu_writes(title, host, old) {
                        crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            give_back(title);
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
        if pending.is_empty() {
            unchanged.clear();
            spans.clear();
            give_back(std::mem::take(pages).into_values().map(|s| s.bytes));
        }
        written
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
        let generation = crate::watch::generation();
        if generation != 0 && memo() {
            {
                let r = self.refreshed.lock().unwrap_or_else(|e| e.into_inner());
                if r.0 == generation && r.1.iter().any(|&(a, e)| a <= from && e >= to) {
                    #[cfg(feature = "diagnostics")]
                    add(C::Memoized, (to - from) as u64);
                    return 0;
                }
            }
            let bytes = self.update_free(from, to);
            let mut r = self.refreshed.lock().unwrap_or_else(|e| e.into_inner());
            if r.0 != generation {
                *r = (generation, Vec::new());
            }
            if r.1.len() < 64 {
                r.1.push((from, to));
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        let watched = self.watched.load(Ordering::Relaxed);
        if watched != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(WATCHED, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                    add(C::WatchBytes, copied as u64);
                }
                return copied;
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::watch_failed(self as *const Self as usize, to - from);
            // Views of a single shadow can cover distinct title allocations.
            // Cache the first negative result only when one region contains
            // the shadow and the query could not have crossed a boundary.
            let start = self.start as usize;
            if watched == WATCH_UNKNOWN
                && crate::watch::private_region_covers(start, start + self.len as usize)
            {
                self.watched.store(NOT_WATCHED, Ordering::Relaxed);
            } else {
                // The range crosses memory the title has not committed (or
                // several allocations): region by region instead of all of
                // it every time (the world's loose windows zero-filled
                // 4.4 GB of uncommitted memory per frame this way).
                return self.update_by_region(from, to);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            add(C::UnwatchedRanges, 1);
            add(C::UnwatchedBytes, (to - from) as u64);
        }
        self.copy(from, to);
        to - from
    }

    /// [`Shadow::update_free`] of a range the write watch could not answer
    /// for as a whole: each committed region is asked on its own (copied
    /// whole when it is not watched either), and each uncommitted one is
    /// zeroed in the copy once, then left alone while it stays uncommitted.
    fn update_by_region(&self, from: usize, to: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
        };
        let start = self.start as usize;
        let base = start / PAGE;
        let (mut at, mut bytes) = (from, 0);
        while at < to {
            // SAFETY: a query of the virtual address, without reading it.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            let size = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            let end = (info.BaseAddress as usize)
                .saturating_add(info.RegionSize)
                .min(to);
            if size != size_of_val(&info) || end <= at {
                // No answer: as before, all the rest copied.
                bytes += self.copy_unwatched(at, to);
                break;
            }
            let readable = info.State == MEM_COMMIT
                && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0
                && info.Protect != 0;
            if readable {
                bytes += match crate::watch::writes(at, end, true) {
                    Some((now, written)) => {
                        let copied = self.copy_stale(at, end, now, &written);
                        #[cfg(feature = "diagnostics")]
                        {
                            add(C::Checked, (end - at) as u64);
                            add(C::Unchanged, (end - at).saturating_sub(copied) as u64);
                            add(C::WatchBytes, copied as u64);
                        }
                        copied
                    }
                    None => self.copy_unwatched(at, end),
                };
            } else {
                // Uncommitted: zeros in the copy. Whole pages already
                // zeroed while uncommitted are skipped; once committed they
                // count as never copied (see `ZEROED` in `copy_stale`).
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                let mut a = at;
                while a < end {
                    let page = a / PAGE;
                    let b = ((page + 1) * PAGE).min(end);
                    let whole = a == page * PAGE && b == (page + 1) * PAGE;
                    let slot = &mut copied[page - base];
                    if !(whole && *slot == ZEROED) {
                        // SAFETY: inside the mapping (offsets in the buffer).
                        unsafe { self.data.add(a - start).write_bytes(0, b - a) };
                        *slot = if whole { ZEROED } else { 0 };
                        #[cfg(feature = "diagnostics")]
                        add(C::Unmapped, (b - a) as u64);
                    }
                    a = b;
                }
            }
            at = end;
        }
        bytes
    }

    /// Copies `[from, to)` whole (the write watch cannot say what changed)
    /// and marks its pages never copied, so a later watched request copies
    /// them again.
    fn copy_unwatched(&self, from: usize, to: usize) -> usize {
        let base = self.start as usize / PAGE;
        self.copied.lock().unwrap_or_else(|e| e.into_inner())
            [from / PAGE - base..=(to - 1) / PAGE - base]
            .fill(0);
        #[cfg(feature = "diagnostics")]
        {
            add(C::UnwatchedRanges, 1);
            add(C::UnwatchedBytes, (to - from) as u64);
        }
        self.copy(from, to)
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        // A page only partly in the range would stay stale for the rest and
        // be copied again by every later range in it (directly indexed SRVs
        // ask for many small ranges per page, each submission). Where no
        // shader of a submission may be writing the rest of an edge page,
        // the range takes in the whole page, within the buffer.
        let (from, to) = self.whole_edge_pages(from, to);
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        #[cfg(feature = "diagnostics")]
        let (mut initial, mut dirty) = (0, 0);
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || *at == ZEROED || write > *at {
                #[cfg(feature = "diagnostics")]
                if *at == 0 {
                    initial += 1;
                } else {
                    dirty += 1;
                }
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
        #[cfg(feature = "diagnostics")]
        {
            add(C::WatchPages, (last + 1 - from / PAGE) as u64);
            add(C::WatchInitialPages, initial);
            add(C::WatchDirtyPages, dirty);
        }
        bytes
    }

    /// `[from, to)` widened to its first and last pages' bounds (cut to the
    /// buffer), each edge only where no pending range lies in what it adds.
    fn whole_edge_pages(&self, from: usize, to: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let lo = (from / PAGE * PAGE).max(start);
        let hi = (to.div_ceil(PAGE) * PAGE).min(end);
        if lo == from && hi == to {
            return (from, to);
        }
        let pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        let free = |a: usize, e: usize| a >= e || pending.iter().all(|&(p, q)| q <= a || p >= e);
        (
            if free(lo, from) { lo } else { from },
            if free(to, hi) { hi } else { to },
        )
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

/// The parts of `[from, to)` outside every range in `ranges`, in order.
fn outside(from: usize, to: usize, mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let (mut at, mut parts) = (from, Vec::new());
    for (a, e) in ranges {
        if at < to && a > at {
            parts.push((at, a.min(to)));
        }
        at = at.max(e);
    }
    if at < to {
        parts.push((at, to));
    }
    parts
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    crate::draw::lock_timed(&BUFFERS, 1)
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
    if address < 0x10000 {
        return None;
    }
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
        let (buffer, data) = allocate_buffer(len.next_multiple_of(1 << 16))?;
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

/// Every copy of title memory made so far (buffers and loose windows),
/// for the memory table: a program that loads an address from memory and
/// reads through it may land in any of them.
pub(crate) fn live() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = buffers()
        .values()
        .filter_map(|e| e.shadow.clone())
        .collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn allocate_buffer(width: u64) -> Option<(ID3D12Resource, *mut u8)> {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let allocation = host()?.shared_buffer(width)?;
    #[cfg(feature = "diagnostics")]
    {
        add(C::ShadowAllocations, 1);
        add(C::ShadowAllocationBytes, width);
        add(C::ShadowAllocationUs, started.elapsed().as_micros() as u64);
    }
    Some(allocation)
}

/// Loose windows copied, and the title memory they cover (less when
/// windows overlap and copy the same bytes more than once).
pub(crate) fn loose_cover() -> (usize, u64) {
    let map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let (mut covered, mut reach) = (0u64, 0u64);
    for &(start, len) in map.keys() {
        let end = start + len;
        if end > reach {
            covered += end - start.max(reach);
            reach = end;
        }
    }
    (map.len(), covered)
}

/// Loose window lookups so far: the clock of [`Shadow::used`].
static LOOSE_USES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The bytes of loose windows kept before the least recently used ones
/// nothing holds are released: `VICTORY_LOOSE_BUDGET_MB`, 1.5 GiB by
/// default. The world's GPU reads over 3 GB of such memory within minutes
/// and kept a copy of all of it for good.
fn loose_budget() -> u64 {
    static BUDGET: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("VICTORY_LOOSE_BUDGET_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(1536)
            << 20
    })
}

/// References to `buffer` besides the caller's: descriptor slots that
/// view it hold one each.
fn com_refs(buffer: &ID3D12Resource) -> u32 {
    use windows::core::Interface;
    // SAFETY: every COM interface's table starts with IUnknown's; an
    // AddRef then a Release leaves the count as it was and returns it.
    unsafe {
        let raw = buffer.as_raw();
        let table = &**raw.cast::<*const windows::core::IUnknown_Vtbl>();
        (table.AddRef)(raw);
        (table.Release)(raw)
    }
}

/// Whether nothing but the window map holds `shadow`: no list (which
/// holds it until reset, after its submission ran), no descriptor slot
/// viewing its buffer, and no GPU writes waiting to go back to title
/// memory. Title memory then holds all it has: it can be made again.
fn idle(shadow: &Arc<Shadow>) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let reason = if Arc::strong_count(shadow) != 1 {
        0
    } else if com_refs(&shadow.buffer) != 1 {
        1
    } else if !shadow.pending.lock().is_ok_and(|p| p.is_empty()) {
        2
    } else if !shadow.before.lock().is_ok_and(|b| b.pages.is_empty() && b.spans.is_empty()) {
        3
    } else {
        return true;
    };
    LOOSE_HELD[reason].fetch_add(1, Relaxed);
    false
}

/// At the last eviction: windows held by a list, by a descriptor slot, by
/// GPU writes waiting, by a snapshot; and windows released so far.
static LOOSE_HELD: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];
static LOOSE_RELEASED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What [`LOOSE_HELD`] and [`LOOSE_RELEASED`] say, for the memory log.
pub(crate) fn loose_eviction() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let h = LOOSE_HELD.each_ref().map(|a| a.load(Relaxed));
    format!(
        "loose released {}, held at last try: list {}, slot {}, gpu writes {}, snapshot {}",
        LOOSE_RELEASED.load(Relaxed),
        h[0],
        h[1],
        h[2],
        h[3]
    )
}

/// Releases the least recently used idle windows until those kept and
/// `more` bytes fit [`loose_budget`].
fn evict_loose(map: &mut BTreeMap<(u64, u64), Arc<Shadow>>, more: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let budget = loose_budget();
    let kept = || crate::memstat::LOOSE.load(Relaxed);
    if kept() + more <= budget {
        return;
    }
    for held in &LOOSE_HELD {
        held.store(0, Relaxed);
    }
    let mut idle: Vec<(u64, (u64, u64))> = map
        .iter()
        .filter(|(_, s)| idle(s))
        .map(|(k, s)| (s.used.load(Relaxed), *k))
        .collect();
    idle.sort_unstable();
    let mut freed = 0;
    for (_, key) in idle {
        if kept() + more <= budget {
            break;
        }
        if let Some(shadow) = map.remove(&key) {
            crate::memstat::sub(&crate::memstat::LOOSE, key.1);
            freed += key.1;
            LOOSE_RELEASED.fetch_add(1, Relaxed);
            crate::host::retire(shadow);
        }
    }
    if freed > 0 {
        tracing::debug!(
            freed_mib = freed >> 20,
            kept_mib = kept() >> 20,
            "loose windows released"
        );
    }
}

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let end = end.max(address.saturating_add(64 << 10));
    let len = end.checked_next_multiple_of(WINDOW)? - start;
    let now = LOOSE_USES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
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
        shadow.used.store(now, std::sync::atomic::Ordering::Relaxed);
        return Some(shadow);
    }
    if !crate::texture::committed(address as *const u8, 1) {
        return None;
    }
    evict_loose(&mut map, len);
    let (buffer, data) = allocate_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    shadow.used.store(now, std::sync::atomic::Ordering::Relaxed);
    map.insert((start, len), shadow.clone());
    crate::memstat::add(&crate::memstat::LOOSE, len);
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
                let mut o1 = old.clone();
                let fast = super::merge_gpu_writes(&mut t1, &mut h1, &mut o1);
                assert_eq!(o1, t1);
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
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

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


    /// Three pages of title memory on page bounds (plain heap memory, so
    /// not write-watched: refreshes copy whole ranges) and their host
    /// copy; `None` without a host GPU. Returns the memory's owner, the
    /// address of the first page and the copy.
    fn title() -> Option<(Vec<u8>, usize, Shadow)> {
        let memory = vec![0u8; 4 * PAGE];
        let base = (memory.as_ptr() as usize).next_multiple_of(PAGE);
        let (buffer, data) = host()?.shared_buffer(3 * PAGE as u64)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu, base as u64, 3 * PAGE as u64);
        Some((memory, base, shadow))
    }

    /// A write of one byte: the title's into its memory, or a shader's
    /// into the copy.
    fn poke(at: usize, value: u8) {
        // SAFETY: the tests pass addresses inside the memory or the copy.
        unsafe { (at as *mut u8).write_volatile(value) }
    }

    fn peek(at: usize) -> u8 {
        // SAFETY: as in `poke`.
        unsafe { (at as *const u8).read_volatile() }
    }

    /// A range declared after ranges on both sides of it, once the snapshot
    /// already spans it: what its refresh brought into the copy is not a
    /// shader's write, and does not go back over what the title wrote since.
    #[test]
    fn a_range_between_earlier_ones_keeps_the_titles_newer_bytes() {
        let Some((_memory, base, s)) = title() else {
            return; // no host GPU here
        };
        let (a, b, c) = (base, base + PAGE, base + 2 * PAGE);
        let copy = |at: usize| s.data as usize + (at - base);
        // An earlier submission read the middle page; the title has written
        // it since.
        s.update(b, PAGE);
        poke(b + 5, 1);
        poke(b + 6, 3);
        // This submission's shaders write the outer pages, then the middle
        // one: each range refreshed, then declared.
        for at in [a, c, b] {
            s.update(at, PAGE);
            s.writing(at, PAGE);
        }
        assert_eq!(peek(copy(b + 5)), 1, "the refresh");
        // While the GPU runs: a shader writes two bytes, the title one.
        poke(copy(a + 7), 9);
        poke(copy(b + 9), 7);
        poke(b + 5, 2);
        for at in [a, c, b] {
            s.copy_back(at, PAGE);
        }
        assert_eq!(peek(b + 5), 2, "the title's newer byte");
        assert_eq!(peek(b + 6), 3, "the title's byte from before");
        // What the shader wrote does go back, in either kind of range.
        assert_eq!((peek(a + 7), peek(b + 9)), (9, 7));
        assert!(
            s.before.lock().unwrap().pages.is_empty(),
            "the snapshot is done"
        );
    }

    /// Two declared ranges that overlap: the first one's copy-back gives
    /// the copy the title's bytes, which the second must not take for a
    /// shader's and put back over what the title wrote in between.
    #[test]
    fn overlapping_ranges_keep_the_titles_newer_bytes() {
        let Some((_memory, base, s)) = title() else {
            return; // no host GPU here
        };
        let copy = |at: usize| s.data as usize + (at - base);
        s.update(base, 2 * PAGE);
        s.writing(base, PAGE);
        s.writing(base, 2 * PAGE);
        // While the GPU runs: a shader's byte and the title's, in one page.
        poke(copy(base + 3), 9);
        poke(base + 8, 1);
        s.copy_back(base, PAGE);
        assert_eq!((peek(base + 3), peek(base + 8)), (9, 1));
        // The title writes its byte again before the second range's turn.
        poke(base + 8, 2);
        s.copy_back(base, 2 * PAGE);
        assert_eq!(peek(base + 8), 2, "the title's newer byte");
        assert_eq!(peek(base + 3), 9, "the shader's byte");
    }

    /// A copy of title memory `[at, at + len)` holding the same bytes, if
    /// there is a host device.
    fn shadow_of(at: usize, len: usize) -> Option<Shadow> {
        let (buffer, data) = host()?.shared_buffer((len as u64).next_multiple_of(PAGE as u64))?;
        // SAFETY: the mapping holds `len` bytes; the title memory is ours.
        unsafe { std::ptr::copy_nonoverlapping(at as *const u8, data, len) };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Some(Shadow::new(buffer, data, gpu, at as u64, len as u64))
    }

    /// Two overlapping ranges shaders write, declared with a refresh in
    /// between: only the bytes the GPU changed go back, the title's own
    /// writes meanwhile stay, and the page buffers return to the pool.
    #[test]
    fn copy_back_takes_only_what_the_gpu_wrote() {
        let mut memory = vec![1u8; 4 * PAGE];
        // Off a page boundary, over three title pages.
        let base = (memory.as_mut_ptr() as usize).next_multiple_of(PAGE) + 100;
        let len = 2 * PAGE + 500;
        let Some(shadow) = shadow_of(base, len) else {
            return;
        };
        // SAFETY: offsets inside the title range and the copy.
        let title = |at: usize| unsafe { (base as *mut u8).add(at) };
        // SAFETY: as above.
        let host = |at: usize| unsafe { shadow.data.add(at) };
        // SAFETY: (all writes below) bytes of the title range and the copy,
        // which nothing else uses.
        unsafe {
            shadow.writing(base + 10, PAGE + 10);
            // A refresh next to the first range, in its last page.
            title(PAGE + 100).write(7);
            host(PAGE + 100).write(7);
            shadow.writing(base + PAGE + 50, PAGE + 50);
            // The GPU writes in both ranges; the title writes meanwhile.
            host(20).write(0xaa);
            host(PAGE + 200).write(0xbb);
            title(30).write(0xcc);
        }
        assert!(shadow.copy_back(base + 10, PAGE + 10) > 0);
        assert!(shadow.copy_back(base + PAGE + 50, PAGE + 50) > 0);
        // SAFETY: as above.
        unsafe {
            assert_eq!(title(20).read(), 0xaa);
            assert_eq!(title(PAGE + 200).read(), 0xbb);
            assert_eq!(title(30).read(), 0xcc, "the title's own write stays");
            assert_eq!(host(30).read(), 0xcc, "and the copy takes it");
            assert_eq!(title(PAGE + 100).read(), 7);
            assert_eq!(title(40).read(), 1);
        }
        assert!(shadow.before.lock().unwrap().pages.is_empty());
        assert!(PAGES.lock().unwrap().len() >= 3);
        drop(memory);
    }

    /// A range declared again, or inside one declared, takes nothing new.
    #[test]
    fn declaring_a_range_again_keeps_its_snapshot() {
        let memory = vec![3u8; 3 * PAGE];
        let base = memory.as_ptr() as usize;
        let Some(shadow) = shadow_of(base, 3 * PAGE) else {
            return;
        };
        shadow.writing(base, 2 * PAGE);
        // SAFETY: a byte of the copy, which nothing else uses: a GPU write
        // between two dispatches declaring the range.
        unsafe { shadow.data.add(5).write(9) };
        shadow.writing(base, 2 * PAGE);
        shadow.writing(base + 4, 100);
        {
            let before = shadow.before.lock().unwrap();
            assert_eq!(before.spans, [(0, 2 * PAGE)]);
            let at = base + 5;
            assert_eq!(before.pages[&(at / PAGE)].bytes[at % PAGE], 3);
        }
        shadow.copy_back(base, 2 * PAGE);
        shadow.copy_back(base + 4, 100);
        // SAFETY: title memory of ours.
        assert_eq!(unsafe { (base as *const u8).add(5).read() }, 9);
        assert!(shadow.before.lock().unwrap().pages.is_empty());
        drop(memory);
    }

    /// A write-watched title allocation of two pages, registered as a
    /// buffer, with its copy.
    struct Watched {
        base: usize,
        shadow: Arc<Shadow>,
    }

    impl Watched {
        fn new() -> Option<Watched> {
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            crate::host::host()?;
            // SAFETY: a new committed, write-watched allocation the test owns.
            let at = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    2 * PAGE,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            };
            assert!(!at.is_null());
            let base = at as usize;
            register(base, base as u64, 2 * PAGE as u64);
            let shadow = find(base as u64, 2 * PAGE as u64).unwrap();
            Some(Watched { base, shadow })
        }
    }

    impl Drop for Watched {
        fn drop(&mut self) {
            unregister(self.base, self.base as u64);
            // SAFETY: allocated with VirtualAlloc in `new`.
            unsafe { VirtualFree(self.base as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// Small ranges in one page (as directly indexed SRVs ask for them): the
    /// first copies the page, the others nothing until the title writes it.
    #[test]
    fn small_ranges_in_one_page_copy_it_once() {
        let Some(t) = Watched::new() else {
            return;
        };
        let page = t.base + PAGE;
        // SAFETY: inside the test's allocation.
        unsafe { (page as *mut u32).write(1) };
        assert_eq!(t.shadow.update(page + 16, 16), PAGE);
        for offset in (256..PAGE).step_by(4096) {
            assert_eq!(t.shadow.update(page + offset, 16), 0);
        }
        // SAFETY: inside the test's allocation.
        unsafe { ((page + 8192) as *mut u32).write(500) };
        assert_eq!(t.shadow.update(page + 16, 16), PAGE);
        // SAFETY: the copy's mapping covers the allocation.
        let copied = unsafe { t.shadow.data.add(PAGE + 8192).cast::<u32>().read() };
        assert_eq!(copied, 500);
    }

    /// Where shaders of a submission may be writing the rest of the page,
    /// the range is copied as asked and the copy there is left alone.
    #[test]
    fn edge_pages_stay_cut_next_to_pending_writes() {
        let Some(t) = Watched::new() else {
            return;
        };
        let page = t.base + PAGE;
        t.shadow.writing(page + 1024, 64);
        // SAFETY: inside the copy's mapping and the test's allocation.
        unsafe {
            t.shadow.data.add(PAGE + 1024).cast::<u32>().write(7);
            ((page + 1024) as *mut u32).write(13);
        }
        assert_eq!(t.shadow.update(page + 16, 16), 32);
        // SAFETY: inside the copy's mapping.
        let kept = unsafe { t.shadow.data.add(PAGE + 1024).cast::<u32>().read() };
        assert_eq!(kept, 7);
    }
}

/// Whether a range refreshed already in the watch generation is skipped:
/// `VICTORY_REFRESH_MEMO=0` turns it off.
fn memo() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_REFRESH_MEMO").map_or(true, |v| v.trim() != "0"))
}
