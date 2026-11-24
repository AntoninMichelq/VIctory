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
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Before>,
}

#[derive(Default)]
struct Before {
    snapshot: Option<(usize, Vec<u8>)>,
    unchanged: UnchangedPages,
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

    #[cfg(test)]
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
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
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
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame). Returns the bytes copied.
    pub(crate) fn writing(&self, address: usize, len: usize) -> usize {
        let mut copied = 0;
        let (from, to) = self.inside(address, len);
        // The parts no earlier range of the submission declared: their
        // refresh was free to bring the title's bytes into the copy (a
        // refresh leaves declared ranges alone), and may just have.
        let fresh = {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            let declared = pending.iter().map(|&(a, e)| self.inside(a, e - a));
            let fresh = outside(from, to, declared.collect());
            pending.push((address, address + len));
            fresh
        };
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let before = &mut before.snapshot;
        let old = match &*before {
            Some((at, bytes)) => (*at, at + bytes.len()),
            None => (from, from),
        };
        let (start, end) = (old.0.min(from), old.1.max(to));
        if (start, end) != old {
            // Parts newly covered from the copy, the rest as it was taken.
            let mut grown = pooled(end - start);
            match before.take() {
                Some((at, bytes)) => {
                    grown.extend_from_slice(host(start, at));
                    grown.extend_from_slice(&bytes);
                    grown.extend_from_slice(host(at + bytes.len(), end));
                    give_back(bytes);
                }
                None => grown.extend_from_slice(host(start, end)),
            }
            *before = Some((start, grown));
            copied += end - start;
        }
        // What the snapshot already held of the fresh parts (it spans the
        // gaps between the ranges declared so far) was taken before their
        // refresh: taken again, or the refresh would pass for a shader's
        // write and go back over what the title writes meanwhile.
        if let Some((at, bytes)) = &mut *before {
            for (a, e) in fresh {
                let (a, e) = (a.max(old.0), e.min(old.1));
                if a < e {
                    bytes[a - *at..e - *at].copy_from_slice(host(a, e));
                    copied += e - a;
                }
            }
        }
        copied
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
        } = &mut *before;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = &mut *snapshot
            && *taken <= from
            && to <= *taken + before.len()
            && from < to
        {
            unchanged.begin(start + *taken, start + *taken + before.len());
            let before = &mut before[from - *taken..to - *taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            // An unchanged page is never modified by copy-back, so later
            // overlapping ranges can skip it for the rest of the submission.
            // Page-aligned parts, each on its own thread when the range is
            // big: pages are independent (own unchanged flag, own bytes).
            let bounds = (unchanged.start, unchanged.end);
            let first_page = unchanged.start / PAGE;
            let data = self.data as usize;
            let snap = before.as_mut_ptr() as usize;
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
                    // SAFETY: inside the snapshot; each page from one part.
                    let old = unsafe { std::slice::from_raw_parts_mut((snap as *mut u8).add(at - from), n) };
                    let flag = &mut flags[(start + at) / PAGE - flags_first];
                    if page_differs(flag, bounds, start + at, host, old)
                        && let title = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n]
                        && crate::host::read_into((start + at) as *const u8, title)
                    {
                        // Title memory with the GPU's bytes put in; the copy
                        // takes the title's bytes in turn, and so does the
                        // snapshot: to a later range that overlaps this one,
                        // they are not a shader's write to put back.
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
            && let Some((_, bytes)) = snapshot.take()
        {
            unchanged.clear();
            give_back(bytes);
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
    } else if !shadow.before.lock().is_ok_and(|b| b.snapshot.is_none()) {
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
        assert!(s.before.lock().unwrap().snapshot.is_none(), "the snapshot is done");
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
}
