//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! unmapped are not asked about again until it maps them: view memory
//! comes from placeholders, which only the runtime's mapping fills.
//! shortcut: so memory committed otherwise into such a block (a range
//! reaching from a view into a plain reservation) is not seen.
//!
//! Writes into physical pages through another view (DirectStorage's page
//! destinations) are not seen at the guarded addresses on their own: the
//! runtime reports them afterwards for every title address showing those
//! pages (xmem's `physical_written`, through [`writing`]). Streamed world
//! models land there too, not only textures. A system call writing into a
//! guarded block fails (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// Whether every watch page of `[start, end)` is in one committed private
/// region. A failed query for a subrange then cannot have crossed a region
/// boundary or encountered an unmapped page elsewhere in the shadow.
pub(crate) fn private_region_covers(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let start = start / PAGE * PAGE;
    let Some(end) = end.checked_next_multiple_of(PAGE) else {
        return false;
    };
    // SAFETY: a query of the virtual address, without reading its memory.
    unsafe {
        let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        VirtualQuery(start as *const _, &mut info, size_of_val(&info)) != 0
            && info.State == MEM_COMMIT
            && info.Type == MEM_PRIVATE
            && (info.BaseAddress as usize)
                .checked_add(info.RegionSize)
                .is_some_and(|limit| limit >= end)
    }
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Title allocations seen: base → (end, whether write-watched). Only
    /// write-watched ones outlive an epoch.
    allocations: BTreeMap<usize, (usize, bool)>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    allocations: BTreeMap::new(),
});

std::thread_local! {
    /// While this thread prepares a submission ([`begin_epoch`]): pages
    /// asked of the kernel since it began, as disjoint runs (first page →
    /// end page). Asks inside them are answered from the journal. Per
    /// thread: another thread's asks are not part of the preparation.
    static EPOCH: std::cell::RefCell<Option<BTreeMap<usize, usize>>> =
        const { std::cell::RefCell::new(None) };
}

fn journal() -> std::sync::MutexGuard<'static, Journal> {
    JOURNAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bytes of title memory around an ask that one kernel ask takes in during
/// an epoch (within the allocation): `VICTORY_WATCH_WINDOW_MB`, 32 by
/// default.
fn window() -> usize {
    static W: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        let mb = std::env::var("VICTORY_WATCH_WINDOW_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(32)
            .min(4096);
        (mb << 20).max(PAGE)
    })
}

/// Whether submissions ask the kernel in epochs; `VICTORY_WATCH_EPOCH=0`
/// asks it for every range as before.
fn epochs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_WATCH_EPOCH").map_or(true, |v| v.trim() != "0"))
}

/// Starts (or restarts) an epoch: a submission's preparation, during which
/// nothing of ours writes title memory. The first ask in a part of a
/// write-watched allocation asks the kernel for a window around it (up to
/// [`window`], within the allocation); later asks inside what was asked
/// are answered from the journal, without a system call (a world
/// submission asks thousands of times: every texture and buffer range).
///
/// Per thread: only the thread that began it answers from the epoch.
///
/// What the title's threads write after the window was asked is not seen
/// until the next epoch's ask, which stamps it newer than any copy made now,
/// so it is copied by the next submission. A write racing the preparation
/// had no order with the GPU's reads before either: the ask it raced might
/// as well have come first. Our own writes into title memory (a
/// submission's copies and work, copy-back) end the epoch first
/// ([`end_epoch`]).
pub(crate) fn begin_epoch() {
    if !epochs() {
        return;
    }
    EPOCH.with_borrow_mut(|e| *e = Some(BTreeMap::new()));
    new_generation();
    let mut j = journal();
    // Memory that was not watched may be by now (remapped, reallocated).
    j.allocations.retain(|_, (_, watched)| *watched);
}

/// Ends the epoch: asks go to the kernel again.
pub(crate) fn end_epoch() {
    EPOCH.with_borrow_mut(|e| *e = None);
    GENERATION.set(0);
}

static GENERATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

std::thread_local! {
    /// This thread's epoch generation: 0 outside an epoch, else a value
    /// unique to the epoch, changed by each of our own title writes in it
    /// ([`uncover`]). Within one generation title memory, as the journal
    /// answers for it, does not change.
    static GENERATION: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn new_generation() {
    GENERATION.set(GENERATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1);
}

/// See [`GENERATION`]: shadows refreshing a range twice in one generation
/// can skip the second time (the world asks for one 174 MB range 22-48
/// times per frame).
pub(crate) fn generation() -> u64 {
    GENERATION.get()
}

/// Our own write of title memory `[at, at + len)` inside an epoch: the
/// pages it touches are asked of the kernel again (it reports our write),
/// the rest of the epoch's answers stand. Outside an epoch, nothing.
pub(crate) fn uncover(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, last) = (at / PAGE, (at + len).div_ceil(PAGE));
    EPOCH.with_borrow_mut(|runs| {
        let Some(runs) = runs else { return };
        new_generation();
        let hit: Vec<(usize, usize)> = runs
            .range(..last)
            .filter(|&(_, &e)| e > first)
            .map(|(&a, &e)| (a, e))
            .collect();
        for (a, e) in hit {
            runs.remove(&a);
            if a < first {
                runs.insert(a, first);
            }
            if e > last {
                runs.insert(last, e);
            }
        }
    });
}

/// Whether pages `[first, last)` were asked in this epoch.
fn covered(runs: &BTreeMap<usize, usize>, first: usize, last: usize) -> bool {
    runs.range(..=first).next_back().is_some_and(|(_, &end)| end >= last)
}

/// Notes pages `[first, last)` asked in this epoch, merging runs.
fn cover(runs: &mut BTreeMap<usize, usize>, mut first: usize, mut last: usize) {
    if let Some((&a, &e)) = runs.range(..=first).next_back()
        && e >= first
    {
        first = a;
        last = last.max(e);
    }
    let inside: Vec<(usize, usize)> = runs.range(first..=last).map(|(&a, &e)| (a, e)).collect();
    for (a, e) in inside {
        runs.remove(&a);
        last = last.max(e);
    }
    runs.insert(first, last);
}

/// The allocation holding `at`: (base, end, write-watched), from what was
/// seen or the system's answer (then taken as watched until an ask fails).
fn allocation(j: &mut Journal, at: usize) -> Option<(usize, usize, bool)> {
    if let Some((&base, &(end, watched))) = j.allocations.range(..=at).next_back()
        && at < end
    {
        return Some((base, end, watched));
    }
    use windows_sys::Win32::System::Memory::{MemoryRegionInfo, QueryVirtualMemoryInformation, WIN32_MEMORY_REGION_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut region = WIN32_MEMORY_REGION_INFORMATION::default();
    // SAFETY: describes an address without dereferencing it; the output
    // buffer matches the information class.
    let queried = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            at as *const _,
            MemoryRegionInfo,
            (&raw mut region).cast(),
            size_of_val(&region),
            std::ptr::null_mut(),
        )
    } != 0;
    let base = region.AllocationBase as usize;
    let end = base.checked_add(region.RegionSize)?;
    if !queried || base > at || end <= at {
        return None;
    }
    // Entries overlapping it are gone (the cache keeps disjoint ones).
    let stale: Vec<usize> = j.allocations.range(..end).filter(|(_, (e, _))| *e > base).map(|(&b, _)| b).collect();
    for b in stale {
        j.allocations.remove(&b);
    }
    j.allocations.insert(base, (end, true));
    Some((base, end, true))
}

/// Asks the kernel for the written pages of `[from, to)` (bytes; whole
/// kernel pages), clearing them, into `j.found`. The count found, or `None`
/// if the range is not write-watched (or spans allocations).
fn ask(j: &mut Journal, from: usize, to: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (to - from).div_ceil(KERNEL_PAGE);
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    ASKS[2].fetch_add(1, Ordering::Relaxed);
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            from as *const _,
            to - from,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    if failed {
        ASKS[3].fetch_add(1, Ordering::Relaxed);
    }
    (!failed).then_some(count)
}

/// Inside an epoch: answers `[first, last)` (pages) from the journal if
/// asked already, else asks the kernel for a window around it. `None`:
/// not answered (ask as outside an epoch); `Some(None)`: the memory is not
/// write-watched.
fn from_epoch(j: &mut Journal, first: usize, last: usize, kend: usize) -> Option<Option<u64>> {
    if EPOCH.with_borrow(|e| e.as_ref().map(|runs| covered(runs, first, last)))? {
        return Some(Some(j.asked));
    }
    let (base, end, watched) = allocation(j, first * PAGE)?;
    // The asked range (to whole kernel pages) must lie in the allocation;
    // its last 64 KiB page may end past it (allocations end on kernel
    // pages): that part is no one's memory.
    if kend > end {
        return None;
    }
    if !watched {
        return Some(None);
    }
    let w = window();
    let from = (first * PAGE / w * w).max(base);
    let to = (last * PAGE).next_multiple_of(w).min(end);
    let Some(count) = ask(j, from, to) else {
        // Not watched, or no longer the allocation it was: as outside an
        // epoch, which notes it unwatched if that fails too.
        j.allocations.remove(&base);
        return None;
    };
    j.asked += 1;
    let now = j.asked;
    let found: Vec<usize> = j.found[..count].iter().map(|&at| at / PAGE).collect();
    for page in found {
        j.written.insert(page, now);
    }
    // Whole pages were asked, and a last one cut by the allocation's end
    // (the rest of it is not allocated: allocations start on 64 KiB).
    // Before, that page was asked of the kernel again on every ask, past
    // the allocation's end, and failed: most of the world's kernel asks.
    let (a, e) = (from.div_ceil(PAGE), if to == end { to.div_ceil(PAGE) } else { to / PAGE });
    if a < e {
        EPOCH.with_borrow_mut(|runs| {
            if let Some(runs) = runs {
                cover(runs, a, e);
            }
        });
    }
    Some(Some(now))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let started = std::time::Instant::now();
    let answer = writes_now(start, end, guarded);
    ASKS[0].fetch_add(1, Ordering::Relaxed);
    ASKS[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    answer
}

/// Since the last [`take_asks`]: calls of [`writes`], the microseconds
/// they took, and the asks of the kernel among them.
static ASKS: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// Calls of [`writes`], their microseconds and the kernel asks they made,
/// since the last call.
pub(crate) fn take_asks() -> [u64; 4] {
    [0, 1, 2, 3].map(|i| ASKS[i].swap(0, Ordering::Relaxed))
}

fn writes_now(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = journal();
    let j = &mut *j;
    // The kernel answers for whole kernel pages inside one allocation.
    let kend = end.next_multiple_of(KERNEL_PAGE);
    let now = match from_epoch(j, first, last, kend) {
        Some(Some(now)) => now,
        Some(None) => {
            // Not write-watched: as when the kernel says so below.
            if !guarded {
                return None;
            }
            let pages = guard(first * PAGE, last * PAGE)?;
            return Some(record(j, pages, first, last));
        }
        None => match ask(j, first * PAGE, kend) {
            Some(count) => {
                let pages: Vec<usize> = j.found[..count].iter().map(|&at| at / PAGE).collect();
                return Some(record(j, pages, first, last));
            }
            None => {
                // Not write-watched: in an epoch, the allocation is noted
                // so, so that its next ranges do not ask again.
                if EPOCH.with_borrow(Option::is_some)
                    && let Some((base, end, _)) = allocation(j, first * PAGE)
                    && kend <= end
                {
                    j.allocations.insert(base, (end, false));
                }
                if !guarded {
                    return None;
                }
                let pages = guard(first * PAGE, last * PAGE)?;
                return Some(record(j, pages, first, last));
            }
        },
    };
    Some((now, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// Notes `pages` written as of a new asking; its number and the journal's
/// entries for pages `[first, last)`.
fn record(j: &mut Journal, pages: Vec<usize>, first: usize, last: usize) -> (u64, Vec<(usize, u64)>) {
    j.asked += 1;
    let now = j.asked;
    for page in pages {
        j.written.insert(page, now);
    }
    (now, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect())
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks found unmapped (not committed) when a copy took them: the
    /// copy holds their zeros until the runtime maps them
    /// ([`title_memory_remapped`]). Without this, every asking found them
    /// "written" again (they cannot be made read-only) and the copy read
    /// them page by page through the kernel's failing reads, hundreds of
    /// MB per world frame.
    absent: BTreeSet<usize>,
    /// Blocks found to be in a view when an asking began there, until the
    /// runtime remaps them. Asking the system each time (`VirtualQuery`)
    /// cost 100-230 ms per world frame, more than the rest of the guard.
    views: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    absent: BTreeSet::new(),
    views: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    install();
    let mut g = guards();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Asked under the lock: a remap in between would leave a stale view.
    if !g.views.contains(&first) {
        if !view(start) {
            return None;
        }
        g.views.insert(first);
    }
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    for block in first..=last {
        let open = block < last && !g.clean.contains(&block) && !g.absent.contains(&block);
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Whether the current allocation at `at` is a mapped view. Asking only for
/// the allocation type avoids enumerating pages with identical protection,
/// which can span an entire large view after its blocks were guarded.
fn mapped(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_MAPPED, MEMORY_BASIC_INFORMATION, MemoryRegionInfo, QueryVirtualMemoryInformation,
        VirtualQuery, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut region = WIN32_MEMORY_REGION_INFORMATION::default();
    // SAFETY: both queries describe an address without dereferencing it. The
    // output buffers match their information classes.
    let queried = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            at as *const _,
            MemoryRegionInfo,
            (&raw mut region).cast(),
            size_of_val(&region),
            std::ptr::null_mut(),
        )
    };
    if queried != 0 {
        // These are the Windows SDK's named bitfields in declaration order.
        // windows-sys exposes only the union's raw bits.
        const PRIVATE: u32 = 1 << 0;
        const MAPPED_DATA_FILE: u32 = 1 << 1;
        const MAPPED_IMAGE: u32 = 1 << 2;
        const MAPPED_PAGE_FILE: u32 = 1 << 3;
        const MAPPED_PHYSICAL: u32 = 1 << 4;
        // SAFETY: this union was filled by the successful query above.
        match unsafe { region.Anonymous.Flags } {
            MAPPED_DATA_FILE | MAPPED_PAGE_FILE | MAPPED_PHYSICAL => return true,
            PRIVATE | MAPPED_IMAGE => return false,
            _ => {}
        }
    }
    // Unknown flag combinations and failed allocation queries keep the
    // VirtualQuery type check, including for a free address.
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: the output buffer has the required layout and size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.Type == MEM_MAPPED
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent.
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else if unmapped(block * BLOCK) {
            g.absent.insert(block);
        }
    }
}

/// Whether `at` is in a view (of a section): [`mapped`], counted in tests.
fn view(at: usize) -> bool {
    #[cfg(test)]
    tests::QUERIED.with(|n| n.set(n.get() + 1));
    mapped(at)
}

/// Whether nothing is committed at `at` (a placeholder between views).
fn unmapped(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State != MEM_COMMIT
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    open(&mut guards(), at, at + len);
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    {
        // Its allocations are new too.
        let end = at.saturating_add(len);
        let mut j = journal();
        let gone: Vec<usize> =
            j.allocations.range(..end).filter(|(_, (e, _))| *e > at).map(|(&b, _)| b).collect();
        for b in gone {
            j.allocations.remove(&b);
        }
    }
    let mut g = guards();
    let range = at / BLOCK..(at + len).div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let views: Vec<usize> = g.views.range(range.clone()).copied().collect();
    for block in views {
        g.views.remove(&block);
    }
    let blocks: Vec<usize> = g.protected.range(range).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
        VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    thread_local! {
        /// The view queries `guard` made on this thread (each test runs on
        /// its own).
        pub(super) static QUERIED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn queried() -> usize {
        QUERIED.with(|n| n.get())
    }

    /// A view is asked about once per block an asking starts in, again
    /// only after the runtime remapped it; other memory every time.
    #[test]
    fn views_are_queried_once_until_remapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let at = queried();
        assert!(guard(view, view + 2 * BLOCK).is_some());
        assert!(guard(view + 8, view + BLOCK).is_some());
        assert_eq!(queried(), at + 1);
        // Another start block is asked about once.
        assert!(guard(view + BLOCK, view + 2 * BLOCK).is_some());
        assert!(guard(view + BLOCK, view + 2 * BLOCK).is_some());
        assert_eq!(queried(), at + 2);
        // The runtime remapped the first block: asked again.
        title_memory_remapped(view, BLOCK);
        assert!(guard(view, view + 2 * BLOCK).is_some());
        assert_eq!(queried(), at + 3);
        // Private memory is not a view, however often asked.
        let plain = vec![0u8; 2 * BLOCK];
        let p = plain.as_ptr() as usize;
        assert!(guard(p, p + 16).is_none());
        assert!(guard(p, p + 16).is_none());
        assert_eq!(queried(), at + 5);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            })
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn allocation_query_recognizes_file_views_and_rejects_nonviews() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY, UnmapViewOfFile,
        };
        let file = tempfile::tempfile().unwrap();
        file.set_len(PAGE as u64).unwrap();
        // SAFETY: a valid, open file and a read-only mapping of its contents.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: an open section containing at least one page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_READ, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        assert!(mapped(view.Value as usize + 13));
        // SAFETY: our successful mapping and its section handle.
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(section);
        }
        assert!(!mapped((&raw const file) as usize)); // private stack allocation
        assert!(!mapped(mapped as *const () as usize)); // executable image
        assert!(!mapped(0)); // no allocation to query
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// In an epoch the first ask takes in the allocation around it; asks
    /// inside it then come from the journal, and a write after it shows at
    /// the next epoch's ask, newer than the answer given before it.
    #[test]
    fn epoch_asks_the_kernel_once_per_window() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                8 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        // SAFETY: committed pages of ours.
        unsafe { ((p + 5 * PAGE) as *mut u8).write(1) };
        begin_epoch();
        let (first, at_first) = writes(p, p + PAGE, false).expect("watched");
        assert!(at_first.iter().all(|&(page, _)| page != p / PAGE + 5));
        // The page written before the epoch, though not asked for above.
        let (now, written) = writes(p + 4 * PAGE, p + 6 * PAGE, false).expect("watched");
        assert!(now >= first);
        assert_eq!(written, [(p / PAGE + 5, first)]);
        // SAFETY: as above.
        unsafe { ((p + 2 * PAGE) as *mut u8).write(1) };
        // Within the epoch, not yet seen.
        let (_, inside) = writes(p + 2 * PAGE, p + 3 * PAGE, false).expect("watched");
        assert!(inside.is_empty());
        end_epoch();
        begin_epoch();
        let (later, seen) = writes(p + 2 * PAGE, p + 3 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(seen, [(p / PAGE + 2, later)]);
        end_epoch();
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        title_memory_remapped(p, 8 * PAGE);
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    pub(crate) struct Reservation(pub(crate) *mut std::ffi::c_void);

    impl Drop for Reservation {
        fn drop(&mut self) {
            // SAFETY: this is the base of a reservation owned by the test.
            assert_ne!(unsafe { VirtualFree(self.0, 0, MEM_RELEASE) }, 0);
        }
    }

    pub(crate) fn adjacent(watched: usize, unwatched: usize) -> (Reservation, Reservation) {
        for _ in 0..16 {
            // SAFETY: these independent test reservations use valid allocation flags.
            unsafe {
                let span = VirtualAlloc(
                    std::ptr::null(),
                    watched + unwatched,
                    MEM_RESERVE,
                    PAGE_NOACCESS,
                );
                assert!(!span.is_null());
                assert_ne!(VirtualFree(span, 0, MEM_RELEASE), 0);
                let first = VirtualAlloc(
                    span,
                    watched,
                    MEM_COMMIT | MEM_RESERVE | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                );
                if first.is_null() {
                    continue;
                }
                let first = Reservation(first);
                let second = VirtualAlloc(
                    span.cast::<u8>().wrapping_add(watched).cast(),
                    unwatched,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
                );
                if !second.is_null() {
                    return (first, Reservation(second));
                }
            }
        }
        panic!("adjacent test reservations");
    }

    #[test]
    fn unwatched_subrange_does_not_disable_watch_elsewhere_in_shadow() {
        if crate::host::host().is_none() {
            return;
        }
        let watched = 16 << 20;
        let (first, second) = adjacent(watched, PAGE);
        let start = first.0 as usize;
        let len = (watched + PAGE) as u64;
        struct RegisteredBuffer(usize);
        impl Drop for RegisteredBuffer {
            fn drop(&mut self) {
                crate::shadow::unregister(self.0, self.0 as u64);
            }
        }
        crate::shadow::register(start, start as u64, len);
        let _registration = RegisteredBuffer(start);
        // SAFETY: both reservations are live. Populate demand-zero pages
        // before the first refresh, since touching them can trip write watch.
        unsafe {
            first.0.cast::<u8>().write_bytes(0, watched);
            first.0.cast::<u32>().write(13);
            second.0.cast::<u32>().write(27);
        }
        let shadow = crate::shadow::find(start as u64, len).unwrap();
        assert_eq!(shadow.update(start, watched), watched);
        assert_eq!(shadow.update(start, watched), 0);

        // A small view in the next allocation needs the CPU fallback. A large
        // conventional view on the same shadow can still use write watching.
        assert_eq!(shadow.update(second.0 as usize, 4), 4);
        let refreshed: usize = (0..4).map(|_| shadow.update(start, watched)).sum();
        eprintln!(
            "4-byte unwatched refresh, then 4 clean watched requests copied {refreshed} bytes"
        );
        assert_eq!(refreshed, 0);

        // SAFETY: the first allocation remains live.
        unsafe { first.0.cast::<u32>().write(500) };
        assert_eq!(shadow.update(start, watched), PAGE);
        // SAFETY: the persistently mapped shadow owns the refreshed word.
        let mut data = std::ptr::null_mut();
        // SAFETY: the live buffer contains the refreshed word. No GPU work is in flight.
        unsafe {
            shadow.buffer.Map(0, None, Some(&mut data)).unwrap();
            let word = data.cast::<u32>().read_unaligned();
            shadow.buffer.Unmap(0, None);
            assert_eq!(word, 500);
        }

        // A later fallback crossing into the watched allocation invalidates
        // that page's generation without affecting the remaining pages.
        assert_eq!(shadow.update(start + watched - 4, 8), 8);
        assert_eq!(shadow.update(start, watched), PAGE);
        assert_eq!(shadow.update(start, watched), 0);
    }
}
