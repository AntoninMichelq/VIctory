//! `VICTORY_SHOW_AFTER=1` (off by default): a title texture slot shows a
//! host texture that is new to it only once the submission carrying that
//! texture's first upload has finished on the card. Until then the slot
//! shows the grey stand-in (`descriptor::host::grey_for`), or goes on
//! showing the previous view of the same placed texture, and it is made
//! again before a later submission (`descriptor::make_waiting`).
//!
//! The translated shaders read the view heap by index: a slot written at
//! once is read by the lists already on the card, while the texture it
//! shows is still being made and filled. Idea from cool-fox-codec `3bab23c`
//! (there it took away a device loss every ~16 minutes: page faults in a
//! texture made and shown in the same tenth of a second).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::resource::HostTexture;

/// `HostTexture::upload`: no upload recorded yet.
pub(crate) const NOT_YET: u64 = 0;
/// `HostTexture::upload`: recorded in a list that is not submitted yet.
pub(crate) const RECORDED: u64 = u64::MAX;
/// `HostTexture::upload` of a texture nothing waits for any more.
const THERE: u64 = 1;
/// How long a slot waits at most (ms): past that it shows the texture as
/// before (an upload that never comes, a list never submitted).
const MOST_MS: u64 = 3000;

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_SHOW_AFTER").is_some_and(|v| v != "0");
        if on {
            tracing::info!("VICTORY_SHOW_AFTER=1: texture slots show a new host texture once its upload has run");
        }
        on
    })
}

/// Whether the first upload of a texture whose `HostTexture::upload` is
/// `upload` has run, with `finished` submissions finished.
pub(crate) fn landed(upload: u64, finished: u64) -> bool {
    upload != NOT_YET && upload != RECORDED && upload <= finished
}

/// Slot writes deferred, stand-ins written, old views kept meanwhile,
/// slots made after the upload ran, slots made after [`MOST_MS`].
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
pub(crate) const STAND_IN: usize = 1;
pub(crate) const KEPT_OLD: usize = 2;

pub(crate) fn count(what: usize) {
    COUNTS[what].fetch_add(1, Relaxed);
}

/// A slot that waits: the record it waits to show, since when (ms), and
/// the texture.
struct Late {
    key: u64,
    since: u64,
    texture: Weak<HostTexture>,
}

/// Slots that wait, by title handle.
static LATE: Mutex<BTreeMap<u64, Late>> = Mutex::new(BTreeMap::new());

/// What [`hold`] says of a slot write.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Hold {
    /// Made as before.
    No,
    /// It waits from now on: the stand-in is written.
    First,
    /// It waits already: nothing to write.
    Again,
}

/// Whether the slot at title `handle`, about to be made from record `key`,
/// a view of `texture`, waits for the texture's first upload.
pub(crate) fn hold(handle: u64, key: u64, texture: &Arc<HostTexture>) -> Hold {
    // Only what the title fills and the pass uploads whole.
    if !texture.filled() || texture.tiled.is_some() {
        return Hold::No;
    }
    let upload = texture.upload.load(Relaxed);
    if landed(upload, crate::host::submissions().1) {
        return Hold::No;
    }
    let now = crate::host::now_ms();
    let mut late = LATE.lock().unwrap_or_else(|e| e.into_inner());
    let (hold, since) = decide(late.get(&handle).map(|l| (l.key, l.since, l.texture.as_ptr())), key, Arc::as_ptr(texture), now);
    match hold {
        Hold::First => {
            late.insert(handle, Late { key, since, texture: Arc::downgrade(texture) });
            COUNTS[0].fetch_add(1, Relaxed);
        }
        Hold::No => {
            // Waited long enough: shown, as every slot of it from now on.
            late.remove(&handle);
            give_up(texture);
        }
        Hold::Again => {}
    }
    hold
}

/// [`hold`]'s choice for a texture not uploaded yet, from the slot's
/// waiting entry (record, since, texture) if it has one; and the entry's
/// start.
fn decide(entry: Option<(u64, u64, *const HostTexture)>, key: u64, texture: *const HostTexture, now: u64) -> (Hold, u64) {
    match entry {
        Some((k, since, t)) if k == key && std::ptr::eq(t, texture) => {
            if now.saturating_sub(since) >= MOST_MS { (Hold::No, since) } else { (Hold::Again, since) }
        }
        _ => (Hold::First, now),
    }
}

/// Nothing waits for `texture`'s upload any more.
fn give_up(texture: &HostTexture) {
    COUNTS[4].fetch_add(1, Relaxed);
    let upload = texture.upload.load(Relaxed);
    let _ = texture.upload.compare_exchange(upload, THERE, Relaxed, Relaxed);
}

/// An upload of `texture` was recorded in `list`: its submission's number
/// is noted when the list is submitted (the first list submitted wins).
pub(crate) fn recorded(list: &crate::host::HostList, texture: &Arc<HostTexture>) {
    if !on() {
        return;
    }
    let upload = texture.upload.load(Relaxed);
    if upload != NOT_YET && upload != RECORDED {
        return;
    }
    let _ = texture.upload.compare_exchange(NOT_YET, RECORDED, Relaxed, Relaxed);
    let weak = Arc::downgrade(texture);
    list.defer_host(move || {
        if let Some(texture) = weak.upgrade() {
            // Run after this submission was counted started: its number
            // or a later one.
            let number = crate::host::submissions().0.max(THERE);
            let _ = texture.upload.compare_exchange(RECORDED, number, Relaxed, Relaxed);
        }
    });
}

/// A texture whose contents the host writes, not an upload: nothing to
/// wait for.
pub(crate) fn not_uploaded(texture: &HostTexture) {
    let _ = texture.upload.compare_exchange(NOT_YET, THERE, Relaxed, Relaxed);
}

/// The waiting slots (title handles) whose texture's upload has run or
/// that waited [`MOST_MS`], taken out of the waiting: to be made again
/// before the next submission. Also the counter line, every 10 s.
pub(crate) fn due() -> Vec<u64> {
    if !on() {
        return Vec::new();
    }
    let finished = crate::host::submissions().1;
    let now = crate::host::now_ms();
    let mut out = Vec::new();
    let waiting = {
        let mut late = LATE.lock().unwrap_or_else(|e| e.into_inner());
        late.retain(|&handle, l| {
            let Some(texture) = l.texture.upgrade() else {
                // Let go meanwhile: the slot is made again as it is now.
                out.push(handle);
                return false;
            };
            if landed(texture.upload.load(Relaxed), finished) {
                COUNTS[3].fetch_add(1, Relaxed);
                out.push(handle);
                return false;
            }
            if now.saturating_sub(l.since) >= MOST_MS {
                give_up(&texture);
                out.push(handle);
                return false;
            }
            true
        });
        late.len()
    };
    static SAID: AtomicU64 = AtomicU64::new(0);
    let said = SAID.load(Relaxed);
    if now.saturating_sub(said) >= 10_000 && SAID.compare_exchange(said, now, Relaxed, Relaxed).is_ok() {
        let [deferred, stand_ins, kept, after, gave_up] = std::array::from_fn(|i| COUNTS[i].load(Relaxed));
        if deferred > 0 {
            tracing::info!(
                deferred,
                stand_ins,
                kept_old = kept,
                shown_after_upload = after,
                shown_after_3s = gave_up,
                waiting,
                "VICTORY_SHOW_AFTER: texture slot writes deferred until their upload ran, since the start"
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_upload_counts_once_its_submission_has_finished() {
        assert!(!landed(NOT_YET, 100));
        assert!(!landed(RECORDED, u64::MAX - 1));
        assert!(!landed(12, 11));
        assert!(landed(12, 12));
        assert!(landed(THERE, 1));
    }

    #[test]
    fn a_slot_waits_once_then_again_then_gives_up() {
        let t = 0x1000 as *const HostTexture;
        assert_eq!(decide(None, 5, t, 100), (Hold::First, 100));
        assert_eq!(decide(Some((5, 100, t)), 5, t, 200), (Hold::Again, 100));
        assert_eq!(decide(Some((5, 100, t)), 5, t, 100 + MOST_MS), (Hold::No, 100));
        // Another record, or another texture, starts again.
        assert_eq!(decide(Some((5, 100, t)), 6, t, 200), (Hold::First, 200));
        assert_eq!(decide(Some((5, 100, t)), 5, 0x2000 as *const HostTexture, 200), (Hold::First, 200));
    }
}
