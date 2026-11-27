//! A menu drawn over the picture: lines of text the input side hands over
//! (the key menu, F1), painted by the presenter's panel when they change.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

/// How a line looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Title,
    Normal,
    /// The line the cursor is on.
    Selected,
    /// Help and messages, in grey.
    Note,
}

pub type Lines = Vec<(String, Tone)>;

static LINES: Mutex<Option<Lines>> = Mutex::new(None);
static VERSION: AtomicU64 = AtomicU64::new(0);

/// Shows these lines (`None`: no menu).
pub fn show(lines: Option<Lines>) {
    *LINES.lock().unwrap_or_else(PoisonError::into_inner) = lines;
    VERSION.fetch_add(1, Ordering::Release);
}

/// Goes up each time the lines change.
pub fn version() -> u64 {
    VERSION.load(Ordering::Acquire)
}

/// The lines shown now.
pub fn lines() -> Option<Lines> {
    LINES.lock().unwrap_or_else(PoisonError::into_inner).clone()
}
