//! The VIctory menu drawn over the picture (F1): tabs of rows that the input
//! side hands over (`victory-hle`'s gameinput), painted by the presenter's
//! panel only when they change. Closed, it costs a version check a frame.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

/// What a row is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A section title inside a tab.
    Header,
    /// A setting: label on the left, value on the right.
    Setting,
    /// Something done on Enter.
    Button,
    /// A line to read (status, help).
    Text,
}

/// One row of a tab.
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    pub label: String,
    pub value: String,
    pub kind: Kind,
}

impl Item {
    pub fn header(label: impl Into<String>) -> Item {
        Item {
            label: label.into(),
            value: String::new(),
            kind: Kind::Header,
        }
    }

    pub fn setting(label: impl Into<String>, value: impl Into<String>) -> Item {
        Item {
            label: label.into(),
            value: value.into(),
            kind: Kind::Setting,
        }
    }

    pub fn button(label: impl Into<String>) -> Item {
        Item {
            label: label.into(),
            value: String::new(),
            kind: Kind::Button,
        }
    }

    pub fn text(label: impl Into<String>, value: impl Into<String>) -> Item {
        Item {
            label: label.into(),
            value: value.into(),
            kind: Kind::Text,
        }
    }
}

/// What the menu shows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Page {
    pub tabs: Vec<String>,
    /// The tab shown.
    pub tab: usize,
    pub items: Vec<Item>,
    /// The row the cursor is on (an index in `items`).
    pub selected: usize,
    /// The keys of this tab, at the bottom.
    pub help: String,
    /// The last thing that happened (a change, a warning).
    pub note: String,
}

static PAGE: Mutex<Option<Page>> = Mutex::new(None);
static VERSION: AtomicU64 = AtomicU64::new(0);

/// Shows this page (`None`: no menu).
pub fn show(page: Option<Page>) {
    *PAGE.lock().unwrap_or_else(PoisonError::into_inner) = page;
    VERSION.fetch_add(1, Ordering::Release);
}

/// Goes up each time the page changes.
pub fn version() -> u64 {
    VERSION.load(Ordering::Acquire)
}

/// The page shown now.
pub fn page() -> Option<Page> {
    PAGE.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// The numbers box (F3) asked on (`Some(true)`) or off by the menu.
static STATS_ASKED: AtomicU8 = AtomicU8::new(0);
/// The numbers box is shown now (as the panel last saw it).
static STATS_SHOWN: AtomicBool = AtomicBool::new(false);

/// Asks the panel to show or hide the numbers box (as F3 does).
pub fn ask_stats(on: bool) {
    STATS_ASKED.store(if on { 1 } else { 2 }, Ordering::Release);
}

/// What the menu asked since the last call (the panel's side).
pub fn take_stats_ask() -> Option<bool> {
    match STATS_ASKED.swap(0, Ordering::AcqRel) {
        1 => Some(true),
        2 => Some(false),
        _ => None,
    }
}

/// Whether the numbers box is shown.
pub fn stats_shown() -> bool {
    STATS_SHOWN.load(Ordering::Acquire)
}

/// The panel's side: the numbers box is shown or not.
pub fn set_stats_shown(on: bool) {
    STATS_SHOWN.store(on, Ordering::Release);
}
