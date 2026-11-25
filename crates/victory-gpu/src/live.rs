//! Switches changed while the title runs (`VICTORY_LIVE=<file>`).
//!
//! With `VICTORY_LIVE` set, a thread reads that file every second: lines
//! `NAME=value` (`VICTORY_` in front or not; `#` starts a comment). A
//! change is applied at the next frame ([`frame`]): from then on [`var`]
//! gives the file's value of a name it holds and the environment's of the
//! others, each applied change is logged (`live switch NAME=value at frame
//! N`) and counts as a new configuration, which the frame-rate line names
//! (`live=<n>`) so a log can be cut into its segments. A line taken out
//! of the file gives the name back to the environment.
//! Without `VICTORY_LIVE`, [`var`] is `std::env::var`: as before.
//!
//! Only switches read through [`Switch`] or [`var`] at every use take a
//! change; the others are read once at the start.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Bumped when a change is applied: what [`Switch`] compares against.
static GENERATION: AtomicU64 = AtomicU64::new(0);
/// Configurations applied so far (0: the start's).
static CONFIG: AtomicU64 = AtomicU64::new(0);
/// The values in force, by full name (`VICTORY_...`).
static ACTIVE: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
/// What the file says now, waiting for the next frame.
static PENDING: Mutex<Option<BTreeMap<String, String>>> = Mutex::new(None);
static HAS_PENDING: AtomicBool = AtomicBool::new(false);

/// The file `VICTORY_LIVE` names, if any (read once).
fn path() -> Option<&'static std::path::Path> {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        let p = std::env::var_os("VICTORY_LIVE").filter(|p| !p.is_empty())?;
        Some(std::path::PathBuf::from(p))
    })
    .as_deref()
}

/// Whether `VICTORY_LIVE` is set.
pub(crate) fn on() -> bool {
    path().is_some()
}

/// The configuration in force (0 at the start; one more per change
/// applied).
pub(crate) fn config() -> u64 {
    CONFIG.load(Ordering::Relaxed)
}

/// The value of switch `name` (full name, `VICTORY_...`): the file's if it
/// holds it, else the environment's.
pub(crate) fn var(name: &str) -> Option<String> {
    if on()
        && let Some(v) = ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).get(name)
    {
        return Some(v.clone());
    }
    std::env::var(name).ok()
}

/// A switch read at every use for almost nothing: its value (a number,
/// parsed from [`var`]) is parsed again only after a change is applied.
pub(crate) struct Switch {
    generation: AtomicU64,
    value: AtomicU64,
}

impl Switch {
    pub(crate) const fn new() -> Self {
        Switch {
            generation: AtomicU64::new(u64::MAX),
            value: AtomicU64::new(0),
        }
    }

    /// The value, from `parse` of the switch's text (`first`: the first
    /// parse, for a line telling what the switch does).
    pub(crate) fn get(&self, name: &str, parse: impl FnOnce(Option<&str>, bool) -> u64) -> u64 {
        let now = GENERATION.load(Ordering::Acquire);
        let seen = self.generation.load(Ordering::Acquire);
        if seen == now {
            return self.value.load(Ordering::Relaxed);
        }
        let value = parse(var(name).as_deref(), seen == u64::MAX);
        self.value.store(value, Ordering::Relaxed);
        self.generation.store(now, Ordering::Release);
        value
    }
}

/// A switch holding a list of words (comma separated), parsed again only
/// after a change is applied.
pub(crate) struct Words {
    generation: AtomicU64,
    words: std::sync::RwLock<Vec<String>>,
}

impl Words {
    pub(crate) const fn new() -> Self {
        Words {
            generation: AtomicU64::new(u64::MAX),
            words: std::sync::RwLock::new(Vec::new()),
        }
    }

    /// Whether any word (lower case) is in `text`.
    pub(crate) fn any_in(&self, name: &str, text: &str) -> bool {
        let now = GENERATION.load(Ordering::Acquire);
        if self.generation.load(Ordering::Acquire) != now {
            let words: Vec<String> = var(name)
                .unwrap_or_default()
                .split(',')
                .map(|w| w.trim().to_lowercase())
                .filter(|w| !w.is_empty())
                .collect();
            *self.words.write().unwrap_or_else(|e| e.into_inner()) = words;
            self.generation.store(now, Ordering::Release);
        }
        let words = self.words.read().unwrap_or_else(|e| e.into_inner());
        !words.is_empty() && words.iter().any(|w| text.contains(w.as_str()))
    }
}

/// `NAME=value` lines; names get `VICTORY_` in front if they lack it.
fn parse(text: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        let name = name.trim().to_uppercase();
        if name.is_empty() {
            continue;
        }
        let name = if name.starts_with("VICTORY_") {
            name
        } else {
            format!("VICTORY_{name}")
        };
        map.insert(name, value.trim().to_string());
    }
    map
}

/// Starts the thread that reads the file, once (with `VICTORY_LIVE`).
fn start() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    let Some(path) = path() else {
        return;
    };
    STARTED.call_once(|| {
        tracing::info!(file = %path.display(), "VICTORY_LIVE: switches read from this file every second, applied at the next frame");
        let spawned = std::thread::Builder::new().name("victory-live".into()).spawn(move || {
            let mut last: Option<BTreeMap<String, String>> = None;
            loop {
                // An unreadable file (being saved) keeps what was there.
                if let Ok(text) = std::fs::read_to_string(path) {
                    let now = parse(&text);
                    if last.as_ref() != Some(&now) {
                        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some(now.clone());
                        HAS_PENDING.store(true, Ordering::Release);
                        last = Some(now);
                    }
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "VICTORY_LIVE: no thread to read the file");
        }
    });
}

/// Frame `frame` begins: a change the file brought is applied now.
pub(crate) fn frame(frame: u64) {
    if !on() {
        return;
    }
    start();
    if !HAS_PENDING.swap(false, Ordering::Acquire) {
        return;
    }
    let Some(new) = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
    if *active == new {
        return;
    }
    let config = CONFIG.fetch_add(1, Ordering::Relaxed) + 1;
    for (name, value) in &new {
        if active.get(name) != Some(value) {
            tracing::info!(config, "live switch {name}={value} at frame {frame}");
        }
    }
    for name in active.keys().filter(|n| !new.contains_key(*n)) {
        let env = std::env::var(name).unwrap_or_default();
        tracing::info!(
            config,
            "live switch {name}={env} at frame {frame} (back to the environment)"
        );
    }
    *active = new;
    drop(active);
    GENERATION.fetch_add(1, Ordering::AcqRel);
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn lines_name_switches() {
        let map = parse("# A/B\nclamp_batch=1\nVICTORY_HOIST_CLAMPS = 8 # mode\n\nnot a line\n");
        assert_eq!(
            map.get("VICTORY_CLAMP_BATCH").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            map.get("VICTORY_HOIST_CLAMPS").map(String::as_str),
            Some("8")
        );
        assert_eq!(map.len(), 2);
    }
}
