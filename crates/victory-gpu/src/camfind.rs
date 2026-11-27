//! Multiplayer helper (`VICTORY_MULTI=1`, off by default): finds the main
//! camera in the title's constant buffers and sends where it is, ten times a
//! second, to `victory-voix` on this machine (UDP 127.0.0.1:47800, or
//! `VICTORY_MULTI_ADDR`) for proximity voice.
//!
//! The title's view constants are plain title memory, and the address a draw
//! names is ours to read too. So, every 100 ms, the constant buffers bound
//! during 15 ms are noted (an address and a size, nothing else is done on
//! the submitting threads; with the switch off, one relaxed load per bind),
//! then a thread of ours reads them (`ReadProcessMemory` on our own process:
//! a page gone since is a failed read, not a fault).
//!
//! What a view's buffer holds (seen in a run, 2026-11-27): the title renders
//! around the camera, so the view matrix has no translation; then the
//! projection (row vectors, reverse depth), the inverse rotation, and the
//! camera's world position as (x, y, z, 1) (Z up, metres). So the view's
//! buffer is the most bound one with a non-square perspective projection
//! (shadow views are orthographic, cube faces square); the camera's axes
//! are the rotation's just before the projection (its columns with row
//! vectors), its position the first (x, y, z, 1) after it.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Once;
use std::time::{Duration, Instant};


const SLOTS: usize = 4096;
static ADDRESSES: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
static SIZES: [AtomicU32; SLOTS] = [const { AtomicU32::new(0) }; SLOTS];
static COUNT: AtomicUsize = AtomicUsize::new(0);
static SAMPLING: AtomicBool = AtomicBool::new(false);
static CHECKED: AtomicBool = AtomicBool::new(false);
static START: Once = Once::new();

/// A constant buffer a draw binds (`size` 0: a root view, size unknown).
#[inline]
pub(crate) fn note(address: u64, size: u32) {
    if !SAMPLING.load(Relaxed) {
        if !CHECKED.load(Relaxed) {
            start();
        }
        return;
    }
    let i = COUNT.fetch_add(1, Relaxed);
    if i < SLOTS {
        ADDRESSES[i].store(address, Relaxed);
        SIZES[i].store(size, Relaxed);
    }
}

fn start() {
    START.call_once(|| {
        CHECKED.store(true, Relaxed);
        if std::env::var("VICTORY_MULTI").is_ok_and(|v| v != "0" && !v.is_empty()) {
            let to = std::env::var("VICTORY_MULTI_ADDR").unwrap_or_else(|_| "127.0.0.1:47800".into());
            let _ = std::thread::Builder::new()
                .name("victory multi camera".into())
                .spawn(move || run(&to));
        }
    });
}

/// Reads title memory without faulting on a page gone since it was noted.
fn read(address: u64, out: &mut [u8]) -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut done = 0;
        // SAFETY: the kernel checks the source; `out` is ours.
        unsafe {
            ReadProcessMemory(GetCurrentProcess(), address as *const _, out.as_mut_ptr().cast(), out.len(), &mut done) != 0
                && done == out.len()
        }
    }
    #[cfg(not(windows))]
    {
        let _ = (address, out);
        false
    }
}

fn run(to: &str) {
    let Ok(socket) = std::net::UdpSocket::bind("127.0.0.1:0") else { return };
    tracing::info!(target: "multi", "multiplayer: looking for the camera, sending to {to}");
    let mut bytes = vec![0u8; 4096];
    let mut last_log = Instant::now();
    let mut packet = Vec::with_capacity(40);
    let mut sent = 0u32;
    loop {
        COUNT.store(0, Relaxed);
        SAMPLING.store(true, Relaxed);
        std::thread::sleep(Duration::from_millis(15));
        SAMPLING.store(false, Relaxed);
        std::thread::sleep(Duration::from_millis(1));
        let n = COUNT.load(Relaxed).min(SLOTS);
        let mut by_address: std::collections::HashMap<u64, (u32, u32)> = std::collections::HashMap::new();
        for i in 0..n {
            let (a, s) = (ADDRESSES[i].load(Relaxed), SIZES[i].load(Relaxed));
            if a != 0 {
                let e = by_address.entry(a).or_insert((s, 0));
                e.0 = e.0.max(s);
                e.1 += 1;
            }
        }
        // address, size, binds; the most bound first.
        let mut seen: Vec<(u64, u32, u32)> = by_address.into_iter().map(|(a, (s, b))| (a, s, b)).collect();
        seen.sort_by_key(|e| std::cmp::Reverse(e.2));
        let mut found: Option<(Pose, u32)> = None;
        let mut views = 0;
        let mut dump: Option<(u64, u32, Vec<f32>)> = None;
        for &(address, size, binds) in seen.iter().take(96) {
            let len = if size == 0 { 1024 } else { (size as usize).clamp(64, 4096) } & !15;
            if !read(address, &mut bytes[..len]) {
                continue;
            }
            let floats: Vec<f32> = bytes[..len].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let Some(projection) = find_projection(&floats) else { continue };
            views += 1;
            if projection.square {
                continue;
            }
            if dump.is_none() {
                dump = Some((address, binds, floats.clone()));
            }
            if let Some(pose) = view_pose(&floats, &projection) {
                found = Some((pose, binds));
                break;
            }
        }
        if let Some((pose, _)) = found {
            packet.clear();
            packet.extend_from_slice(b"VXP1");
            for v in pose.pos.iter().chain(&pose.right).chain(&pose.up) {
                packet.extend_from_slice(&v.to_le_bytes());
            }
            let _ = socket.send_to(&packet, to);
            sent += 1;
        }
        if last_log.elapsed() > Duration::from_secs(5) {
            last_log = Instant::now();
            match found {
                Some((p, binds)) => tracing::info!(
                    target: "multi",
                    "multiplayer camera: buffers {}, views {views}, bound {binds}x, position [{:.2}, {:.2}, {:.2}], right [{:.2}, {:.2}, {:.2}], up [{:.2}, {:.2}, {:.2}], sent {sent}",
                    seen.len(), p.pos[0], p.pos[1], p.pos[2], p.right[0], p.right[1], p.right[2], p.up[0], p.up[1], p.up[2]
                ),
                None => tracing::info!(target: "multi", "multiplayer camera: buffers {}, views {views}, no camera found, sent {sent}", seen.len()),
            }
            // When none is found: what the view's buffer holds, to learn why.
            if found.is_none()
                && let Some((address, binds, floats)) = &dump
            {
                let text: Vec<String> = floats.iter().take(96).map(|x| format!("{x:.3}")).collect();
                tracing::info!(target: "multi", "multiplayer view buffer {address:#x} bound {binds}x, {} floats: {}", floats.len(), text.join(" "));
            }
        }
        std::thread::sleep(Duration::from_millis(84));
    }
}

/// Position, right and up of a camera, in world units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pose {
    pub pos: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
}

/// A projection found in a buffer: where (in floats), whether square (a cube
/// face's), whether written for row vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Projection {
    pub at: usize,
    pub square: bool,
    pub row: bool,
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The camera of a view's buffer: the rotation just before the projection
/// (not mirrored), the first (x, y, z, 1) after it.
pub(crate) fn view_pose(f: &[f32], p: &Projection) -> Option<Pose> {
    let k = p.at.checked_sub(16)?;
    let m = |r: usize, c: usize| f[k + r * 4 + c];
    // The view's axes in the world: with row vectors its columns.
    let axis = |a: usize| if p.row { [m(0, a), m(1, a), m(2, a)] } else { [m(a, 0), m(a, 1), m(a, 2)] };
    let axes = [axis(0), axis(1), axis(2)];
    let unit = axes.iter().all(|v| (dot(*v, *v) - 1.0).abs() < 0.02);
    let square = dot(axes[0], axes[1]).abs() < 0.02 && dot(axes[0], axes[2]).abs() < 0.02 && dot(axes[1], axes[2]).abs() < 0.02;
    // A mirrored view (a reflection's) turns the other way.
    let [a, b, c] = axes;
    let det = dot([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], c);
    if !unit || !square || det < 0.9 {
        return None;
    }
    let mut at = p.at + 16;
    while at + 4 <= f.len() {
        let v = [f[at], f[at + 1], f[at + 2]];
        if (f[at + 3] - 1.0).abs() < 1e-5 && v.iter().all(|x| x.is_finite() && x.abs() < 1e6) && dot(v, v) >= 1.0 {
            return Some(Pose { pos: v, right: axes[0], up: axes[1] });
        }
        at += 4;
    }
    None
}

/// The first perspective projection in a buffer's floats (row or column
/// vectors, any depth convention, a small jitter allowed), and whether it is
/// square (a cube face's).
pub(crate) fn find_projection(f: &[f32]) -> Option<Projection> {
    let zero = |x: f32| x.abs() < 1e-6;
    let scale = |x: f32| x > 0.1 && x < 20.0;
    let mut k = 0;
    while k + 16 <= f.len() {
        let m = |r: usize, c: usize| f[k + r * 4 + c];
        let common = scale(m(0, 0)) && scale(m(1, 1)) && zero(m(0, 1)) && zero(m(1, 0)) && zero(m(3, 3));
        if common {
            // Row vectors: w in (2, 3), depth bias in (3, 2).
            let row = zero(m(0, 2)) && zero(m(0, 3)) && zero(m(1, 2)) && zero(m(1, 3))
                && m(2, 0).abs() < 0.1 && m(2, 1).abs() < 0.1
                && (m(2, 3).abs() - 1.0).abs() < 1e-3
                && zero(m(3, 0)) && zero(m(3, 1)) && !zero(m(3, 2));
            // Column vectors: the transpose.
            let column = zero(m(2, 0)) && zero(m(3, 0)) && zero(m(2, 1)) && zero(m(3, 1))
                && m(0, 2).abs() < 0.1 && m(1, 2).abs() < 0.1
                && (m(3, 2).abs() - 1.0).abs() < 1e-3
                && zero(m(0, 3)) && zero(m(1, 3)) && !zero(m(2, 3));
            if row || column {
                return Some(Projection { at: k, square: (m(0, 0) - m(1, 1)).abs() < 1e-3, row });
            }
        }
        k += 4;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projections_are_found() {
        // 16:9, 60 degrees, reverse depth, row vectors, with jitter.
        let (sy, sx) = (1.732, 1.732 / 1.777);
        let row = [sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0003, -0.0002, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0];
        let mut buffer = vec![0.5; 8];
        buffer.extend(row);
        assert_eq!(find_projection(&buffer), Some(Projection { at: 8, square: false, row: true }));
        let column: Vec<f32> = (0..16).map(|i| row[(i % 4) * 4 + i / 4]).collect();
        assert_eq!(find_projection(&column), Some(Projection { at: 0, square: false, row: false }));
        let cube = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -0.1, 0.0];
        assert!(find_projection(&cube).is_some_and(|p| p.square));
        let ortho = [0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.5, 1.0];
        assert_eq!(find_projection(&ortho), None);
    }

    /// The first 60 floats of a view's buffer from a run (2026-11-27 13:4x).
    #[test]
    fn a_buffer_from_the_title() {
        let f: Vec<f32> = "0.286 -0.108 0.952 0.000 0.958 0.037 -0.283 0.000 -0.005 0.993 0.115 0.000 0.000 0.000 0.000 1.000 \
            2.344 0.000 0.000 0.000 0.000 4.167 0.000 0.000 0.000 0.000 0.000 -1.000 0.000 0.000 0.150 0.000 \
            0.286 0.958 -0.005 0.000 -0.108 0.037 0.993 0.000 0.952 -0.283 0.115 0.000 0.000 0.000 0.000 1.000 \
            1266.388 -301.635 -1.572 1.000 -18.388 13.635 1.572 0.000 0.284 -0.108 0.953 0.000"
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        let p = find_projection(&f).unwrap();
        assert_eq!(p, Projection { at: 16, square: false, row: true });
        let pose = view_pose(&f, &p).unwrap();
        assert_eq!(pose.pos, [1266.388, -301.635, -1.572]);
        assert!((pose.right[1] - 0.958).abs() < 1e-3 && pose.right[2].abs() < 0.01);
        assert!(pose.up[2] > 0.99);
    }
}
