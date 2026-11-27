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
//! a page gone since is a failed read, not a fault) and looks for a rigid
//! transform: three orthonormal rows of a 4x4 or 3x4 matrix with a
//! translation. The buffer bound most often that holds one is the view's.
//!
//! Such a matrix is camera-to-world (its translation is the camera) or
//! world-to-camera (the camera is minus the rotated translation). Which one
//! is learnt while the title runs: when the camera turns, the camera's
//! position barely moves (it orbits the player by a few metres) while the
//! other reading swings by the distance to the world's origin times the
//! angle. Votes are kept per place (offset in the buffer, layout); nothing is
//! sent before one reading leads by three votes.

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
    let mut finder = Finder::default();
    let mut bytes = vec![0u8; 4096];
    let mut last_log = Instant::now();
    let mut packet = Vec::with_capacity(40);
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
        let mut found = None;
        for &(address, size, binds) in seen.iter().take(64) {
            let len = if size == 0 { 1024 } else { (size as usize).clamp(64, 4096) } & !15;
            if !read(address, &mut bytes[..len]) {
                continue;
            }
            let floats: Vec<f32> = bytes[..len].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            if let Some(m) = find_rigid(&floats) {
                found = Some((m, binds));
                break;
            }
        }
        if let Some((m, _)) = found
            && let Some(pose) = finder.sample(&m)
        {
            packet.clear();
            packet.extend_from_slice(b"VXP1");
            for v in pose.pos.iter().chain(&pose.right).chain(&pose.up) {
                packet.extend_from_slice(&v.to_le_bytes());
            }
            let _ = socket.send_to(&packet, to);
        }
        if last_log.elapsed() > Duration::from_secs(5) {
            last_log = Instant::now();
            match found {
                Some((m, binds)) => tracing::info!(
                    target: "multi",
                    "multiplayer camera: buffers {}, best bound {binds}x, matrix at +{} ({:?}), as camera {:?}, as view {:?}, votes {:?}, sending {}",
                    seen.len(), m.offset * 4, m.layout, m.position(false), m.position(true), finder.votes(&m), finder.decided(&m).is_some()
                ),
                None => tracing::info!(target: "multi", "multiplayer camera: buffers {}, no camera matrix found", seen.len()),
            }
        }
        std::thread::sleep(Duration::from_millis(84));
    }
}

/// Where the translation of a rigid matrix lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Layout {
    /// `[R | t]`, translation in the fourth column (column vectors; HLSL
    /// float3x4 / transposed float4x4).
    Column,
    /// `[R ; t]`, translation in the fourth row (row vectors).
    Row,
}

/// A rigid transform found in a buffer: rotation rows, translation, where.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rigid {
    pub r: [[f32; 3]; 3],
    pub t: [f32; 3],
    pub offset: usize,
    pub layout: Layout,
}

/// Position, right and up of a camera, in world units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pose {
    pub pos: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
}

impl Rigid {
    /// The camera's axes in the world (right, up, forward) and its position,
    /// reading the matrix as world-to-camera (`inverse`) or camera-to-world.
    pub fn pose(&self, inverse: bool) -> ([[f32; 3]; 3], [f32; 3]) {
        let r = self.r;
        let transposed = [[r[0][0], r[1][0], r[2][0]], [r[0][1], r[1][1], r[2][1]], [r[0][2], r[1][2], r[2][2]]];
        // Camera-to-world with column vectors: the axes are R's columns.
        let axes = match (self.layout == Layout::Column, inverse) {
            (true, false) | (false, true) => transposed,
            (false, false) | (true, true) => r,
        };
        let pos = if inverse {
            let t = self.t;
            [0, 1, 2].map(|j| -(t[0] * axes[0][j] + t[1] * axes[1][j] + t[2] * axes[2][j]))
        } else {
            self.t
        };
        (axes, pos)
    }

    pub fn position(&self, inverse: bool) -> [f32; 3] {
        self.pose(inverse).1
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The first rigid transform in a buffer's floats that is not trivial (an
/// identity, a swap of axes, a matrix at the origin).
pub(crate) fn find_rigid(f: &[f32]) -> Option<Rigid> {
    let near = |a: f32, b: f32, e: f32| (a - b).abs() <= e;
    let mut k = 0;
    while k + 12 <= f.len() {
        let row = |i: usize| [f[k + i * 4], f[k + i * 4 + 1], f[k + i * 4 + 2]];
        let r = [row(0), row(1), row(2)];
        let unit = r.iter().all(|v| near(dot(*v, *v), 1.0, 0.02));
        let square = unit
            && near(dot(r[0], r[1]), 0.0, 0.02)
            && near(dot(r[0], r[2]), 0.0, 0.02)
            && near(dot(r[1], r[2]), 0.0, 0.02);
        // Axis-aligned: identities and swaps are everywhere and say nothing.
        let aligned = r.iter().flatten().filter(|x| x.abs() > 0.999).count() >= 3;
        if square && !aligned {
            let column = [f[k + 3], f[k + 7], f[k + 11]];
            let fourth = (k + 16 <= f.len()).then(|| [f[k + 12], f[k + 13], f[k + 14], f[k + 15]]);
            let found = if column.iter().all(|x| x.abs() < 1e-6) {
                // Row vectors: translation in the fourth row, w 1.
                fourth.filter(|w| near(w[3], 1.0, 1e-4)).map(|w| ([w[0], w[1], w[2]], Layout::Row))
            } else {
                // Column vectors: the fourth row, if any, is (0, 0, 0, 1)
                // or belongs to something else (a 3x4 array).
                let bottom = fourth.is_some_and(|w| w[..3].iter().all(|x| x.abs() < 1e-6) && near(w[3], 1.0, 1e-4));
                let packed = fourth.is_none_or(|w| !(w.iter().all(|x| x.abs() < 1e-6)));
                (bottom || packed).then_some((column, Layout::Column))
            };
            if let Some((t, layout)) = found
                && t.iter().all(|x| x.is_finite() && x.abs() < 1e6)
                && dot(t, t) > 1e-4
            {
                return Some(Rigid { r, t, offset: k, layout });
            }
        }
        k += 4;
    }
    None
}

/// Learns which reading of the view's matrix is the camera.
#[derive(Default)]
pub(crate) struct Finder {
    last: Option<Rigid>,
    /// Per place (offset, layout): votes for camera-to-world, for world-to-camera.
    votes: Vec<((usize, Layout), [u32; 2])>,
}

impl Finder {
    pub fn votes(&self, m: &Rigid) -> [u32; 2] {
        self.votes.iter().find(|v| v.0 == (m.offset, m.layout)).map_or([0, 0], |v| v.1)
    }

    /// The reading decided for this place, if any (true: world-to-camera).
    pub fn decided(&self, m: &Rigid) -> Option<bool> {
        let [direct, inverse] = self.votes(m);
        if direct >= inverse + 3 {
            Some(false)
        } else if inverse >= direct + 3 {
            Some(true)
        } else {
            None
        }
    }

    /// One look at the view's matrix; the camera's pose once decided.
    pub fn sample(&mut self, m: &Rigid) -> Option<Pose> {
        if let Some(last) = self.last.filter(|l| (l.offset, l.layout) == (m.offset, m.layout)) {
            // How far the rotation turned: the smallest agreement of an axis.
            let turned = (0..3).map(|i| dot(last.r[i], m.r[i])).fold(1.0f32, f32::min);
            if turned < 0.9986 {
                // > ~3 degrees.
                let moved = |inverse: bool| {
                    let (a, b) = (last.position(inverse), m.position(inverse));
                    dot([a[0] - b[0], a[1] - b[1], a[2] - b[2]], [a[0] - b[0], a[1] - b[1], a[2] - b[2]])
                };
                let (direct, inverse) = (moved(false), moved(true));
                // Only clear verdicts: one reading moved four times less.
                let vote = if direct * 16.0 < inverse {
                    Some(0)
                } else if inverse * 16.0 < direct {
                    Some(1)
                } else {
                    None
                };
                if let Some(v) = vote {
                    let place = (m.offset, m.layout);
                    match self.votes.iter_mut().find(|e| e.0 == place) {
                        Some(e) => e.1[v] += 1,
                        None => {
                            let mut n = [0, 0];
                            n[v] = 1;
                            self.votes.push((place, n));
                        }
                    }
                }
            }
        }
        self.last = Some(*m);
        let inverse = self.decided(m)?;
        let (axes, pos) = m.pose(inverse);
        Some(Pose { pos, right: axes[0], up: axes[1] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera at `pos` turned by `yaw` about Z (Z up), written the four
    /// ways a title may.
    fn matrices(pos: [f32; 3], yaw: f32) -> Vec<(Vec<f32>, bool)> {
        let (s, c) = yaw.sin_cos();
        // Camera axes in the world: right, up (Z), forward.
        let right = [c, s, 0.0];
        let up = [0.0, 0.0, 1.0];
        let fwd = [-s, c, 0.0];
        let axes = [right, up, fwd];
        let mut out = Vec::new();
        // Camera-to-world, column vectors: columns are the axes.
        let cw_col: Vec<f32> = (0..3)
            .flat_map(|i| [axes[0][i], axes[1][i], axes[2][i], pos[i]])
            .chain([0.0, 0.0, 0.0, 1.0])
            .collect();
        out.push((cw_col, false));
        // Camera-to-world, row vectors: rows are the axes, translation last.
        let cw_row: Vec<f32> = axes.iter().flat_map(|a| [a[0], a[1], a[2], 0.0]).chain([pos[0], pos[1], pos[2], 1.0]).collect();
        out.push((cw_row, false));
        // World-to-camera, column vectors: rows are the axes, t = -R p.
        let t: Vec<f32> = axes.iter().map(|a| -dot(*a, pos)).collect();
        let wc_col: Vec<f32> = (0..3).flat_map(|i| [axes[i][0], axes[i][1], axes[i][2], t[i]]).chain([0.0, 0.0, 0.0, 1.0]).collect();
        out.push((wc_col, true));
        // World-to-camera, row vectors: columns are the axes.
        let wc_row: Vec<f32> = (0..3).flat_map(|i| [axes[0][i], axes[1][i], axes[2][i], 0.0]).chain([t[0], t[1], t[2], 1.0]).collect();
        out.push((wc_row, true));
        out
    }

    #[test]
    fn every_layout_gives_the_camera() {
        let pos = [1234.5, -820.25, 37.0];
        for (i, (m, inverse)) in matrices(pos, 0.7).into_iter().enumerate() {
            let mut buffer = vec![0.0; 8];
            buffer.extend(m);
            buffer.extend([0.5; 8]);
            let r = find_rigid(&buffer).unwrap_or_else(|| panic!("layout {i}"));
            assert_eq!(r.offset, 8);
            let (axes, p) = r.pose(inverse);
            for j in 0..3 {
                assert!((p[j] - pos[j]).abs() < 0.05, "layout {i}: {p:?}");
            }
            assert!((axes[1][2] - 1.0).abs() < 1e-4, "layout {i}: up {:?}", axes[1]);
            assert!((axes[0][0] - 0.7f32.cos()).abs() < 1e-4, "layout {i}: right {:?}", axes[0]);
        }
        assert!(find_rigid(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 5.0, 5.0, 1.0]).is_none());
    }

    #[test]
    fn turning_decides_the_reading() {
        for which in 0..4 {
            let mut finder = Finder::default();
            let mut last = None;
            for step in 0..8 {
                // Orbiting a player at (500, 300, 20) by 4 m while turning.
                let yaw = 0.1 + step as f32 * 0.2;
                let pos = [500.0 + 4.0 * yaw.sin(), 300.0 - 4.0 * yaw.cos(), 22.0];
                let (m, _) = matrices(pos, yaw).swap_remove(which);
                let r = find_rigid(&m).unwrap();
                last = finder.sample(&r).map(|p| (p, pos));
            }
            let (pose, pos) = last.unwrap_or_else(|| panic!("layout {which} undecided"));
            for j in 0..3 {
                assert!((pose.pos[j] - pos[j]).abs() < 0.05, "layout {which}: {:?} vs {pos:?}", pose.pos);
            }
        }
    }
}
