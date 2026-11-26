//! `VICTORY_HOT_SHADERS`: host forms of a compute program that tell the
//! host's compiler what the console's code says by its registers.
//!
//! On the console a scalar register holds one value for the whole wave, and
//! a branch on SCC sends the whole wave one way. The lowering gives every
//! register dword one `i32` per lane, so the host compiler sees a scalar
//! load's dword (a raw buffer load: a lane's own memory read, as far as it
//! knows) and everything computed from it as values that may differ per
//! lane. It then keeps them in VGPRs, and every uniform branch on them in
//! a divergent form: EXEC saved and restored around each side, and the
//! joins of many branches turned into ladders of per-lane compares. The
//! title's ScreenProbes program (`39f0e9aeb156a61c`, ~10,000 uniform
//! branches) came out of the AMD compiler at 256 VGPRs with 5 KiB of
//! scratch per lane, ~25,000 scratch reads and ~25,000 VGPR swaps.
//!
//! A set of bits (0 or unset: none, every program as before):
//! - 1: a compute program's scalar loads from buffers (`s_buffer_load_dword*`
//!   and `s_load_dword*` from title memory) give the first lane's dword
//!   (`WaveReadLaneFirst`), so their value and what is computed from it
//!   are uniform to the host compiler (scalar registers and scalar ALU, as
//!   on the console). Every lane that runs the load reads the same address
//!   (scalar registers), so every lane reads the same dword.
//! - 2: a compute program's uniform branches (on SCC: a condition over
//!   scalar values only) branch on the first lane's condition. Every lane
//!   that reaches the branch computed the same condition from the same
//!   scalar values; the host compiler is told so and branches on SCC.
//! - 4: a scalar load of several dwords from a constant buffer at a
//!   computed offset (`s_buffer_load_dwordx4/x8/x16`, a record picked by
//!   an index: the volumetric gathers' lights) reads the 16-byte rows the
//!   dwords lie in, one read each, instead of one read per dword: 5 reads
//!   for 16 dwords, not 16. Each read was a vector load the next one
//!   waited for (`s_waitcnt vmcnt(0)`); the console makes one scalar load.
//!   The same dwords, whatever the offset's alignment; any program.
//! - 8: compute and mesh programs get every clean-up pass of
//!   `bitcode::opt` (dead code, constants, one instruction per value,
//!   branch facts, loads) whatever `VICTORY_SHADER_OPT` says (the Jouer
//!   line runs with it at 0). The passes are exact. ScreenProbes
//!   (`39f0e9ae`, RGA gfx1102): scratch per lane 5,376 -> 3,584 bytes,
//!   scratch reads 12,430 -> 6,977, instructions 81k -> 72k.
//!
//! Part of the translation: the shader cache keys on it (`victory-gpu`'s
//! does).

use super::Lower;
use super::super::bitcode::{Attr, Bin, Cast, V, pred};
use super::super::shader::flags;
use crate::structure::{Atom, Bool};

/// Scalar loads from buffers give a wave-uniform value.
pub const UNIFORM_LOADS: u32 = 1;
/// Uniform branches branch on a wave-uniform condition.
pub const UNIFORM_BRANCHES: u32 = 2;
/// Constant-buffer dwords at a computed offset read by rows.
pub const GROUPED_ROWS: u32 = 4;
/// Every clean-up pass for compute and mesh programs.
pub const CLEAN_COMPUTE: u32 = 8;
/// Every bit.
pub const ALL: u32 = UNIFORM_LOADS | UNIFORM_BRANCHES | GROUPED_ROWS | CLEAN_COMPUTE;

/// The clean-up passes bit [`CLEAN_COMPUTE`] adds for a program with
/// thread groups (`compute`: compute or mesh): every module pass, or none.
pub(crate) fn module_passes(compute: bool) -> u32 {
    if compute && hot_shaders() & CLEAN_COMPUTE != 0 { super::super::bitcode::opt::MODULE } else { 0 }
}

/// `VICTORY_HOT_SHADERS`'s bits (see the module).
pub fn hot_shaders() -> u32 {
    static BITS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BITS.get_or_init(|| parse(std::env::var("VICTORY_HOT_SHADERS").ok().as_deref()))
}

/// The switch's value; none when unset or unreadable. Pure, for testing.
pub(super) fn parse(value: Option<&str>) -> u32 {
    value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        })
        .map_or(0, |n| n & ALL)
}

impl Lower<'_> {
    /// Whether `bit` of [`hot_shaders`] applies to this program: compute
    /// programs only (no helper lanes, every lane a thread of its own).
    fn hot(&self, bit: u32) -> bool {
        hot_shaders() & bit != 0 && self.s.threads.is_some()
    }

    /// The `count` dwords of constant buffer `h` from byte `a` on, read by
    /// 16-byte rows (bit [`GROUPED_ROWS`]); `None` without the bit or for
    /// fewer than four dwords (then dword by dword, `cbuffer_dword`).
    pub(super) fn cbuffer_rows(&mut self, h: V, a: V, count: u32) -> Option<Vec<V>> {
        if hot_shaders() & GROUPED_ROWS == 0 || count < 4 {
            return None;
        }
        let c = self.cur;
        let (two, three, four) = (self.s.m.i32(2), self.s.m.i32(3), self.s.m.i32(4));
        let row = self.s.m.bin(c, Bin::LShr, a, four);
        let first = self.s.m.bin(c, Bin::LShr, a, two);
        let first = self.s.m.bin(c, Bin::And, first, three);
        // Dword k is component (first + k) & 3 of row (a >> 4) + (first + k) >> 2.
        let mut flat = Vec::new();
        for j in 0..(count + 2) / 4 + 1 {
            let jc = self.s.m.i32(j);
            let ri = self.s.m.bin(c, Bin::Add, row, jc);
            let r = self.s.op(c, "cbufferLoadLegacy.i32", Attr::ReadOnly, self.s.t.cbuf_ret, 59, &[h, ri]);
            for e in 0..4 {
                flat.push(self.s.m.extract(c, r, e));
            }
        }
        let mut out = Vec::with_capacity(count as usize);
        for k in 0..count as usize {
            let mut x = flat[k];
            for s in 1..4 {
                let sc = self.s.m.i32(s as u32);
                let is = self.s.m.cmp(c, pred::EQ, first, sc);
                x = self.s.m.select(c, is, flat[k + s], x);
            }
            out.push(x);
        }
        Some(out)
    }

    /// A dword a scalar load read from a buffer, as the first lane read it
    /// (bit [`UNIFORM_LOADS`]).
    pub(super) fn scalar_loaded(&mut self, x: V) -> V {
        if !self.hot(UNIFORM_LOADS) || matches!(x, V::Const(_)) {
            return x;
        }
        self.s.flags |= flags::WAVE_OPS;
        let i32t = self.s.t.i32;
        self.s.op(self.cur, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x])
    }

    /// A branch's condition `cond` (the lane's `when`, with the loop
    /// limit's test): where `when` is over scalar values only (a uniform
    /// branch), as the first lane computed it (bit [`UNIFORM_BRANCHES`]).
    pub(super) fn uniform_branch(&mut self, when: Bool, cond: V) -> V {
        let uniform = self.x.st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_)));
        if !uniform || !self.hot(UNIFORM_BRANCHES) || matches!(cond, V::Const(_)) {
            return cond;
        }
        self.s.flags |= flags::WAVE_OPS;
        let (c, i32t) = (self.cur, self.s.t.i32);
        let n = self.s.m.cast(c, Cast::ZExt, cond, i32t);
        let first = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[n]);
        let zero = self.s.m.i32(0);
        self.s.m.cmp(c, pred::NE, first, zero)
    }
}

#[cfg(test)]
mod tests {
    use super::{ALL, parse};

    #[test]
    fn reads_the_switch() {
        assert_eq!(parse(None), 0);
        assert_eq!(parse(Some("x")), 0);
        assert_eq!(parse(Some(" 3 ")), 3);
        assert_eq!(parse(Some("0x2")), 2);
        assert_eq!(parse(Some("255")), ALL);
    }
}
