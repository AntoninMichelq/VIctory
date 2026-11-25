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
//!
//! Part of the translation: the shader cache keys on it (`victory-gpu`'s
//! does).

use super::Lower;
use super::super::bitcode::{Attr, Cast, V, pred};
use super::super::shader::flags;
use crate::structure::{Atom, Bool};

/// Scalar loads from buffers give a wave-uniform value.
pub const UNIFORM_LOADS: u32 = 1;
/// Uniform branches branch on a wave-uniform condition.
pub const UNIFORM_BRANCHES: u32 = 2;
/// Every bit.
pub const ALL: u32 = UNIFORM_LOADS | UNIFORM_BRANCHES;

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
