//! Where a program's constant table starts (`Lower::code_dword`).
//!
//! The table is the program's code from the first byte a load names on: a
//! constant array in the module, indexed by the byte a load computes. The
//! console compiler puts a program's constants after its instructions,
//! and a load at a computed offset reads them, but the lowest offset a
//! load names can be one the compiler wrote relative to an earlier point
//! (`s_getpc_b64` plus an offset that a computed term then carries past
//! the instructions): the table then held all the instructions after that
//! point too, as constants the host compiler must keep.
//!
//! `VICTORY_CODE_TABLE_AFTER=1` (port of upstream #2143 / fxceless#1006,
//! justr4mzi `d2efdbb3`): the table starts where the instructions end,
//! unless a load at a fixed offset reads below that (then at the lowest
//! offset, as before). Unset: the lowest offset, as before. Part of the
//! translation: the shader cache keys on it (`victory-gpu`'s does).

use crate::ir::Program;
use crate::resource::Address;

/// Whether the switch is on.
fn after() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CODE_TABLE_AFTER").is_some_and(|v| v == "1"))
}

/// The table's first byte for a program whose loads from its code name
/// `addresses`; `None` without any.
pub(super) fn first(p: &Program, addresses: &[Address]) -> Option<u32> {
    start(p, addresses, after())
}

/// [`first`] with the switch given. Pure, for testing.
fn start(p: &Program, addresses: &[Address], after: bool) -> Option<u32> {
    let lowest = addresses.iter().map(|a| a.offset & !3).min()?;
    if !after {
        return Some(lowest);
    }
    let code_end = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .map(|i| i.machine.offset + 4 * i.machine.words.len())
        .max()
        .unwrap_or(0) as u32;
    let fixed_below = addresses.iter().any(|a| a.dynamic.is_none() && a.offset < code_end);
    Some(if fixed_below { lowest } else { lowest.max(code_end & !3) })
}

#[cfg(test)]
mod tests {
    use super::start;
    use crate::ir::{self, Program};
    use crate::resource::{Address, CODE};

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn program(words: &[u32]) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        ir::build(&code, false, 256).expect("builds")
    }

    fn address(offset: u32, dynamic: bool) -> Address {
        Address { param: CODE, offset, dynamic: dynamic.then_some(ir::Value(1)) }
    }

    #[test]
    fn starts_after_the_instructions_unless_a_fixed_load_reads_below() {
        // Two instructions: the code ends at byte 8.
        let p = program(&[0xBF80_0000, S_ENDPGM]);
        let computed = [address(0, true)];
        assert_eq!(start(&p, &computed, false), Some(0));
        assert_eq!(start(&p, &computed, true), Some(8));
        let fixed = [address(4, false), address(0, true)];
        assert_eq!(start(&p, &fixed, true), Some(0));
        let past = [address(12, true)];
        assert_eq!(start(&p, &past, true), Some(12));
        assert_eq!(start(&p, &[], true), None);
    }
}
