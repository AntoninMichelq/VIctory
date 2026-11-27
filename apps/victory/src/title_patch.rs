//! A patch of the title's code before it runs, for its FidelityFX path
//! (ported from the forkAMD line, commits 0425b6e7 and 13fd3039):
//!
//! - `VICTORY_FFX_SCALE=<scale>[,<rva>]` sets the render scale of that path
//!   (Series S at 1080p: 0.6667 is the title's own 720p, 0.5 renders 540p).
//!   The method that picks it (1.0.2.1: 28EF680, the default) takes 2/3 on
//!   Series S and 1.0 on Series X.
//!
//! It checks the code it patches first and patches nothing in other code
//! (another build of the title): the method must have its two loads and its
//! padding where expected. `VICTORY_FSR_4K` patches the same method its own
//! way; the caller does not apply this one then.

use victory_loader::LoadedImage;

/// Applies the patches the environment asks for.
pub(crate) fn apply(image: &LoadedImage) {
    ffx_scale(image);
}

fn hex(s: &str) -> Option<usize> {
    usize::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

/// Writes `bytes` at title+`rva`, inside the image (checked), through a
/// writable mapping for the moment. False if Windows refused it.
fn write_code(image: &LoadedImage, rva: usize, bytes: &[u8]) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    if rva
        .checked_add(bytes.len())
        .is_none_or(|end| end > image.info().size_of_image as usize)
    {
        return false;
    }
    let at = (image.base() + rva) as *mut u8;
    let mut old = 0;
    // SAFETY: bytes inside the mapped image; no title code runs yet.
    unsafe {
        if VirtualProtect(at.cast(), bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return false;
        }
        at.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
        VirtualProtect(at.cast(), bytes.len(), old, &mut old);
    }
    true
}

/// The bytes of the scale method [`patch_scale`] reads and writes.
const SCALE_SPAN: usize = 0x68;
/// Offset of the `f32` scale slot inside the patched scale method.
const SCALE_SLOT: usize = 0x64;

fn ffx_scale(image: &LoadedImage) {
    let Ok(spec) = std::env::var("VICTORY_FFX_SCALE") else {
        return;
    };
    if spec.trim().is_empty() {
        return;
    }
    let (scale, rva) = spec.split_once(',').unwrap_or((&spec, "28EF680"));
    let scale = scale
        .trim()
        .parse::<f32>()
        .ok()
        .filter(|s| *s > 0.0 && *s <= 1.0);
    let (Some(scale), Some(rva)) = (scale, hex(rva)) else {
        eprintln!("VIctory: VICTORY_FFX_SCALE={spec}: expected <scale in (0, 1]>[,<hex rva>]");
        return;
    };
    if rva
        .checked_add(SCALE_SPAN)
        .is_none_or(|end| end > image.info().size_of_image as usize)
    {
        eprintln!("VIctory: VICTORY_FFX_SCALE: title+{rva:#x} is outside the image");
        return;
    }
    // SAFETY: the method's bytes and its padding, inside the mapped image
    // (checked above), read before any title code runs.
    let mut code =
        unsafe { std::slice::from_raw_parts((image.base() + rva) as *const u8, SCALE_SPAN) }
            .to_vec();
    if !patch_scale(&mut code, scale) {
        eprintln!(
            "VIctory: FidelityFX scale not patched: unexpected code at title+{rva:#x} (wrong rva?)"
        );
    } else if write_code(image, rva, &code) {
        tracing::info!(rva = format_args!("{rva:#x}"), scale, "FidelityFX render scale set (VICTORY_FFX_SCALE)");
        eprintln!("VIctory: FidelityFX render scale at title+{rva:#x} set to {scale}");
    } else {
        eprintln!("VIctory: FidelityFX scale at title+{rva:#x} not patched: not writable");
    }
}

/// Points both scale loads of the method `code` starts with
/// (`vmovss xmm0, [rip+d]` at +0x28 and +0x32) at `scale`, written into the
/// method's `int3` padding at +0x64. False, with nothing written, for other
/// code.
fn patch_scale(code: &mut [u8], scale: f32) -> bool {
    const LOADS: [usize; 2] = [0x28, 0x32];
    const VMOVSS_RIP: [u8; 4] = [0xC5, 0xFA, 0x10, 0x05];
    if code.len() < SCALE_SPAN
        || LOADS.iter().any(|&l| code[l..l + 4] != VMOVSS_RIP)
        || code[SCALE_SLOT..SCALE_SLOT + 4] != [0xCC; 4]
    {
        return false;
    }
    code[SCALE_SLOT..SCALE_SLOT + 4].copy_from_slice(&scale.to_le_bytes());
    for l in LOADS {
        let disp = (SCALE_SLOT - (l + 8)) as i32;
        code[l + 4..l + 8].copy_from_slice(&disp.to_le_bytes());
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scale_loads_point_at_the_constant() {
        // Synthetic code: the two loads, filler, and int3 padding.
        let mut code = vec![0x90u8; SCALE_SPAN];
        for at in [0x28, 0x32] {
            code[at..at + 8].copy_from_slice(&[0xC5, 0xFA, 0x10, 0x05, 1, 2, 3, 4]);
        }
        code[0x60..].fill(0xCC);
        assert!(patch_scale(&mut code, 0.5));
        assert_eq!(code[0x64..0x68], 0.5f32.to_le_bytes());
        // Each load's next instruction plus its displacement is the constant.
        for at in [0x28, 0x32] {
            let disp = i32::from_le_bytes(code[at + 4..at + 8].try_into().unwrap());
            assert_eq!(at as i32 + 8 + disp, 0x64);
        }
        // Already patched, so no padding left: nothing is written again.
        let before = code.clone();
        assert!(!patch_scale(&mut code, 0.25));
        assert_eq!(code, before);
    }

    #[test]
    fn other_code_is_left_alone() {
        let mut code = vec![0xCCu8; SCALE_SPAN];
        assert!(!patch_scale(&mut code, 0.5));
        assert!(code.iter().all(|&b| b == 0xCC));
    }
}
