//! Samplers as the host describes them, from the console's sampler
//! description (`CreateSamplerX`), which names the hardware's sampler
//! fields. (Immediate samplers in shader code come decoded with the
//! translation.)

use std::sync::OnceLock;
use windows::Win32::Graphics::Direct3D12::*;

/// Optional mip LOD bias delta from `VICTORY_MIP_BIAS` (`<float>` or `"auto"`).
/// Unset or `0` = `None` (no change to any sampler or root signature).
/// `"auto"` uses AMD's FidelityFX recommendation `log2(scale) - 1.0` from
/// `VICTORY_FFX_SCALE` (e.g. `-2.0` at 0.5, `-2.25` at 0.42).
pub(crate) fn mip_bias_delta() -> Option<f32> {
    static DELTA: OnceLock<Option<f32>> = OnceLock::new();
    *DELTA.get_or_init(|| {
        let raw = std::env::var("VICTORY_MIP_BIAS").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed == "0" {
            return None;
        }
        let delta = if trimmed.eq_ignore_ascii_case("auto") {
            let scale = std::env::var("VICTORY_FFX_SCALE")
                .ok()
                .and_then(|s| {
                    let head = s.split_once(',').map_or(s.as_str(), |(a, _)| a).trim();
                    if head.eq_ignore_ascii_case("auto") {
                        Some(0.42f32)
                    } else {
                        head.parse::<f32>().ok()
                    }
                })
                .filter(|s| s.is_finite() && *s > 0.0 && *s <= 1.0)
                .unwrap_or(0.5);
            (scale.log2() - 1.0).clamp(-4.0, 0.0)
        } else {
            match trimmed.parse::<f32>() {
                Ok(v) if v.is_finite() && v != 0.0 => v.clamp(-8.0, 8.0),
                _ => {
                    tracing::warn!(
                        value = trimmed,
                        "VICTORY_MIP_BIAS: expected float (e.g. -1.5) or auto; disabled"
                    );
                    return None;
                }
            }
        };
        tracing::info!(
            delta = format_args!("{delta:.2}"),
            "sampler mip LOD bias active (VICTORY_MIP_BIAS)"
        );
        Some(delta)
    })
}

/// `VICTORY_MIP_BIAS_SCOPE=aniso`: restricts `VICTORY_MIP_BIAS` to anisotropic
/// samplers (`D3D12_FILTER_ANISOTROPIC`, `0x55`) so bilinear/trilinear samplers
/// used by bloom, lens flare, auto-exposure pyramids and reflection cubemaps
/// keep their original `MipLODBias`. Default (`all`): all mipmapped linear and
/// anisotropic samplers, as before.
fn mip_bias_aniso_only() -> bool {
    static ANISO: OnceLock<bool> = OnceLock::new();
    *ANISO.get_or_init(|| {
        let on = std::env::var("VICTORY_MIP_BIAS_SCOPE")
            .is_ok_and(|v| v.trim().eq_ignore_ascii_case("aniso"));
        if on {
            tracing::info!("VICTORY_MIP_BIAS restricted to anisotropic samplers only (VICTORY_MIP_BIAS_SCOPE=aniso)");
        }
        on
    })
}

/// Pure helper: applies `delta` to `orig_bias` only on standard mipmapped
/// linear/anisotropic samplers (never comparison/shadow, min/max Hi-Z,
/// point-only, or single-mip samplers), clamped to D3D12's `[-16.0, 15.99]`.
/// When `aniso_only` is true, only `D3D12_FILTER_ANISOTROPIC` (`0x55`) is biased.
fn biased_lod_scoped(
    filter: D3D12_FILTER,
    min_lod: f32,
    max_lod: f32,
    orig_bias: f32,
    delta: f32,
    aniso_only: bool,
) -> f32 {
    let raw = filter.0 as u32;
    let reduction = raw >> 7;
    let filter_bits = raw & 0x7F;
    if reduction != 0 || filter_bits == 0 || max_lod <= min_lod {
        return orig_bias;
    }
    if aniso_only && filter_bits != 0x55 {
        return orig_bias;
    }
    (orig_bias + delta).clamp(-16.0, 15.99)
}

/// Applies `VICTORY_MIP_BIAS` (if set) to a host sampler's `MipLODBias`.
pub(crate) fn apply_mip_bias(
    filter: D3D12_FILTER,
    min_lod: f32,
    max_lod: f32,
    orig_bias: f32,
) -> f32 {
    let Some(delta) = mip_bias_delta() else {
        return orig_bias;
    };
    let next = biased_lod_scoped(filter, min_lod, max_lod, orig_bias, delta, mip_bias_aniso_only());
    if next != orig_bias {
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| {
            tracing::info!(
                filter = format_args!("{:#x}", filter.0),
                orig_bias = format_args!("{orig_bias:.2}"),
                new_bias = format_args!("{next:.2}"),
                delta = format_args!("{delta:.2}"),
                "first sampler biased by VICTORY_MIP_BIAS"
            );
        });
    }
    next
}

/// The fields both forms share.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Fields {
    /// Point 0, bilinear 1, anisotropic point 2, anisotropic bilinear 3.
    mag: u32,
    min: u32,
    /// None 0, point 1, linear 2.
    mip: u32,
    /// Plain 0, minimum 1, maximum 2.
    mode: u32,
    /// Host address modes.
    address: [u32; 3],
    anisotropy: u32,
    /// Host comparison function.
    compare: u32,
    min_lod: f32,
    max_lod: f32,
    bias: f32,
    border: [f32; 4],
}

impl Fields {
    fn host(&self) -> D3D12_SAMPLER_DESC {
        let linear = |f: u32| u32::from(f == 1 || f == 3);
        let aniso = self.mag >= 2 || self.min >= 2;
        // The console compares when the instruction says so, with the
        // sampler's function; the host needs a comparison sampler for that.
        // A sampler given a real test (less ... greater-or-equal) is one;
        // never, always and none are plain samplers.
        const LESS: u32 = 2;
        const GREATER_EQUAL: u32 = 7;
        let compares = (LESS..=GREATER_EQUAL).contains(&self.compare);
        let reduction = match self.mode {
            1 => 2,
            2 => 3,
            _ => u32::from(compares),
        };
        let filter = if aniso {
            0x55
        } else {
            linear(self.min) << 4 | linear(self.mag) << 2 | u32::from(self.mip == 2)
        };
        D3D12_SAMPLER_DESC {
            Filter: D3D12_FILTER((reduction << 7 | filter) as i32),
            AddressU: D3D12_TEXTURE_ADDRESS_MODE(self.address[0] as i32),
            AddressV: D3D12_TEXTURE_ADDRESS_MODE(self.address[1] as i32),
            AddressW: D3D12_TEXTURE_ADDRESS_MODE(self.address[2] as i32),
            MipLODBias: self.bias,
            MaxAnisotropy: if aniso {
                self.anisotropy.clamp(1, 16)
            } else {
                1
            },
            ComparisonFunc: D3D12_COMPARISON_FUNC(self.compare.max(1) as i32),
            BorderColor: self.border,
            MinLOD: self.min_lod,
            // Without a mip filter the console samples mip 0 only.
            MaxLOD: if self.mip == 0 {
                self.min_lod
            } else {
                self.max_lod
            },
        }
    }
}

/// The console's sampler description (27 words): flags, magnification,
/// minification, mip and depth filters, filter mode, three address modes,
/// minimum and maximum LOD, LOD bias, secondary bias, anisotropy, its bias
/// and threshold, comparison, border color, then tuning words.
pub(crate) fn console(words: &[u32; 27]) -> D3D12_SAMPLER_DESC {
    let f = |i: usize| f32::from_bits(words[i]);
    Fields {
        mag: words[1],
        min: words[2],
        mip: words[3],
        mode: words[5],
        address: [words[6], words[7], words[8]],
        min_lod: f(9),
        max_lod: f(10),
        bias: f(11),
        anisotropy: words[13],
        compare: words[16],
        border: [f(17), f(18), f(19), f(20)],
    }
    .host()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_descriptions_decode() {
        let mut w = [0u32; 27];
        w[1] = 1; // bilinear magnification
        w[2] = 0; // point minification
        w[3] = 2; // linear mips
        w[6..9].copy_from_slice(&[3, 3, 3]);
        w[10] = 1000.0f32.to_bits();
        let d = console(&w);
        assert_eq!(d.Filter.0, 0x5, "min point, mag and mip linear");
        assert_eq!(d.AddressW, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
        assert_eq!(d.MaxLOD, 1000.0);
        // A less-or-equal test makes a comparison sampler; never does not.
        w[16] = 4;
        assert_eq!(console(&w).Filter.0, 0x85);
        assert_eq!(console(&w).ComparisonFunc, D3D12_COMPARISON_FUNC_LESS_EQUAL);
        w[16] = 1;
        assert_eq!(console(&w).Filter.0, 0x5);
    }

    #[test]
    fn mip_bias_only_affects_standard_mipmapped_linear_samplers() {
        // Anisotropic mipmapped sampler: gets biased in both modes.
        assert!((biased_lod_scoped(D3D12_FILTER(0x55), 0.0, 1000.0, 0.0, -2.25, false) - -2.25).abs() < 1e-5);
        assert!((biased_lod_scoped(D3D12_FILTER(0x55), 0.0, 1000.0, 0.0, -2.25, true) - -2.25).abs() < 1e-5);
        // Trilinear mipmapped sampler: biased in `all`, untouched in `aniso` (bloom/exposure/cubemaps).
        assert!((biased_lod_scoped(D3D12_FILTER(0x15), 0.0, 12.0, -1.0, -2.0, false) - -3.0).abs() < 1e-5);
        assert_eq!(biased_lod_scoped(D3D12_FILTER(0x15), 0.0, 12.0, -1.0, -2.0, true), -1.0);
        assert_eq!(biased_lod_scoped(D3D12_FILTER(0x15), 0.0, 12.0, -15.0, -3.0, false), -16.0);
        // Comparison (shadow) sampler: untouched.
        assert_eq!(biased_lod_scoped(D3D12_FILTER(0x85), 0.0, 1000.0, 0.0, -2.25, false), 0.0);
        // Min/Max reduction (Hi-Z) sampler: untouched.
        assert_eq!(biased_lod_scoped(D3D12_FILTER(0x115), 0.0, 1000.0, 0.0, -2.25, false), 0.0);
        // Point-only sampler: untouched.
        assert_eq!(biased_lod_scoped(D3D12_FILTER(0x0), 0.0, 1000.0, 0.0, -2.25, false), 0.0);
        // Single-mip sampler (MaxLOD <= MinLOD): untouched.
        assert_eq!(biased_lod_scoped(D3D12_FILTER(0x55), 0.0, 0.0, 0.0, -2.25, false), 0.0);
    }
}
