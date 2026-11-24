//! `VICTORY_PRESENT_TARGET`: show one of the frame's render targets in
//! place of the title's image (in every build, so a play build can show
//! the G-buffer colours while the title's lighting is still black).

use windows::Win32::Graphics::Direct3D12::*;

/// `VICTORY_PRESENT_TARGET=<width>x<height>:<format>[:<n>]`: present the
/// `n`-th (first-use order, default 0) texture of that size and DXGI format
/// the frame renders to or writes, in place of what the title presents, to
/// see a pass the final image does not show (e.g. the G-buffer while lighting
/// is still black). Frames without such a texture present as usual.
fn present_spec() -> Option<(u64, u32, i32, usize)> {
    static SPEC: std::sync::OnceLock<Option<(u64, u32, i32, usize)>> = std::sync::OnceLock::new();
    *SPEC.get_or_init(|| {
        let v = std::env::var("VICTORY_PRESENT_TARGET").ok()?;
        let mut parts = v.split(':');
        let (w, h) = parts.next()?.split_once('x')?;
        let format = parts.next()?.trim().parse().ok()?;
        let n = parts.next().map_or(Some(0), |n| n.trim().parse().ok())?;
        let spec = (w.trim().parse().ok()?, h.trim().parse().ok()?, format, n);
        tracing::info!(
            ?spec,
            "presenting a render target in place of the title's image"
        );
        Some(spec)
    })
}

/// The frame being built and the textures it used that match
/// [`present_spec`], in first-use order.
#[allow(clippy::type_complexity)]
static PRESENTED: std::sync::Mutex<(
    u64,
    Vec<(usize, std::sync::Arc<crate::resource::HostTexture>)>,
)> = std::sync::Mutex::new((0, Vec::new()));

/// Notes a title texture the frame being built renders to or writes.
pub(crate) fn note(resource: crate::object::This) {
    if resource.is_null() {
        return;
    }
    let Some((w, h, format, _)) = present_spec() else {
        return;
    };
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    // A format the host makes targets in a stand-in for (R9G9B9E5_SHAREDEXP
    // as R16G16B16A16_FLOAT) matches its stand-in.
    let stand_in = crate::resource::target_format(format as u32) as i32;
    if (d.Width, d.Height) != (w, h)
        || (d.Format.0 != format && d.Format.0 != stand_in)
        || d.SampleDesc.Count > 1
        || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
    {
        return;
    }
    let frame = crate::queue::presented() + 1;
    let mut p = PRESENTED.lock().unwrap_or_else(|e| e.into_inner());
    if p.0 != frame {
        *p = (frame, Vec::new());
    }
    if !p.1.iter().any(|&(r, _)| r == resource as usize) {
        p.1.push((resource as usize, copy));
    }
}

/// At the present of `frame`: the texture to show in place of the title's,
/// if [`present_spec`] names one the frame used.
pub(crate) fn present_override(frame: u64) -> Option<std::sync::Arc<crate::resource::HostTexture>> {
    let (.., n) = present_spec()?;
    let mut p = PRESENTED.lock().unwrap_or_else(|e| e.into_inner());
    if p.0 != frame {
        return None;
    }
    let (_, list) = std::mem::take(&mut *p);
    list.into_iter().nth(n).map(|(_, t)| t)
}

