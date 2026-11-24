//! Render targets the frame used, picked by size and format for the
//! present (in every build, so a play build can show the G-buffer colours
//! while the title's lighting is still black):
//! - `VICTORY_PRESENT_TARGET=<width>x<height>:<format>[:<n>]` presents the
//!   `n`-th (first-use order, default 0; `busiest` for the one the frame
//!   binds most often) texture of that size and DXGI
//!   format the frame renders to or writes, in place of what the title
//!   presents, to see a pass the final image does not show.
//! - `VICTORY_FLAT_LIGHT` (same spec, see [`crate::flat_light`]) adds that
//!   texture to the title's image instead.
//!
//! Frames without such a texture present as usual.

use windows::Win32::Graphics::Direct3D12::*;

type Spec = (u64, u32, i32, usize);

/// The `n` that picks the matching texture the frame bound most often.
const BUSIEST: usize = usize::MAX;

fn parse(var: &str, what: &str) -> Option<Spec> {
    let v = std::env::var(var).ok()?;
    let mut parts = v.split(':');
    let (w, h) = parts.next()?.split_once('x')?;
    let format = parts.next()?.trim().parse().ok()?;
    // `busiest`: the matching texture the frame bound most often.
    let n = parts.next().map_or(Some(0), |n| match n.trim() {
        "busiest" => Some(BUSIEST),
        n => n.parse().ok(),
    })?;
    let spec = (w.trim().parse().ok()?, h.trim().parse().ok()?, format, n);
    tracing::info!(?spec, "{what}");
    Some(spec)
}

fn present_spec() -> Option<Spec> {
    static SPEC: std::sync::OnceLock<Option<Spec>> = std::sync::OnceLock::new();
    *SPEC.get_or_init(|| {
        parse(
            "VICTORY_PRESENT_TARGET",
            "presenting a render target in place of the title's image",
        )
    })
}

fn flat_spec() -> Option<Spec> {
    static SPEC: std::sync::OnceLock<Option<Spec>> = std::sync::OnceLock::new();
    *SPEC.get_or_init(|| {
        parse(
            "VICTORY_FLAT_LIGHT",
            "adding a render target to the title's image (stand-in lighting)",
        )
    })
}

/// The frame being built and the textures it used that match a spec, in
/// first-use order.
#[allow(clippy::type_complexity)]
struct Used(std::sync::Mutex<(u64, Vec<(usize, std::sync::Arc<crate::resource::HostTexture>, u32)>)>);

static PRESENTED: Used = Used(std::sync::Mutex::new((0, Vec::new())));
static FLAT: Used = Used(std::sync::Mutex::new((0, Vec::new())));
/// Depth buffers of the stand-in lighting's size, for its shading.
static FLAT_DEPTH: Used = Used(std::sync::Mutex::new((0, Vec::new())));

/// Notes `resource` in [`FLAT_DEPTH`] when it is a single-sample 2D depth
/// buffer of the stand-in lighting's size.
fn note_depth(resource: crate::object::This) {
    let Some((w, h, ..)) = flat_spec() else {
        return;
    };
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    if (d.Width, d.Height) != (w, h)
        || d.SampleDesc.Count > 1
        || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 == 0
        || crate::flat_light::depth_format(d.Format).is_none()
    {
        return;
    }
    let frame = crate::queue::presented() + 1;
    let mut p = FLAT_DEPTH.0.lock().unwrap_or_else(|e| e.into_inner());
    if p.0 != frame {
        *p = (frame, Vec::new());
    }
    if !p.1.iter().any(|&(r, ..)| r == resource as usize) {
        p.1.push((resource as usize, copy, 1));
    }
}

impl Used {
    fn note(&self, spec: Spec, resource: crate::object::This) {
        let (w, h, format, _) = spec;
        let Some(copy) = crate::resource::host_copy(resource) else {
            return;
        };
        // SAFETY: a live resource.
        let d = unsafe { copy.texture.GetDesc() };
        // A format the host makes targets in a stand-in for
        // (R9G9B9E5_SHAREDEXP as R16G16B16A16_FLOAT) matches its stand-in.
        let stand_in = crate::resource::target_format(format as u32) as i32;
        if (d.Width, d.Height) != (w, h)
            || (d.Format.0 != format && d.Format.0 != stand_in)
            || d.SampleDesc.Count > 1
            || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        {
            return;
        }
        let frame = crate::queue::presented() + 1;
        let mut p = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if p.0 != frame {
            *p = (frame, Vec::new());
        }
        match p.1.iter_mut().find(|(r, ..)| *r == resource as usize) {
            Some((.., uses)) => *uses += 1,
            None => p.1.push((resource as usize, copy, 1)),
        }
    }

    fn take(&self, spec: Spec, frame: u64) -> Option<std::sync::Arc<crate::resource::HostTexture>> {
        let mut p = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if p.0 != frame {
            return None;
        }
        let (_, list) = std::mem::take(&mut *p);
        if spec.3 == BUSIEST {
            // The first of the most bound, on a tie.
            return list
                .into_iter()
                .rev()
                .max_by_key(|&(.., uses)| uses)
                .map(|(_, t, _)| t);
        }
        list.into_iter().nth(spec.3).map(|(_, t, _)| t)
    }
}

/// Notes a title texture the frame being built renders to or writes.
pub(crate) fn note(resource: crate::object::This) {
    if resource.is_null() {
        return;
    }
    if let Some(spec) = present_spec() {
        PRESENTED.note(spec, resource);
    }
    if let Some(spec) = flat_spec() {
        FLAT.note(spec, resource);
        note_depth(resource);
    }
}

/// At the present of `frame`: the texture to show in place of the title's,
/// if `VICTORY_PRESENT_TARGET` names one the frame used.
pub(crate) fn present_override(frame: u64) -> Option<std::sync::Arc<crate::resource::HostTexture>> {
    PRESENTED.take(present_spec()?, frame)
}

/// At the present of `frame`: the texture to add to the title's image, if
/// `VICTORY_FLAT_LIGHT` names one the frame used.
/// With the frame's first depth buffer of that size, if any (for shading).
#[allow(clippy::type_complexity)]
pub(crate) fn flat_light(
    frame: u64,
) -> Option<(
    std::sync::Arc<crate::resource::HostTexture>,
    Option<std::sync::Arc<crate::resource::HostTexture>>,
)> {
    let spec = flat_spec()?;
    let depth = FLAT_DEPTH.take((spec.0, spec.1, spec.2, 0), frame);
    Some((FLAT.take(spec, frame)?, depth))
}
