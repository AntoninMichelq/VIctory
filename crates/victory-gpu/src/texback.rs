//! Textures shaders write that the title's own code then reads on the CPU.
//!
//! On the console the CPU and GPU share memory, so what a compute pass stores
//! into a texture is visible to CPU reads in title memory. On PC it stays in
//! the host texture copy. The title's ocean simulation writes a 256x256 height
//! map (`R32_FLOAT`, DXGI 41) and displacement maps (`R32G32_FLOAT`, DXGI 16;
//! `R16G16_FLOAT`, DXGI 36) that CPU physics samples for buoyancy — without
//! copying them back, boats sink in the first mission.
//!
//! Controlled by `VICTORY_TEXTURE_BACK`: comma-separated `<w>x<h>f<format>`
//! specs (`1` or unset: the four water simulation formats; `0`: disabled).
//! Readback buffers live in system RAM (`D3D12_HEAP_TYPE_READBACK`, ~0.9 MiB
//! per list for the four default formats, 0 bytes of VRAM) and are released
//! when the list resets.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

const DEFAULT_KINDS: &str = "256x256f41,256x256f16,64x64f16,128x128f36";

fn parse_kinds(spec: &str) -> Vec<(u64, u32, u32)> {
    spec.split(',')
        .filter_map(|k| {
            let (size, format) = k.trim().split_once('f')?;
            let (w, h) = size.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
        })
        .collect()
}

/// The kinds copied back: `(width, height, dxgi_format)`.
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let list = match std::env::var("VICTORY_TEXTURE_BACK").as_deref() {
            Ok("1") | Err(_) => DEFAULT_KINDS.to_owned(),
            Ok(spec) => spec.to_owned(),
        };
        let kinds = parse_kinds(&list);
        if !kinds.is_empty() {
            tracing::info!(
                ?kinds,
                "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)"
            );
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding write to a specific allocation. Mapping and resource
/// changes invalidate only the writes that could touch the replaced memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket {
        owner,
        at,
        end,
        valid: AtomicBool::new(true),
    });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS.lock().unwrap_or_else(|e| e.into_inner()).retain(|weak| {
        let Some(ticket) = weak.upgrade() else {
            return false;
        };
        if matches(&ticket) {
            ticket.valid.store(false, Ordering::Release);
        }
        true
    });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        let copy = crate::resource::host_copy(resource)?;
        let _mapping = crate::watch::title_memory_reading();
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self {
            copy,
            layout,
            ticket,
        })
    }

    pub(crate) fn owner(&self) -> usize {
        self.ticket.owner
    }
}

/// The readback resources and the allocation the pixels belong to.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

/// Records a host copy from a texture already in `COPY_SOURCE`.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live texture kept through the command list's state tracking.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    // The last footprint's advertised extent can omit row-end padding.
    // The tiler reads a pitched slice, so the mapped buffer must include it.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let buffer = host.read_back_buffer(total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back {
        buffer,
        footprints,
        rows,
        mips,
        layout: target.layout,
        ticket: target.ticket,
    })
}

/// Once the list has run, tile its texels back into the same title allocation.
pub(crate) fn land(back: &Back) {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    if !back.ticket.valid.load(Ordering::Acquire) {
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(
                skipped,
                "skipping texture readbacks whose title allocation changed"
            );
        }
        return;
    }
    let _mapping = crate::watch::title_memory_reading();
    let (at, end) = back.range();
    let size = end - at;
    SCRATCH.with_borrow_mut(|tiled| {
        tiled.resize(size, 0);
        // Keep title bytes the subresources do not cover.
        if !crate::host::read_into(at as *const u8, &mut tiled[..size]) {
            return;
        }
        let mut data = std::ptr::null_mut();
        // SAFETY: our read-back buffer, written by the list that has run.
        unsafe {
            if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
                return;
            }
        }
        for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
            let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped buffer.
            let linear = unsafe {
                std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len)
            };
            if let Err(e) = back.layout.tile(
                &mut tiled[..size],
                mip,
                slice,
                linear,
                pitch,
                pitch * rows as usize,
            ) {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled")
                });
                // SAFETY: mapped above.
                unsafe { back.buffer.Unmap(0, None) };
                return;
            }
        }
        // SAFETY: mapped above.
        unsafe { back.buffer.Unmap(0, None) };
        let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
        if !back.ticket.valid.load(Ordering::Acquire) {
            let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
            if skipped.is_power_of_two() {
                tracing::info!(
                    skipped,
                    "skipping texture readbacks whose title allocation changed"
                );
            }
            return;
        }
        crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, true);
        drop(_ticket_guard);
        let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
        if copied == 1 || copied.is_multiple_of(128) {
            tracing::info!(copied, bytes = size, "texture readback landed");
        }
    });
}

/// Lands a flight's texture readbacks in order, skipping earlier readbacks to
/// the exact same allocation and (under `VICTORY_RDNA_CPU=1`) tiling
/// non-overlapping textures in parallel without dropping COM objects on the
/// relay thread.
pub(crate) fn land_many(backs: &[&Back]) {
    if backs.is_empty() {
        return;
    }
    if backs.len() == 1 || !crate::host::rdna_cpu() {
        for &back in backs {
            land(back);
        }
        return;
    }
    // Keep only the last readback for any identical `(at, end)` range since
    // `record` copies all subresources of the texture.
    let mut active: Vec<&Back> = Vec::with_capacity(backs.len());
    for (i, &back) in backs.iter().enumerate() {
        let range = back.range();
        if backs[i + 1..].iter().any(|later| later.range() == range) {
            continue;
        }
        active.push(back);
    }
    if active.len() <= 1 {
        if let Some(&back) = active.first() {
            land(back);
        }
        return;
    }
    // If any two distinct ranges overlap, fall back to sequential order.
    let mut sorted: Vec<(usize, usize)> = active.iter().map(|b| b.range()).collect();
    sorted.sort_unstable();
    let overlaps = sorted.windows(2).any(|w| w[0].1 > w[1].0);
    if overlaps {
        for back in active {
            land(back);
        }
    } else {
        // SAFETY: `wide_run` joins all jobs before returning, so every `&Back`
        // stays alive for the duration of the parallel run, and each job writes
        // to a disjoint title-memory range.
        let jobs: Vec<Box<dyn FnOnce() + Send>> = active
            .iter()
            .map(|&back| {
                let ptr = std::ptr::from_ref(back) as usize;
                Box::new(move || {
                    let back = unsafe { &*(ptr as *const Back) };
                    land(back);
                }) as Box<dyn FnOnce() + Send>
            })
            .collect();
        crate::shadow::wide_run(jobs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_and_disabled_kind_specs() {
        assert_eq!(
            parse_kinds(DEFAULT_KINDS),
            vec![(256, 256, 41), (256, 256, 16), (64, 64, 16), (128, 128, 36)]
        );
        assert!(parse_kinds("0").is_empty());
        assert!(parse_kinds("").is_empty());
    }

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }
}
