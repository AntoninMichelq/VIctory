//! `amd_fidelityfx_loader_xbox.dll`: AMD FidelityFX's API, served by the
//! host's PC build of it on the host GPU when the user names a folder that
//! holds one (`VICTORY_FIDELITYFX_DIR`). Without it, or if it offers no
//! upscaler for the host GPU, this module stays out of the registry and the
//! title's own console DLLs run as before.
//!
//! Both builds have the same API (the FidelityFX SDK's, MIT): five functions
//! over chains of descriptions, each starting with a type id and the next
//! one's address; contexts are opaque pointers. The console-specific parts
//! are translated: the backend description names the console device (type
//! 0x0100_0002; on PC type 2 with the host device), and a dispatch names the
//! title's command list and resources (their host list and host copies, in
//! the states the title gives, see `victory_gpu::external`). The rest passes
//! through unchanged: sizes, flags, jitter, motion-vector scale, camera
//! values, the resource descriptions, the context. Only the upscaler: other
//! effects (frame generation) are refused.
#![allow(non_snake_case)]
// Title code calls these through its import table.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use victory_gpu::object::Object;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::core::Interface;

use crate::{HleModule, hle_export};

/// The PC build's upscaler, which has the five functions itself.
const DLL: &str = "amd_fidelityfx_upscaler_dx12.dll";

// Return codes.
const OK: u32 = 0;
const UNKNOWN_TYPE: u32 = 2;
const NO_PROVIDER: u32 = 4;
const PARAMETER: u32 = 6;

// Description types.
const CREATE_UPSCALE: u64 = 0x1_0000;
const DISPATCH_UPSCALE: u64 = 0x1_0001;
const CONSOLE_BACKEND: u64 = 0x100_0002;
const HOST_BACKEND: u64 = 2;
const GET_VERSIONS: u64 = 4;
const GET_PROVIDER_VERSION: u64 = 6;
const MEMORY_USAGE_WITH_DEVICE: u64 = 0x1_0009;

/// `FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION` (`1 << 6`) in FidelityFX SDK 2.x.
const UPSCALE_ENABLE_DYNAMIC_RESOLUTION: u32 = 1 << 6;

/// Default dynamic scale bounds when `VICTORY_FFX_DYNAMIC` is set.
const DEFAULT_SCALE_MAX: f32 = 0.5;
const DEFAULT_SCALE_MIN: f32 = 0.38;
const ABS_SCALE_MIN: f32 = 0.25;

#[repr(C)]
#[derive(Clone, Copy)]
struct Header {
    kind: u64,
    next: *mut Header,
}

/// An upscaler context: flags, largest render and output sizes, a message
/// callback.
#[repr(C)]
#[derive(Clone, Copy)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render: [u32; 2],
    max_output: [u32; 2],
    message: usize,
}

/// The backend description: a device.
#[repr(C)]
struct Backend {
    header: Header,
    device: *mut c_void,
}

/// A resource, its description (type, format, sizes, mips, flags, usage)
/// and the state it is in.
#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    resource: *mut c_void,
    description: [u32; 8],
    state: u32,
}

/// One upscaling: the command list, then colour, depth, motion vectors,
/// exposure, reactive mask, transparency mask and output; then jitter,
/// motion-vector scale, sizes, sharpening, frame time, pre-exposure, reset,
/// camera and flags, which pass through.
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchUpscale {
    header: Header,
    list: *mut c_void,
    resources: [Resource; 7],
    rest: [u32; 18],
}

const _: () = assert!(size_of::<CreateUpscale>() == 0x30);
const _: () = assert!(size_of::<Resource>() == 0x30);
const _: () = assert!(size_of::<DispatchUpscale>() == 0x1b0);

/// The versions a backend offers: count in and out, ids, names.
#[repr(C)]
struct GetVersions {
    header: Header,
    create_kind: u64,
    device: *mut c_void,
    count: *mut u64,
    ids: *mut u64,
    names: *mut *const c_char,
}

#[repr(C)]
struct ProviderVersion {
    header: Header,
    id: u64,
    name: *const c_char,
}

type Create = unsafe extern "C" fn(*mut *mut c_void, *mut Header, *const c_void) -> u32;
type Destroy = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> u32;
/// Configure, query and dispatch.
type Call = unsafe extern "C" fn(*mut *mut c_void, *mut Header) -> u32;

struct Host {
    create: Create,
    destroy: Destroy,
    configure: Call,
    query: Call,
    dispatch: Call,
    device: ID3D12Device,
}

// SAFETY: D3D12 devices are free-threaded; the rest is code.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

static HOST: OnceLock<Host> = OnceLock::new();

/// Writable 32-bit float slot in the title's scale-getter (`title_patch`),
/// registered before title execution when `VICTORY_FFX_DYNAMIC` is active.
static SCALE_SLOT: OnceLock<usize> = OnceLock::new();

/// Target GPU frame time in ms from `VICTORY_FFX_DYNAMIC` (`None` = disabled,
/// default fixed-scale behavior).
pub fn dynamic_target_ms() -> Option<f64> {
    static TARGET: OnceLock<Option<f64>> = OnceLock::new();
    *TARGET.get_or_init(|| parse_target(std::env::var("VICTORY_FFX_DYNAMIC").ok().as_deref()))
}

/// The target of `VICTORY_FFX_DYNAMIC` from its text.
fn parse_target(raw: Option<&str>) -> Option<f64> {
    let trimmed = raw?.trim();
    if trimmed.is_empty() || trimmed == "0" {
        return None;
    }
    match trimmed.parse::<f64>() {
        Ok(ms) if ms.is_finite() && ms > 0.0 => Some(ms),
        _ => {
            tracing::warn!(
                value = trimmed,
                "VICTORY_FFX_DYNAMIC: expected positive target GPU time in ms (e.g. 33); disabled"
            );
            None
        }
    }
}

/// Live (`victory_gpu::live`), once dynamic scaling is on at the start
/// (the scale slot is patched in then): the target (`VICTORY_FFX_DYNAMIC`;
/// `0` holds the scale at its maximum) and the minimum scale
/// (`VICTORY_FFX_SCALE_MIN`). The maximum (`VICTORY_FFX_SCALE`) stays.
fn live_dynamic(ctrl: &mut DynamicController) -> bool {
    static TARGET: victory_gpu::live::Switch = victory_gpu::live::Switch::new();
    static MIN: victory_gpu::live::Switch = victory_gpu::live::Switch::new();
    let target = TARGET.get("VICTORY_FFX_DYNAMIC", |v, _| parse_target(v).map_or(0, f64::to_bits));
    let min = MIN.get("VICTORY_FFX_SCALE_MIN", |v, _| {
        let min = v
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|s| s.is_finite() && *s > 0.0 && *s <= 1.0)
            .unwrap_or(DEFAULT_SCALE_MIN);
        u64::from(min.to_bits())
    });
    ctrl.scale_min = f32::from_bits(min as u32).clamp(ABS_SCALE_MIN.min(ctrl.scale_max), ctrl.scale_max);
    if target == 0 {
        if ctrl.current_scale != ctrl.scale_max {
            ctrl.current_scale = ctrl.scale_max;
            write_scale_slot(ctrl.scale_max);
        }
        return false;
    }
    ctrl.target_ms = f64::from_bits(target);
    if ctrl.current_scale < ctrl.scale_min {
        ctrl.current_scale = ctrl.scale_min;
        write_scale_slot(ctrl.scale_min);
    }
    true
}

/// Registers the title's writable scale `f32` slot (4-byte aligned, mapped for
/// the process lifetime) so dynamic FSR can update it atomically between frames.
///
/// # Safety
/// `slot` must point to a 4-byte-aligned, writable `f32` location that remains
/// valid for the rest of the process lifetime.
pub unsafe fn register_scale_slot(slot: *const AtomicU32) {
    let addr = slot as usize;
    if addr != 0 && addr.is_multiple_of(align_of::<AtomicU32>()) {
        let _ = SCALE_SLOT.set(addr);
    }
}

fn write_scale_slot(scale: f32) -> bool {
    let Some(&addr) = SCALE_SLOT.get() else {
        return false;
    };
    // SAFETY: registered by `title_patch` as a 4-byte-aligned writable slot
    // inside the loaded title image, alive for the process lifetime.
    let slot = unsafe { &*(addr as *const AtomicU32) };
    slot.store(scale.to_bits(), Ordering::Relaxed);
    true
}

/// Parses `VICTORY_FFX_SCALE` (max scale, default 0.5) and
/// `VICTORY_FFX_SCALE_MIN` (min scale, default 0.38).
fn dynamic_bounds() -> (f32, f32) {
    let max = std::env::var("VICTORY_FFX_SCALE")
        .ok()
        .and_then(|s| {
            let head = s.split_once(',').map_or(s.as_str(), |(a, _)| a);
            head.trim().parse::<f32>().ok()
        })
        .filter(|s| s.is_finite() && *s > 0.0 && *s <= 1.0)
        .unwrap_or(DEFAULT_SCALE_MAX);
    let min = std::env::var("VICTORY_FFX_SCALE_MIN")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|s| s.is_finite() && *s > 0.0 && *s <= 1.0)
        .unwrap_or(DEFAULT_SCALE_MIN)
        .clamp(ABS_SCALE_MIN.min(max), max);
    (min, max)
}

/// Pure hysteresis controller for dynamic FSR render scale.
struct DynamicController {
    target_ms: f64,
    scale_min: f32,
    scale_max: f32,
    current_scale: f32,
    ema_ms: Option<f64>,
    wait_ema_ms: Option<f64>,
    frames_seen: u64,
    frames_since_step: u32,
    adjustments: u64,
    max_render: [u32; 2],
    max_output: [u32; 2],
    first_render: Option<[u32; 2]>,
    title_followed: bool,
}

impl DynamicController {
    /// Smoothing factor (~6–7 frames time constant).
    const EMA_ALPHA: f64 = 0.15;
    /// Deadband around `target_ms`: no scale change in `[0.88 * target, 1.05 * target]`.
    const HIGH_RATIO: f64 = 1.05;
    const LOW_RATIO: f64 = 0.88;
    /// Frames after context creation before adjusting (keeps startup at `scale_max`).
    const WARMUP_FRAMES: u64 = 12;
    /// Minimum frames between consecutive scale adjustments (prevents pumping).
    const STEP_COOLDOWN: u32 = 6;
    /// Asymmetric steps: drop faster than we raise.
    const MIN_STEP_DOWN: f32 = 0.005;
    const MAX_STEP_DOWN: f32 = 0.02;
    const STEP_UP: f32 = 0.005;
    /// Clamp individual frame samples so streaming spikes do not skew the EMA.
    const MAX_SAMPLE_MS: f64 = 250.0;

    fn new(target_ms: f64, scale_min: f32, scale_max: f32) -> Self {
        Self {
            target_ms,
            scale_min,
            scale_max,
            current_scale: scale_max,
            ema_ms: None,
            wait_ema_ms: None,
            frames_seen: 0,
            frames_since_step: 0,
            adjustments: 0,
            max_render: [0, 0],
            max_output: [0, 0],
            first_render: None,
            title_followed: false,
        }
    }

    fn on_context_created(&mut self, max_render: [u32; 2], max_output: [u32; 2]) {
        self.max_render = max_render;
        self.max_output = max_output;
        self.frames_seen = 0;
        self.frames_since_step = 0;
    }

    fn quantize_scale(&self, scale: f32) -> f32 {
        let clamped = scale.clamp(self.scale_min, self.scale_max);
        if clamped <= self.scale_min + 1e-4 {
            return self.scale_min;
        }
        if clamped >= self.scale_max - 1e-4 {
            return self.scale_max;
        }
        let out_w = self.max_output[0];
        if out_w >= 64 {
            let aligned_w = ((clamped * out_w as f32) / 8.0).round() * 8.0;
            (aligned_w / out_w as f32).clamp(self.scale_min, self.scale_max)
        } else {
            clamped
        }
    }

    /// Feeds one frame's GPU time sample and returns `Some(new_scale)` when the
    /// render scale should be updated.
    fn on_frame(&mut self, gpu_ms: f64, wait_ms: f64, render_size: [u32; 2]) -> Option<f32> {
        if render_size[0] > 0 && render_size[1] > 0 {
            match self.first_render {
                None => self.first_render = Some(render_size),
                Some(first) if render_size != first => self.title_followed = true,
                _ => {}
            }
        }
        if !gpu_ms.is_finite() || gpu_ms < 0.0 {
            return None;
        }
        let sample = gpu_ms.clamp(0.0, Self::MAX_SAMPLE_MS);
        self.ema_ms = Some(match self.ema_ms {
            Some(prev) => prev * (1.0 - Self::EMA_ALPHA) + sample * Self::EMA_ALPHA,
            None => sample,
        });
        if wait_ms.is_finite() && wait_ms >= 0.0 {
            let w = wait_ms.clamp(0.0, Self::MAX_SAMPLE_MS);
            self.wait_ema_ms = Some(match self.wait_ema_ms {
                Some(prev) => prev * (1.0 - Self::EMA_ALPHA) + w * Self::EMA_ALPHA,
                None => w,
            });
        }
        self.frames_seen += 1;
        self.frames_since_step = self.frames_since_step.saturating_add(1);
        if self.frames_seen <= Self::WARMUP_FRAMES || self.frames_since_step < Self::STEP_COOLDOWN {
            return None;
        }
        let ema = self.ema_ms?;
        let high_ms = self.target_ms * Self::HIGH_RATIO;
        let low_ms = self.target_ms * Self::LOW_RATIO;
        let candidate = if ema > high_ms && self.current_scale > self.scale_min {
            let ideal = self.current_scale * ((self.target_ms * 0.97) / ema).sqrt() as f32;
            let step = (self.current_scale - ideal).clamp(Self::MIN_STEP_DOWN, Self::MAX_STEP_DOWN);
            self.quantize_scale(self.current_scale - step)
        } else if ema < low_ms && self.current_scale < self.scale_max {
            self.quantize_scale(self.current_scale + Self::STEP_UP)
        } else {
            return None;
        };
        if (candidate - self.current_scale).abs() >= 1e-4 {
            self.current_scale = candidate;
            self.frames_since_step = 0;
            self.adjustments += 1;
            Some(candidate)
        } else {
            None
        }
    }
}

struct DynamicState {
    ctrl: DynamicController,
    last_sample: Option<(Instant, u64, u64)>,
    last_log: Instant,
}

fn dynamic_state() -> Option<&'static Mutex<DynamicState>> {
    static STATE: OnceLock<Option<Mutex<DynamicState>>> = OnceLock::new();
    STATE
        .get_or_init(|| {
            let target_ms = dynamic_target_ms()?;
            let (scale_min, scale_max) = dynamic_bounds();
            tracing::info!(
                target_ms = format_args!("{target_ms:.1}"),
                scale_min = format_args!("{scale_min:.3}"),
                scale_max = format_args!("{scale_max:.3}"),
                slot_wired = SCALE_SLOT.get().is_some(),
                "FidelityFX: dynamic resolution scaling enabled (VICTORY_FFX_DYNAMIC)"
            );
            Some(Mutex::new(DynamicState {
                ctrl: DynamicController::new(target_ms, scale_min, scale_max),
                last_sample: None,
                last_log: Instant::now(),
            }))
        })
        .as_ref()
}

fn dynamic_on_dispatch(render_size: [u32; 2]) {
    let Some(lock) = dynamic_state() else {
        return;
    };
    let now = Instant::now();
    let (idle_us, wait_us, signals) = victory_gpu::external::card_totals_us();
    let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
    if !live_dynamic(&mut state.ctrl) {
        state.last_sample = None;
        return;
    }
    let sample = state.last_sample.replace((now, idle_us, wait_us));
    if let Some((prev_now, prev_idle, prev_wait)) = sample {
        let dt_wall_us = now.duration_since(prev_now).as_micros() as u64;
        // Skip pauses / loading stalls longer than 1 s.
        if dt_wall_us > 0 && dt_wall_us < 1_000_000 {
            let dt_idle_us = idle_us.saturating_sub(prev_idle);
            let dt_wait_us = wait_us.saturating_sub(prev_wait);
            let wait_ms = dt_wait_us as f64 / 1000.0;
            let gpu_ms = if signals > 0 {
                (dt_wall_us.saturating_sub(dt_idle_us) as f64 / 1000.0).max(wait_ms)
            } else {
                wait_ms
            };
            if let Some(new_scale) = state.ctrl.on_frame(gpu_ms, wait_ms, render_size) {
                write_scale_slot(new_scale);
            }
        }
    }
    if state.last_log.elapsed() >= Duration::from_secs(5) {
        state.last_log = now;
        let c = &state.ctrl;
        tracing::info!(
            scale = format_args!("{:.3}", c.current_scale),
            min = format_args!("{:.3}", c.scale_min),
            max = format_args!("{:.3}", c.scale_max),
            gpu_ms = format_args!("{:.1}", c.ema_ms.unwrap_or(0.0)),
            wait_ms = format_args!("{:.1}", c.wait_ema_ms.unwrap_or(0.0)),
            target_ms = format_args!("{:.1}", c.target_ms),
            render = ?render_size,
            max_render = ?c.max_render,
            steps = c.adjustments,
            slot_wired = SCALE_SLOT.get().is_some(),
            title_followed = c.title_followed,
            "FidelityFX dynamic scale"
        );
    }
}

macro_rules! once {
    ($level:ident, $($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::$level!($($arg)*));
    }};
}

/// Serves the title's FidelityFX calls with the PC upscaler in `dir`
/// (`VICTORY_FIDELITYFX_DIR`) if it offers one for the host GPU; else the
/// title's own DLLs run. Call once, before the import registry is first
/// used: it holds this module only when serving.
pub fn use_host_dir(dir: Option<&Path>) {
    let Some(dir) = dir else {
        tracing::info!(
            "no FidelityFX folder (VICTORY_FIDELITYFX_DIR): a title's own FidelityFX DLLs run"
        );
        return;
    };
    match load(dir) {
        Ok((host, versions)) => {
            tracing::info!(folder = %dir.display(), versions, "FidelityFX: the host's upscaler serves the title's");
            let _ = HOST.set(host);
        }
        Err(why) => tracing::warn!(
            folder = %dir.display(),
            "FidelityFX: {why}; the title's own DLLs run"
        ),
    }
}

/// Whether the host's FidelityFX serves the title's.
pub(crate) fn serving() -> bool {
    HOST.get().is_some()
}

fn load(dir: &Path) -> Result<(Host, String), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LoadLibraryExW,
    };
    // The search flags below want a full path.
    let path = std::path::absolute(dir.join(DLL)).map_err(|e| e.to_string())?;
    if !path.is_file() {
        return Err(format!("no {DLL} there"));
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    // Its own folder and System32 only, never the current folder.
    let search = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
    // SAFETY: a NUL-terminated path to the user's FidelityFX build.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), null_mut(), search) };
    if module.is_null() {
        return Err(format!(
            "{DLL} did not load ({})",
            std::io::Error::last_os_error()
        ));
    }
    let find = |name: &CStr| {
        // SAFETY: a loaded module and a NUL-terminated name.
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
            .map(|f| f as *const c_void)
            .ok_or_else(|| format!("{DLL} has no {}", name.to_string_lossy()))
    };
    let device = victory_gpu::external::host_device().ok_or("there is no host GPU")?;
    // SAFETY: the API's exports, with its signatures.
    let host = unsafe {
        Host {
            create: std::mem::transmute::<*const c_void, Create>(find(c"ffxCreateContext")?),
            destroy: std::mem::transmute::<*const c_void, Destroy>(find(c"ffxDestroyContext")?),
            configure: std::mem::transmute::<*const c_void, Call>(find(c"ffxConfigure")?),
            query: std::mem::transmute::<*const c_void, Call>(find(c"ffxQuery")?),
            dispatch: std::mem::transmute::<*const c_void, Call>(find(c"ffxDispatch")?),
            device,
        }
    };
    let versions = versions(&host)?;
    Ok((host, versions))
}

/// The upscaler versions the host DLL offers for the host GPU, by name; an
/// error if none.
fn versions(host: &Host) -> Result<String, String> {
    let mut count = 0u64;
    let mut query = GetVersions {
        header: Header {
            kind: GET_VERSIONS,
            next: null_mut(),
        },
        create_kind: CREATE_UPSCALE,
        device: host.device.as_raw(),
        count: &mut count,
        ids: null_mut(),
        names: null_mut(),
    };
    // SAFETY: a version query naming the host device.
    let code = unsafe { (host.query)(null_mut(), &mut query.header) };
    if code != OK || count == 0 {
        return Err(format!(
            "it offers no upscaler for the host GPU (code {code})"
        ));
    }
    let mut names = vec![null::<c_char>(); count as usize];
    query.names = names.as_mut_ptr();
    // SAFETY: as above, with room for `count` names.
    unsafe { (host.query)(null_mut(), &mut query.header) };
    Ok(names
        .iter()
        .filter(|n| !n.is_null())
        // SAFETY: NUL-terminated names the DLL keeps.
        .map(|&n| unsafe { CStr::from_ptr(n) }.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(", "))
}

/// Puts the host backend after the title's upscaler description, in place
/// of the console's backend that follows it (where the title puts it);
/// anywhere else the console's is left for the host DLL to ignore, ours
/// coming first.
fn link(upscale: &mut CreateUpscale, backend: &mut Backend) {
    let after = upscale.header.next;
    // SAFETY: the title's chain.
    backend.header.next = match unsafe { after.as_ref() } {
        Some(h) if h.kind == CONSOLE_BACKEND => h.next,
        _ => after,
    };
    backend.header.kind = HOST_BACKEND;
    upscale.header.next = &raw mut backend.header;
}

/// A FidelityFX resource state as the host resource state it means (as the
/// PC backend reads it); `None` for generic read and indirect arguments,
/// which a host texture cannot be in.
fn host_state(state: u32) -> Option<u32> {
    Some(match state {
        0x1 | 0x80 => 0,  // common, present
        0x2 => 0x8,       // unordered access
        0x4 => 0x40,      // compute read
        0x8 => 0x80,      // pixel read
        0xc => 0xc0,      // both
        0x10 => 0x800,    // copy source
        0x20 => 0x400,    // copy destination
        0x100 => 0x4,     // render target
        0x200 => 0x10,    // depth write
        _ => return None, // generic read, indirect arguments
    })
}

extern "C" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *mut Header,
    memory: *const c_void,
) -> u32 {
    let Some(host) = HOST.get() else {
        return NO_PROVIDER;
    };
    if context.is_null() || desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a chain of descriptions.
    let kind = unsafe { (*desc).kind };
    if kind != CREATE_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: only the upscaler runs on the host; another effect was refused"
        );
        return NO_PROVIDER;
    }
    // SAFETY: an upscaler description, by its type.
    let mut upscale = unsafe { desc.cast::<CreateUpscale>().read() };
    if let Some(lock) = dynamic_state() {
        upscale.flags |= UPSCALE_ENABLE_DYNAMIC_RESOLUTION;
        let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
        state
            .ctrl
            .on_context_created(upscale.max_render, upscale.max_output);
    }
    let mut backend = Backend {
        header: Header {
            kind: HOST_BACKEND,
            next: null_mut(),
        },
        device: host.device.as_raw(),
    };
    link(&mut upscale, &mut backend);
    // SAFETY: a PC chain; the title's later descriptions live for the call.
    let code = unsafe { (host.create)(context, &mut upscale.header, memory) };
    if code != OK {
        tracing::error!(code, render = ?upscale.max_render, output = ?upscale.max_output, flags = format_args!("{:#x}", upscale.flags), "FidelityFX: the host refused the title's upscaler");
        return code;
    }
    let mut version = ProviderVersion {
        header: Header {
            kind: GET_PROVIDER_VERSION,
            next: null_mut(),
        },
        id: 0,
        name: null(),
    };
    // SAFETY: a provider query on the context just made.
    unsafe { (host.query)(context, &mut version.header) };
    // SAFETY: a NUL-terminated name the DLL keeps, if any.
    let name = (!version.name.is_null())
        .then(|| unsafe { CStr::from_ptr(version.name) }.to_string_lossy());
    tracing::info!(
        provider = name.as_deref().unwrap_or("?"),
        id = format_args!("{:#x}", version.id),
        render = ?upscale.max_render,
        output = ?upscale.max_output,
        flags = format_args!("{:#x}", upscale.flags),
        "FidelityFX: upscaler context on the host"
    );
    OK
}

extern "C" fn ffxDestroyContext(context: *mut *mut c_void, memory: *const c_void) -> u32 {
    // SAFETY: the title passes its context.
    HOST.get()
        .map_or(NO_PROVIDER, |h| unsafe { (h.destroy)(context, memory) })
}

/// Settings (key-value pairs, debug messages) name nothing console-specific.
extern "C" fn ffxConfigure(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    // SAFETY: the title passes its context and descriptions.
    HOST.get()
        .map_or(NO_PROVIDER, |h| unsafe { (h.configure)(context, desc) })
}

extern "C" fn ffxQuery(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(host) = HOST.get() else {
        return NO_PROVIDER;
    };
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    // These carry the console device.
    if kind == GET_VERSIONS || kind == MEMORY_USAGE_WITH_DEVICE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: a query naming a device was refused"
        );
        return PARAMETER;
    }
    // SAFETY: as above.
    unsafe { (host.query)(context, desc) }
}

extern "C" fn ffxDispatch(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(host) = HOST.get() else {
        return NO_PROVIDER;
    };
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    if kind != DISPATCH_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: only upscaling runs on the host; a dispatch was refused"
        );
        return UNKNOWN_TYPE;
    }
    // SAFETY: an upscaling, by its type.
    let mut dispatch = unsafe { desc.cast::<DispatchUpscale>().read() };
    // `renderSize` sits at `rest[4..6]` in `ffxDispatchDescUpscale`.
    dynamic_on_dispatch([dispatch.rest[4], dispatch.rest[5]]);
    let list = dispatch.list.cast::<Object>().cast_const();
    // Alive for the call (the list holds them too).
    let mut textures = Vec::new();
    for r in dispatch
        .resources
        .iter_mut()
        .filter(|r| !r.resource.is_null())
    {
        let wanted = host_state(r.state);
        let texture = wanted.and_then(|w| {
            victory_gpu::external::host_texture(list, r.resource.cast::<Object>(), w)
                .filter(|&(_, state)| state == w)
        });
        let Some((texture, _)) = texture else {
            once!(
                warn,
                state = r.state,
                "FidelityFX: a resource has no host texture in its state; the upscaling did not run"
            );
            return PARAMETER;
        };
        r.resource = texture.as_raw();
        textures.push(texture);
    }
    once!(info, formats = ?textures.iter().map(|t| {
        // SAFETY: a live texture.
        unsafe { t.GetDesc() }.Format.0
    }).collect::<Vec<_>>(), "FidelityFX: first upscaling, host formats of its textures");
    victory_gpu::external::record(list, |host_list| {
        dispatch.list = host_list.as_raw();
        // SAFETY: the PC dispatch on host objects; the context is the host's.
        unsafe { (host.dispatch)(context, &raw mut dispatch.header) }
    })
    .unwrap_or(PARAMETER)
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "fidelityfx",
        dlls: &["amd_fidelityfx_loader_xbox.dll"],
        exports: vec![
            hle_export!(ffxConfigure),
            hle_export!(ffxCreateContext),
            hle_export!(ffxDestroyContext),
            hle_export!(ffxDispatch),
            hle_export!(ffxQuery),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(kind: u64, next: *mut Header) -> Header {
        Header { kind, next }
    }

    /// The title's chain: upscaler, console backend, version.
    #[test]
    fn the_console_backend_becomes_the_hosts() {
        let mut version = header(0x1_000b, null_mut());
        let mut console = Backend {
            header: header(CONSOLE_BACKEND, &raw mut version),
            device: 0x1234 as *mut c_void,
        };
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut console.header),
            flags: 0x29,
            max_render: [1280, 720],
            max_output: [1920, 1080],
            message: 0,
        };
        let mut backend = Backend {
            header: header(0, null_mut()),
            device: 0x5678 as *mut c_void,
        };
        link(&mut upscale, &mut backend);
        assert_eq!(upscale.header.next, &raw mut backend.header);
        assert_eq!(backend.header.kind, HOST_BACKEND);
        assert_eq!(backend.header.next, &raw mut version);
        assert_eq!(backend.device, 0x5678 as *mut c_void);
        // Without a console backend right after it, ours goes in front.
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut version),
            ..upscale
        };
        link(&mut upscale, &mut backend);
        assert_eq!(backend.header.next, &raw mut version);
    }

    /// Without the PC build the title's own DLL serves its imports.
    #[test]
    fn without_the_dll_the_titles_own_runs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load(dir.path()).err().unwrap().contains(DLL));
        assert!(!serving());
        let symbol = victory_pe::ImportSymbol::Name {
            hint: 0,
            name: "ffxCreateContext".into(),
        };
        let resolver = crate::HleResolver::new(crate::HostPolicy::none());
        assert_eq!(
            resolver.classify("amd_fidelityfx_loader_xbox.dll", &symbol),
            None
        );
    }

    #[test]
    fn resource_states_read_as_the_pc_backend_reads_them() {
        assert_eq!(host_state(0xc), Some(0xc0));
        assert_eq!(host_state(0x2), Some(0x8));
        assert_eq!(host_state(0x200), Some(0x10));
        assert_eq!(host_state(0x14), None);
    }

    #[test]
    fn dynamic_controller_holds_steady_in_deadband() {
        let mut ctrl = DynamicController::new(33.0, 0.38, 0.5);
        ctrl.on_context_created([1920, 1080], [3840, 2160]);
        // Oscillating inside [29.04, 34.65] ms must never pump the scale.
        for i in 0..60 {
            let gpu_ms = if i % 2 == 0 { 30.0 } else { 34.0 };
            assert_eq!(ctrl.on_frame(gpu_ms, gpu_ms, [1920, 1080]), None);
        }
        assert_eq!(ctrl.current_scale, 0.5);
        assert_eq!(ctrl.adjustments, 0);
    }

    #[test]
    fn dynamic_controller_drops_to_min_and_recovers_slowly_to_max() {
        let mut ctrl = DynamicController::new(33.0, 0.38, 0.5);
        ctrl.on_context_created([1920, 1080], [3840, 2160]);
        // Warmup keeps scale at max even when overloaded.
        for _ in 0..DynamicController::WARMUP_FRAMES {
            assert_eq!(ctrl.on_frame(70.0, 50.0, [1920, 1080]), None);
        }
        assert_eq!(ctrl.current_scale, 0.5);
        // Sustained overload drops scale smoothly down to `scale_min`.
        let mut drops = 0;
        for _ in 0..120 {
            if let Some(s) = ctrl.on_frame(70.0, 50.0, [1920, 1080]) {
                assert!(s >= 0.38 && s <= 0.5);
                drops += 1;
            }
        }
        assert!(drops > 0);
        assert!((ctrl.current_scale - 0.38).abs() < 1e-4);
        assert!(!ctrl.title_followed);
        // Simulate title following with a smaller render size and GPU time falling below deadband.
        let mut raises = 0;
        for _ in 0..300 {
            if let Some(s) = ctrl.on_frame(20.0, 10.0, [1464, 824]) {
                assert!(s >= 0.38 && s <= 0.5);
                raises += 1;
            }
        }
        assert!(ctrl.title_followed);
        assert!(raises > drops, "scale should rise slower than it drops");
        assert!((ctrl.current_scale - 0.5).abs() < 1e-4);
    }
}
