//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! The title creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    container: Vec<u8>,
    /// The program that resolves the indirect diffuse light
    /// (`crate::ambient::PROGRAM`), which the ambient light follows.
    #[cfg(windows)]
    pub(crate) resolves_light: bool,
    #[cfg(windows)]
    translated: OnceLock<Option<Translated>>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// For the memory lines: containers kept and their bytes, translations
/// kept and the bytes of their host code. None of them is ever let go.
static KEPT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What the shader table and the pipeline library keep in memory:
/// containers and their bytes, translations and their bytes, the bytes of
/// the library's file (kept whole while the library lives) and what its
/// last save wrote.
#[cfg(windows)]
pub(crate) fn kept() -> [u64; 7] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        KEPT[0].load(Relaxed),
        KEPT[1].load(Relaxed),
        KEPT[2].load(Relaxed),
        KEPT[3].load(Relaxed),
        library::BLOB.load(Relaxed),
        library::SAVED.load(Relaxed),
        library::TRIMMED_BYTES.load(Relaxed),
    ]
}

/// The shared container for `bytes`.
pub(crate) fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(*full.as_bytes())
        .or_insert_with(|| {
            KEPT[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            KEPT[1].fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
            Arc::new(Shader {
                stage,
                hash: full.to_hex()[..16].to_owned(),
                container: bytes.to_vec(),
                #[cfg(windows)]
                resolves_light: bytes
                    .windows(crate::ambient::PROGRAM.len())
                    .any(|w| w == crate::ambient::PROGRAM),
                #[cfg(windows)]
                translated: OnceLock::new(),
            })
        })
        .clone()
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    pub(crate) used: Vec<victory_shader::UsedResource>,
    /// Whether the shader's code can store to buffer memory ([`stores`]);
    /// true when its code is not known.
    pub(crate) stores: bool,
}

#[cfg(windows)]
impl Shader {
    /// The translation, made on first use; a shader that does not
    /// translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        self.translated
            .get_or_init(|| {
                let done = translate(&self.container);
                // `VICTORY_SHADER_SAVE=<dir>`: each shader as the title gave it (`.cdxil`) and as
                // translated (`.dxil`), by the hash crash reports name it with
                if let Some(dir) = std::env::var_os("VICTORY_SHADER_SAVE") {
                    let dir = std::path::PathBuf::from(dir);
                    let _ = std::fs::create_dir_all(&dir);
                    let stem = format!("{}.{}", self.hash, self.stage);
                    let _ = std::fs::write(dir.join(format!("{stem}.cdxil")), &self.container);
                    if let Ok(t) = &done {
                        let _ = std::fs::write(dir.join(format!("{stem}.dxil")), &t.host.dxil);
                    }
                }
                if let Ok(t) = &done {
                    KEPT[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    KEPT[3].fetch_add(t.host.dxil.len() as u64, std::sync::atomic::Ordering::Relaxed);
                }
                match &done {
                    Ok(t) => tracing::info!(
                        stage = self.stage,
                        hash = %self.hash,
                        user_data = t.host.user_data.len(),
                        samplers = t.host.samplers.len(),
                        "shader translated"
                    ),
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok()
            })
            .as_ref()
    }
}

#[cfg(windows)]
fn translate(container: &[u8]) -> Result<Translated, String> {
    use victory_shader::{Container, FourCC, HwStage};
    // The translation depends on it (a view index past a capped heap).
    crate::descriptor::host::probe_view_limit();
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let programs = match c.part(FourCC::DXIL) {
        Some(part) => victory_shader::native_programs(part.data).map_err(|e| e.to_string())?,
        None => Vec::new(),
    };
    let host = crate::shader_cache::host_shader(container)?;
    let stores = programs.is_empty() || programs.iter().any(stores);
    let used = programs
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
            )
        })
        .map(|p| p.used)
        .unwrap_or_default();
    Ok(Translated { host, used, stores })
}

/// The vertex and geometry stages of a pipeline with a geometry shader: the
/// console links the ES program of the vertex container with the geometry
/// container's GS program into one merged stage, which translates as a host
/// vertex shader and a host geometry shader
/// ([`victory_shader::emit::translate_es_gs`]). Two shaders of their own
/// (the title's vertex container also translates on its own, for the
/// pipelines without a geometry stage), made once per pair.
#[cfg(windows)]
pub(crate) fn merged_halves(vs: &Arc<Shader>, gs: &Arc<Shader>) -> Option<(Arc<Shader>, Arc<Shader>)> {
    type Pairs = HashMap<(String, String), Option<(Arc<Shader>, Arc<Shader>)>>;
    static PAIRS: Mutex<Option<Pairs>> = Mutex::new(None);
    let key = (vs.hash.clone(), gs.hash.clone());
    if let Some(found) = PAIRS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).get(&key) {
        return found.clone();
    }
    let made = (|| -> Result<(Arc<Shader>, Arc<Shader>), String> {
        use victory_shader::{Container, FourCC, HwStage};
        crate::descriptor::host::probe_view_limit();
        let (host_vs, host_gs) =
            victory_shader::emit::translate_es_gs(&vs.container, &gs.container).map_err(|e| e.to_string())?;
        let program = |bytes: &[u8], stage| {
            Container::parse(bytes)
                .ok()
                .and_then(|c| c.part(FourCC::DXIL).and_then(|part| victory_shader::native_programs(part.data).ok()))
                .and_then(|programs| programs.into_iter().find(|p| p.stage == stage))
        };
        let half = |stage: &'static str, of: &Shader, host, program: Option<victory_shader::NativeProgram>| {
            KEPT[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Arc::new(Shader {
                stage,
                hash: format!("{}.{stage}", of.hash),
                container: of.container.clone(),
                resolves_light: false,
                translated: OnceLock::from(Some(Translated {
                    host,
                    stores: program.as_ref().is_none_or(stores),
                    used: program.map(|p| p.used).unwrap_or_default(),
                })),
            })
        };
        Ok((
            half("vertex", vs, host_vs, program(&vs.container, HwStage::Es)),
            half("geometry", gs, host_gs, program(&gs.container, HwStage::Gs)),
        ))
    })();
    match &made {
        Ok(_) => tracing::info!(vertex = %vs.hash, geometry = %gs.hash, "merged ES+GS stage translated as vertex and geometry shaders"),
        Err(e) => tracing::warn!(
            vertex = %vs.hash,
            geometry = %gs.hash,
            error = %e,
            "merged ES+GS stage does not translate; its pipelines' draws are dropped"
        ),
    }
    let made = made.ok();
    PAIRS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(key, made.clone());
    made
}

/// The compute pipelines of the console's ray-tracing structure builder
/// (`XboxBvhBuilder.dll`; its programs are the ones with a spilled
/// user-data table) run **all or none**. Its stages feed each other
/// through memory: a stage that runs on what a dropped stage never wrote
/// computes its store offsets from garbage. The structures lie in plain
/// title memory, between the title's own objects, and a root view has no
/// size, so those stores land on the neighbours (run 98, with 14 of 35
/// stages translated: the title died within a second of the first builder
/// submission, on pointers that had become bounding-box bit patterns).
///
/// So a builder dispatch runs only while every builder pipeline the title
/// has created has a host pipeline; the first builder dispatch makes them
/// all on the spot instead of over the next frames (a stage "still being
/// made" would be a dropped stage). One that cannot be made turns the
/// builder off for the rest of the run.
///
/// shortcut: and only with `VICTORY_BVH_BUILDER=1`, until a run has shown
/// the whole builder and the host side of title memory by address
/// (`memory.rs`) working on a card; nothing reads the structures before
/// the programs that walk them translate.
#[cfg(windows)]
struct Builder {
    pipelines: Vec<Arc<make::Pending>>,
    /// Every one of `pipelines` has a host pipeline.
    whole: bool,
    /// One could not be made.
    off: bool,
}

#[cfg(windows)]
static BUILDER: Mutex<Builder> = Mutex::new(Builder {
    pipelines: Vec::new(),
    whole: false,
    off: false,
});

/// Whether `shaders` are a pipeline of the structure builder.
#[cfg(windows)]
fn of_builder(shaders: &[Arc<Shader>]) -> bool {
    use victory_shader::{Container, FourCC};
    let [shader] = shaders else { return false };
    if shader.stage != "compute" {
        return false;
    }
    let Ok(c) = Container::parse(&shader.container) else {
        return false;
    };
    c.part(FourCC::DXIL)
        .and_then(|part| victory_shader::native_programs(part.data).ok())
        .is_some_and(|programs| programs.iter().any(victory_shader::spill::is_spilled))
}

/// `VICTORY_BVH_QUARANTINE=1`: what the structure builder's stages store
/// stays in the host's copies and is not copied back into title memory. A
/// check for a first run of the whole builder: a stage that stores where
/// it should not can then not reach the title's own objects (run 98), and
/// the programs that walk the structures read the host's copies anyway.
/// shortcut: not a way to run: the title also reads results back (sizes,
/// counters) and gets none, and a copy's pages are overwritten from title
/// memory when the title writes them.
#[cfg(windows)]
pub(crate) fn builder_quarantined(this: This) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    is_builder(this) && *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_QUARANTINE").is_some_and(|v| v == "1"))
}

/// Whether the title's pipeline `this` is one of the structure builder's.
#[cfg(windows)]
pub(crate) fn is_builder(this: This) -> bool {
    !this.is_null() && pipeline(this).builder
}

/// A stage of the builder was dropped after all: none runs from here on.
#[cfg(windows)]
pub(crate) fn builder_off(why: &'static str) {
    let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
    if !b.off {
        b.off = true;
        tracing::warn!(why, "the ray-tracing structure builder is turned off for the rest of the run: its stages run all or none");
    }
}

/// At a dispatch of a builder pipeline: whether the builder runs (see
/// [`Builder`]).
#[cfg(windows)]
pub(crate) fn builder_runs() -> bool {
    static ALLOWED: OnceLock<bool> = OnceLock::new();
    if !*ALLOWED.get_or_init(|| std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1")) {
        return false;
    }
    let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
    if b.off {
        return false;
    }
    if b.whole {
        return true;
    }
    let missing = b.pipelines.iter().filter(|p| !make::now(p)).count();
    if missing > 0 {
        b.off = true;
        tracing::warn!(
            missing,
            of = b.pipelines.len(),
            "the ray-tracing structure builder stays off: not all of its programs have a host pipeline, and its stages run all or none"
        );
        return false;
    }
    tracing::info!(
        pipelines = b.pipelines.len(),
        "the ray-tracing structure builder runs: all of its programs have host pipelines"
    );
    b.whole = true;
    true
}

/// Whether a console program can store to buffer memory: it has a buffer,
/// typed-buffer, scalar-memory, flat or global store or atomic, or code
/// that is not all known (an instruction of those kinds outside our table,
/// an indirect jump, code that does not decode). Stores to textures
/// (image instructions) and to the program's own scratch memory do not
/// count: neither reaches a buffer's host copy. After upstream #413,
/// which leaves scalar-memory stores out and counts image stores.
#[cfg(windows)]
fn stores(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        let name = i.name();
        match i.encoding {
            Encoding::Mubuf
            | Encoding::Mtbuf
            | Encoding::Smem
            | Encoding::Flat
            | Encoding::Global => name.is_none_or(|n| {
                (n.contains("store") || n.contains("atomic")) && !n.contains("scratch")
            }),
            Encoding::Sop1 => name.is_none_or(|n| matches!(n, "s_setpc_b64" | "s_swappc_b64")),
            _ => false,
        }
    })
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
    /// One of the ray-tracing structure builder's ([`Builder`]).
    #[cfg(windows)]
    builder: bool,
    /// A graphics pipeline of method 0x1c8 ([`create_from_words`]): its
    /// host pipelines are made per set of targets drawn into, not `host`.
    #[cfg(windows)]
    precompiled: Option<Box<Precompiled>>,
}

/// A graphics pipeline described by its precompiled parts: the state its
/// registers section holds, and a host pipeline per set of targets and
/// topology type it is drawn with (the description names neither).
#[cfg(windows)]
struct Precompiled {
    registers: crate::precompiled::GraphicsRegisters,
    /// The input layout of the info section.
    inputs: Vec<crate::precompiled::InputElement>,
    variants: crate::precompiled::Variants<make::Pending>,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.container.len(), hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline drawn into `targets` (which only
/// precompiled graphics pipelines are made for, see [`Precompiled`]);
/// `None` if one of its shaders does not translate, the host refuses it,
/// or it is still being made (its first use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(
    this: This,
    targets: &crate::precompiled::Targets,
) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    let p = pipeline(this);
    let Some(pre) = &p.precompiled else {
        return make::ready(&p.host);
    };
    let (variant, new) = pre.variants.get(targets, || {
        let state = host::GraphicsState::precompiled(&pre.registers, &pre.inputs, targets);
        make::Pending::new(p.shaders.clone(), Some(Box::new(state)))
    });
    if new {
        tracing::info!(pipeline = ?this, shaders = %make::names(&variant), ?targets, "precompiled graphics pipeline: a host pipeline for these targets");
    }
    let made = make::ready(&variant)?;
    // SAFETY: `pre.variants` keeps the variant while the pipeline lives,
    // as `p.host` is kept for the others.
    Some(unsafe { &*std::ptr::from_ref(made) })
}

/// The host pipeline of a title pipeline with its vertex shader translated
/// for instances drawn flat (`victory_shader::emit::with_flat_instances`,
/// four vertices an instance), made on first use and kept; `None` if it
/// cannot be made, or its root signature is not the pipeline's own (the
/// draw binds the pipeline's roots, then this one's state).
#[cfg(windows)]
pub(crate) fn flat_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    type Made = HashMap<usize, (Arc<make::Pending>, Option<&'static host::HostPipeline>)>;
    static MADE: Mutex<Option<Made>> = Mutex::new(None);
    let p = pipeline(this);
    // A precompiled pipeline's state is per set of targets: none drawn flat.
    if p.precompiled.is_some() {
        return None;
    }
    let base = host_pipeline(this, &crate::precompiled::Targets::default())?;
    let key = Arc::as_ptr(&p.host) as usize;
    if let Some((_, made)) = MADE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).get(&key) {
        return *made;
    }
    let made = (|| {
        let vs = p.shaders.iter().find(|s| s.stage == "vertex")?;
        crate::descriptor::host::probe_view_limit();
        let host = victory_shader::emit::with_flat_instances(2, || victory_shader::emit::translate(&vs.container));
        let host = match host {
            Ok(host) => host,
            Err(e) => {
                tracing::warn!(hash = %vs.hash, error = %e, "vertex shader does not translate for flat instances");
                return None;
            }
        };
        let base_vs = vs.translated()?;
        let flat = Arc::new(Shader {
            stage: "vertex",
            hash: format!("{}.flat", vs.hash),
            container: vs.container.clone(),
            resolves_light: false,
            translated: OnceLock::from(Some(Translated { host, used: base_vs.used.clone(), stores: base_vs.stores })),
        });
        let shaders: Vec<Arc<Shader>> =
            p.shaders.iter().map(|s| if s.stage == "vertex" { flat.clone() } else { s.clone() }).collect();
        let made = victory_shader::emit::with_flat_rects(|| host::HostPipeline::new(&shaders, p.host.graphics.as_deref()))?;
        if made.root != base.root {
            tracing::warn!(?shaders, "flat-instance pipeline has another root signature; not used");
            return None;
        }
        // What the check of a wrong image needs: that the vertex program
        // differs from the title's, and whether the pixel shader numbers
        // its primitives (then 0 for each rectangle, `with_flat_rects`).
        let ps_primitive = p.shaders.iter().find(|s| s.stage == "pixel").and_then(|s| s.translated()).is_some_and(|t| {
            victory_shader::Container::parse(&t.host.dxil)
                .ok()
                .and_then(|c| c.part(victory_shader::FourCC::ISG1).map(|part| part.data.windows(14).any(|w| w == b"SV_PrimitiveID")))
                .unwrap_or(false)
        });
        let flat_dxil = flat.translated().map_or(0, |t| t.host.dxil.len());
        let same = flat.translated().is_some_and(|t| t.host.dxil == base_vs.host.dxil);
        tracing::info!(
            ?shaders,
            base_vs_bytes = base_vs.host.dxil.len(),
            flat_vs_bytes = flat_dxil,
            same_vs = same,
            ps_primitive,
            rects = made.rects,
            "flat-instance pipeline made"
        );
        let made: &'static host::HostPipeline = Box::leak(Box::new(made));
        Some(made)
    })();
    MADE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(key, (p.host.clone(), made));
    made
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    let hashes = HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        parse_skip_hashes(&value)
    });
    if hashes.is_empty() {
        return None;
    }
    p.shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && skipped_shader(pipeline(this)).is_some()
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        let p = pipeline(this);
        skipped_shader(p)
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| match &p.precompiled {
                Some(pre) => pre.variants.all().iter().find_map(|v| make::missing(v)),
                None => make::missing(&p.host),
            })
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    crate::capture::op(0, || format!("DROPPED {why}"));
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops.get_or_insert_with(HashMap::new).entry(why).or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The reason a dispatch of the ray-tracing structure builder is dropped
/// with: on purpose, the builder is off (`VICTORY_BVH_BUILDER`).
#[cfg(windows)]
pub(crate) const BUILDER_OFF: &str = "the ray-tracing structure builder, off on purpose (VICTORY_BVH_BUILDER)";

/// Not a drop, counted with them: a draw run with a vertex buffer its
/// input layout reads unset or without a host copy (it reads zeros there).
#[cfg(windows)]
pub(crate) const DRAWN_UNBOUND: &str = "DRAWN all the same, a vertex buffer unset or without a host copy (reads zeros)";

/// Draws and dispatches dropped since the last [`drop_report`], by title
/// pipeline and reason: its shaders as named in the log, and the count.
#[cfg(windows)]
type Tally = HashMap<(usize, &'static str), (String, u64)>;
#[cfg(windows)]
static TALLY: Mutex<Option<Tally>> = Mutex::new(None);

/// Whether dropped draws are counted by pipeline and reported
/// (`VICTORY_DROP_REPORT=0`: not).
#[cfg(windows)]
fn drop_report_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_DROP_REPORT").as_deref() != Ok("0"))
}

/// Counts a draw or dispatch of `this` dropped for `why`; an empty `why`
/// is the pipeline's own: none set, not made yet, a shader that does not
/// translate, refused, or (with a host pipeline) something else the draw
/// needs.
#[cfg(windows)]
pub(crate) fn tally_drop(this: This, why: &'static str) {
    if !drop_report_on() {
        return;
    }
    let why = match why {
        "" if this.is_null() => "no pipeline set",
        "" => make::lacks(&pipeline(this).host).unwrap_or("it could not be prepared (no reason noted)"),
        why => why,
    };
    let mut tally = TALLY.lock().unwrap_or_else(|e| e.into_inner());
    tally
        .get_or_insert_with(HashMap::new)
        .entry((this as usize, why))
        .or_insert_with(|| {
            let shaders = if this.is_null() { String::new() } else { format!("{:?}", pipeline(this).shaders) };
            (shaders, 0)
        })
        .1 += 1;
}

/// Counts `frames` more frames for the dropped draws' report; with `log`,
/// logs the draws and dispatches dropped since the last report, per frame
/// by reason, then the 20 pipelines that dropped most.
#[cfg(windows)]
pub(crate) fn drop_report(frames: f64, log: bool) {
    static FRAMES: Mutex<f64> = Mutex::new(0.0);
    let frames = {
        let mut kept = FRAMES.lock().unwrap_or_else(|e| e.into_inner());
        *kept += frames;
        if !log {
            return;
        }
        std::mem::take(&mut *kept)
    };
    let Some(tally) = TALLY.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    if tally.is_empty() || frames <= 0.0 {
        return;
    }
    let mut reasons: HashMap<&'static str, u64> = HashMap::new();
    for (&(_, why), &(_, n)) in &tally {
        *reasons.entry(why).or_default() += n;
    }
    let mut reasons: Vec<_> = reasons.into_iter().collect();
    reasons.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let mut top: Vec<_> = tally.into_iter().map(|((_, why), (shaders, n))| (n, why, shaders)).collect();
    top.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
    let pipelines = top.len();
    let by_reason: Vec<String> =
        reasons.iter().map(|(why, n)| format!("{:.1} {why}", *n as f64 / frames)).collect();
    let by_pipeline: Vec<String> = top
        .iter()
        .take(20)
        .map(|(n, why, shaders)| format!("{:.1} {shaders} ({why})", *n as f64 / frames))
        .collect();
    tracing::info!(
        "dropped draws and dispatches per frame, by reason: {}; {pipelines} pipelines, the most: {}",
        by_reason.join("; "),
        by_pipeline.join("; ")
    );
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code.
    Some(shader(stage, unsafe {
        std::slice::from_raw_parts(code.code, code.size)
    }))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    #[cfg(windows)] precompiled: Option<Box<Precompiled>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    // A precompiled pipeline's host pipelines are its variants'; this one
    // stays empty (nothing to start early).
    #[cfg(windows)]
    let host = Arc::new(if precompiled.is_some() {
        make::Pending::new(Vec::new(), None)
    } else {
        make::Pending::new(shaders.clone(), graphics)
    });
    #[cfg(windows)]
    make::early(&host);
    #[cfg(windows)]
    let builder = of_builder(&shaders);
    #[cfg(windows)]
    if builder {
        // Another stage: the group is looked at again at its next dispatch.
        let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
        b.pipelines.push(host.clone());
        b.whole = false;
    }
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        #[cfg(windows)]
        builder,
        #[cfg(windows)]
        precompiled,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// The console stage of a shader container's program, as the stage names
/// of [`Shader::stage`]; `None` if it is not a container we read.
#[cfg(windows)]
fn container_stage(bytes: &[u8]) -> Option<&'static str> {
    use victory_shader::{Container, FourCC, HwStage};
    let c = Container::parse(bytes).ok()?;
    let part = c.part(FourCC::DXIL)?;
    let programs = victory_shader::native_programs(part.data).ok()?;
    programs.iter().find_map(|p| match p.stage {
        HwStage::Cs => Some("compute"),
        HwStage::Ms => Some("mesh"),
        HwStage::Ps => Some("pixel"),
        HwStage::Vs => Some("vertex"),
        _ => None,
    })
}

/// The shader whose container holds `code` as a program's GFX10 machine
/// code (or begins with it), among the shaders the title has given us so
/// far (the index is brought up to date with the shaders added since the
/// last look), else among the containers of the shader pack ([`pack`]).
#[cfg(windows)]
fn by_code(code: &[u8]) -> Option<Arc<Shader>> {
    use victory_shader::{Container, FourCC};
    struct Index {
        /// Shaders looked at (by container hash).
        seen: std::collections::HashSet<[u8; 32]>,
        /// A program's whole code by hash, and by its key
        /// ([`crate::precompiled::program_key`]: the hash in its prologue).
        whole: HashMap<[u8; 32], Arc<Shader>>,
        head: HashMap<[u8; 32], Arc<Shader>>,
    }
    static INDEX: Mutex<Option<Index>> = Mutex::new(None);
    let all: Vec<([u8; 32], Arc<Shader>)> = SHADERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|m| m.iter().map(|(k, v)| (*k, v.clone())).collect())
        .unwrap_or_default();
    let mut guard = INDEX.lock().unwrap_or_else(|e| e.into_inner());
    let index = guard.get_or_insert_with(|| Index {
        seen: Default::default(),
        whole: HashMap::new(),
        head: HashMap::new(),
    });
    for (key, s) in all {
        if !index.seen.insert(key) {
            continue;
        }
        let Ok(c) = Container::parse(&s.container) else { continue };
        let Some(part) = c.part(FourCC::DXIL) else { continue };
        let Ok(programs) = victory_shader::native_programs(part.data) else { continue };
        for p in programs {
            index.whole.insert(*blake3::hash(&p.code).as_bytes(), s.clone());
            index
                .head
                .entry(crate::precompiled::program_key(&p.code))
                .or_insert_with(|| s.clone());
        }
    }
    let found = index
        .whole
        .get(blake3::hash(code).as_bytes())
        .or_else(|| index.head.get(&crate::precompiled::program_key(code)))
        .cloned();
    drop(guard);
    // Else the container in the shader pack, if one is named.
    found.or_else(|| {
        let bytes = pack::container(code)?;
        container_stage(&bytes).map(|stage| shader(stage, &bytes))
    })
}

/// The title's shader pack (`VICTORY_SHADER_PACK`, the path of its
/// `shaders.rpf`; unset, none is read): the containers of the programs that
/// precompiled descriptions name ([`crate::precompiled`]), which the title
/// never hands over itself.
#[cfg(windows)]
mod pack {
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    use crate::precompiled::PackIndex;

    struct Pack {
        path: PathBuf,
        index: PackIndex,
    }

    static PACK: OnceLock<Option<Pack>> = OnceLock::new();

    /// The container holding `code` (padding and all), read from the pack;
    /// the pack is indexed on first use.
    pub(super) fn container(code: &[u8]) -> Option<Vec<u8>> {
        let pack = PACK
            .get_or_init(|| {
                let path = PathBuf::from(std::env::var_os("VICTORY_SHADER_PACK")?);
                let started = std::time::Instant::now();
                match index(&path) {
                    Ok(index) => {
                        tracing::info!(path = %path.display(), programs = index.programs(), ms = started.elapsed().as_millis() as u64, "shader pack indexed for precompiled pipelines");
                        Some(Pack { path, index })
                    }
                    Err(e) => {
                        tracing::warn!(path = %path.display(), %e, "shader pack not read: precompiled pipelines stay unmade");
                        None
                    }
                }
            })
            .as_ref()?;
        let at = pack.index.find(code)?;
        record(&pack.path, at).inspect_err(|e| tracing::warn!(%e, at, "shader pack record not read")).ok().flatten()
    }

    fn index(path: &Path) -> std::io::Result<PackIndex> {
        let file = std::fs::read(path)?;
        let records: Vec<_> = victory_game::sgd5::records(&file).collect();
        if records.is_empty() {
            return Err(std::io::Error::other("no SGD5 records"));
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunk = records.len().div_ceil(threads);
        let parts: Vec<PackIndex> = std::thread::scope(|s| {
            let handles: Vec<_> = records
                .chunks(chunk)
                .map(|part| {
                    s.spawn(move || {
                        PackIndex::build(part.iter().map(|r| (r.offset, r.unpack().unwrap_or_default())), programs)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
        });
        let mut index = PackIndex::default();
        for part in parts {
            index.merge(part);
        }
        Ok(index)
    }

    /// The native programs of a container.
    fn programs(container: &[u8]) -> Vec<Vec<u8>> {
        use victory_shader::{Container, FourCC};
        let Ok(c) = Container::parse(container) else { return Vec::new() };
        let Some(part) = c.part(FourCC::DXIL) else { return Vec::new() };
        victory_shader::native_programs(part.data)
            .map(|ps| ps.into_iter().map(|p| p.code).collect())
            .unwrap_or_default()
    }

    /// The container of the record whose marker is at `at`: its sizes, then
    /// the record, read from the file (with the six bytes before the marker
    /// a record is found by).
    fn record(path: &Path, at: usize) -> std::io::Result<Option<Vec<u8>>> {
        let mut f = std::fs::File::open(path)?;
        let start = at.checked_sub(6).ok_or_else(|| std::io::Error::other("record at the start"))?;
        f.seek(SeekFrom::Start(start as u64))?;
        let mut head = [0u8; 18];
        f.read_exact(&mut head)?;
        let n = u32::from_le_bytes(head[2..6].try_into().expect("4 bytes")) as usize;
        let mut buf = head.to_vec();
        buf.resize(6 + n, 0);
        f.read_exact(&mut buf[18..])?;
        Ok(victory_game::sgd5::records(&buf).next().and_then(|r| r.unpack()))
    }
}

/// Device methods 0x1c8 and 0x218 (`slot`), whose descriptions name
/// precompiled parts ([`crate::precompiled`]): the shaders of the
/// description's programs (or, if the words do not read as a description,
/// of any (address, bytes) pair among them), each a container we read or
/// GFX10 code whose container we know ([`by_code`]; `read` reads title
/// memory through the kernel, neither pointer is known to be mapped). A single compute shader is made into a compute
/// pipeline as [`create_compute`] makes one, which reads nothing of its
/// description but the shader either; a vertex and a pixel shader of a
/// graphics description (with `VICTORY_SHADER_PACK` set) into a pipeline
/// whose host pipelines are made at its draws, its state read from the
/// registers section ([`Precompiled`]); anything else is left (`None`),
/// and the stages found are returned for the log.
/// shortcut: the map sections (which the container has in its own form)
/// are ignored, and what of the registers and info sections
/// [`crate::precompiled::GraphicsRegisters`] and
/// [`crate::precompiled::input_layout`] do not read.
#[cfg(windows)]
pub(crate) fn create_from_words(
    slot: usize,
    words: &[u64],
    read: impl Fn(usize, &mut [u8]) -> bool,
    iid: *const Iid,
    out: *mut This,
) -> (Option<HRESULT>, Vec<String>) {
    let mut found: Vec<Arc<Shader>> = Vec::new();
    // For the log: each readable candidate's size and first bytes.
    let mut seen: Vec<String> = Vec::new();
    // The programs of a description read as one (`crate::precompiled`),
    // else every (address, size) pair among the words.
    let parsed = crate::precompiled::Description::parse(slot, words);
    // Diagnostics: `VICTORY_PRECOMPILED_DUMP=<dir>` saves every section of
    // the first 64 descriptions of each method as a file of its own
    // (`<slot>-<n>-<role>.bin`), to work out what the graphics ones' state
    // sections hold.
    if let (Some(d), Some(dir)) = (&parsed, std::env::var_os("VICTORY_PRECOMPILED_DUMP")) {
        static DUMPED: [std::sync::atomic::AtomicU32; 2] = [const { std::sync::atomic::AtomicU32::new(0) }; 2];
        let n = DUMPED[usize::from(slot == crate::precompiled::GRAPHICS_SLOT)]
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 64 {
            let dir = std::path::PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            for s in &d.sections {
                let mut bytes = vec![0u8; s.size as usize];
                if read(s.address as usize, &mut bytes) {
                    let _ = std::fs::write(dir.join(format!("{slot:x}-{n:02}-{:?}.bin", s.role)), &bytes);
                }
            }
        }
    }
    let pairs: Vec<(u64, u64)> = match &parsed {
        Some(d) => d.code().map(|s| (s.address, s.size)).collect(),
        None => words.windows(2).map(|p| (p[0], p[1])).collect(),
    };
    for (at, len) in pairs {
        let (at, len) = (at as usize, len as usize);
        // Containers of the title's run from a few hundred bytes to a few
        // MB, machine code from 64 bytes; addresses are user-mode and
        // 4-byte aligned.
        if !(64..=(8 << 20)).contains(&len) || !(0x10000..1 << 47).contains(&at) || !at.is_multiple_of(4) {
            continue;
        }
        let mut bytes = vec![0u8; len];
        if !read(at, &mut bytes) {
            continue;
        }
        let head: String = bytes.iter().take(16).map(|b| format!("{b:02x}")).collect();
        seen.push(format!("{len:#x}:{head}"));
        let known = if bytes.starts_with(b"DXBC") {
            container_stage(&bytes).map(|stage| shader(stage, &bytes))
        } else {
            // GFX10 machine code: the container the title made another
            // pipeline from that holds this code.
            by_code(&bytes)
        };
        if let Some(s) = known
            && !found.iter().any(|f| Arc::ptr_eq(f, &s))
        {
            found.push(s);
        }
    }
    let mut stages: Vec<String> = found.iter().map(|s| format!("{} {}", s.stage, s.hash)).collect();
    if found.is_empty() {
        stages = seen;
    }
    if let [cs] = found.as_slice()
        && cs.stage == "compute"
    {
        return (Some(hand_out(found.clone(), None, None, iid, out)), stages);
    }
    // A vertex and a pixel program of a graphics description, with the
    // shader pack named: a pipeline whose host pipelines are made at its
    // draws, for the targets bound then ([`Precompiled`]).
    if slot == crate::precompiled::GRAPHICS_SLOT
        && std::env::var_os("VICTORY_SHADER_PACK").is_some()
        && precompiled_graphics()
        && let Some(d) = &parsed
        && let (Some(vs), Some(ps)) = (
            found.iter().find(|s| s.stage == "vertex"),
            found.iter().find(|s| s.stage == "pixel"),
        )
        && found.len() == 2
    {
        let section = |role| {
            let s = d.sections.iter().find(|s| s.role == role).filter(|s| s.size <= 0x1000)?;
            let mut bytes = vec![0u8; s.size as usize];
            read(s.address as usize, &mut bytes).then_some(bytes)
        };
        let dwords: Vec<u32> = section(crate::precompiled::Role::Registers)
            .map(|b| b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
            .unwrap_or_default();
        let inputs = section(crate::precompiled::Role::Info)
            .and_then(|b| crate::precompiled::input_layout(&b))
            .unwrap_or_else(|| {
                note_once("an info section without a readable input layout: none");
                Vec::new()
            });
        // Diagnostics: with `VICTORY_PRECOMPILED_DUMP=<dir>`, the registers
        // and info sections of every graphics description made, by its
        // shaders (`gfx-<vertex>-<pixel>-<n>-registers.bin`, a new `n` for
        // each different registers section), to check the state decoded.
        if let Some(dir) = std::env::var_os("VICTORY_PRECOMPILED_DUMP") {
            let dir = std::path::PathBuf::from(dir);
            let bytes: Vec<u8> = dwords.iter().flat_map(|d| d.to_le_bytes()).collect();
            for n in 0..16 {
                let stem = format!("gfx-{}-{}-{n}", vs.hash, ps.hash);
                let path = dir.join(format!("{stem}-registers.bin"));
                match std::fs::read(&path) {
                    Ok(same) if same == bytes => break,
                    Ok(_) => continue,
                    Err(_) => {
                        let _ = std::fs::create_dir_all(&dir);
                        let _ = std::fs::write(&path, &bytes);
                        if let Some(info) = section(crate::precompiled::Role::Info) {
                            let _ = std::fs::write(dir.join(format!("{stem}-info.bin")), info);
                        }
                        break;
                    }
                }
            }
        }
        let (registers, notes) = crate::precompiled::GraphicsRegisters::decode(&dwords);
        for note in notes {
            note_once(note);
        }
        let shaders = vec![vs.clone(), ps.clone()];
        let precompiled = Box::new(Precompiled {
            registers,
            inputs,
            variants: Default::default(),
        });
        return (Some(hand_out(shaders, None, Some(precompiled), iid, out)), stages);
    }
    (None, stages)
}

/// Whether graphics pipelines of method 0x1c8 are made (with the shader
/// pack named); `VICTORY_PRECOMPILED_GRAPHICS=0` leaves them unmade, as
/// before they were, and keeps the compute ones of 0x218.
#[cfg(windows)]
fn precompiled_graphics() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_PRECOMPILED_GRAPHICS").is_none_or(|v| v != "0");
        if !on {
            tracing::info!("VICTORY_PRECOMPILED_GRAPHICS=0: precompiled graphics pipelines are left unmade");
        }
        on
    })
}

/// Logs what the host cannot have of a precompiled graphics pipeline's
/// state, once for each kind of thing.
#[cfg(windows)]
fn note_once(note: &'static str) {
    static SEEN: Mutex<Option<std::collections::HashSet<&'static str>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.get_or_insert_with(Default::default).insert(note) {
        tracing::warn!(note, "precompiled graphics pipeline state the host has otherwise (logged once)");
    }
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut shaders = Vec::new();
    #[cfg(windows)]
    let mut graphics = host::GraphicsState::default();
    let mut at = 0;
    while at + 8 <= desc.size {
        // SAFETY: inside the stream the title passes.
        let kind = unsafe { desc.stream.add(at).cast::<u32>().read_unaligned() };
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8.
            let code = unsafe { &*desc.stream.add(at + 8).cast::<Bytecode>() };
            shaders.extend(copy(stage, code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, desc.stream.add(at))
        };
        at += size;
    }
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;
#[cfg(windows)]
pub mod prewarm;
#[cfg(windows)]
mod recipe;

/// Lets the pipeline library's mapped file go from memory when no pipeline
/// has loaded for a while (`library::trim`); called by the host's sweep.
#[cfg(windows)]
pub(crate) fn trim_library() {
    library::trim();
}

/// The pipeline library's part of the report when the host device is
/// removed (`library::removal_note`).
#[cfg(windows)]
pub(crate) fn library_removal_note() -> String {
    library::removal_note()
}

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
}

#[cfg(test)]
mod tests {
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.clone()))
            .collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    /// The stream layout the title builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }
}

/// Diagnostics, run by hand: the containers of the programs of dumped
/// descriptions (`VICTORY_PRECOMPILED_DUMP`), looked up in the shader pack
/// as the device does: `VICTORY_SHADER_PACK=<shaders.rpf>
/// VICTORY_PRECOMPILED_DIR=<dumps> cargo test -p victory-gpu --lib
/// precompiled_dumps_find_their_containers -- --ignored --nocapture`.
#[cfg(all(test, windows))]
mod pack_diag {
    #[test]
    #[ignore = "reads the title's shader pack and dumped descriptions"]
    fn precompiled_dumps_find_their_containers() {
        use victory_shader::{Container, FourCC};
        let (Some(pack), Some(dir)) = (
            std::env::var_os("VICTORY_SHADER_PACK"),
            std::env::var_os("VICTORY_PRECOMPILED_DIR"),
        ) else {
            return;
        };
        let _ = tracing_subscriber::fmt().with_writer(std::io::stderr).try_init();
        let file = std::fs::read(&pack).unwrap();
        let records: Vec<_> = victory_game::sgd5::records(&file).collect();
        let index = crate::precompiled::PackIndex::build(
            records.iter().map(|r| (r.offset, r.unpack().unwrap_or_default())),
            |c: &[u8]| {
                let Ok(c) = Container::parse(c) else { return Vec::new() };
                let Some(part) = c.part(FourCC::DXIL) else { return Vec::new() };
                victory_shader::native_programs(part.data)
                    .map(|ps| ps.into_iter().map(|p| p.code).collect::<Vec<_>>())
                    .unwrap_or_default()
            },
        );
        let mut names: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).collect();
        names.sort();
        let (mut looked, mut missing) = (0, Vec::new());
        for path in names {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if !name.contains("-Code(") {
                continue;
            }
            looked += 1;
            let code = std::fs::read(&path).unwrap();
            let stage = index
                .find(&code)
                .and_then(|at| records.iter().find(|r| r.offset == at)?.unpack())
                .and_then(|c| super::container_stage(&c));
            println!("{name}: {stage:?}");
            if stage.is_none() {
                missing.push(name);
            }
        }
        println!("programs {looked}, not found {missing:?}");
        assert!(missing.is_empty());
        // Each graphics description's input layout reads, and names every
        // input its vertex shader reads.
        #[cfg(feature = "diagnostics")]
        for n in 0..64 {
            let file = |part: &str| std::fs::read(std::path::Path::new(&dir).join(format!("1c8-{n:02}-{part}.bin")));
            let (Ok(info), Ok(code)) = (file("Info"), file("Code(1)")) else { continue };
            let layout = crate::precompiled::input_layout(&info).unwrap_or_else(|| panic!("{n}: no layout"));
            let container = index
                .find(&code)
                .and_then(|at| records.iter().find(|r| r.offset == at)?.unpack())
                .unwrap();
            let vs = super::shader("vertex", &container);
            let reads = super::rect_diag::vertex_inputs(&vs.translated().unwrap().host.dxil);
            let named = |e: &crate::precompiled::InputElement| {
                layout.iter().any(|l| l.name == e.name && l.index == e.index)
            };
            let summary: Vec<String> = layout.iter().map(|e| format!("{}{} {} @{}", e.name, e.index, e.format, e.offset)).collect();
            let ps_hash = file("Code(0)")
                .ok()
                .and_then(|c| index.find(&c))
                .and_then(|at| records.iter().find(|r| r.offset == at)?.unpack())
                .map(|c| blake3::hash(&c).to_hex()[..16].to_owned());
            println!("1c8-{n:02} shaders {} {ps_hash:?} layout {summary:?}", vs.hash);
            assert!(reads.iter().all(named), "{n}: the vertex shader reads {reads:?}");
            // The host pipeline of the whole description, drawn into one
            // RGBA8 target as triangles and as rectangles.
            let (Ok(regs), Ok(ps_code)) = (file("Registers"), file("Code(0)")) else { continue };
            let dwords: Vec<u32> = regs.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            let (registers, _) = crate::precompiled::GraphicsRegisters::decode(&dwords);
            let ps_container = index
                .find(&ps_code)
                .and_then(|at| records.iter().find(|r| r.offset == at)?.unpack())
                .unwrap();
            let ps = super::shader("pixel", &ps_container);
            if crate::host::host().is_none() {
                continue;
            }
            for topology in [4, 17] {
                let targets = crate::precompiled::Targets::new(&[28], 0, 1).drawing(topology);
                let g = super::host::GraphicsState::precompiled(&registers, &layout, &targets);
                let made = super::host::HostPipeline::new(&[vs.clone(), ps.clone()], Some(&g));
                let made = made.unwrap_or_else(|| panic!("1c8-{n:02} topology {topology}: refused"));
                assert!(topology != 17 || made.rects, "1c8-{n:02}: no rectangle geometry stage");
            }
        }
    }
}

/// Diagnostics, run by hand on the host GPU: the rectangle pipelines of the
/// precompiled vertex and pixel shaders named by `VICTORY_RECT_PAIRS`
/// (`<vertex hash>/<pixel hash>,...`, as the log names them), taken from
/// the shader pack, with an input layout made from the vertex shader's
/// inputs; they are to be made with their rectangle geometry stage
/// (`VICTORY_D3D12_DEBUG=1` adds the debug layer's messages, where it is
/// installed): `VICTORY_SHADER_PACK=<shaders.rpf> VICTORY_RECT_PAIRS=...
/// cargo test -p victory-gpu --features diagnostics --lib rect_pairs --
/// --ignored --nocapture`.
#[cfg(all(test, windows, feature = "diagnostics"))]
mod rect_diag {
    /// The non-system inputs of a DXIL vertex shader's input signature as
    /// one buffer of 32-bit floats, packed in order.
    pub(super) fn vertex_inputs(dxil: &[u8]) -> Vec<crate::precompiled::InputElement> {
        let Ok(c) = victory_shader::Container::parse(dxil) else { return Vec::new() };
        let Some(part) = c.part(victory_shader::FourCC::ISG1) else { return Vec::new() };
        let d = part.data;
        let word = |at: usize| u32::from_le_bytes(d[at..at + 4].try_into().unwrap());
        let mut offset = 0;
        let mut out = Vec::new();
        for i in 0..word(0) as usize {
            let at = 8 + 32 * i;
            let (name_at, index, system) = (word(at + 4) as usize, word(at + 8), word(at + 12));
            if system != 0 {
                continue;
            }
            let end = d[name_at..].iter().position(|&b| b == 0).unwrap();
            let name = String::from_utf8_lossy(&d[name_at..name_at + end]).into_owned();
            let columns = (d[at + 24] & 0xf).count_ones();
            // R32_FLOAT, R32G32_FLOAT, R32G32B32_FLOAT, R32G32B32A32_FLOAT.
            let format = [41, 16, 6, 2][columns.clamp(1, 4) as usize - 1];
            out.push(crate::precompiled::InputElement {
                name,
                index,
                format,
                slot: 0,
                offset,
                per_instance: false,
                step_rate: 0,
            });
            offset += 4 * columns;
        }
        out
    }

    #[test]
    #[ignore = "reads the title's shader pack and needs the host GPU"]
    fn rect_pairs_make_their_pipelines() {
        let (Some(pack), Ok(pairs)) = (
            std::env::var_os("VICTORY_SHADER_PACK"),
            std::env::var("VICTORY_RECT_PAIRS"),
        ) else {
            return;
        };
        // The pipelines' own log lines, the host's messages among them.
        let _ = tracing_subscriber::fmt().with_writer(std::io::stderr).try_init();
        if crate::host::host().is_none() {
            return;
        }
        let wanted: Vec<(String, String)> = pairs
            .split(',')
            .filter_map(|p| p.split_once('/'))
            .map(|(v, p)| (v.trim().to_owned(), p.trim().to_owned()))
            .collect();
        let file = std::fs::read(&pack).unwrap();
        let mut by_hash = std::collections::HashMap::new();
        for r in victory_game::sgd5::records(&file) {
            let Some(c) = r.unpack() else { continue };
            let hash = blake3::hash(&c).to_hex()[..16].to_owned();
            if wanted.iter().any(|(v, p)| *v == hash || *p == hash) {
                by_hash.insert(hash, c);
            }
        }
        let targets = crate::precompiled::Targets::new(&[28], 0, 1);
        let registers = crate::precompiled::GraphicsRegisters::console_defaults();
        for (v, p) in &wanted {
            let (Some(vc), Some(pc)) = (by_hash.get(v), by_hash.get(p)) else {
                println!("{v}/{p}: not in the pack");
                continue;
            };
            let vs = super::shader("vertex", vc);
            let ps = super::shader("pixel", pc);
            // The vertex shader's inputs as one buffer of floats, in order
            // (the description's own layout is not at hand here).
            let inputs = vs
                .translated()
                .map(|t| vertex_inputs(&t.host.dxil))
                .unwrap_or_default();
            for topology in [4, 17] {
                let targets = targets.drawing(topology);
                let g = super::host::GraphicsState::precompiled(&registers, &inputs, &targets);
                let Some(made) = super::host::HostPipeline::new(&[vs.clone(), ps.clone()], Some(&g)) else {
                    panic!(
                        "{v}/{p} topology {topology}: no pipeline; translated {} {}",
                        vs.translated().is_some(),
                        ps.translated().is_some()
                    );
                };
                println!("{v}/{p} topology {topology}: made, rectangle geometry stage {}", made.rects);
                assert!(topology != 17 || made.rects, "{v}/{p}: made without its rectangle geometry stage");
            }
        }
    }
}
