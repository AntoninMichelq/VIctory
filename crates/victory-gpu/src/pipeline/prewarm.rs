//! Offline pre-warming of a cache's translated shaders and host pipeline
//! library (`host-pipelines.bin`) without starting the title.
//!
//! Uses the exact same translation cache (`crate::shader_cache`), root
//! signature builder, library key (`library::name`), and `HostPipeline::new`
//! path as a live run so every pre-built PSO is found on the next launch.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::HostPipeline;
use super::library::{self, EARLY_FILE, LIBRARY_FILE, LOADED, MADE, RECIPES_FILE};
use super::recipe::{self, Recipe};

/// Options for [`prewarm`].
pub struct PrewarmOptions {
    /// Root cache folder (`VICTORY_CACHE_DIR` layout: `shaders/` and `pipelines/`).
    pub cache_dir: PathBuf,
    /// Folders holding console shader containers (`.bin` / `.cdxil`).
    pub shader_dirs: Vec<PathBuf>,
    /// Optional source cache (or `pipelines/` folder) to seed `early-pipelines.txt`
    /// and `early-recipes.bin` from.
    pub from: Option<PathBuf>,
    /// Number of worker threads (default: `VICTORY_PIPELINE_WORKERS` or half the
    /// logical cores, clamped to 2..=8 for pipelines, up to 16 for `--all-shaders`).
    pub jobs: Option<usize>,
    /// Optional pre-extracted unique DXBC containers (`--all-shaders`): every
    /// container is translated into `cache_dir/shaders/` in parallel before
    /// recipe pipeline compilation, without needing pipeline recipes.
    pub all_shaders: Vec<Vec<u8>>,
}

/// Summary of an offline pre-warm run.
pub struct PrewarmReport {
    pub all_shaders_total: usize,
    pub all_shaders_hits: u64,
    pub all_shaders_translated: u64,
    pub all_shaders_failed: usize,
    pub recipes_total: usize,
    pub from_recipes_file: usize,
    pub compute_matched_from_early: usize,
    pub early_without_recipe: usize,
    pub loaded: u64,
    pub made: u64,
    pub failed: usize,
    pub missing_containers: usize,
    pub shader_hits: u64,
    pub shader_misses: u64,
    pub library_bytes: u64,
    pub elapsed: Duration,
}

/// Translates missing shaders and builds host pipelines into `host-pipelines.bin`.
pub fn prewarm(opts: PrewarmOptions) -> Result<PrewarmReport, String> {
    let started = Instant::now();
    let pipelines_dir = opts.cache_dir.join("pipelines");
    let shaders_dir = opts.cache_dir.join("shaders");
    std::fs::create_dir_all(&pipelines_dir)
        .map_err(|e| format!("creating {}: {e}", pipelines_dir.display()))?;
    std::fs::create_dir_all(&shaders_dir)
        .map_err(|e| format!("creating {}: {e}", shaders_dir.display()))?;

    // Initialize the shader and pipeline caches before touching the device.
    super::use_pipeline_cache(pipelines_dir.clone());
    crate::shader_cache::use_shader_cache(shaders_dir);

    // Initialize the host D3D12 device (sets host_max_wave) and probe the view
    // heap limit (sets view_heap_limit), exactly as a live run does.
    let Some(_host) = crate::host::host() else {
        return Err("no host D3D12 device available".into());
    };
    crate::descriptor::host::probe_view_limit();

    // Optional `--all-shaders` pass: translate and cache every unique DXBC
    // container in parallel into `cache_dir/shaders/` (DXIL cache only; no
    // graphics render state required). Uses the exact same host device wave
    // width and view heap limit probed above so cache keys match live runs.
    let all_shaders_total = opts.all_shaders.len();
    let (all_hits_before, all_misses_before) = crate::shader_cache::counts();
    let all_shaders_failed = AtomicUsize::new(0);
    if !opts.all_shaders.is_empty() {
        let shader_threads = opts
            .jobs
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(4, |n| n.get().clamp(2, 16))
            })
            .clamp(1, 32);
        let next_shader = AtomicUsize::new(0);
        let done_shader = AtomicUsize::new(0);
        let last_shader_log = Mutex::new(Instant::now());
        let all_started = Instant::now();
        tracing::info!(
            shaders = all_shaders_total,
            threads = shader_threads,
            "prewarm translating all shaders into DXIL cache"
        );
        std::thread::scope(|s| {
            for i in 0..shader_threads.min(all_shaders_total) {
                let all_shaders = &opts.all_shaders;
                let next_shader = &next_shader;
                let done_shader = &done_shader;
                let all_shaders_failed = &all_shaders_failed;
                let last_shader_log = &last_shader_log;
                std::thread::Builder::new()
                    .name(format!("victory-shaders-{i}"))
                    .spawn_scoped(s, move || {
                        while let Some(bytes) =
                            all_shaders.get(next_shader.fetch_add(1, Ordering::Relaxed))
                        {
                            if let Err(e) = crate::shader_cache::host_shader(bytes) {
                                all_shaders_failed.fetch_add(1, Ordering::Relaxed);
                                tracing::debug!(error = %e, bytes = bytes.len(), "prewarm shader skipped");
                            }
                            let n = done_shader.fetch_add(1, Ordering::Relaxed) + 1;
                            if n == all_shaders_total
                                || n.is_multiple_of(1000)
                                || last_shader_log
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .elapsed()
                                    >= Duration::from_secs(10)
                            {
                                let mut t = last_shader_log
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                *t = Instant::now();
                                drop(t);
                                let (h, m) = crate::shader_cache::counts();
                                tracing::info!(
                                    done = n,
                                    total = all_shaders_total,
                                    hits = h - all_hits_before,
                                    translated = m - all_misses_before,
                                    failed = all_shaders_failed.load(Ordering::Relaxed),
                                    elapsed_s = all_started.elapsed().as_secs(),
                                    "prewarm all-shaders progress"
                                );
                            }
                        }
                    })
                    .ok();
            }
        });
    }
    let (all_hits_after, all_misses_after) = crate::shader_cache::counts();
    let all_shaders_hits = all_hits_after - all_hits_before;
    let all_shaders_translated = all_misses_after - all_misses_before;

    // Index in-memory containers by 16-hex BLAKE3 hash so recipes can also
    // resolve shaders extracted from game archives without `_work/prison/shaders`.
    let memory_index: HashMap<String, usize> = opts
        .all_shaders
        .iter()
        .enumerate()
        .map(|(idx, bytes)| (blake3::hash(bytes).to_hex()[..16].to_owned(), idx))
        .collect();

    // Load recipes and early keys from the target cache and optional `--from`.
    let mut recipes_by_key: HashMap<String, Recipe> = HashMap::new();
    let mut early_keys: HashSet<String> = HashSet::new();
    let mut sources = vec![pipelines_dir.clone()];
    if let Some(from) = &opts.from {
        let src = if from.join("pipelines").is_dir() {
            from.join("pipelines")
        } else {
            from.clone()
        };
        if src != pipelines_dir {
            sources.push(src);
        }
    }
    for dir in &sources {
        if let Ok(bytes) = std::fs::read(dir.join(RECIPES_FILE)) {
            match recipe::decode_all(&bytes) {
                Some(list) => {
                    for r in list {
                        recipes_by_key.entry(r.key.clone()).or_insert(r);
                    }
                }
                None => tracing::warn!(path = %dir.join(RECIPES_FILE).display(), "unreadable early-recipes.bin"),
            }
        }
        if let Ok(text) = std::fs::read_to_string(dir.join(EARLY_FILE)) {
            for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
                early_keys.insert(line.to_owned());
            }
        }
    }
    let from_recipes_file = recipes_by_key.len();

    // Index console shader containers from `--shaders`.
    let container_index = index_containers(&opts.shader_dirs);
    tracing::info!(
        containers = container_index.len(),
        recipes = from_recipes_file,
        early_keys = early_keys.len(),
        "prewarm inputs loaded"
    );

    // Any key in `early-pipelines.txt` not yet in `early-recipes.bin` may be a
    // compute pipeline: its Pending::key is blake3("compute" || hash)[..32].
    let uncovered: HashSet<&str> = early_keys
        .iter()
        .map(String::as_str)
        .filter(|k| !recipes_by_key.contains_key(*k))
        .collect();
    let mut compute_matched_from_early = 0usize;
    if !uncovered.is_empty() && (!container_index.is_empty() || !memory_index.is_empty()) {
        for hash in container_index.keys().chain(memory_index.keys()) {
            let mut h = blake3::Hasher::new();
            h.update(b"compute");
            h.update(hash.as_bytes());
            let key = h.finalize().to_hex()[..32].to_owned();
            if uncovered.contains(key.as_str()) {
                recipes_by_key.entry(key.clone()).or_insert(Recipe {
                    key,
                    shaders: vec![("compute".to_owned(), hash.clone())],
                    graphics: None,
                });
                compute_matched_from_early += 1;
            }
        }
    }
    let early_without_recipe = early_keys
        .iter()
        .filter(|k| !recipes_by_key.contains_key(k.as_str()))
        .count();

    let recipes: Vec<Recipe> = recipes_by_key.into_values().collect();
    let recipes_total = recipes.len();

    let loaded_before = LOADED.load(Ordering::Relaxed);
    let made_before = MADE.load(Ordering::Relaxed);
    let (hits_before, misses_before) = crate::shader_cache::counts();

    let threads = opts
        .jobs
        .or_else(|| {
            std::env::var("VICTORY_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
        })
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8))
        })
        .clamp(1, 16);

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let missing_containers = AtomicUsize::new(0);
    let loaded_shaders: Mutex<HashMap<(String, String), Option<Arc<Shader>>>> =
        Mutex::new(HashMap::new());
    let last_log = Mutex::new(Instant::now());

    if !recipes.is_empty() {
        tracing::info!(
            recipes = recipes_total,
            threads,
            "prewarm starting pipeline compilation"
        );
        std::thread::scope(|s| {
            for i in 0..threads.min(recipes_total) {
                let recipes = &recipes;
                let container_index = &container_index;
                let memory_index = &memory_index;
                let all_shaders = &opts.all_shaders;
                let next = &next;
                let done = &done;
                let failed = &failed;
                let missing_containers = &missing_containers;
                let loaded_shaders = &loaded_shaders;
                let last_log = &last_log;
                // Name workers `victory-pipelines-*` so `make::wait_to_retry`
                // handles driver E_OUTOFMEMORY retries identically to runtime.
                std::thread::Builder::new()
                    .name(format!("victory-pipelines-{i}"))
                    .spawn_scoped(s, move || {
                        while let Some(r) = recipes.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let mut shaders = Vec::with_capacity(r.shaders.len());
                            let mut all_found = true;
                            for (stage, hash) in &r.shaders {
                                match load_shader(
                                    stage,
                                    hash,
                                    container_index,
                                    memory_index,
                                    all_shaders,
                                    loaded_shaders,
                                ) {
                                    Some(sh) => shaders.push(sh),
                                    None => {
                                        all_found = false;
                                        break;
                                    }
                                }
                            }
                            if !all_found {
                                missing_containers.fetch_add(1, Ordering::Relaxed);
                            } else {
                                let _making = super::make::room_to_make();
                                if HostPipeline::new(&shaders, r.graphics.as_ref()).is_none() {
                                    failed.fetch_add(1, Ordering::Relaxed);
                                }
                                library::save(false);
                            }
                            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                            if n == recipes_total
                                || n.is_multiple_of(100)
                                || last_log
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .elapsed()
                                    >= Duration::from_secs(10)
                            {
                                let mut t = last_log.lock().unwrap_or_else(|e| e.into_inner());
                                *t = Instant::now();
                                drop(t);
                                tracing::info!(
                                    done = n,
                                    total = recipes_total,
                                    loaded = LOADED.load(Ordering::Relaxed) - loaded_before,
                                    made = MADE.load(Ordering::Relaxed) - made_before,
                                    failed = failed.load(Ordering::Relaxed),
                                    elapsed_s = started.elapsed().as_secs(),
                                    "prewarm progress"
                                );
                            }
                        }
                    })
                    .ok();
            }
        });
    }

    // Flush the library to disk (`host-pipelines.bin`).
    library::save(true);

    // Ensure `early-pipelines.txt` and `early-recipes.bin` are up to date in
    // the target cache so the next game run starts every pre-warmed pipeline early.
    let mut out_keys: HashSet<String> = early_keys;
    for r in &recipes {
        out_keys.insert(r.key.clone());
        out_keys.insert(r.current_key());
    }
    if !out_keys.is_empty() {
        let mut sorted: Vec<String> = out_keys.into_iter().collect();
        sorted.sort();
        library::write_atomically(
            &pipelines_dir.join(EARLY_FILE),
            sorted.join("\n").as_bytes(),
        )
        .map_err(|e| format!("writing {EARLY_FILE}: {e}"))?;
    }
    if !recipes.is_empty() {
        let refs: Vec<&Recipe> = recipes.iter().collect();
        library::write_atomically(
            &pipelines_dir.join(RECIPES_FILE),
            &recipe::encode_all(&refs),
        )
        .map_err(|e| format!("writing {RECIPES_FILE}: {e}"))?;
    }

    let (hits_after, misses_after) = crate::shader_cache::counts();
    let library_bytes = std::fs::metadata(pipelines_dir.join(LIBRARY_FILE))
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(PrewarmReport {
        all_shaders_total,
        all_shaders_hits,
        all_shaders_translated,
        all_shaders_failed: all_shaders_failed.load(Ordering::Relaxed),
        recipes_total,
        from_recipes_file,
        compute_matched_from_early,
        early_without_recipe,
        loaded: LOADED.load(Ordering::Relaxed) - loaded_before,
        made: MADE.load(Ordering::Relaxed) - made_before,
        failed: failed.load(Ordering::Relaxed),
        missing_containers: missing_containers.load(Ordering::Relaxed),
        shader_hits: hits_after - hits_before,
        shader_misses: misses_after - misses_before,
        library_bytes,
        elapsed: started.elapsed(),
    })
}

fn static_stage(stage: &str) -> Option<&'static str> {
    match stage {
        "vertex" => Some("vertex"),
        "pixel" => Some("pixel"),
        "domain" => Some("domain"),
        "hull" => Some("hull"),
        "geometry" => Some("geometry"),
        "compute" => Some("compute"),
        "amplification" => Some("amplification"),
        "mesh" => Some("mesh"),
        _ => None,
    }
}

fn load_shader(
    stage: &str,
    hash: &str,
    index: &HashMap<String, PathBuf>,
    memory_index: &HashMap<String, usize>,
    all_shaders: &[Vec<u8>],
    cache: &Mutex<HashMap<(String, String), Option<Arc<Shader>>>>,
) -> Option<Arc<Shader>> {
    let key = (stage.to_owned(), hash.to_owned());
    if let Some(found) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return found.clone();
    }
    let stage_static = static_stage(stage)?;
    let loaded = index
        .get(hash)
        .and_then(|path| {
            let bytes = std::fs::read(path).ok()?;
            let sh = super::shader(stage_static, &bytes);
            if sh.hash == hash {
                Some(sh)
            } else {
                tracing::warn!(
                    path = %path.display(),
                    expected = hash,
                    actual = %sh.hash,
                    "shader container hash mismatch"
                );
                None
            }
        })
        .or_else(|| {
            let &idx = memory_index.get(hash)?;
            let sh = super::shader(stage_static, &all_shaders[idx]);
            (sh.hash == hash).then_some(sh)
        });
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, loaded.clone());
    loaded
}

fn is_hex16(s: &str) -> bool {
    s.len() == 16 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn index_containers(dirs: &[PathBuf]) -> HashMap<String, PathBuf> {
    let mut map = HashMap::new();
    for root in dirs {
        if !root.exists() {
            continue;
        }
        walk_containers(root, &mut map);
    }
    map
}

fn walk_containers(dir: &Path, out: &mut HashMap<String, PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_containers(&path, out);
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !ext.eq_ignore_ascii_case("bin") && !ext.eq_ignore_ascii_case("cdxil") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            // Fast path: `<16-hex>.bin` (as in `_work/prison/shaders`) or
            // `<name>.<16-hex>.cdxil`.
            let candidate = stem.rsplit('.').next().unwrap_or(stem);
            if is_hex16(candidate) {
                out.entry(candidate.to_ascii_lowercase()).or_insert(path);
                continue;
            }
        }
        if let Ok(bytes) = std::fs::read(&path) {
            let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
            out.entry(hash).or_insert(path);
        }
    }
}
