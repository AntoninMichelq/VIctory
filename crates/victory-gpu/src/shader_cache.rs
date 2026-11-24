//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the VIctory build
//! that translated it (the BLAKE3 of our executable), so a rebuilt
//! translator never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub use victory_pipeline::Cache;
use victory_pipeline::Converter;
use victory_shader::emit::{HostShader, StaticSampler, Target, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies this build's translations: the BLAKE3 of the
/// translator's source (`victory_shader::SOURCE_HASH`, so a rebuild that
/// does not touch the translator keeps the cache), of this file (the entry
/// encoding), of the build profile, and of the translation settings it runs
/// with (the loop and spin-wait limits, which `VICTORY_SHADER_LOOP_LIMIT`
/// and `VICTORY_SHADER_SPIN_LIMIT` change at run time).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        let spin = victory_shader::emit::spin_limit();
        if spin != victory_shader::emit::DEFAULT_SPIN_LIMIT {
            tracing::warn!(
                spin,
                "VICTORY_SHADER_SPIN_LIMIT: spin-wait loops stop at this count"
            );
        }
        let mut h = blake3::Hasher::new();
        h.update(b"victory shader cache v2");
        h.update(victory_shader::SOURCE_HASH.as_bytes());
        h.update(include_str!("shader_cache.rs").as_bytes());
        h.update(&[u8::from(cfg!(debug_assertions)), u8::from(cfg!(feature = "diagnostics"))]);
        h.update(&limit.to_le_bytes());
        h.update(&spin.to_le_bytes());
        h.update(&[u8::from(victory_shader::emit::root_uavs())]);
        if let Some(list) = victory_shader::emit::root_uavs_only() {
            tracing::warn!(
                shaders = list.split(',').count(),
                list,
                "VICTORY_ROOT_UAVS_ONLY: root UAVs only for these shaders"
            );
            h.update(b"root uavs only:");
            h.update(list.as_bytes());
        }
        let except = victory_shader::emit::root_uavs_except();
        if except != victory_shader::emit::ROOT_UAVS_EXCEPT_DEFAULT {
            tracing::warn!(
                list = except,
                "VICTORY_ROOT_UAVS_EXCEPT: never root UAVs for these shaders"
            );
        }
        h.update(b"root uavs except:");
        h.update(except.as_bytes());
        h.update(&[u8::from(victory_shader::emit::pointer_views())]);
        h.update(&[u8::from(victory_shader::emit::pointer_chase())]);
        h.update(&[u8::from(victory_shader::emit::fold_branches())]);
        h.update(&[u8::from(victory_shader::emit::merge_repeats())]);
        *h.finalize().as_bytes()
    })
}

/// The wave sizes the runtime translates for: its host device's
/// ([`Target::ANY`] without one). `victory translate --write-cache` asks
/// the same function, so its entries carry the runtime's key.
pub fn runtime_target() -> Target {
    crate::host::host().map_or(Target::ANY, |h| h.target())
}

/// The key part of every entry this build translates for `target` (see
/// [`key`]): the one [`host_shader`] looks entries up with and [`store`]
/// writes them with.
pub fn cache_key(target: Target) -> [u8; 32] {
    key(build(), target)
}

/// The translation the runtime makes of a container (and keeps).
fn translate_for(target: Target) -> impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync {
    move |c: &[u8]| {
        let hash = blake3::hash(c).to_hex();
        victory_shader::emit::translate_named(c, target, &hash[..16]).map_err(|e| e.to_string())
    }
}

/// The translation of a console shader container for a host that runs
/// `target`'s wave sizes, from the cache when there is one.
pub(crate) fn host_shader(container: &[u8], target: Target) -> Result<HostShader, String> {
    let translate = translate_for(target);
    match CACHE.get() {
        Some(cache) => cached(cache, &cache_key(target), container, translate),
        None => translate(container),
    }
}

/// What [`store`] did with a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stored {
    /// Translated and written.
    Written,
    /// A readable entry was there already.
    AlreadyCached,
}

/// Opens the shader cache in `dir` for [`store`] (the runtime's is the
/// `shaders` folder of the configured cache).
pub fn open(dir: PathBuf) -> anyhow::Result<Cache> {
    Cache::open(dir)
}

/// Ahead of time: translates `container` as the runtime would for
/// `target` and keeps it in `cache` under `key` (from [`cache_key`]), so a
/// run with the same build, settings and host finds it instead of
/// translating it. The same converter and entry encoding as the runtime's
/// lookup, so the two cannot disagree on where an entry lives.
pub fn store(
    cache: &Cache,
    key: &[u8; 32],
    container: &[u8],
    target: Target,
) -> Result<Stored, String> {
    store_with(cache, key, container, translate_for(target))
}

fn store_with(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<Stored, String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
        translated: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        let bytes = std::fs::read(&path).ok();
        if let Some(e) = bytes.as_deref().and_then(failure) {
            return Err(e);
        }
        if bytes.as_deref().and_then(decode).is_some() {
            return Ok(if converter.ran.load(Ordering::Relaxed) {
                Stored::Written
            } else {
                Stored::AlreadyCached
            });
        }
        // An entry the runtime could not read: replace it.
        let _ = std::fs::remove_file(&path);
    }
    Err("the cache entry does not read back".to_owned())
}

/// The part of an entry's key before the container: the build, and the
/// host's wave sizes (a translation for one host's waves is not another's).
fn key(build: &[u8; 32], target: Target) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(build);
    h.update(&target.key());
    *h.finalize().as_bytes()
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
    /// The translation it ran worked: a cache error after that is the
    /// cache's (it could not keep the entry), not the shader's.
    translated: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let entry = match (self.translate)(&input[32..]) {
            Ok(shader) => encode(&shader),
            // A container this build cannot translate is kept as its
            // error, so later runs do not translate it again (some take
            // seconds to fail).
            Err(e) => [FAILED.as_slice(), e.as_bytes()].concat(),
        };
        self.translated.store(true, Ordering::Relaxed);
        Ok(entry)
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
        translated: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            // A full disk, a file in the way, a scanner holding the new
            // entry: the shader is still good, only not kept.
            Err(e) if converter.translated.load(Ordering::Relaxed) => {
                tracing::warn!(error = format!("{e:#}"), "shader cache entry not written");
                break;
            }
            Err(e) => return Err(format!("{e:#}")),
        };
        let bytes = std::fs::read(&path).ok();
        if let Some(e) = bytes.as_deref().and_then(failure) {
            let counter = if converter.ran.load(Ordering::Relaxed) { &MISSES } else { &HITS };
            counter.fetch_add(1, Ordering::Relaxed);
            return Err(e);
        }
        if let Some(shader) = bytes.as_deref().and_then(decode) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

// "VSH3": root UAVs (PR #294) joined the stride-table flag of "VSH2".
const MAGIC: &[u8; 4] = b"VSH3";
/// The start of an entry that keeps a translation's error (the rest).
const FAILED: &[u8; 4] = b"VSHF";

/// The error a failed translation's entry keeps.
fn failure(bytes: &[u8]) -> Option<String> {
    let rest = bytes.strip_prefix(FAILED.as_slice())?;
    Some(String::from_utf8_lossy(rest).into_owned())
}
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, u32::from(s.stride_table));
    word(&mut out, u32::from(s.memory_table));
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let stride_table = r.word()? != 0;
    let memory_table = r.word()? != 0;
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        stride_table,
        root_uavs,
        memory_table,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            stride_table: true,
            root_uavs: vec![1],
            memory_table: true,
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH3 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are stored with their error and not translated again.
        let fails = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            fails.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported".to_owned())
        };
        assert_eq!(cached(&cache, &build, b"other", fail), Err("unsupported".to_owned()));
        assert_eq!(cached(&cache, &build, b"other", fail), Err("unsupported".to_owned()));
        assert_eq!(fails.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn stored_entries_are_what_the_runtime_finds() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = key(&[7; 32], Target::ANY);
        let container = b"a console container".as_slice();
        let ok = |_: &[u8]| Ok(shader());
        assert_eq!(store_with(&cache, &build, container, ok), Ok(Stored::Written));
        assert_eq!(store_with(&cache, &build, container, ok), Ok(Stored::AlreadyCached));
        // The runtime's lookup with the same key does not translate.
        let never = |_: &[u8]| -> Result<HostShader, String> { panic!("translated again") };
        assert_eq!(cached(&cache, &build, container, never), Ok(shader()));
        // Another host's waves are another entry.
        let other = key(&[7; 32], Target { lanes_min: 32, lanes_max: 32 });
        assert_eq!(store_with(&cache, &other, container, ok), Ok(Stored::Written));
        // A damaged entry is written again; a failure is kept as its error.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH3 broken").unwrap();
        }
        assert_eq!(store_with(&cache, &build, container, ok), Ok(Stored::Written));
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(store_with(&cache, &build, b"other", fail).is_err());
        let never = |_: &[u8]| -> Result<HostShader, String> { panic!("translated again") };
        assert_eq!(cached(&cache, &build, b"other", never), Err("unsupported".to_owned()));
    }

    #[test]
    fn keys_tell_hosts_wave_sizes_apart() {
        let build = [7u8; 32];
        let nvidia = Target {
            lanes_min: 32,
            lanes_max: 32,
        };
        let rdna = Target {
            lanes_min: 32,
            lanes_max: 64,
        };
        assert_eq!(key(&build, nvidia), key(&build, nvidia));
        assert_ne!(key(&build, nvidia), key(&build, rdna));
        assert_ne!(key(&build, nvidia), key(&[8; 32], nvidia));
    }

    #[test]
    fn a_cache_that_cannot_keep_an_entry_still_gives_the_translation() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        // A file where every entry's folder would go: nothing can be written.
        for shard in 0..=0xff_u8 {
            std::fs::write(dir.path().join(format!("{shard:02x}")), b"").unwrap();
        }
        let translate = |_: &[u8]| Ok(shader());
        let container = b"a console container".as_slice();
        assert_eq!(cached(&cache, &[7; 32], container, translate), Ok(shader()));
        // A shader that does not translate still fails, with its own error.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        let error = cached(&cache, &[7; 32], b"other", fail).unwrap_err();
        assert!(error.contains("unsupported"), "{error}");
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }
}
