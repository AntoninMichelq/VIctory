//! Offline pre-warming of a cache's translated shaders and host pipeline
//! library (`host-pipelines.bin`) without starting the title.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use victory_game::sgd5;
use victory_shader::Container;

use crate::shader::MAX_FILE;

const DEFAULT_SHADERS_DIR: &str = r"C:\Users\4nto\Desktop\VIctory\_work\prison\shaders";
const DEFAULT_GAME_DIR: &str = r"C:\Users\4nto\Downloads\GameFolder";

/// Summary of containers discovered by [`collect_all_shaders`].
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct ScanSummary {
    files_scanned: usize,
    packed_found: usize,
    raw_found: usize,
    duplicates_skipped: usize,
    corrupt_skipped: usize,
}

#[cfg(windows)]
pub fn run(
    cache: PathBuf,
    mut shaders: Vec<PathBuf>,
    from: Option<PathBuf>,
    jobs: Option<usize>,
    all_shaders: bool,
    game_dir: Option<PathBuf>,
) -> Result<()> {
    let mut extracted = Vec::new();
    let mut scan = ScanSummary::default();
    if all_shaders {
        let targets = resolve_all_shader_targets(&shaders, game_dir.as_deref());
        anyhow::ensure!(
            !targets.is_empty(),
            "no shader archives or folders found for `--all-shaders` (pass `--game-dir` or `--shaders`)"
        );
        let (containers, summary) = collect_all_shaders(&targets)?;
        println!(
            "scan           {} unique containers ({} packed, {} raw, {} duplicates, {} corrupt) across {} files",
            containers.len(),
            summary.packed_found,
            summary.raw_found,
            summary.duplicates_skipped,
            summary.corrupt_skipped,
            summary.files_scanned
        );
        extracted = containers;
        scan = summary;
    }

    if shaders.is_empty() {
        let default = PathBuf::from(DEFAULT_SHADERS_DIR);
        if default.is_dir() {
            shaders.push(default);
        }
    }

    let shader_dirs: Vec<PathBuf> = shaders.into_iter().filter(|p| p.is_dir()).collect();
    let report = victory_gpu::pipeline::prewarm::prewarm(
        victory_gpu::pipeline::prewarm::PrewarmOptions {
            cache_dir: cache.clone(),
            shader_dirs,
            from,
            jobs,
            all_shaders: extracted,
        },
    )
    .map_err(anyhow::Error::msg)?;

    println!("cache          {}", cache.display());
    if all_shaders {
        let _ = scan;
        println!(
            "all-shaders    {} unique ({} cache hits, {} translated, {} failed)",
            report.all_shaders_total,
            report.all_shaders_hits,
            report.all_shaders_translated,
            report.all_shaders_failed
        );
    }
    println!(
        "recipes        {} total ({} from early-recipes.bin, {} compute matched from early-pipelines.txt)",
        report.recipes_total, report.from_recipes_file, report.compute_matched_from_early
    );
    if report.early_without_recipe > 0 {
        println!(
            "missing recipe {} keys in early-pipelines.txt have no recipe yet (one warm run with this build will record early-recipes.bin)",
            report.early_without_recipe
        );
    }
    println!(
        "pipelines      {} loaded, {} made, {} failed, {} missing containers",
        report.loaded, report.made, report.failed, report.missing_containers
    );
    println!(
        "shaders        {} cache hits, {} translated",
        report.shader_hits, report.shader_misses
    );
    println!(
        "library        {} bytes ({:.1} KiB) in {:.1} s",
        report.library_bytes,
        report.library_bytes as f64 / 1024.0,
        report.elapsed.as_secs_f64()
    );
    Ok(())
}

#[cfg(not(windows))]
pub fn run(
    _cache: PathBuf,
    _shaders: Vec<PathBuf>,
    _from: Option<PathBuf>,
    _jobs: Option<usize>,
    _all_shaders: bool,
    _game_dir: Option<PathBuf>,
) -> Result<()> {
    anyhow::bail!("`victory prewarm` requires Windows (host D3D12)")
}

/// Picks the files/folders to scan when `--all-shaders` is requested.
///
/// If explicit `--shaders` paths were passed, scans those; otherwise targets
/// the shader-bearing top-level files of `--game-dir` (or `game_dir` from
/// `victory.toml`, or `DEFAULT_GAME_DIR`): `shaders.rpf`, `*.exe`, `*.dll`,
/// avoiding multi-gigabyte audio/level/texture archives.
fn resolve_all_shader_targets(shaders: &[PathBuf], game_dir: Option<&Path>) -> Vec<PathBuf> {
    if !shaders.is_empty() {
        return shaders.to_vec();
    }
    let root = game_dir
        .map(Path::to_path_buf)
        .or_else(|| victory_core::Config::load().ok().and_then(|c| c.game_dir))
        .or_else(|| {
            let p = PathBuf::from(DEFAULT_GAME_DIR);
            p.is_dir().then_some(p)
        });
    if let Some(dir) = root {
        let mut files = top_level_shader_files(&dir);
        if !files.is_empty() {
            files.sort();
            return files;
        }
        return vec![dir];
    }
    let fallback = PathBuf::from(DEFAULT_SHADERS_DIR);
    if fallback.is_dir() {
        vec![fallback]
    } else {
        Vec::new()
    }
}

/// Top-level files in a game directory that hold shader containers
/// (`shaders.rpf`, `*.exe`, `*.dll`).
fn top_level_shader_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if name == "shaders.rpf" || ext == "exe" || ext == "dll" {
            out.push(path);
        }
    }
    out
}

/// Extracts every unique, exact-sized DXBC shader container from `targets`
/// (individual files or folders).
fn collect_all_shaders(targets: &[PathBuf]) -> Result<(Vec<Vec<u8>>, ScanSummary)> {
    let mut containers = Vec::new();
    let mut seen: HashSet<[u8; 32]> = HashSet::new();
    let mut summary = ScanSummary::default();

    for target in targets {
        if target.is_file() {
            scan_file(target, &mut containers, &mut seen, &mut summary);
            continue;
        }
        if target.is_dir() {
            // If this looks like a game root (contains `shaders.rpf`), scan
            // only top-level shader files so we never read 100+ GiB of levels/audio.
            if target.join("shaders.rpf").is_file() {
                for file in top_level_shader_files(target) {
                    scan_file(&file, &mut containers, &mut seen, &mut summary);
                }
                continue;
            }
            for entry in walkdir::WalkDir::new(target)
                .into_iter()
                .filter_map(Result::ok)
            {
                if !entry.file_type().is_file() {
                    continue;
                }
                let path = entry.path();
                if entry.metadata().map(|m| m.len()).unwrap_or(0) > MAX_FILE {
                    continue;
                }
                scan_file(path, &mut containers, &mut seen, &mut summary);
            }
        }
    }
    Ok((containers, summary))
}

fn scan_file(
    path: &Path,
    out: &mut Vec<Vec<u8>>,
    seen: &mut HashSet<[u8; 32]>,
    summary: &mut ScanSummary,
) {
    let Ok(meta) = std::fs::metadata(path) else {
        return;
    };
    if meta.len() == 0 || meta.len() > MAX_FILE {
        return;
    }
    let Ok(data) = std::fs::read(path) else {
        return;
    };
    summary.files_scanned += 1;
    extract_containers(&data, out, seen, summary);
}

/// Extracts all packed SGD5 and raw DXBC containers from `data`, slicing each
/// to its exact DXBC header total size so BLAKE3 cache keys match live runs.
fn extract_containers(
    data: &[u8],
    out: &mut Vec<Vec<u8>>,
    seen: &mut HashSet<[u8; 32]>,
    summary: &mut ScanSummary,
) {
    if sgd5::is_sgd5(data) {
        for record in sgd5::records(data) {
            let Some(unpacked) = record.unpack() else {
                summary.corrupt_skipped += 1;
                continue;
            };
            let Some(exact) = exact_dxbc_slice(&unpacked) else {
                summary.corrupt_skipped += 1;
                continue;
            };
            summary.packed_found += 1;
            let hash = *blake3::hash(exact).as_bytes();
            if seen.insert(hash) {
                out.push(exact.to_vec());
            } else {
                summary.duplicates_skipped += 1;
            }
        }
    }
    for offset in memchr::memmem::find_iter(data, b"DXBC") {
        let Some(exact) = exact_dxbc_slice(&data[offset..]) else {
            continue;
        };
        summary.raw_found += 1;
        let hash = *blake3::hash(exact).as_bytes();
        if seen.insert(hash) {
            out.push(exact.to_vec());
        } else {
            summary.duplicates_skipped += 1;
        }
    }
}

/// Validates a DXBC container and returns its exact `[..total_size]` slice.
fn exact_dxbc_slice(data: &[u8]) -> Option<&[u8]> {
    Container::parse(data).ok()?;
    let total = u32::from_le_bytes(data.get(0x18..0x1c)?.try_into().ok()?) as usize;
    data.get(..total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_dxbc(payload: &[u8]) -> Vec<u8> {
        // Minimal valid DXBC container with 1 part (`DXIL`).
        let part_offset = 0x24u32;
        let total = part_offset + 8 + payload.len() as u32;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(b"DXBC");
        out.extend_from_slice(&[0u8; 16]); // digest
        out.extend_from_slice(&1u16.to_le_bytes()); // major
        out.extend_from_slice(&0u16.to_le_bytes()); // minor
        out.extend_from_slice(&total.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes()); // part count
        out.extend_from_slice(&part_offset.to_le_bytes());
        out.extend_from_slice(b"DXIL");
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn exact_dxbc_slice_strips_trailing_bytes() {
        let mut blob = fake_dxbc(b"hello-shader");
        let expected = blob.clone();
        blob.extend_from_slice(b"trailing-garbage-from-archive");
        assert_eq!(exact_dxbc_slice(&blob), Some(expected.as_slice()));
        assert_eq!(exact_dxbc_slice(b"not-a-dxbc"), None);
    }

    #[test]
    fn extract_containers_deduplicates_raw_and_ignores_garbage() {
        let a = fake_dxbc(b"shader-A");
        let b = fake_dxbc(b"shader-B");
        let mut blob = Vec::new();
        blob.extend_from_slice(b"prefix");
        blob.extend_from_slice(&a);
        blob.extend_from_slice(b"DXBC\x00\x00"); // corrupt header
        blob.extend_from_slice(&b);
        blob.extend_from_slice(&a); // duplicate of A

        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut summary = ScanSummary::default();
        extract_containers(&blob, &mut out, &mut seen, &mut summary);

        assert_eq!(out.len(), 2);
        assert_eq!(out[0], a);
        assert_eq!(out[1], b);
        assert_eq!(summary.raw_found, 3);
        assert_eq!(summary.duplicates_skipped, 1);
    }
}
