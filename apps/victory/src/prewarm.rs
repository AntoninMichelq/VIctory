//! Offline pre-warming of a cache's translated shaders and host pipeline
//! library (`host-pipelines.bin`) without starting the title.

use std::path::PathBuf;

use anyhow::Result;

const DEFAULT_SHADERS_DIR: &str = r"C:\Users\4nto\Desktop\VIctory\_work\prison\shaders";

#[cfg(windows)]
pub fn run(
    cache: PathBuf,
    mut shaders: Vec<PathBuf>,
    from: Option<PathBuf>,
    jobs: Option<usize>,
) -> Result<()> {
    if shaders.is_empty() {
        let default = PathBuf::from(DEFAULT_SHADERS_DIR);
        if default.is_dir() {
            shaders.push(default);
        }
    }
    let report = victory_gpu::pipeline::prewarm::prewarm(
        victory_gpu::pipeline::prewarm::PrewarmOptions {
            cache_dir: cache.clone(),
            shader_dirs: shaders,
            from,
            jobs,
        },
    )
    .map_err(anyhow::Error::msg)?;

    println!("cache          {}", cache.display());
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
) -> Result<()> {
    anyhow::bail!("`victory prewarm` requires Windows (host D3D12)")
}
