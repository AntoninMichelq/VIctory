# VIctory on an RX 7600 — project handoff (2026-11-25)

## Résumé en français (pour 4nto)

- **Ce qui marche aujourd'hui** : le raccourci du Bureau « Jouer VIctory (tout cumulé).bat » lance **Peau3** (commit `b74ac64`, branche `jouer/peau3`). La prison est éclairée, avec bâtiments, personnages, peau et cheveux. Environ **6,2 à 6,5 images/s**, environ 21 Go de RAM, sur la RX 7600 avec FSR (rendu 1080p).
- **Le gros défaut qui reste** : des reflets « chrome » ou des taches noires au hasard sur les habits et la peau. Cause trouvée : quand la carte graphique (8 Go) est pleine, des textures attendent ou sont retirées, et le jeu lisait des zéros à leur place. Premier correctif essayé le 25/11 au soir (branche `fix/wait-grey`, `89e0bb6`) : la chemise redevient mate, mais il reste des affiches noires, des visages noirs et du chrome. Il n'est donc pas dans Jouer. La vraie solution : retirer beaucoup moins de textures (voir section 8).
- **Pour aller plus vite** : l'« essai groupé » (`essai/groupe`) passe de ~6,4 à ~7 images/s, mais il donne une tête flottante et des taches argentées. Il n'est donc pas encore dans Jouer.
- **Règles à garder** : un seul jeu lancé à la fois. Ne jamais fermer le jeu de force pendant que tu joues. Toujours garder un « Jouer precedent ». Ne jamais publier de fichiers du jeu.
- Les **prompts à coller dans T3 Code** sont tout en bas (section 12).

---

Everything below is written for coding agents (Claude, Gemini/Antigravity) that pick this
project up in **T3 Code** on the user's PC. Read it fully before touching the build.

## 0. Where things are

| What | Where |
|---|---|
| Fork (all our work) | https://github.com/AntoninMichelq/VIctory-forkAMD (old name AntoninMichelq/VIctory redirects) |
| Upstream | https://github.com/Marvisak/VIctory (GPL-3). We cannot push there or comment via API. |
| Our real base | fxceless/victory-fork `4e17478` + our commits (user decision 2026-11-24 21:55 Paris) |
| PC clone | `C:\Users\4nto\Desktop\VIctory` — **its `origin` is Marvisak upstream, not the fork** |
| Scratch / test launchers / exes / logs | `C:\Users\4nto\Desktop\VIctory\_work` (`_work\runs\run-*.log`) |
| Game (never delete, never publish) | `C:\Users\4nto\Downloads\GameFolder` (`game_scarlett_master.exe`, game version 1.0.2.1) |
| Shader caches | `C:\Users\4nto\victory-cache\<name>` (must be outside AppData because of MSIX virtualisation) |
| Notes copied from the old project | `docs/notes/` in this repo (text only) |

Get the fork on the PC (one time):

```bat
cd /d C:\Users\4nto\Desktop\VIctory
git remote add fork https://github.com/AntoninMichelq/VIctory-forkAMD
git fetch fork
git show fork/main:HANDOFF.md
```

A branch from the fork is then `fork/<branch>`, e.g. `git switch -c work fork/jouer/peau3`.

## 1. The PC

- Windows 11, AMD **RX 7600 (8 GB VRAM)**, 8C/16T CPU, **32 GB RAM**, pagefile **32768 MB** (needed: world load peaks ~22 GB commit), registry `TdrDelay=10`.
- Keyboard = pad: Enter=A, Backspace=B, arrows=d-pad, WASD=left stick, Esc=Menu.
- `DISABLE_VULKAN_OBS_CAPTURE=1` always.
- AMD's `amd_fidelityfx_upscaler_dx12.dll` (FidelityFX SDK 2.x) sits in `C:\Users\4nto\Desktop\VIctory` for host FSR.

## 2. User rules (non-negotiable)

The user speaks **French**, is not very technical: answer in simple French, step by step.

1. **One VIctory instance at a time.** Check no `victory*.exe` runs before launching. Never kill the user's game while they play. Close test runs with Alt+F4 (no taskkill).
2. Only come back to the user with a build that shows **something new in game**, through the **single Desktop shortcut** `Jouer VIctory (tout cumulé).bat`. Everything else (test bats, exes) goes in `_work`. One checkout, one build folder; don't create new folders on the PC; never delete without asking.
3. Each Jouer update = **all** accumulated working fixes; always keep a fallback `_work\Jouer precedent.bat`.
4. Only **proven** builds go into Jouer: no crash, fps >= the current Jouer, image at least as good (judge image defects over **repeated** shots, single runs are flaky). More RAM/VRAM is fine if it buys speed.
5. Prefer grouped trials over one-by-one; parallelise work across cores.
6. Before each build round, check upstream + reference forks for new useful commits (section 9) and merge them; don't ask the user to check.
7. **Never attach or commit game files**, extracted assets, shader dumps, generated shaders, or screenshots of private tools to issues/PRs/messages. Reports that list shader hashes are fine.
8. Read-only on other people's repos unless the user OKs a port.
9. Unattended test runs are allowed (one at a time), but a *new* run needs the user's written OK naming it.

## 3. Build & check — lessons that cost hours

- Build: `cargo build --release -p victory`, copy `target\release\victory.exe` to `_work\victory-play-<name>.exe`. Only **release** builds count for fps (diagnostics builds overstate costs ~10x).
- Each new exe needs a warm-up launch (AMD driver cache is per exe). A new **shader cache key** (translator change or new `VICTORY_LEAN_MATH`) = cold run, ~1,800 pipelines, ~10 min before fps settles. Cache is keyed by translator sources (not the exe), so reuse one cache dir per translator line (`tout3q2` for Peau3).
- Build trial exes **on the current Jouer commit with the same env**, otherwise comparisons are meaningless.
- **Never `cargo fmt -p victory-shader` or `-p victory-gpu`** (upstream isn't fmt-clean; it rewrites unrelated files). rustfmt only the files you touched.
- `victory-gpu` does not build on Linux. Cross-check: `CARGO_FEATURE_PURE=1 cargo clippy -p victory-gpu --target x86_64-pc-windows-msvc -- -D warnings` (from Linux also set `CC_x86_64_pc_windows_msvc=clang-cl-18 AR_x86_64_pc_windows_msvc=llvm-lib`). Shader tests: `cargo test -p victory-shader` (Linux: `--target x86_64-unknown-linux-gnu`).
- CI runs clippy `-D warnings` (default and all features) + `cargo test --workspace`, which includes xtask's **asset policy: no file > 512 KiB**, only text extensions, no long hex/base64 runs. `crates/victory-shader/src/emit/lower.rs` hit the limit once: its tests now live in `emit/lower/tests.rs`; keep `lower.rs` small.
- Any DirectStorage change needs a real game run before Jouer (e39be20 regressed).
- "Missing" objects: wait for full stream-in before calling an object missing.
- Offline translation check of the whole title shader set: `victory translate --pipelines` over `_work\prison\shaders\*.bin` (19,105 dumps; private, never publish).

## 4. Current Jouer = cumul9b (promoted 2026-11-26 06:54)

- Exe `_work\victory-play-cumul9b.exe` built from **`jouer/cumul9b` = `7479227`** = `jouer/cumul9` cf5a068 + prep4 (`perf/prep3` 6842074,
  `JOURNAL_SHARED=2` with review-4 fixes, review 8: safe). Line of integrations: cumul (groupe-fix, overhead, loading-full, depth-bounds,
  uploads, evict-less) → cumul2 (copies, FSR dynamic) → cumul3 (live switches, async compute, hot shaders, dropped-draw report, prewarm)
  → cumul4 (plan-reuse, copies3, cpu-frame, loading-full streak, missing-walls diag) → cumul5 (copies4, levels) → cumul6 (copies5, quality, levels2)
  → cumul7 (loading/PHYS_INDEX, ports1/CODE_TABLE_AFTER, lower-size, copies6 fixed, prep2) → cumul8 (ports2, prep3, fsrdyn2, texuploads)
  → cumul9 (timeline3 with ALIAS_ON_CARD **off**, fsrsharp) → cumul9b (prep4). Every fiche: `_work\essais\cumulN.md`.
- **Cache `C:\Users\4nto\victory-cache\cumul7`** (translator changed in cumul7: ports1 adds `CODE_TABLE_AFTER` to the cache key).
- Measured (T41, 15 min): **8.87 img/s, 112.8 ms/frame, card ~51 ms**, no crash; quality stats: 0 texture reads null / partial / waiting.
  A/B live: `JOURNAL_SHARED=2` −5.5 to −5.9 ms, `INDIRECT_COMPACT` neutral. (Peau3 was 6.15–6.5 fps, card ~70 ms.)
- Fallback: `_work\Jouer precedent.bat` = **Peau3** (`b74ac64`, `victory-play-peau3.exe`, cache tout3q2); Peau2 kept as `_work\Jouer precedent (peau2).bat`.
- Candidate not promoted: cumul9c (`jouer/cumul9c` ceb0abc = 9b + review-7 fix of ALIAS_ON_CARD, launcher `_work\Jouer candidat cumul9c (temporaire).bat`) —
  **suspended**: timeline3 (ALIAS_ON_CARD) removed the device ~60 s into the world. Procedure for a promotion: `_work\essais\promotion.md`.
- Next integration built, not promoted: cumul10 (`jouer/cumul10` f1fbbb5 = cumul9c + volum + walls9, new cache `cumul10`, pre-warmed offline;
  ALIAS_ON_CARD=0 there).

Jouer env (exact, from the Desktop shortcut; `CARD_BUFFERS_MB` is set twice, the second, 1024, wins):

```bat
set DISABLE_VULKAN_OBS_CAPTURE=1
set VICTORY_LOG=info
set VICTORY_CACHE_DIR=C:\Users\4nto\victory-cache\cumul7
set VICTORY_LATE_WAIT=3
set VICTORY_WIDE_BACK=1
set VICTORY_WIDE_MERGE=1
set VICTORY_JOIN_COPIES=1
set VICTORY_WIDE_REFRESH=1
set VICTORY_RELAY=2
set VICTORY_TEXTURE_NO_WAIT=1
set VICTORY_CARD_BUFFERS_MIN_MB=1
set VICTORY_CARD_BUFFERS_MB=512
set VICTORY_STAGING_POOL_MB=1024
set VICTORY_VRAM_HYSTERESIS=85,75
set VICTORY_AMBIENT=off
set VICTORY_TEXTURE_HALF=2
set VICTORY_WAVE_WIDE=1
set VICTORY_SHADER_OPT=0
set VICTORY_FIDELITYFX_DIR=C:\Users\4nto\Desktop\VIctory
set VICTORY_UPSCALER=7E93F0
set VICTORY_FFX_SCALE=0.42
set VICTORY_WIDE_TEXTURES=1
set VICTORY_CARD_GATHER=1
set VICTORY_NO_COMPILE=none
set VICTORY_LEAN_MATH=247
set VICTORY_HOIST_CLAMPS=9
set VICTORY_FLAT_INSTANCES=e732a8b4
set VICTORY_CLAMP_BATCH=1
set VICTORY_CLAMP_BIND_ONCE=1
set VICTORY_UPLOAD_BARRIERS=1
set VICTORY_DEPTH_BOUNDS=1
set VICTORY_EVICT_LESS=1
set VICTORY_RESIDENCY_PRIORITY=1
set VICTORY_CARD_BUFFERS_MB=1024
set VICTORY_CARD_BUFFERS_ROOM=95
set VICTORY_FFX_DYNAMIC=33
set VICTORY_FFX_SCALE_MIN=0.38
set VICTORY_ASYNC_COMPUTE=1
set VICTORY_HOT_SHADERS=3
set VICTORY_CARD_SEND_CHANGED=1
set VICTORY_FAST_TITLE_COPY=1
set VICTORY_REFRESH_RANGES=1
set VICTORY_BACK_BLOCKS=1
set VICTORY_FAST_TITLE_WRITE=1
set VICTORY_WIDE_THREADS=12
set VICTORY_REFRESH_SPLIT_MB=4
set VICTORY_SNAP_STREAM=1
set VICTORY_PLAN_REUSE=1
set VICTORY_CARD_GATHER_PIECES=640
set VICTORY_LEVELS_ON_DEMAND=0
set VICTORY_SETTLE_PARTIAL=0
set VICTORY_CODE_TABLE_AFTER=1
set VICTORY_PHYS_INDEX=1
set VICTORY_FLUSH_PART=1
set VICTORY_FLIGHT_CHECKS_MAP=1
set VICTORY_FENCE_PARTIAL=0
set VICTORY_INDIRECT_COMPACT=1
set VICTORY_TEX_UPLOAD_POOL=1
set VICTORY_SCOPED_UAV=0
set VICTORY_TEXTURE_UPLOAD_BANDS=0
set VICTORY_FFX_SHARPNESS=auto
set VICTORY_MIP_BIAS=auto
set VICTORY_ASK_ONCE=1
set VICTORY_ALIAS_ON_CARD=0
set VICTORY_JOURNAL_SHARED=2
"_work\victory-play-cumul9b.exe" run "C:\Users\4nto\Downloads\GameFolder\game_scarlett_master.exe" --execute > "_work\runs\run-%TS%-jouer.log" 2>&1
```

Notes on that env:
- The shortcut refuses to start only if a **game** runs (`victory-play-*.exe` with ` run ` on its command line), not for `victory.exe` tools.
- `FFX_SCALE=0.42` (1612x907) + `FFX_SHARPNESS=auto` (RCAS) + `MIP_BIAS=auto` (−2.25): card 55 → 38 ms against 0.5, sharpness back to the 0.5 level
  (gradient energy −0.6 %). `FFX_DYNAMIC=33` does nothing useful: the title reads its scale once (fsrdyn2.md).
- OFF on purpose (reviews 4/5, crash-cumul7): `LEVELS_ON_DEMAND`, `SETTLE_PARTIAL`, `FENCE_PARTIAL`, `SCOPED_UAV`, `TEXTURE_UPLOAD_BANDS`,
  `ALIAS_ON_CARD`; absent: `MERGE_MORE`, `ALIAS_PARTIAL`, `VRS`. All are read live (below) for A/B once their authors fix them.
- `TEXTURE_HALF=2` stays: `TEXTURE_HALF=0` does not fit 8 GB (cumul5 L0 device removed at 6,948/6,965 MB; cumul6 L0 ran but over budget, 7,147/6,910 MB).
- `NO_COMPILE=none`, `CARD_BUFFERS_MIN_MB=1` + `CARD_BUFFERS_MB` must stay, as before.
- **Live switches** (`VICTORY_LIVE=<file>`, tools/live-switches): `NAME=value` lines read every second, applied at the next frame, logged
  `live switch NAME=value at frame N`; the fps line carries `live=<n>`. Which switches are live and which are read at the start: each cumulN.md
  (`À chaud` section). Translation switches (`LEAN_MATH`, `HOT_SHADERS`, `CODE_TABLE_AFTER`), `DEPTH_BOUNDS`, `FFX_SCALE`, `ASYNC_COMPUTE`,
  `WIDE_THREADS`, `CARD_GATHER`, `ALIAS_COPIES`, `LATE_WAIT`, `PHYS_INDEX`, `MIP_BIAS` are start-only.

## 5. Rejected / never do

| Thing | Why |
|---|---|
| `VICTORY_DISPLAY_MODE` (any value) | OOM; title output is fixed 4K anyway |
| `FFX_SCALE=0.6667` | E_OUTOFMEMORY on PSOs |
| `VICTORY_INPLACE=1/2` | missing torso on characters; never combine LATE_WAIT with INPLACE=2 |
| `VICTORY_AMBIENT` on (port `343827a`) | removes walls/glass; keep `AMBIENT=off` |
| `--console series-s` | stalls at boot (never opens core_restofworld); fixes a92bc87/c3ab396 did not help; paused by user |
| `VICTORY_SHADER_OPT_BIG` | device removed |
| dstorage `e39be20` | regression, reverted |
| `VICTORY_PIPELINES_ALL_EARLY=1` cold | never drains |
| `VICTORY_ASYNC_SUBMIT` on our line | does not exist in our tree (exists only on the fx merge) |
| fork merge line `nuit/tout-4` (105ee06) / `tout-5` | bald player / hair strands lost; tout-5 crashes the AMD compiler |
| quad-helpers `3024779` alone | crashes amdxc64; use narrowed bd85106 (already in Jouer line) |
| RT on (`BVH_BUILDER=1` + `QUARANTINE=1`) | works but ~1–1.9 fps (copies); `_work\"Jouer avec RT"` only |
| root-UAV copy-back for `caa78b8c`/`28918b71` | title crash; they stay on `VICTORY_ROOT_UAV_NO_COPY_BACK_FOR` |
| `VICTORY_RELAY=2` w/o `LATE_WAIT=3` | relay needs stage 3 |
| Running a downloaded dxc in cloud, or `cargo fmt` whole crate | refused / rewrites the tree |

## 6. Candidates not yet in Jouer (all pushed to the fork)

| Branch / commit | What | Status |
|---|---|---|
| `fix/evict-grey` `056a9b5` (on b74ac64) | texture slots given up for VRAM room point at a 4x4 grey stand-in instead of a NULL view (`VICTORY_EVICT_GREY=0` = old) | superseded by the next line |
| **`fix/wait-grey` `89e0bb6`** (on 056a9b5) | also slots still **waiting** for their host copy, and placed/streamed textures; integer formats stay 0 (`VICTORY_WAIT_GREY=0` = control) | Run 2026-11-25 20:48 (run-20261125-204839-jouer-evictgrey): shirt matte at the window, BUT posters solid black, face/neck black in corridor A102, NPCs in cells still chrome; given-up count doubled (65.6k at w+300); no "grey" line in the log, so the stand-in use is unconfirmed; 4.90 fps (heavy scene), 21.8 GB. **Not clean, not promoted.** |
| **`fix/partly-clamp` `d1e021b5`** (on 89e0bb6) | ~1,700 slots stay "uploaded in part" (streamed textures, memory partly mapped): their missing top levels read as zeros. Remakes those slots with `ResourceMinLODClamp` at the first fully mapped level, drops the clamp once whole (`VICTORY_PARTLY_CLAMP=0` = off). Expect INFO "texture slots uploaded in part remade…" | **Not built yet.** Needs Windows clippy -D warnings, then a common-room run with shots at w+360/400 |
| `fix/loading-sharp` `2cde495` | TEXTURE_HALF skips non-power-of-two textures (loading art sharp again); `VICTORY_TEXTURE_HALF_NPOT=1` = old | compiled, not run alone |
| `jouer/peau3-overlay` `bbc847a` (aeebeec + fix) | overlay only shows console mode + FSR lines; box sized to its lines | compiled |
| `perf/lean-math` `4237bce` | `VICTORY_LEAN_MATH` bitmask: 1 legacy mul in 3 ops (+0 for zero product), 2 plain max/min, 4 med3 w/o NaN, 8 plain mul + operand clamps, 16 finite pixel outputs, 32 v_mad as FMad, 64 fast rcp. In the cache key. | L7 6.68 fps (+4%, GPU 70.7 ms, grey-white hair at w+120 before the 4237bce fix); L31 6.93 (+8%, GPU 61.6); L127 in the group. **Chrome seen on L31 was also on the Peau3 control → not proof against it.** L1/L6/L97 split pending |
| `perf/clamp-ledger` `7068acc` (c07e4ac, 8a89721) | hoisted clamps without barriers + ledger (`VICTORY_HOIST_CLAMPS=8/9`, 7 = Peau3) | 9 lost the torso once (fixed 7068acc); suspect for the floating head |
| `perf/tiles-probe` `142aa83` (8e09544) | flat instances for the tile shaders (`VICTORY_FLAT_INSTANCES=e732a8b4`), `VICTORY_DISPATCH_PEEK` diag | suspect for the floating head |
| **`essai/groupe` `f5a52e9`** | = lean 4237bce + ledger 7068acc + flat 142aa83 + loading-sharp 2cde495 + overlay aeebeec. `essai/groupe-base` `9beab0a` = without the last two; `essai/groupe-overlay` `0bdc220` = + bbc847a. Env: Jouer + `LEAN_MATH=127`, `FLAT_INSTANCES=e732a8b4`, `DISPATCH_PEEK=39f0e9ae,5cf00dc3`, cache `victory-cache\groupe` | ~7.0 fps steady, **GPU ~53–57 ms vs 70**, +1.5 GB; silver patches on characters and a **floating head** in the common room. Splits A/B/C (each part removed) were queued: see section 7 |
| `port/fx-d510ee4` `23e5a09` | Peau3 merged with fxceless main d510ee4 (fx claims 9.6–10.3 fps on RTX 4070). Plan: `docs/notes/fx-d510ee4-essai.md` (F0 = regression check, F1 = fx switches `ASYNC_SUBMIT QUEUE2 COPY_PLAIN JOIN_PARTS BACK_OWN_VIEW CARD_REHASH_RUNS`, F2 = + `INDIRECT_WINDOW INDIRECT_GATHER HOIST_CLAMPS=0`; never with CARD_GATHER/CARD_FETCH_INFLIGHT) | Windows cargo check OK, **never run in game** |
| `port/fx-heap-sync` `a898679` | fx descriptor-sync fingerprint on perf/30fps-pipeline | not run |
| `feat/map-tour-2` / `nuit/tour-3b` `54d8982` | teleport tour of the map to pre-warm shaders (`VICTORY_TOUR`) | paused by user; resume on tour-3b |
| `fix/series-s-boot` `a92bc87` | Series S memory figures | did not fix the stall |

## 7. Last results (night of 2026-11-25/26, ~07:00 Paris)

- **Promoted cumul9b** (see section 4): 8.87 img/s, 112.8 ms/frame, card ~51 ms, 15 min stable. The user asked to stop after it:
  remaining work in `_work\essais\A-FAIRE.md`; every event of the night in `_work\essais\EVENEMENTS.log`.
- **ALIAS_ON_CARD** (perf/copies6 → timeline2/3/4): aliased refreshes take the flights' bytes by card copies instead of a wait.
  timeline2: **−20 ms/frame** (104 vs 125 ms, wait 48 → 1 ms). But **timeline3 removed the device ~60 s into the world**; the review-7 fix
  (7345a5da) is in cumul9c/cumul10, untested in a long run. Keep it off until a review + run say otherwise.
- **FSR 0.42** (cumul7): card 55 → 38 ms, frame 130 → 116 ms (8.64 img/s); with fsrsharp the image is as sharp as 0.5.
- **prep4** `JOURNAL_SHARED=2` (kernel write-watch call outside the journal lock): −5.5 to −7 ms/frame, review 8 safe → in Jouer.
- **Crashes LEVELS / HALF=0** (`_work\essais\crash-cumul7.md`): cumul5 L0 and cumul7's cold first launch died the same way — DEVICE_HUNG,
  no breadcrumbs, the frame collapsing to 0.1–0.4 img/s under extreme load (full-size textures or cold pipeline compiles), with
  LEVELS_ON_DEMAND and ASYNC_COMPUTE on. Suspect (not proven): LEVELS tiles given back to the pool and remapped on the direct queue while a
  compute-queue list still reads the old texture. The device-removed line said `VICTORY_RELAY=1` whatever the mode (text fixed in cumul8).
  For the next cold run set `VICTORY_D3D12_DRED=1` (works in release).
- **Missing walls** (walls2–walls9 fiches): the title does submit the walls; they are lost **on our side** at render time, depending on timing
  (not the indirect budget: 0 records emptied; not untranslated mesh shaders: 6,685/6,685 translate). walls9 (`RECOPY_ALL_EVERY` without
  touching card-written copies) is in cumul10, not yet run. Live A/B of 8 switches on cumul6 moved fps ±4 %, captures still to compare.
- Frame now ≈ preparing ~72 ms + card ~50–55 ms **in series** (copies7 A/B): the CPU side is the next target (prep/copies threads).
- **Tools** (all in `_work\outils`): `live_ab.ps1` (one run, A/B steps written into the VICTORY_LIVE file, one switch per step, `-Avant 180`),
  `live_stats.ps1 -Log <run> -Etapes <runs\<tag>-live.txt>` (per step: medians of ms/frame, GPU, preparing, copies, copying back, draws;
  B vs the A around it; log times are UTC, local = UTC+1), `comparer_captures.py` (image comparison of captures, gradient energy),
  `derive_bat.py` (derive a test/candidate launcher from another), `essai.ps1` (fixed-name test runs), `prewarm` (`victory-play-<x>.exe prewarm
  --cache <new> --from <old>`, offline pipeline pre-warm, paused during tests).
- Builds: sccache in `~/.cargo/config.toml` (its `rustc-wrapper` path must be a TOML literal string, single quotes); compiling is allowed at any
  time (the test script pauses rustc/link during a run).

## 8. Open problems, ranked

1. **Random chrome / rainbow / black patches on characters and floors** (also on Jouer). Cause: texture slots evicted or waiting under VRAM pressure were NULL views → zeros (black albedo, normal (-1,-1), roughness 0 = mirror). `fix/wait-grey` only partly helps (it also doubled given-up slots). Next leads: (a) `fix/partly-clamp` d1e021b5 for partly mapped streamed textures; (b) black faces/necks are probably the **baked skin** (PS_BakeDecorations render target), separate from eviction; (c) evict less. Real fix next: give up far fewer slots — evict least-recently-sampled instead of oldest-made, or keep/remake small mips instead of emptying the slot. Code: `crates/victory-gpu` texture queue / `texture_slots_evict`, `resource::room()` (90% of the VRAM budget).
2. **Floating head / torso missing**: seen only on trial builds with the clamp ledger (mode 9, 8a89721/7068acc) — its logs show 49–80k "a flight's shaders write it" refreshes vs 0–72 on other builds; flat tiles (`FLAT_INSTANCES`) not cleared either. Bisect with the group splits A/B/C; logs of ledger `8a89721` show 49–80k "a flight's shaders write it".
3. **Speed: GPU-bound ~70 ms/frame** (~6.4 fps, user's goal was 10). Biggest card costs (F0d, 105.7 ms/frame diag): ScreenProbes `39f0e9ae` 8.8 ms (1 dispatch, scalar-bound; idea: async compute queue), tiles `e732a8b4`/`456f02b5` 8.5, hoisted clamps 7.0, scattering gathers ~9, shadowmask `5cf00dc3` 2.6. Levers: lean math (−8..−18 ms), ledger, flat tiles, fx merge, `FFX_SCALE=0.4`.
4. Eyes glowing white in cutscenes, satin streaks on shirts, blotchy inmate faces (some may be #1).
5. Loading screens blurry (fix ready) and overlay too long (fix ready).
6. Untranslated: 195 `GS_VoxelizeGeneric` (voxel GI; fxceless PR #309 is doing it), `water_shading` (f64), 32 KiB mesh-output fire/decal shaders.
7. Intro video stutter (host Media Foundation, `media.rs`): low priority, paused.
8. Series S mode stall (paused).

## 9. Reference repos — check before every build round

- https://github.com/fxceless/victory-fork — our base (4e17478). Last seen main d510ee4.
- https://github.com/Belobug/VIctory-Belobug-AMD — RX 9070 XT; on upstream 8e79c80 (ports need rewriting). Hoisted clamps came from there.
- https://github.com/pragyaangaur/VIctory-macOS — behind us, universal fixes only.
- https://github.com/ZSlayerHQ/zslayer-pc-port-runtime — = fx 310c6eb + victory-amd, nothing faster.
- Upstream Marvisak/VIctory (gets ~1 PR/min; most conflict with our base).
- ivall/xcl is GPL-2: ideas only, never copy code.

Method: one blobless clone with remotes (`git fetch --filter=blob:none`), `git log 4e17478..fx/main`. Comparison: `docs/notes/3-depots-reference-2026-11-25.md`, `docs/notes/nouveautes-repos-2026-11-25.md`.

## 10. Every thread of the old project, in short

Newest first. "Branch" = fork branch.

- **Écrire le document de reprise** — this file.
- **Sync with latest VIctory and xcl** (the PC-driving thread) — built every Jouer, ran all trials (L7/L31/L127, control, group, sans-sondes, evict-grey). Owns `_work` bats. Pushed `essai/groupe*`.
- **Make the texture and lighting calculations work** — `fix/wait-grey` 89e0bb6 (chrome cause). Earlier: copy-back runs 379f21a, root-UAV bisect, `fix/root-uav-gpu-owned`.
- **Éclairer le corps et les habits du personnage** (characters) — proven in Peau3: evict requeue deee1941, alias copies cff6214b/d8780f2a (INPLACE=1 loses torsos). New: `fix/partly-clamp` d1e021b5. Earlier skin/hair: bake textures 63a1e8c8, evict requeue deee1941 (`VICTORY_EVICT_REQUEUE_FRAMES=30`), alias copies refreshed d8780f2a (ponytail, no chrome shirts), `fix/evict-grey` 056a9b5. Offline tool pedtool (private). Next: evict fewer slots.
- **Accélérer les shaders traduits** — `perf/lean-math` (section 6). Method notes in `docs/notes` and memory: corpus `_work\shaders-corpus.tgz`, `examples/dump.rs`, a local-only DXIL datalayout hack (never commit).
- **Reprendre les optimisations AMD de deux forks** — `port/fx-d510ee4` 23e5a09, plan `docs/notes/fx-d510ee4-essai.md`. Not run.
- **Simplifier la liste de l'overlay** — aeebeec + bbc847a (`jouer/peau3-overlay`).
- **Rendre nettes les images de chargement** — `fix/loading-sharp` 2cde495.
- **Raise VIctory frame rate in the world** — from 0.6 fps to 6.4: LATE_WAIT=3/WIDE_*/JOIN_COPIES, RELAY=2, card hash #253, texture detile workers auto (35231204), host FSR (`feat/host-fsr` 3e4f369), shader cache key by sources (7bf6cfc), CARD_GATHER, heap pass, clamp ledger, flat tiles. Full history `docs/notes/nuit-2026-11-25/fps-detail-2026-11-25.md`.
- **Comparer nos avancées aux 3 forks principaux** — `docs/notes/3-depots-reference-2026-11-25.md`; ported hoisted clamps + self-copy skip (`port/refs-2026-11-25`), ambient (rejected).
- **Publier mes changements sur main** — fork main = Peau3 + README (results on RX 7600). Updates: merge the new Jouer branch into main + README commit, plain push (no force push). Old main backed up as `ancien-main` (70564ae).
- **Bring in latest fork and xcl updates** — `port/forks-2026-11-25` a8622f4 / `port/forks-on-tout-3` 915e38b (fx a26859f + victory-amd + VItoriaBr); its follow-up line tout-4 regressed hair.
- **Afficher le sol et les bâtiments** — ground/buildings came from the fxceless base + 9-bit mesh index fix 5173d96 (needle triangles). Culling computes need root UAVs.
- **Preload shaders across the whole map** — `feat/map-tour`, `perf/preload` (cdfbe3a, 44add24, `VICTORY_PRELOAD_DRAWS`). Paused.
- **Answer new comments on upstream PRs** — drafts in `docs/notes/upstream-prs/`; the user posts them by hand.
- **Lister les nouveautés des repos** — `docs/notes/nouveautes-repos-2026-11-25.md`.
- **Afficher la première cinématique** — early PC thread (first image of the cutscene, G-buffer view, launcher scheme).
- **Faire apparaître les objets du monde** (blocked, old) — vehicle bodies missing; mesh-root-reach hypothesis disproven; tool vtool (private).
- **Find and triage VIctory's open problems** (blocked, old) — triage `docs/notes/triage-2026-11-24.md`, shader gap fixes, fork PR #1 `integrate/all` (old d-line).
- **Reprendre l'existant pour dessiner le décor** — `docs/notes/decor-reuse/`.
- **Afficher les véhicules et objets** — see "objets du monde".
- **Learn from VIctory and xcl forks** / **Learn from the xcl project** — `docs/notes/xcl-idees-2026-11-24.md`, `forks-idees-2026-11-24.md`.
- **Prevent AMD GPU hang when entering the world** — DRED use-after-free fixes (textures 7dddec2, list holds 0674874, pooled lists 99dd25b), bounded root SRVs a88b689, per-lane loop limit 2431719.
- **Reduce VIctory memory use** — `memory/lazy-host-textures` (commit 24.3 → 17.4 GB), memory log lines, later RAM guards (`VICTORY_MEMORY_LOW_MB`, `VICTORY_PIPELINE_MIN_FREE_MB`, `VICTORY_NO_COMPILE`).
- 2026-11-23 threads (Leonida project, abandoned for VIctory on 2026-11-24): boot to menu, VS input layout, levels.rpf models, D3D12→Vulkan. Not relevant any more.

## 11. Useful switches (all `VICTORY_*`)

Speed: `LATE_WAIT=3`, `WIDE_BACK`, `WIDE_MERGE`, `JOIN_COPIES`, `WIDE_REFRESH`, `RELAY=2`, `CARD_GATHER` (+`CARD_GATHER_PIECES`), `WIDE_TEXTURES`, `HOIST_CLAMPS` (0 old / 7 Peau3 / 8 / 9 ledger), `LEAN_MATH`, `FLAT_INSTANCES`, `WAVE_WIDE=1` (avoids AMD driver hang), `CARD_HASH_WIDE`, `TEXTURE_DETILE_WORKERS` (auto).
Memory/VRAM: `TEXTURE_HALF` (+`_MIN`, `_NPOT`), `TEXTURE_FILL_LEVELS`, `CARD_BUFFERS_MB`, `CARD_BUFFERS_MIN_MB`, `STAGING_POOL_MB`, `VRAM_HYSTERESIS`, `EVICT_REQUEUE_FRAMES`, `EVICT_GREY`, `WAIT_GREY`, `MEMORY_LOW_MB`, `PIPELINE_MIN_FREE_MB`, `PIPELINE_WORKERS`, `NO_COMPILE`, `ALIAS_COPIES`, `SELF_COPY`.
Image/diag: `PRESENT_TARGET`, `FLAT_LIGHT`, `OVERLAY=0`, `DISPATCH_PEEK`, `GPU_TIMING`, `D3D12_DEBUG`, `D3D12_DRED` (diagnostics build), `INDIRECT_STATS`, `STACKS`.
FSR: `FIDELITYFX_DIR`, `UPSCALER=7E93F0`, `FFX_SCALE`.

## 12. Starter prompts for T3 Code

Paste these as the first message of each agent. All agents work in `C:\Users\4nto\Desktop\VIctory`.

**Prompt 0 — every agent (prepend to the others):**
```
Tu travailles sur VIctory (émulateur Xbox Series pour GTA VI) dans C:\Users\4nto\Desktop\VIctory.
Commence par : git remote add fork https://github.com/AntoninMichelq/VIctory-forkAMD (si absent), git fetch fork,
puis lis HANDOFF.md (git show fork/main:HANDOFF.md) en entier et docs/notes au besoin. Respecte ses règles :
un seul jeu lancé à la fois et seulement par l'agent "PC", ne jamais tuer mon jeu, ne jamais publier de fichiers du jeu,
pas de cargo fmt sur victory-shader/victory-gpu, fichiers < 512 KiB. Réponds-moi en français simple.
Travaille sur ta propre branche (git worktree si un autre agent utilise le dossier), pousse sur le remote fork.
```

**Agent A — "PC / tests" (Claude, the only one that builds Jouer and runs the game):**
```
Tu es le seul agent qui lance le jeu. 1) Jouer reste Peau3 (b74ac64). Lis la section 7 de HANDOFF.md (essais du 25/11 au soir).
2) Propose-moi d'abord un Jouer = jouer/peau3 + fix/loading-sharp + jouer/peau3-overlay (images de chargement nettes, overlay court),
même env que Peau3, garde l'ancien en "Jouer precedent". 3) Ensuite les essais de vitesse (section 6 et 8 de HANDOFF.md),
un à la fois, en groupant, et seulement après mon OK écrit pour chaque nouveau lancement.
```

**Agent B — "textures / chrome" (Claude):**
```
Problème n°1 de HANDOFF.md : les textures retirées ou en attente quand la carte (8 Go) est pleine.
Pars de fork/fix/wait-grey. Fais en sorte d'abandonner beaucoup moins d'emplacements (évincer le moins récemment utilisé,
ou garder/refaire les petits mips). Vérifie avec clippy cible x86_64-pc-windows-msvc, pousse une branche, et donne à l'agent PC
un essai précis (bat + quoi regarder). Ne lance pas le jeu toi-même.
```

**Agent C — "vitesse" (Claude or Gemini):**
```
Problème n°3 de HANDOFF.md : on est limité par la carte (~70 ms/image). Trie l'essai groupé (fork/essai/groupe) :
trouve ce qui cause la tête flottante (ledger HOIST_CLAMPS=9 ou FLAT_INSTANCES) et ce qui garde le gain sans défaut.
Puis prépare l'essai port/fx-d510ee4 (plan docs/notes/fx-d510ee4-essai.md). Donne les essais à l'agent PC, ne lance pas le jeu.
```

**Agent D — "veille des forks" (Gemini is fine):**
```
Section 9 de HANDOFF.md : regarde les nouveaux commits de fxceless/victory-fork, Belobug/VIctory-Belobug-AMD,
pragyaangaur/VIctory-macOS, ZSlayerHQ/zslayer-pc-port-runtime et Marvisak/VIctory depuis le 25/11/2026.
Liste ce qui peut aider une RX 7600 8 Go (vitesse, image, mémoire), classé, avec les commits. Lecture seule, rien à fusionner sans mon OK.
```

## 13. Branches kept, and the archive

The fork had ~310 branches. Kept (everything an agent needs):
`main`, `docs/handoff`, `jouer/peau3` (current Jouer), `jouer/peau2` (fallback), `jouer/peau3-overlay`,
`essai/groupe`, `essai/groupe-base`, `essai/groupe-overlay`, `perf/lean-math`, `perf/clamp-ledger`, `perf/tiles-probe`,
`fix/loading-sharp`, `fix/evict-grey`, `fix/wait-grey`, `fix/partly-clamp`, `port/fx-d510ee4`, `scratch/smem` (PC worktree `_work\VIctory-fix`), and `archive/branches`.

**`archive/branches`** is one commit whose parents are the tips of every other branch (296 at first), with
`ARCHIVE.md` listing `branch | commit`. Nothing is lost: `git fetch fork archive/branches`, then
`git branch <name> <sha>` restores any of them (e.g. `nuit/tour-3b` 54d8982 for the map tour, `ancien-main` 70564ae,
`perf/preload`, `feat/host-fsr`, `perf/heap-pass`...). Old fork PRs #1–#13 point at archived branches.

Done on 2026-11-25 ~21:05 Paris from the PC: fork 313 → 18 branches (the list above + `fix/ci`), PC local branches 70 → 2 (`main`, `scratch/smem`), worktrees untouched. `archive/branches` = **090920e** (parents: cd3126a + local build-B e00f127f, build-C 60d35c8c, pr25 f1485bc1). ARCHIVE.md does not name every local branch (e.g. build-A), but all their commits are inside the archive.

The cloud could create the archive but not delete branches. `scripts/nettoyer-branches.ps1` does the deletions from the PC
(fork + local branches + clean extra worktrees, archiving any tip not yet saved first). Dry run by default, `-Apply` to do it;
it refuses to run while VIctory is running.
