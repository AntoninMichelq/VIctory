# VIctory fork (work in progress)

A work-in-progress fork of [VIctory](https://github.com/Marvisak/VIctory), a
compatibility layer that runs an Xbox Series title natively on Windows from
files the user supplies. This fork follows upstream
`main` and adds fixes aimed at one goal: getting the game's world to render
correctly and playably on one particular PC.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's
developer or publisher.

## Credits

- **[VIctory](https://github.com/Marvisak/VIctory)** by Marvisak and its
  contributors is the base of everything here: the loader, the runtime
  services, the Direct3D 12 back end and the shader recompiler. Licensed
  GPL-3.0-or-later; so is this fork. Upstream's history is kept intact below
  the single commit that carries this fork's changes.
- **[XCL](https://github.com/ivall/xcl)** by ivall and its contributors
  (GPL-2.0-only) was used as a reference for how the same title behaves under
  another layer. Fixes inspired by it were written anew in VIctory's code.
- This fork's changes were developed with [Claude Code](https://claude.com/claude-code).

## En bref (English summary)

This fork targets **one modest AMD PC: Radeon RX 7600 8 GB, 32 GB of RAM**. On top of
upstream VIctory and the fxceless line it adds: a fix for walls and ceilings that were not
drawn (the title read back GPU culling results before they were handed back to it), a
lifetime fix for host copies freed while a submission still used them (device removals),
precompiled pipelines read from the title's shader pack (ported from Felix2180), an
offline pre-warm of every shader of the game, CPU-side savings and memory guards ported
from cool-fox-codec, faster shader math for small GPUs (`VICTORY_LEAN_MATH`) and host FSR
upscaling with sharpening. Measured results are in the table below. Everything is opt-in
through `VICTORY_*` environment switches; the recommended set is listed further down.

## Ce que ce fork apporte (AMD 8 Go)

Ce fork vise **un PC AMD modeste : Radeon RX 7600 8 Go, 32 Go de RAM**. Il part de VIctory et de la
ligne fxceless, puis ajoute :

- **Murs et plafonds qui manquaient** : le jeu relit des résultats calculés par la carte (culling,
  visibilité) en boucle, sans attendre de fence. Nos soumissions envoyées en avance les lui rendaient
  trop tard : des pièces entières étaient jugées cachées et pas dessinées. Une soumission qui écrit ces
  valeurs de fin de pipe est maintenant réglée tout de suite (`VICTORY_LATE_WAIT_WRITES=0`).
- **Pipelines précompilées** du jeu (méthodes du device `0x1c8` et `0x218`) lues dans son pack de shaders
  (`VICTORY_SHADER_PACK`) : le jeu s'en passait en silence. Portage du travail de Felix2180.
- **Correctif de durée de vie** : une copie hôte libérée par un nettoyage alors qu'une soumission pas encore
  partie l'utilisait encore faisait perdre la carte (page fault). La libération attend désormais
  16 soumissions de plus (`VICTORY_RETIRE_AHEAD`, `0` = comme avant).
- **Pré-chauffe hors jeu de tous les shaders** du jeu (`victory prewarm --all-shaders`, ~26 000 conteneurs)
  et de toutes les recettes de pipelines connues : moins de saccades en arrivant dans une zone nouvelle.
- **Gains CPU et garde-fous mémoire de cool-fox-codec**, réécrits sur notre arbre : rendus de cibles non
  répétés, constantes non alignées copiées une fois, déclarations dédupliquées, et `VICTORY_MEM_GUARD`
  (attente après un refus de mémoire, garde sur le commit Windows, plafond du staging sous 32 Go).
- **Eau et missions suivantes** : recopie vers la mémoire du jeu des textures d'eau écrites par la carte
  (`VICTORY_TEXTURE_BACK`, les bateaux ne coulent plus).
- **Petite carte** : maths des shaders allégées (`VICTORY_LEAN_MATH`, −8 à −18 ms sur la RX 7600),
  textures en demi-résolution, éviction et hystérésis VRAM pour 8 Go, **FSR hôte** (`VICTORY_FFX_SCALE`,
  netteté RCAS `VICTORY_FFX_SHARPNESS`, biais de mip `VICTORY_MIP_BIAS`).
- **Préparation CPU** : copies, recopies et rafraîchissements regroupés ou parallélisés (relais, recopie par
  blocs, journal d'écriture partagé), pour qu'image après image le CPU attende moins la carte.

## Performances mesurées (2026-11-26)

PC de test : Windows 11, **AMD Radeon RX 7600 8 Go**, 32 Go de RAM, profil Series X, dans la prison au
début du jeu (promenade WASD), images par seconde moyennes sur la durée indiquée.

| Version | Réglages | img/s | ms/image | Carte | Remarques |
|---|---|---|---|---|---|
| Peau3 (25/11) | FSR 0,5 | 6,2-6,4 | ~160 | ~70 ms | ancien Jouer |
| cumul9b (Jouer du 26/11) | FSR 0,42 + netteté, `JOURNAL_SHARED=2` | **8,87** | 112,8 | ~51 ms | 15 min sans plantage ; murs parfois absents |
| cumul11e | + pack de shaders (murs) | 7,51 | – | – | 7 min sans plantage |
| cumul12 + E1 | + `LATE_WAIT_WRITES=0` | 5,5 / 6,6 | – | ~90 ms | murs présents : beaucoup plus de décor dessiné |
| **cumul14** | E1 + gains cool-fox + `MEM_GUARD` | _à compléter_ | _à compléter_ | _à compléter_ | _3 essais en cours_ |

Mémoire : ~21 Go de RAM pour le Jouer. Le premier lancement d'un nouvel exe refait le cache du pilote AMD
(plus long) ; avec un cache pré-chauffé, l'arrivée dans le monde prend environ une minute.

## Réglages recommandés (lanceur « tout regroupé »)

Base du lanceur du Bureau, plus :

```bat
set VICTORY_LATE_WAIT=3
set VICTORY_RELAY=2
set VICTORY_LATE_WAIT_WRITES=0
set VICTORY_SHADER_PACK=<dossier du jeu>\shaders.rpf
set VICTORY_MEM_GUARD=1
set VICTORY_MIP_BIAS_SCOPE=aniso
set VICTORY_LEAN_MATH=247
set VICTORY_FFX_SCALE=0.42
set VICTORY_TEXTURE_HALF=2
set VICTORY_VRAM_HYSTERESIS=85,75
```

À ne pas utiliser pour l'instant : `VICTORY_LATE_WAIT=2` (blocages de la carte), `VICTORY_ALIAS_AFTER_UPLOAD=1`
(pertes de la carte), `VICTORY_ALIAS_ON_CARD=1` (non validé), `VICTORY_PLANE_PASS=written` (murs absents),
`VICTORY_INDIRECT_WIDE=1` (au-delà de la limite D3D12, non essayé).

## Problèmes connus

- Tout ce qui est dans l'ombre reste noir : l'éclairage par rayons du jeu ne tourne pas encore.
- Reflets « chrome » sur certains vêtements clairs (cause à l'étude : matériau ou réflexions).
- Les pertes de la carte sont devenues rares, mais un plantage intermittent reste possible lors de longues sessions.
- Avec tous les murs dessinés, la carte travaille plus (~90 ms) : la vitesse dépend surtout d'elle sur une RX 7600.

## Crédits supplémentaires

- **fxceless/victory-fork** : la ligne dont ce fork part (attentes tardives, relais, copies sur la carte) et les correctifs
  plafond des vues de constantes, lookup cache, recopie de textures écrites par la carte.
- **Felix2180/VIctory** : lecture des pipelines précompilées (`0x1c8`/`0x218`), tests de rayons « structures vides ».
- **ComputerScape04/cool-fox-codec** (codec_d, avatarmanz) : idées des gains CPU, des garde-fous mémoire et de `TEXTURE_BACK`
  (Geckomacabre via fxceless), réécrites ici.
- **Belobug**, **myokex/victory-amd** : premiers travaux AMD (éclairage ambiant de secours, garde des vues).

## Historique : les premiers correctifs du fork (2026-11-25)

Rendering:
- Meshlet culling read its scene buffers as the wrong kind of view, so the world drew almost nothing; fixed.
- Per-draw descriptor tables could run out, after which draws used slots never made for them and the card hung; fixed.
- Light packed by the game's compute shaders was read back through empty copies (sky, probes, fog); fixed for every mip level and layer.
- The fog reconstruction, shadow-mask, hair, blend-shape skinning, fire and several material programs did not translate; fixed.
- Gathers at a named mip level (with and without an offset) read level 0; fixed.
- Implicit-level samples outside pixel shaders produced invalid DXIL in 730 compute programs; fixed.
- Rectangle lists, depth and stencil twins for depth textures written by compute, texture read-backs.
- Ray tracing, in progress and off by default: the game's own structure-builder programs translate, title memory can be read by raw address from shaders, the hardware ray/box and ray/triangle test is emulated in DXIL (checked against a reference on the GPU), and 59 of the 64 trace programs translate. Nothing is lit by it yet. See [docs/fork-ray-tracing.md](docs/fork-ray-tracing.md).

Speed and memory:
- Waits for memory values, write-watch queries and page queries no longer walk page tables per call.
- Large title buffers get a copy in card memory, with shader writes found by hashing on the card.
- Submissions can be sent ahead of the wait for the card (`VICTORY_LATE_WAIT`), with a relay thread that lands results as the card finishes (`VICTORY_RELAY`): in a first five-minute run the card's idle time behind the submitting thread fell from about 136 to 30 ms per frame.
- Title memory can be opened as a graphics heap so that buffers are used in place instead of copied (`VICTORY_INPLACE`, first stage only; a probe confirmed the approach on the test GPU).
- The bindless texture queue no longer uploads stale views; the view table no longer grows without bound; the pipeline library is mapped instead of read into memory.
- An intermittent device loss was traced to the pipeline library file being replaced while mapped; fixed.

- Translated shaders get clean-up passes over the finished module (dead code, constant folding, reuse of values, branch facts); the pack's host instruction count falls by about 43 %, worth about 20 ms of GPU time per frame on the test machine (`VICTORY_SHADER_OPT=0` turns them off).

Most of the newer speed changes sit behind environment switches that are off
by default (`VICTORY_LATE_WAIT`, `VICTORY_RELAY`, `VICTORY_WIDE_BACK`,
`VICTORY_CARD_BUFFERS_MIN_MB`, `VICTORY_INPLACE`, `VICTORY_BVH_BUILDER`, ...).
Each is described in a comment where it is read.

Everything below this line is upstream VIctory's own README, unchanged; its
status section describes upstream, not this fork.

---

# VIctory

Runs the title (Xbox Series X|S) natively on Windows, from files
the user supplies.

The title's x86-64 code runs unchanged inside our process; VIctory supplies
the platform underneath it and converts the title's data, ahead of time where
it can and on first use where it must (shaders). See
[docs/architecture.md](docs/architecture.md).

## Status (2026-11-23)

**Not playable yet.** The title boots, runs its menus at its own 30 fps and
loads the world, where rendering fails.

| Stage | State |
|-------|-------|
| Boot, first-boot screens (accessibility, language, settings, legal) | works, 30 fps |
| Intro video with sound | works |
| Start screen, profile and saves | works |
| Loading into the world (about 50 s) | works |
| The world (first mission's intro) | reached and runs with indirect draws; its title card renders, the scene after it is still dark; 0.5-1.8 s per frame |

Tested on one machine: Windows 11, AMD RX 6700 XT.

> **Warning.** Entering the world on a real GPU can currently hang it, and
> Windows then resets the display driver (every monitor goes black for a
> moment). To look further, use a diagnostics build with
> `VICTORY_D3D12_WARP=1`, which renders on the software adapter.

What each part does today and what it still needs:

| Subsystem | Today | Needs |
|-----------|-------|-------|
| Loader | Runs the title's executable and its DLLs in our process; crash reports name the faulting code and are kept in `crashes.log` | – |
| Runtime services | One signed-in offline user, licences, packages, saves that survive a kill at any step, text entry dialog | Speech is silent; the dialog and relaunch are untested live |
| Network | Offline by design: the title cannot reach the network | – |
| Files and streaming | Archives and streamed data read, zlib and BCPack decoded in parallel | Loading is limited by writes into title memory |
| Input | XInput pads, keyboard, keyboard stand-in pad | Trigger rumble |
| Audio | Menu, video and loading sound; XMA voices and Opus streams decode | Reverb is silent; Opus results unverified against the title |
| GPU: menus | Draws, compute and presentation on host D3D12; only changed title memory is copied | Frame pacing drifts on one screen in release builds |
| GPU: world | Mesh pipelines, shader-written buffers, descriptor lifetimes, indirect execution (arguments clamped), pipelines made by worker threads and kept on disk | Most of the scene is still missing (untranslated shaders drop draws); frames take seconds; real queries, depth bounds, HDR and scaling are missing |
| Shaders | Console GPU code recompiled to DXIL at first use, cached on disk. Menus: all. World: mesh 8,191 of 8,506, vertex/pixel 10,617 of 12,346, compute 1,643 of 2,539 | 64-bit image atomics, merged geometry stages, the ray-tracing builder's shaders; some compute programs need a GPU with 64-lane waves |
| Textures | Console tiled layouts and detiling for the modes the title uses | Render-target metadata planes |
| Series S mode | Runs through the menus to the loading screen | The world, as above |

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
copy victory.example.toml victory.toml   # then set your paths (outside the repo)
cargo test --workspace                   # needs no game data
cargo run -p victory -- config           # check your paths
```

Run the title:

```powershell
cargo build --release -p victory
target\release\victory.exe run <game dir>\<the title's exe> --execute
```

The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\victory\data`), derived
data under the cache folder.

Input: XInput pads. With none connected the keyboard is a pad while the
title's window has focus:

| Keys | Pad |
|------|-----|
| W A S D / I J K L | left / right stick |
| arrows | D-pad |
| Enter, Backspace, X, Y | A, B, X, Y |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Esc, Tab | Menu, View |

## Commands

```
victory inspect <exe|dll> [--imports] [--exports]
victory census [paths...] [--missing] [--strict]
victory lib-exports <import.lib> [--list]
victory shader-info [paths...] [--verbose] [--disasm]
victory translate [paths...] [--validate] [--pipelines] [--list] [--jobs N]
victory run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-x|series-s] [-- <title arguments>]
victory config
```

Logging: `VICTORY_LOG=victory_loader=trace,info` (a release build logs
info and above).

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

| Variable | Effect |
|----------|--------|
| `VICTORY_LOG=<target>=debug,...` | debug and trace output |
| `VICTORY_STACKS=20,60` | every thread's stack at those seconds |
| `VICTORY_WATCH=<rva>,...` | logs a title u32 when it changes |
| `VICTORY_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (powers of two, or `VICTORY_CAPTURE_FRAMES=<list>`) |
| `VICTORY_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `VICTORY_D3D12_DEBUG=1` | host debug layer |
| `VICTORY_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, the allocation hit and the commands in flight |
| `VICTORY_PIPELINE_WAIT_MS=24` | reduce the per-frame wait for background pipeline creation from the default of 500 ms. A draw can be omitted until its pipeline is ready |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | omit draws and dispatches, including indirect ones, whose pipeline contains a listed 16-digit shader hash |
| `VICTORY_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands |
| `VICTORY_CARD_HASH_WIDE=0` | the passes that find what shaders wrote to copies in the card's memory hash with one thread per 64 KiB, as before |
| `VICTORY_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | translated shader loops stop after `n` iterations per lane instead of 2^20 (own shader cache entries); if a lower `n` makes world frames much faster, loops hitting the limit are what makes them slow |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `VICTORY_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |

Every build appends crashes and host GPU failures to `crashes.log` in the
title's data folder.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
