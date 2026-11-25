# Precompiled pipelines (device methods 0x1c8 and 0x218)

The title creates hundreds of pipelines through two device methods whose
descriptions the console headers we work from do not give: vtable slots
`0x1c8` (graphics) and `0x218` (compute), both called as
`(device, description, interface id, out)`. This note records what their
descriptions hold, as worked out on 2026-11-25 from the title's call sites,
the descriptions it passed in runs, its shader pack and the FidelityFX
upscaler library it ships. `crates/victory-gpu/src/precompiled.rs` reads
them; its module comment has the same layout in short.

## Call sites

- `0x218` (title+0x24800d5): the description is part of a node of a hash
  map the title keeps per pipeline (node: +0x08 next, +0x10 a 64-bit key,
  +0x18 the description, 0x60 bytes in all). The title writes the root
  signature into the node's first description word just before the call.
  So the 12 words the device logged were one description (9 words) and the
  start of the next node.
- `0x1c8` (title+0x2477578): the description sits at +0x40 in an object of
  the title's; its first word is likewise written just before the call.

Both ask for `ID3D12PipelineState`; on failure the title goes without the
pipeline, silently.

## Description

```text
u64 root signature (one of our host root signature objects)
(u64 address, u64 size) per section:
  0x218  registers, info, code, map                      4 pairs
  0x1c8  registers, info, code, map, code, map, (zeros)  6 pairs
```

In every graphics description looked at, the first program was a pixel
shader and the second a vertex shader. The metadata sections of one
description lie back to back in title memory; the code sections lie apart,
in memory the GPU reads (256-byte aligned).

| section   | compute                         | graphics                       |
|-----------|---------------------------------|--------------------------------|
| registers | 0x1c bytes, register values     | 0x184 or 0x18c bytes           |
| info      | 0x14: u32 hash, u32 3, zeros    | 0x50-0xbc: u32 hash, u32 0x1000, ... |
| code      | GFX10 machine code              | GFX10 machine code             |
| map       | see below                       | u32 2, u32 size - 0xc, u32 (code dwords - 10 in both samples), ... |

Compute map: u32 0x403, u32 size - 0x14, u32 code size in dwords (padding
included), u32 0, u32 0, u16 0x440, u16 the number of 0x41 entries, then
32-byte entries: 0x41 (user-data mapping), 0x42 (resource use), and a
closing 0x43 entry holding the program's 128-bit hash.

## The code

The code sections are not DXBC containers, nor a different encoding: they
are the GFX10 native programs (`FSSM`, see `crates/victory-shader/src/native.rs`)
of the title's own shader containers in `shaders.rpf`, byte for byte, with
`s_code_end` (0xbf9f0000) padding after them (0x40 bytes more than the
container's program for compute programs, 0x80 for vertex programs, none
for pixel programs in the samples). Of 196 distinct program heads logged in
runs, 194 were found in containers of `shaders.rpf` (for example a compute
program of a texture-array mip generator, and a pixel/vertex pair of a VFX
material whose containers are named after the same material hash).

Every console program begins with three 12-byte words: `v_nop` in its VOP3
form (`xx xx 80 d5 xx xx fe/ff 03`, the fields the instruction ignores
carrying the program's hash) and a dword more; then `s_inst_prefetch` and
the program proper. That is where the `80 d5` at bytes 2-3 and 14-15 come
from.

The title never hands these containers to the known creation methods, which
is why they did not match any container the device had seen.

## FSPX

The FidelityFX library (`amd_fidelityfx_upscaler_xbox.dll`) stores the same
sections for its compute shaders in files of their own:

```text
"FSPX" u32 version (1), u32 section count, u32 file size,
       then per section: u32 kind, u32 offset, u32 size, u32 0
kinds  0 registers (0x1c), 1 info (0x14), 6 code, 7 map
```

No `FSPX` file, nor any whole pipeline, is stored in the title's packs: the
title assembles the descriptions at run time from its containers.

## What the device does with them

`VICTORY_SHADER_PACK=<path of shaders.rpf>` (off by default) has the device
index the pack's containers by their programs' first 256 bytes on the first
such call (the 411 MB pack is read whole and its 26,503 containers
unpacked in parallel: about 0.45 s offline on the test machine, logged with
its time), and find each description's
program there when the title has not given us its container. A compute
description whose program is found is made into a compute pipeline as
`create_compute` makes one (the root signature is bound at dispatch, as for
other pipelines). Graphics descriptions are still not made: their pipeline
state (blend, depth, raster, render target formats, input layout) is in the
registers and info sections, which are not decoded; the stages found are
logged. `VICTORY_UNKNOWN_PIPELINES=0` leaves both methods failing, as before.

## Graphics registers (first look, 64 descriptions)

`VICTORY_PRECOMPILED_DUMP=<dir>` (diagnostics) saves every section of the
first 64 descriptions of each method. The graphics registers section is 97
dwords (0x184). Most never change across the 64; the ones that do, read
against GFX10's public register layouts:

| dwords | values seen | reading |
|--------|-------------|---------|
| 14, 15 | `0x082c034x`, `0x12`/`0x14`/`0x18` | `SPI_SHADER_PGM_RSRC1/2` of the pixel program (VGPR/SGPR blocks, float mode `0xc0`, user SGPRs) |
| 19, 20 | `0x022c[c]34x`, same | the same for the vertex program |
| 46 | `0x007007f0` (57), `0x007002f7`, `0x00200291`, ... | `DB_DEPTH_CONTROL`: `0x007007f0` = no depth test, no stencil; `0x2f7` = stencil, depth test and write |
| 48, 49 | `0x0100ff00` (61), ... | `DB_STENCILREFMASK` and its back-face twin (test value, mask `0xff`, write mask, op value) |
| 51 | `0xffffffff` / `0` | a colour write mask over the eight targets (`CB_TARGET_MASK`-like) |
| 52-59 | `0x80000000` (23), `0xe5040504` (25), ... | `CB_BLEND0..7_CONTROL`: bit 30 enable, bit 31 no ROP3; `0xe5040504` = source alpha / one minus source alpha, separate alpha the same |
| 64-71 | `0x06000600` (23), `0x01540154` (25), ... | `SX_MRT0..7_BLEND_OPT`, matching dwords 52-59 (`6` = blending off) |
| 63 | `0` / `0x33333333` | likely `SPI_SHADER_COL_FORMAT` (4 bits a target) |
| 74 | `0x19fc0130` (47), `...01c0`, `...0120` | not identified (rasterizer or interpolation control) |
| 92-96 | small counts, `0x10000a00` | not identified (vertex output or stream setup) |

**No render target or depth formats are in the description.** On the
console they come with the targets bound at draw time; a host pipeline
needs them at creation (`D3D12_RT_FORMAT_ARRAY`, the depth format). So a
graphics pipeline of this method has to be made at its first draw, keyed
by the formats bound then (and remade for other formats), with blend,
depth-stencil and write masks taken from the dwords above and the
rasterizer from the console defaults until dword 74 is read.

## Open

- Making graphics pipelines at first draw from the bound formats (above).
- Dwords 74 and 92-96 of the graphics registers, and the info section.
- Two of the logged program heads were not found in `shaders.rpf`.
