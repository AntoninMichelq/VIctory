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
are GFX10 machine code made from the native programs (`FSSM`, see
`crates/victory-shader/src/native.rs`) of the title's own shader containers
in `shaders.rpf`, with `s_code_end` (0xbf9f0000) padding after them. They
are not those programs byte for byte (first believed, from 16-byte heads
in logs; the dumped sections showed otherwise):

- A program in a container begins with two 12-byte words, `v_nop` in its
  VOP3 form (`xx xx 80 d5 xx xx fe/ff 03`) and a dword more, whose fields
  the instruction ignores carry the program's 128-bit hash (the hash of
  the map's closing 0x43 entry); then `s_inst_prefetch` and the program.
  That is where the `80 d5` at bytes 2-3 and 14-15 come from.
- In a description the same two words come first, then a third
  (`d5800000 03fe0100 00000ac0` in all 192 dumped) and code that differs
  from the container's from its first instructions on (the title's
  run-time form of the program).

So a description's program is found by its two hashed words. Offline, all
192 programs of the 64 graphics and 64 compute descriptions dumped are
found that way, each in a container of its own stage (pixel, vertex,
compute); none of them by its first bytes past the hash. Of the pack's
26,698 programs the hashed words name 25,321: 195 of them name two
programs, an `Es` and a `Gs` one.

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
index the pack's containers by their programs' hashed words
(`precompiled::program_key`) on the first such call (the 411 MB pack is read whole and its 26,503 containers
unpacked in parallel: about 0.45 s offline on the test machine, logged with
its time), and find each description's
program there when the title has not given us its container. A compute
description whose program is found is made into a compute pipeline as
`create_compute` makes one (the root signature is bound at dispatch, as for
other pipelines).

A graphics description whose vertex and pixel programs are both found (and
only with `VICTORY_SHADER_PACK` set) gets a pipeline object at once, whose
host pipelines are made at its draws (`pipeline::Precompiled`): one per set
of what the draw has bound, the render target formats as the host has them
(the views' formats, else their textures'), their number, the depth format,
the sample count, and the topology type of the topology set (rectangles
their own; `precompiled::Targets`). Another set met later gets a host
pipeline of its own; each is made by the pipeline workers as any other
(`make::Pending`), so a first draw may be dropped while it is made. The
state comes from the registers section as the table below reads it
(`precompiled::GraphicsRegisters::decode`); what D3D12 cannot express
exactly is logged once per kind ("precompiled graphics pipeline state the
host has otherwise"). Descriptions not made log the stages found, as
before. `VICTORY_UNKNOWN_PIPELINES=0` leaves both methods failing, as
before.

## Graphics registers (first look, 64 descriptions)

`VICTORY_PRECOMPILED_DUMP=<dir>` (diagnostics) saves every section of the
first 64 descriptions of each method. The graphics registers section is 97
dwords (0x184). Most never change across the 64; the ones that do, read
against GFX10's public register layouts:

| dwords | values seen | reading |
|--------|-------------|---------|
| 14, 15 | `0x082c034x`, `0x12`/`0x14`/`0x18` | `SPI_SHADER_PGM_RSRC1/2` of the pixel program (VGPR/SGPR blocks, float mode `0xc0`, user SGPRs) |
| 19, 20 | `0x022c[c]34x`, same | the same for the vertex program |
| 31 | `0x00cc0010` (all) | `CB_COLOR_CONTROL`: normal mode, ROP3 copy (not read) |
| 39 | `0x00080000` (all) | `PA_CL_CLIP_CNTL`: DX clip space; bits 26/27 (near/far clipping off) read as `DepthClipEnable` |
| 40 | `0x00201a40` (54), `0x00201a42` (10) | `PA_SU_SC_MODE_CNTL`: bits 0/1 cull front/back (read), bit 2 front face (read: 0 counter-clockwise, as RADV writes Vulkan's numbering), bits 3-10 polygon mode and types (read: lines as wireframe), bits 11/12 polygon offset on (not read: the offsets are not in the section) |
| 46 | `0x007007f0` (57), `0x007002f7`, `0x00200291`, ... | `DB_DEPTH_CONTROL` (read): stencil, depth, depth write, depth bounds, depth func (bits 4-6), two-sided stencil (7), stencil funcs (8-10, 20-22); `0x007007f0` = no depth test, no stencil; `0x7e2` = greater-or-equal test, no write |
| 47 | `0` (62), `0x1001`, `0x80080` | `DB_STENCIL_CONTROL` (read): fail, depth-pass, depth-fail ops, front then back, 4 bits each (`0x80080` = add one wrapping on pass) |
| 48, 49 | `0x0100ff00` (61), ... | `DB_STENCILREFMASK` and its back-face twin (read: mask, write mask; the test value is the draw's reference on the host; op value 1 = D3D12's increments) |
| 51 | `0xffffffff` / `0` | `CB_TARGET_MASK` (read): 4 bits (RGBA) a target |
| 52-59 | `0x80000000` (23), `0xe5040504` (25), ... | `CB_BLEND0..7_CONTROL` (read): colour source, function, destination (bits 0-4, 5-7, 8-12), the same for alpha (16-28), bit 29 separate alpha, bit 30 enable, bit 31 no ROP3; `0xe5040504` = source alpha / one minus source alpha |
| 64-71 | `0x06000600` (23), `0x01540154` (25), ... | `SX_MRT0..7_BLEND_OPT`, matching dwords 52-59 (`6` = blending off; not read) |
| 63 | `0` / `0x33333333` | not identified (0 where blending is on, `0x33333333` where it is off) |
| 74 | `0x19fc0130` (47), `...01c0`, `...0120` | likely `DB_SHADER_CONTROL`: Z order in bits 4-5 (late Z with kill, bit 6, in `0x1c0`); nothing D3D12 needs |
| 92-96 | small counts, `0x10000a00` | not identified (vertex output or stream setup); the 3 sections of 99 dwords have 2 more here, the dwords above are where they are in the others |

What the host pipeline gets beyond these: no input layout (none is in the
description; see Open), sample mask all, no depth bias, no alpha to
coverage, D3D12's nearest for what has no exact equal (constant alpha
blend factors as the blend factor, stencil ops `ONES` and
replace-by-op-value as replace, logic stencil ops as keep, both faces
culled as none). A registers section shorter than 60 dwords or unreadable
gets the console's reset state (no blending, all written, no depth or
stencil test, no culling, solid, clipped), logged once.

**No render target or depth formats are in the description.** On the
console they come with the targets bound at draw time; a host pipeline
needs them at creation (`D3D12_RT_FORMAT_ARRAY`, the depth format). So a
graphics pipeline of this method is made at its draws, keyed by the
formats bound then (above).

## Open

- Not run in the game yet. Whether the front face reading (bit 2 of dword
  40) is right shows in the 10 of 64 descriptions that cull back faces.
- The info section (0x50-0xbc bytes) holds the names of the interface
  between the programs (`POSITION`, `COLOR`, `TEXCOORD`, ...); whether a
  vertex program reads vertex inputs that need an input layout is not
  known. The host pipeline has none.
- Dwords 63 and 92-96 of the graphics registers.
- Host pipelines of precompiled pipelines are not started early in later
  runs (their variants appear only at draws).
- Whether the run-time code differs from the container's program in more
  than its form (the container's program is what is translated).
