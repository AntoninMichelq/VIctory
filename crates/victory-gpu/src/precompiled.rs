//! Pipelines the title describes by their precompiled console parts:
//! device methods 0x1c8 (graphics) and 0x218 (compute), whose descriptions
//! the console's headers do not give. What is known, from the title's call
//! sites, the descriptions it passed in runs, its shader pack and the
//! FidelityFX upscaler library it ships (which stores the same parts in
//! files); docs/precompiled-pipelines.md has the longer account.
//!
//! ```text
//! description   u64 root signature (a host object of ours)
//!               then (u64 address, u64 size) pairs, one per section:
//!   0x218       registers, info, code, map            (4 pairs, 0x48 bytes)
//!   0x1c8       registers, info, code, map, code, map (6 pairs; zero
//!               words follow); the first program was a pixel shader and
//!               the second a vertex shader in every description seen
//! registers     0x1c bytes (compute: register values, e.g. 0x4005cf00
//!               first); 0x184 or 0x18c (graphics)
//! info          0x14 bytes (compute: u32 hash, u32 3, zeros); 0x50-0xbc
//!               (graphics: u32 hash, u32 0x1000, ..., the input layout,
//!               see [`input_layout`])
//! code          GFX10 machine code: made from the native program of one
//!               of the title's shader containers (its hashed prologue
//!               words kept, see [`program_key`]), followed by s_code_end
//!               (0xbf9f0000) padding (0x40 bytes more for compute
//!               programs, 0x80 for vertex programs, none for pixel
//!               programs in the samples)
//! map           compute: u32 0x403, u32 size - 0x14, u32 code size in
//!               dwords (padding included), u32 0, u32 0, u16 0x440, u16
//!               number of 0x41 entries, then 32-byte entries tagged 0x41
//!               (user-data mapping), 0x42 (resource use) and a closing
//!               0x43 holding the program's 128-bit hash
//!               graphics: u32 2, u32 size - 0xc, u32 (code dwords - 10 in
//!               both samples), ...
//! ```
//!
//! Every program begins with three 12-byte words: a `v_nop` in its VOP3
//! form (`xx xx 80 d5 xx xx fe/ff 03`), whose ignored fields carry the
//! program's hash, and a dword more; then `s_inst_prefetch` and the code.
//!
//! The FidelityFX library keeps the same sections in `FSPX` files:
//!
//! ```text
//! "FSPX" u32 version (1), u32 section count, u32 file size, then per
//!        section: u32 kind, u32 offset, u32 size, u32 0
//! kinds  0 registers, 1 info, 6 compute code, 7 compute map
//! ```
//!
//! The programs are made from the native programs (FSSM) the title's
//! containers in its shader pack hold, and keep their hash; the title
//! never hands those containers to the known creation methods. So a
//! description's program is matched to its container by that hash, among
//! the containers of the pack ([`PackIndex`]), and the container is
//! translated as any other. Of a graphics description's registers section
//! the depth, stencil, blend, write mask and rasterizer registers are read
//! ([`GraphicsRegisters`]); its host pipelines are made per set of targets
//! drawn into ([`Targets`], [`Variants`]), which the description does not
//! name. What else the registers, and the info and map sections, hold is
//! not read: the container has most of it in its own form.

use std::collections::HashMap;

/// Device method of graphics pipelines described this way.
pub const GRAPHICS_SLOT: usize = 0x1c8;
/// Device method of compute pipelines described this way.
pub const COMPUTE_SLOT: usize = 0x218;
/// `s_code_end`, which pads programs.
const CODE_END: u32 = 0xbf9f_0000;
/// Programs without the hashed prologue are matched by this many first
/// bytes (whole if shorter).
pub const HEAD: usize = 256;
/// Sections larger than this are not believed.
const MAX_SECTION: u64 = 8 << 20;

/// What a section of a description holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Registers,
    Info,
    /// The program at this position (0 first).
    Code(usize),
    /// The map of the program at this position.
    Map(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Section {
    pub role: Role,
    pub address: u64,
    pub size: u64,
}

/// A description of either method, read from its words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Description {
    pub root_signature: u64,
    pub sections: Vec<Section>,
}

impl Description {
    /// The description `words` hold for method `slot`; `None` if they do
    /// not read as one (a section missing, or of an implausible size).
    pub fn parse(slot: usize, words: &[u64]) -> Option<Self> {
        let programs = match slot {
            COMPUTE_SLOT => 1,
            GRAPHICS_SLOT => 2,
            _ => return None,
        };
        let root_signature = *words.first()?;
        if root_signature == 0 {
            return None;
        }
        let mut roles = vec![Role::Registers, Role::Info];
        for p in 0..programs {
            roles.extend([Role::Code(p), Role::Map(p)]);
        }
        let pairs = words.get(1..1 + 2 * roles.len())?;
        let sections: Vec<Section> = roles
            .into_iter()
            .zip(pairs.chunks_exact(2))
            .map(|(role, p)| Section {
                role,
                address: p[0],
                size: p[1],
            })
            .collect();
        let plausible = |s: &Section| {
            s.address != 0
                && (1..=MAX_SECTION).contains(&s.size)
                && (!matches!(s.role, Role::Code(_)) || (s.size >= 64 && s.size.is_multiple_of(4)))
        };
        sections.iter().all(plausible).then_some(Self {
            root_signature,
            sections,
        })
    }

    /// The program sections, first program first.
    pub fn code(&self) -> impl Iterator<Item = &Section> {
        self.sections
            .iter()
            .filter(|s| matches!(s.role, Role::Code(_)))
    }
}

/// The first words of a map section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapHeader {
    /// Kind 0x403: the program's size in dwords, padding included.
    Compute { code_dwords: u32 },
    /// Kind 2.
    Graphics,
}

impl MapHeader {
    /// The header of `map` (the whole section); `None` if its kind is not
    /// one seen or its stated size disagrees with the section's.
    pub fn parse(map: &[u8]) -> Option<Self> {
        let word = |i: usize| {
            map.get(4 * i..4 * i + 4)
                .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        };
        let body = word(1)? as usize;
        match word(0)? {
            0x403 if body + 0x14 == map.len() => Some(Self::Compute {
                code_dwords: word(2)?,
            }),
            2 if body + 0xc == map.len() => Some(Self::Graphics),
            _ => None,
        }
    }
}

/// Whether `code` begins as the console's programs do: `v_nop` in its
/// VOP3 form at bytes 0 and 12.
pub fn is_program(code: &[u8]) -> bool {
    let high = |at: usize| code.get(at + 2..at + 4) == Some(&[0x80, 0xd5][..]);
    high(0) && high(12)
}

/// `code` without the `s_code_end` dwords that pad its end.
pub fn trim_padding(code: &[u8]) -> &[u8] {
    let mut end = code.len() - code.len() % 4;
    while end >= 4 && code[end - 4..end] == CODE_END.to_le_bytes() {
        end -= 4;
    }
    &code[..end]
}

/// Bytes of a program's prologue that carry its hash: two `v_nop` words.
const HASH_WORDS: usize = 24;

/// The key a program is matched by: its hash, carried in the first two
/// words of its prologue ([`is_program`]); else a hash of its first
/// [`HEAD`] bytes, or of the whole if it is shorter, padding trimmed first
/// (programs in containers are padded too, by other amounts).
///
/// The code a description holds is not the container's program byte for
/// byte: the title puts a third `v_nop` (`d5800000 03fe0100 xxxxxxxx`)
/// after the two hashed ones and changes the code after it, so in the
/// title's run no graphics program (and offline none of 64 compute ones)
/// was found by its first bytes. The hashed words are the same, and name
/// one program in the pack (195 hashes name two: an `Es` and a `Gs`
/// program, of one container).
pub fn program_key(code: &[u8]) -> [u8; 32] {
    if is_program(code) && code.len() >= HASH_WORDS {
        let mut h = blake3::Hasher::new();
        h.update(b"prologue hash");
        h.update(&code[..HASH_WORDS]);
        return *h.finalize().as_bytes();
    }
    let code = trim_padding(code);
    *blake3::hash(&code[..code.len().min(HEAD)]).as_bytes()
}

/// The sections of an `FSPX` file, as (kind, bytes); `None` if it is not
/// one or a section lies outside it.
pub fn fspx_sections(file: &[u8]) -> Option<Vec<(u32, &[u8])>> {
    const MAX_SECTIONS: usize = 64;
    let word = |at: usize| {
        file.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
    };
    if file.get(..4)? != b"FSPX" || word(4)? != 1 {
        return None;
    }
    let count = word(8)? as usize;
    let size = word(12)? as usize;
    if count > MAX_SECTIONS || size > file.len() {
        return None;
    }
    (0..count)
        .map(|i| {
            let at = 0x10 + 0x10 * i;
            let (offset, len) = (word(at + 4)? as usize, word(at + 8)? as usize);
            let end = offset.checked_add(len).filter(|&e| e <= size)?;
            Some((word(at)?, &file[offset..end]))
        })
        .collect()
}

/// The containers of a shader pack, by the programs they hold
/// ([`program_key`]): where each container's record starts in the pack.
#[derive(Debug, Default)]
pub struct PackIndex {
    by_key: HashMap<[u8; 32], usize>,
}

impl PackIndex {
    /// The index of `records`: each a record's offset and its container,
    /// whose programs `programs` lists (none if it does not read).
    pub fn build<C: AsRef<[u8]>, P: AsRef<[u8]>>(
        records: impl IntoIterator<Item = (usize, C)>,
        programs: impl Fn(&[u8]) -> Vec<P>,
    ) -> Self {
        let mut by_key = HashMap::new();
        for (offset, container) in records {
            for program in programs(container.as_ref()) {
                by_key
                    .entry(program_key(program.as_ref()))
                    .or_insert(offset);
            }
        }
        Self { by_key }
    }

    /// Where the container holding `code` (a description's program, padding
    /// and all) starts in the pack.
    pub fn find(&self, code: &[u8]) -> Option<usize> {
        self.by_key.get(&program_key(code)).copied()
    }

    pub fn programs(&self) -> usize {
        self.by_key.len()
    }

    /// Joins indexes built over parts of a pack (the first wins).
    pub fn merge(&mut self, other: Self) {
        for (k, v) in other.by_key {
            self.by_key.entry(k).or_insert(v);
        }
    }
}

/// Where the graphics registers section keeps what is read of it (dword
/// positions; docs/precompiled-pipelines.md has the table), read against
/// GFX10's public register layouts and the 64 descriptions logged.
mod at {
    /// `PA_CL_CLIP_CNTL` (`0x00080000` in all 64: DX clip space, near and
    /// far clipping on).
    pub const CLIP_CNTL: usize = 39;
    /// `PA_SU_SC_MODE_CNTL` (`0x00201a40`; `0x00201a42` culls back faces).
    pub const SC_MODE_CNTL: usize = 40;
    pub const DEPTH_CONTROL: usize = 46;
    /// `DB_STENCIL_CONTROL` (`0`, `0x1001`, `0x80080`).
    pub const STENCIL_CONTROL: usize = 47;
    pub const STENCIL_REF_MASK: usize = 48;
    pub const STENCIL_REF_MASK_BF: usize = 49;
    /// `PA_SU_POLY_OFFSET_CLAMP`, then `_FRONT_SCALE` and `_FRONT_OFFSET`.
    pub const POLY_OFFSET: usize = 43;
    /// `PA_SC_CONSERVATIVE_RASTERIZATION_CNTL`.
    pub const CONSERVATIVE: usize = 72;
    /// `CB_TARGET_MASK`: four bits (RGBA) a target.
    pub const TARGET_MASK: usize = 51;
    /// `CB_BLEND0_CONTROL`, seven more after it.
    pub const BLEND_CONTROL: usize = 52;
    /// Dwords a registers section has at least for the above to be read.
    pub const NEEDED: usize = BLEND_CONTROL + 8;
}

/// One target's blending, in D3D12's numbers (`D3D12_BLEND`,
/// `D3D12_BLEND_OP`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetBlend {
    pub enable: bool,
    pub src: u32,
    pub dest: u32,
    pub op: u32,
    pub src_alpha: u32,
    pub dest_alpha: u32,
    pub op_alpha: u32,
}

impl TargetBlend {
    /// Blending off: one, zero, add.
    pub const OFF: Self = Self {
        enable: false,
        src: 2,
        dest: 1,
        op: 1,
        src_alpha: 2,
        dest_alpha: 1,
        op_alpha: 1,
    };
}

/// One face's stencil test, in D3D12's numbers (`D3D12_STENCIL_OP`,
/// `D3D12_COMPARISON_FUNC`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilFace {
    pub fail: u32,
    pub depth_fail: u32,
    pub pass: u32,
    pub func: u32,
}

impl StencilFace {
    /// Keep, keep, keep, always.
    pub const KEEP: Self = Self {
        fail: 1,
        depth_fail: 1,
        pass: 1,
        func: 8,
    };
}

/// Depth and stencil state, in D3D12's numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthStencil {
    pub depth_enable: bool,
    pub depth_write: bool,
    /// `D3D12_COMPARISON_FUNC`.
    pub depth_func: u32,
    pub depth_bounds: bool,
    pub stencil_enable: bool,
    pub read_mask: u8,
    pub write_mask: u8,
    pub front: StencilFace,
    pub back: StencilFace,
}

/// Rasterizer state, in D3D12's numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Raster {
    /// `D3D12_CULL_MODE`: 1 none, 2 front, 3 back.
    pub cull: u32,
    pub front_ccw: bool,
    pub wireframe: bool,
    pub depth_clip: bool,
    /// `DepthBias`, `SlopeScaledDepthBias`, `DepthBiasClamp`.
    pub depth_bias: i32,
    pub slope_bias: f32,
    pub bias_clamp: f32,
    /// Overestimating conservative rasterization.
    pub conservative: bool,
}

/// The pipeline state a graphics description's registers section holds,
/// as far as it is read ([`GraphicsRegisters::decode`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphicsRegisters {
    pub blend: [TargetBlend; 8],
    /// `D3D12_COLOR_WRITE_ENABLE` bits per target.
    pub write_masks: [u8; 8],
    pub depth: DepthStencil,
    pub raster: Raster,
}

impl GraphicsRegisters {
    /// The console's reset state, for a section that does not read: no
    /// blending, every channel written, no depth or stencil test, no
    /// culling, solid, clipped at near and far.
    pub fn console_defaults() -> Self {
        Self {
            blend: [TargetBlend::OFF; 8],
            write_masks: [0xf; 8],
            depth: DepthStencil {
                depth_enable: false,
                depth_write: false,
                depth_func: 8,
                depth_bounds: false,
                stencil_enable: false,
                read_mask: 0xff,
                write_mask: 0xff,
                front: StencilFace::KEEP,
                back: StencilFace::KEEP,
            },
            raster: Raster {
                cull: 1,
                front_ccw: true,
                wireframe: false,
                depth_clip: true,
                depth_bias: 0,
                slope_bias: 0.0,
                bias_clamp: 0.0,
                conservative: false,
            },
        }
    }

    /// The state the registers section `dwords` holds, and what of it the
    /// host cannot have exactly (each a fixed text, for a log line once).
    /// A section too short to hold the registers read gets
    /// [`GraphicsRegisters::console_defaults`].
    /// shortcut: alpha to coverage, the sample mask and the other dwords
    /// keep D3D12's defaults; the back faces' polygon offset is taken to
    /// be the front faces'.
    pub fn decode(dwords: &[u32]) -> (Self, Vec<&'static str>) {
        let mut notes = Vec::new();
        let mut r = Self::console_defaults();
        if dwords.len() < at::NEEDED {
            notes.push("a graphics registers section too short to read: console defaults");
            return (r, notes);
        }
        let bits = |v: u32, at: u32, n: u32| v >> at & ((1 << n) - 1);

        // DB_DEPTH_CONTROL: 0 stencil, 1 depth, 2 depth write, 3 bounds,
        // 4-6 depth func, 7 back face stencil, 8-10 stencil func, 20-22
        // back face stencil func.
        let dc = dwords[at::DEPTH_CONTROL];
        // DB_STENCIL_CONTROL: fail, depth pass, depth fail, then the same
        // for back faces, four bits each.
        let sc = dwords[at::STENCIL_CONTROL];
        // DB_STENCILREFMASK: test value, mask, write mask, op value.
        let refmask = dwords[at::STENCIL_REF_MASK];
        let refmask_bf = dwords[at::STENCIL_REF_MASK_BF];
        let face =
            |ops: u32, func: u32, op_value: u32, notes: &mut Vec<&'static str>| StencilFace {
                fail: stencil_op(bits(ops, 0, 4), op_value, notes),
                pass: stencil_op(bits(ops, 4, 4), op_value, notes),
                depth_fail: stencil_op(bits(ops, 8, 4), op_value, notes),
                func: func + 1,
            };
        let stencil_enable = bits(dc, 0, 1) != 0;
        let two_sided = bits(dc, 7, 1) != 0;
        let (front, back) = if stencil_enable {
            let front = face(sc, bits(dc, 8, 3), bits(refmask, 24, 8), &mut notes);
            let back = if two_sided {
                face(
                    sc >> 12,
                    bits(dc, 20, 3),
                    bits(refmask_bf, 24, 8),
                    &mut notes,
                )
            } else {
                front
            };
            if two_sided && bits(refmask, 8, 16) != bits(refmask_bf, 8, 16) {
                notes.push("back face stencil masks differ from the front's: the front's are used");
            }
            (front, back)
        } else {
            (StencilFace::KEEP, StencilFace::KEEP)
        };
        r.depth = DepthStencil {
            depth_enable: bits(dc, 1, 1) != 0,
            depth_write: bits(dc, 2, 1) != 0,
            depth_func: bits(dc, 4, 3) + 1,
            depth_bounds: bits(dc, 3, 1) != 0,
            stencil_enable,
            read_mask: bits(refmask, 8, 8) as u8,
            write_mask: bits(refmask, 16, 8) as u8,
            front,
            back,
        };

        // CB_TARGET_MASK, CB_BLENDn_CONTROL: 0-4 colour source, 5-7
        // colour function, 8-12 colour destination, 16-20, 21-23, 24-28
        // the same for alpha, 29 separate alpha, 30 enable.
        let mask = dwords[at::TARGET_MASK];
        for i in 0..8 {
            r.write_masks[i] = bits(mask, 4 * i as u32, 4) as u8;
            let b = dwords[at::BLEND_CONTROL + i];
            if bits(b, 30, 1) == 0 {
                continue;
            }
            let (src, op, dest) = (bits(b, 0, 5), bits(b, 5, 3), bits(b, 8, 5));
            let (src_a, op_a, dest_a) = if bits(b, 29, 1) != 0 {
                (bits(b, 16, 5), bits(b, 21, 3), bits(b, 24, 5))
            } else {
                (src, op, dest)
            };
            r.blend[i] = TargetBlend {
                enable: true,
                src: blend_factor(src, false, &mut notes),
                dest: blend_factor(dest, false, &mut notes),
                op: blend_op(op, &mut notes),
                src_alpha: blend_factor(src_a, true, &mut notes),
                dest_alpha: blend_factor(dest_a, true, &mut notes),
                op_alpha: blend_op(op_a, &mut notes),
            };
        }

        // PA_SU_SC_MODE_CNTL: 0 cull front, 1 cull back, 2 front face
        // (0 counter-clockwise: Vulkan's numbering, which RADV writes as
        // is), 3-4 polygon mode, 5-7 front and 8-10 back polygon type
        // (0 points, 1 lines, 2 triangles).
        let sc_mode = dwords[at::SC_MODE_CNTL];
        r.raster.cull = match bits(sc_mode, 0, 2) {
            0 => 1,
            1 => 2,
            2 => 3,
            _ => {
                notes.push("both faces culled, which D3D12 cannot: drawn unculled");
                1
            }
        };
        r.raster.front_ccw = bits(sc_mode, 2, 1) == 0;
        if bits(sc_mode, 3, 2) != 0 {
            match bits(sc_mode, 5, 3) {
                2 => {}
                1 => r.raster.wireframe = true,
                _ => notes.push("polygons drawn as points: drawn solid"),
            }
            if bits(sc_mode, 5, 3) != bits(sc_mode, 8, 3) {
                notes.push("front and back polygon modes differ: the front's is used");
            }
        }
        // PA_CL_CLIP_CNTL: 26 near, 27 far clipping off.
        let clip = dwords[at::CLIP_CNTL];
        let (near_off, far_off) = (bits(clip, 26, 1) != 0, bits(clip, 27, 1) != 0);
        r.raster.depth_clip = !(near_off || far_off);
        if near_off != far_off {
            notes.push("near or far clipping off but not both: neither is clipped");
        }
        // PA_SU_POLY_OFFSET_CLAMP, _FRONT_SCALE, _FRONT_OFFSET (floats),
        // used when PA_SU_SC_MODE_CNTL bit 11 enables the front offset (in
        // every description seen): the scale is D3D12's slope-scaled bias
        // times 16 (as RADV writes it), the offset its constant bias. Seen:
        // (0, 16, 1) and (0.003, 29, 0) on shadow passes, 0 otherwise.
        if bits(sc_mode, 11, 1) != 0 {
            let float = |i: usize| f32::from_bits(dwords[i]);
            let (clamp, scale, offset) = (
                float(at::POLY_OFFSET),
                float(at::POLY_OFFSET + 1),
                float(at::POLY_OFFSET + 2),
            );
            if [clamp, scale, offset].iter().all(|v| v.is_finite()) {
                r.raster.bias_clamp = clamp;
                r.raster.slope_bias = scale / 16.0;
                r.raster.depth_bias = offset.round() as i32;
            } else {
                notes.push("a polygon offset that is not a number: none");
            }
        }
        // PA_SC_CONSERVATIVE_RASTERIZATION_CNTL: `0x00100000` (RADV's
        // default, NULL_SQUAD_AA_MASK_ENABLE) in nearly all; `0x00e00441`,
        // bit 0 OVER_RAST_ENABLE, on fog volume proxies, trail maps and a
        // pixel shader named `ps_light_proxy_conservative` (run 50).
        if let Some(&cntl) = dwords.get(at::CONSERVATIVE) {
            r.raster.conservative = bits(cntl, 0, 1) != 0;
        }
        (r, notes)
    }
}

/// D3D12's `D3D12_BLEND` for GFX10's blend factor `f`; for the alpha
/// channel (`alpha`) colour factors become their alpha ones, which is what
/// they are there and all D3D12 takes.
fn blend_factor(f: u32, alpha: bool, notes: &mut Vec<&'static str>) -> u32 {
    let (colour, alpha_one) = match f {
        0 => (1, 1),
        1 => (2, 2),
        2 => (3, 5),
        3 => (4, 6),
        4 => (5, 5),
        5 => (6, 6),
        6 => (7, 7),
        7 => (8, 8),
        8 => (9, 7),
        9 => (10, 8),
        10 => (11, 11),
        11 => {
            notes.push("blend factor both-source-alpha: source alpha");
            (5, 5)
        }
        12 => {
            notes.push("blend factor both-inverse-source-alpha: inverse source alpha");
            (6, 6)
        }
        13 => (14, 14),
        14 => (15, 15),
        15 => (16, 18),
        16 => (17, 19),
        17 => (18, 18),
        18 => (19, 19),
        19 => {
            notes.push("blend factor constant alpha: the blend factor");
            (14, 14)
        }
        20 => {
            notes.push("blend factor inverse constant alpha: the inverse blend factor");
            (15, 15)
        }
        _ => {
            notes.push("an unknown blend factor: one");
            (2, 2)
        }
    };
    if alpha { alpha_one } else { colour }
}

/// D3D12's `D3D12_BLEND_OP` for GFX10's combine function.
fn blend_op(f: u32, notes: &mut Vec<&'static str>) -> u32 {
    match f {
        0 => 1,
        1 => 2,
        2 => 4,
        3 => 5,
        4 => 3,
        _ => {
            notes.push("an unknown blend function: add");
            1
        }
    }
}

/// D3D12's `D3D12_STENCIL_OP` for GFX10's stencil op; `op_value` is what
/// the adding ops add on the console (D3D12's add one), and the
/// replacing ones take the reference set at draw time on the host.
fn stencil_op(op: u32, op_value: u32, notes: &mut Vec<&'static str>) -> u32 {
    let adds = |d3d12: u32, notes: &mut Vec<&'static str>| {
        if op_value != 1 {
            notes.push("a stencil op adding other than one: adds one");
        }
        d3d12
    };
    match op {
        0 => 1,
        1 => 2,
        2 => {
            notes.push("stencil op ones: replace");
            3
        }
        3 => 3,
        4 => {
            notes.push("stencil op replace by the op value: replace by the reference");
            3
        }
        5 => adds(4, notes),
        6 => adds(5, notes),
        7 => 6,
        8 => adds(7, notes),
        9 => adds(8, notes),
        _ => {
            notes.push("a logic stencil op: keep");
            1
        }
    }
}

/// One vertex input of a graphics description's input layout
/// ([`input_layout`]), as `D3D12_INPUT_ELEMENT_DESC` has it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputElement {
    pub name: String,
    pub index: u32,
    /// The title's DXGI format.
    pub format: u32,
    pub slot: u32,
    pub offset: u32,
    pub per_instance: bool,
    pub step_rate: u32,
}

/// The input layout a graphics description's info section holds; `None`
/// if the section does not read as one.
///
/// ```text
/// 0x38  u32 bytes of the layout (from 0x4c, padding not counted)
/// 0x4c  u32 element count, then per element:
///       u32 name length (its zero included), the name,
///       u32 semantic index, u32 format, u32 input slot,
///       u32 byte offset, u32 classification (1 per instance),
///       u32 instance step rate
/// ```
///
/// Read from the 64 descriptions dumped: e.g. `POSITION` R32G32B32_FLOAT
/// at 0, `COLOR` R8G8B8A8_UNORM at 12, `TEXCOORD` R32G32_FLOAT at 16; a
/// section of 0x50 bytes has a count of 0. Without it a host pipeline
/// whose vertex shader reads inputs is refused (`E_INVALIDARG`, run 47).
pub fn input_layout(info: &[u8]) -> Option<Vec<InputElement>> {
    const START: usize = 0x4c;
    const MAX_ELEMENTS: usize = 32;
    const MAX_NAME: usize = 64;
    let word = |at: usize| {
        info.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
    };
    let bytes = word(0x38)? as usize;
    let end = START.checked_add(bytes).filter(|&e| e <= info.len())?;
    let count = word(START)? as usize;
    if count > MAX_ELEMENTS {
        return None;
    }
    let mut at = START + 4;
    let mut elements = Vec::with_capacity(count);
    for _ in 0..count {
        let len = word(at)? as usize;
        if !(2..=MAX_NAME).contains(&len) {
            return None;
        }
        let name = info.get(at + 4..at + 4 + len)?;
        let (&zero, name) = name.split_last()?;
        if zero != 0 || !name.iter().all(|c| c.is_ascii_graphic()) {
            return None;
        }
        let fields = at + 4 + len;
        let field = |i: usize| word(fields + 4 * i);
        let class = field(4)?;
        if class > 1 {
            return None;
        }
        elements.push(InputElement {
            name: String::from_utf8(name.to_vec()).ok()?,
            index: field(0)?,
            format: field(1)?,
            slot: field(2)?,
            offset: field(3)?,
            per_instance: class == 1,
            step_rate: field(5)?,
        });
        at = fields + 24;
    }
    (at <= end).then_some(elements)
}

/// What a graphics pipeline of method 0x1c8 is made for besides its
/// description: what the draw that uses it has bound (render target and
/// depth formats as the host has them, their number, samples) and the
/// kind of primitive it draws. Each such set gets a host pipeline of its
/// own ([`Variants`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Targets {
    /// DXGI formats; past `count` zero.
    pub formats: [u32; 8],
    pub count: u8,
    pub depth: u32,
    pub samples: u32,
    /// `D3D12_PRIMITIVE_TOPOLOGY_TYPE`, or the console's rectangles
    /// ([`RECTANGLES_TYPE`]).
    pub topology: u32,
}

/// The console's rectangle list, in its topology numbering.
const CONSOLE_RECTANGLES: u32 = 17;
/// The console's rectangles as a pipeline's topology type (what its
/// stream pipelines name them).
pub const RECTANGLES_TYPE: u32 = 19;

impl Targets {
    /// The targets bound: their formats (at most 8 are kept), the depth
    /// format (0 for none) and the sample count (0 is taken as 1).
    pub fn new(formats: &[u32], depth: u32, samples: u32) -> Self {
        let count = formats.len().min(8);
        let mut f = [0; 8];
        f[..count].copy_from_slice(&formats[..count]);
        Self {
            formats: f,
            count: count as u8,
            depth,
            samples: samples.max(1),
            topology: 0,
        }
    }

    /// The same, drawn with the console's primitive topology `topology`.
    pub fn drawing(self, topology: u32) -> Self {
        Self {
            topology: topology_type(topology),
            ..self
        }
    }
}

/// The topology type a pipeline needs for the console's primitive
/// topology `t` (the console's fans, loops and quads draw as triangle
/// lists on the host; its rectangles get their own type).
pub fn topology_type(t: u32) -> u32 {
    match t {
        1 => 1,
        2 | 3 | 10 | 11 => 2,
        CONSOLE_RECTANGLES => RECTANGLES_TYPE,
        33..=64 => 4,
        _ => 3,
    }
}

/// The variants of one pipeline by what they are drawn with, each made
/// once and kept while the pipeline lives.
#[derive(Debug)]
pub struct Variants<V> {
    by_targets: std::sync::Mutex<HashMap<Targets, std::sync::Arc<V>>>,
}

impl<V> Default for Variants<V> {
    fn default() -> Self {
        Self {
            by_targets: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl<V> Variants<V> {
    /// The variant for `targets`, made by `make` if there is none yet;
    /// and whether it was made now.
    pub fn get(&self, targets: &Targets, make: impl FnOnce() -> V) -> (std::sync::Arc<V>, bool) {
        let mut map = self.by_targets.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = map.get(targets) {
            return (v.clone(), false);
        }
        let v = std::sync::Arc::new(make());
        map.insert(*targets, v.clone());
        (v, true)
    }

    /// Every variant made so far.
    pub fn all(&self) -> Vec<std::sync::Arc<V>> {
        let map = self.by_targets.lock().unwrap_or_else(|e| e.into_inner());
        map.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A made-up program: the prologue's shape, `n` dwords of body.
    fn program(seed: u32, n: usize) -> Vec<u8> {
        let mut words = vec![
            0xd580_0000 | (seed & 0xffff),
            0x03ff_0000 | seed >> 16,
            seed,
            0xd580_1234,
        ];
        words.extend((0..n as u32).map(|i| i.wrapping_mul(0x9e37_79b9) ^ seed));
        le(&words)
    }

    #[test]
    fn reads_compute_description() {
        // The node's next words (another node) follow; they are not read.
        let words = [
            0xaa00, 0x1000, 0x1c, 0x2000, 0x14, 0x3000, 0x240, 0x4000, 0x198, 0x55, 0x66, 0x77,
        ];
        let d = Description::parse(COMPUTE_SLOT, &words).unwrap();
        assert_eq!(d.root_signature, 0xaa00);
        let roles: Vec<Role> = d.sections.iter().map(|s| s.role).collect();
        assert_eq!(
            roles,
            [Role::Registers, Role::Info, Role::Code(0), Role::Map(0)]
        );
        let code: Vec<_> = d.code().map(|s| (s.address, s.size)).collect();
        assert_eq!(code, [(0x3000, 0x240)]);
    }

    #[test]
    fn reads_graphics_description() {
        let words = [
            0xbb00, 0x10, 0x184, 0x194, 0xbc, 0x8000, 0x300, 0x250, 0xd0, 0x8300, 0x140, 0x320,
            0x2f0, 0, 0, 0,
        ];
        let d = Description::parse(GRAPHICS_SLOT, &words).unwrap();
        let code: Vec<_> = d.code().map(|s| (s.role, s.address, s.size)).collect();
        assert_eq!(
            code,
            [
                (Role::Code(0), 0x8000, 0x300),
                (Role::Code(1), 0x8300, 0x140)
            ]
        );
        assert_eq!(d.sections[5].role, Role::Map(1));
    }

    #[test]
    fn rejects_what_is_not_a_description() {
        let good = [
            0xaa00u64, 0x1000, 0x1c, 0x2000, 0x14, 0x3000, 0x240, 0x4000, 0x198,
        ];
        assert!(Description::parse(0x1f0, &good).is_none());
        assert!(Description::parse(COMPUTE_SLOT, &good[..8]).is_none());
        let mut no_root = good;
        no_root[0] = 0;
        assert!(Description::parse(COMPUTE_SLOT, &no_root).is_none());
        let mut odd_code = good;
        odd_code[6] = 0x241;
        assert!(Description::parse(COMPUTE_SLOT, &odd_code).is_none());
        let mut missing = good;
        missing[7] = 0;
        assert!(Description::parse(COMPUTE_SLOT, &missing).is_none());
        // Graphics needs both programs.
        assert!(Description::parse(GRAPHICS_SLOT, &good).is_none());
    }

    #[test]
    fn reads_map_headers() {
        let mut compute = le(&[0x403, 0x84, 0xc0, 0, 0, 0x0002_0440]);
        compute.resize(0x98, 0);
        assert_eq!(
            MapHeader::parse(&compute),
            Some(MapHeader::Compute { code_dwords: 0xc0 })
        );
        let mut graphics = le(&[2, 0x84, 0x86, 0x0002_0440]);
        graphics.resize(0x90, 0);
        assert_eq!(MapHeader::parse(&graphics), Some(MapHeader::Graphics));
        // Size disagreeing, unknown kind, too short.
        assert_eq!(MapHeader::parse(&compute[..0x94]), None);
        assert_eq!(MapHeader::parse(&le(&[5, 0, 0])), None);
        assert_eq!(MapHeader::parse(&[3, 4]), None);
    }

    #[test]
    fn programs_and_padding() {
        let p = program(0x1234_5678, 20);
        assert!(is_program(&p));
        assert!(!is_program(&p[4..]));
        let mut padded = p.clone();
        padded.extend(le(&[CODE_END; 16]));
        assert_eq!(trim_padding(&padded), &p[..]);
        assert_eq!(trim_padding(&p), &p[..]);
        assert_eq!(trim_padding(&le(&[CODE_END; 3])), &[] as &[u8]);
        // Short programs match padded or not; long ones by their head.
        assert_eq!(program_key(&padded), program_key(&p));
        let long = program(7, 100);
        let mut longer = long.clone();
        longer.extend(le(&[CODE_END; 32]));
        assert_eq!(program_key(&long), program_key(&longer));
        assert_ne!(program_key(&long), program_key(&program(8, 100)));
        // A short program padded in its container and padded past HEAD in
        // the description.
        let short = program(9, 20);
        let mut in_container = short.clone();
        in_container.extend(le(&[CODE_END; 8]));
        let mut in_description = short.clone();
        in_description.extend(le(&[CODE_END; 64]));
        assert!(in_description.len() > HEAD);
        assert_eq!(program_key(&in_container), program_key(&in_description));
    }

    #[test]
    fn reads_fspx() {
        let mut file = b"FSPX".to_vec();
        file.extend(le(&[1, 3, 0x80]));
        for (kind, offset, size) in [(0, 0x40, 0x1c), (1, 0x5c, 0x14), (6, 0x70, 0x10)] {
            file.extend(le(&[kind, offset, size, 0]));
        }
        file.resize(0x80, 0);
        file[0x70..0x80].copy_from_slice(&le(&[1, 2, 3, CODE_END]));
        let s = fspx_sections(&file).unwrap();
        assert_eq!(
            s.iter().map(|(k, b)| (*k, b.len())).collect::<Vec<_>>(),
            [(0, 0x1c), (1, 0x14), (6, 0x10)]
        );
        assert_eq!(trim_padding(s[2].1), &le(&[1, 2, 3])[..]);
        // Out of bounds, wrong version, not FSPX.
        let mut bad = file.clone();
        bad[0x38] = 0x80;
        assert!(fspx_sections(&bad).is_none());
        let mut v2 = file.clone();
        v2[4] = 2;
        assert!(fspx_sections(&v2).is_none());
        assert!(fspx_sections(b"DXBC").is_none());
    }

    #[test]
    fn finds_containers_by_program() {
        let (a, b, c) = (program(1, 100), program(2, 100), program(3, 10));
        // A container here is its programs' bytes joined, 200 dwords apart.
        let containers = vec![
            (0x100, [a.clone(), b.clone()].concat()),
            (0x900, c.clone()),
            (0x1000, a.clone()),
        ];
        let split = |bytes: &[u8]| {
            bytes
                .chunks(a.len())
                .map(<[u8]>::to_vec)
                .collect::<Vec<_>>()
        };
        let index = PackIndex::build(containers, split);
        assert_eq!(index.programs(), 3);
        let mut padded = b.clone();
        padded.extend(le(&[CODE_END; 16]));
        assert_eq!(index.find(&padded), Some(0x100));
        assert_eq!(
            index.find(&a),
            Some(0x100),
            "the first container holding it"
        );
        let mut short = c.clone();
        short.extend(le(&[CODE_END; 4]));
        assert_eq!(index.find(&short), Some(0x900));
        assert_eq!(index.find(&program(4, 100)), None);
        let mut more = PackIndex::build([(0x2000, program(4, 100))], |b: &[u8]| vec![b.to_vec()]);
        more.merge(index);
        assert_eq!(more.find(&program(4, 100)), Some(0x2000));
        assert_eq!(more.find(&padded), Some(0x100));
    }

    /// An info section holding `elements` (name, index, format, slot,
    /// offset, classification, step rate), laid out as the title's.
    fn info(elements: &[(&str, [u32; 6])]) -> Vec<u8> {
        let mut layout = le(&[elements.len() as u32]);
        for (name, fields) in elements {
            layout.extend(le(&[name.len() as u32 + 1]));
            layout.extend(name.as_bytes());
            layout.push(0);
            layout.extend(le(fields));
        }
        let mut info = le(&[
            0x76e7_bc07,
            0x1000,
            0,
            1,
            0x1c,
            0,
            0,
            u32::MAX,
            0,
            0x3_0100,
            1,
            0,
            0,
            0,
        ]);
        info.extend(le(&[layout.len() as u32, 0xc, 0, 0, 0]));
        assert_eq!(info.len(), 0x4c);
        info.extend(&layout);
        info.resize(info.len().next_multiple_of(4), 0);
        info
    }

    #[test]
    fn reads_input_layouts() {
        let section = info(&[
            ("POSITION", [0, 6, 0, 0, 0, 0]),
            ("COLOR", [0, 28, 0, 12, 0, 0]),
            ("TEXCOORD", [0, 16, 0, 16, 0, 0]),
        ]);
        // As description 00 holds it, byte for byte from 0x38 on.
        assert_eq!(section.len(), 188);
        assert_eq!(&section[0x50..0x5d], b"\x09\0\0\0POSITION\0");
        let layout = input_layout(&section).unwrap();
        let summary: Vec<_> = layout
            .iter()
            .map(|e| (e.name.as_str(), e.format, e.offset))
            .collect();
        assert_eq!(
            summary,
            [("POSITION", 6, 0), ("COLOR", 28, 12), ("TEXCOORD", 16, 16)]
        );
        assert!(
            layout
                .iter()
                .all(|e| e.slot == 0 && !e.per_instance && e.index == 0)
        );
        // Per instance, a second slot, a semantic index.
        let layout = input_layout(&info(&[("TEXCOORD", [3, 2, 1, 8, 1, 1])])).unwrap();
        assert_eq!(
            layout,
            [InputElement {
                name: "TEXCOORD".into(),
                index: 3,
                format: 2,
                slot: 1,
                offset: 8,
                per_instance: true,
                step_rate: 1
            }]
        );
        // None (a section of 0x50 bytes).
        assert_eq!(input_layout(&info(&[])), Some(Vec::new()));
        // What does not read: cut short, a name without its zero, an
        // unknown classification, a stated size past the section.
        assert_eq!(input_layout(&section[..0x90]), None);
        let mut no_zero = section.clone();
        no_zero[0x5c] = b'X';
        assert_eq!(input_layout(&no_zero), None);
        assert_eq!(input_layout(&info(&[("COLOR", [0, 28, 0, 0, 2, 0])])), None);
        let mut long = section.clone();
        long[0x38] = 0xff;
        assert_eq!(input_layout(&long), None);
    }

    /// What the title hands a method of a container's program: a third
    /// `v_nop` after the two hashed ones, the code after it changed, and
    /// padding (as in every description dumped, compute ones included).
    fn as_described(program: &[u8]) -> Vec<u8> {
        let mut code = program[..HASH_WORDS].to_vec();
        code.extend(le(&[0xd580_0000, 0x03fe_0100, 0x0000_0ac0]));
        code.extend(program[HASH_WORDS..].iter().map(|b| b ^ 0x5a));
        code.extend(le(&[CODE_END; 32]));
        code
    }

    #[test]
    fn finds_containers_of_described_programs_by_their_hash() {
        // Vertex and pixel programs of a material, each in a container of
        // its own, and a compute one; the pack holds them as they are.
        let (vs, ps, cs) = (
            program(0x1111, 60),
            program(0x2222, 90),
            program(0x3333, 40),
        );
        let pack = vec![
            (0x100, vs.clone()),
            (0x900, ps.clone()),
            (0x1400, cs.clone()),
        ];
        let index = PackIndex::build(pack, |b: &[u8]| vec![b.to_vec()]);
        assert_eq!(index.find(&as_described(&vs)), Some(0x100));
        assert_eq!(index.find(&as_described(&ps)), Some(0x900));
        assert_eq!(index.find(&as_described(&cs)), Some(0x1400));
        // The program as it is (and padded) is found as before.
        assert_eq!(index.find(&ps), Some(0x900));
        // Another program's hash is another program, whatever follows.
        let mut other = as_described(&ps);
        other[8] ^= 1;
        assert_eq!(index.find(&other), None);
        // Code without the prologue is still matched by its first bytes.
        let plain = le(&(0..80u32).collect::<Vec<_>>());
        let index = PackIndex::build([(0x40, plain.clone())], |b: &[u8]| vec![b.to_vec()]);
        assert!(!is_program(&plain));
        assert_eq!(index.find(&plain), Some(0x40));
    }

    /// A graphics registers section of 97 dwords as most descriptions
    /// logged have it, with `set` (position, value) put in.
    fn registers(set: &[(usize, u32)]) -> Vec<u32> {
        let mut r = vec![0u32; 97];
        r[at::CLIP_CNTL] = 0x0008_0000;
        r[at::SC_MODE_CNTL] = 0x0020_1a40;
        r[at::DEPTH_CONTROL] = 0x0070_07f0;
        r[at::STENCIL_REF_MASK] = 0x0100_ff00;
        r[at::STENCIL_REF_MASK_BF] = 0x0100_ff00;
        r[at::TARGET_MASK] = 0xffff_ffff;
        for b in &mut r[at::BLEND_CONTROL..at::BLEND_CONTROL + 8] {
            *b = 0x8000_0000;
        }
        for &(at, v) in set {
            r[at] = v;
        }
        r
    }

    #[test]
    fn decodes_the_common_description() {
        let (r, notes) = GraphicsRegisters::decode(&registers(&[]));
        assert!(notes.is_empty(), "{notes:?}");
        // No depth or stencil test, no blending, all written, no culling.
        let defaults = GraphicsRegisters::console_defaults();
        assert_eq!(
            (r.blend, r.write_masks, r.raster),
            (defaults.blend, defaults.write_masks, defaults.raster)
        );
        assert!(!r.depth.depth_enable && !r.depth.stencil_enable && !r.depth.depth_bounds);
        assert_eq!(
            (r.depth.front, r.depth.back),
            (StencilFace::KEEP, StencilFace::KEEP)
        );
    }

    #[test]
    fn decodes_depth_and_stencil() {
        // Reverse depth test, no write (0x007007e2).
        let (r, _) = GraphicsRegisters::decode(&registers(&[(at::DEPTH_CONTROL, 0x0070_07e2)]));
        assert!(r.depth.depth_enable && !r.depth.depth_write && !r.depth.stencil_enable);
        assert_eq!(r.depth.depth_func, 7, "greater or equal");
        // Depth test and write always, stencil equal (0x007002f7).
        let (r, _) = GraphicsRegisters::decode(&registers(&[
            (at::DEPTH_CONTROL, 0x0070_02f7),
            (at::STENCIL_REF_MASK, 0x0100_0700),
        ]));
        assert!(r.depth.depth_enable && r.depth.depth_write && r.depth.stencil_enable);
        assert_eq!((r.depth.depth_func, r.depth.front.func), (8, 3));
        assert_eq!((r.depth.read_mask, r.depth.write_mask), (0x07, 0));
        // Back faces as their own: always.
        assert_eq!(r.depth.back.func, 8);
        // Stencil equal, add one wrapping on pass, both faces (description
        // 56: 0x002002f1, 0x00080080, 0x01ffff00).
        let (r, notes) = GraphicsRegisters::decode(&registers(&[
            (at::DEPTH_CONTROL, 0x0020_02f1),
            (at::STENCIL_CONTROL, 0x0008_0080),
            (at::STENCIL_REF_MASK, 0x01ff_ff00),
            (at::STENCIL_REF_MASK_BF, 0x01ff_ff00),
        ]));
        assert!(notes.is_empty(), "{notes:?}");
        let face = StencilFace {
            fail: 1,
            depth_fail: 1,
            pass: 7,
            func: 3,
        };
        assert_eq!((r.depth.front, r.depth.back), (face, face));
        assert!(!r.depth.depth_enable);
        // Zero on fail (description 8: 0x00200291, 0x00001001, 0x01202800).
        let (r, _) = GraphicsRegisters::decode(&registers(&[
            (at::DEPTH_CONTROL, 0x0020_0291),
            (at::STENCIL_CONTROL, 0x0000_1001),
            (at::STENCIL_REF_MASK, 0x0120_2800),
            (at::STENCIL_REF_MASK_BF, 0x0120_2800),
        ]));
        assert_eq!(
            r.depth.front,
            StencilFace {
                fail: 2,
                depth_fail: 1,
                pass: 1,
                func: 3
            }
        );
        assert_eq!((r.depth.read_mask, r.depth.write_mask), (0x28, 0x20));
        // Depth bounds; a stencil op D3D12 lacks is noted.
        let (r, notes) = GraphicsRegisters::decode(&registers(&[
            (at::DEPTH_CONTROL, 0x0000_000b),
            (at::STENCIL_CONTROL, 0x0000_000a),
        ]));
        assert!(r.depth.depth_bounds);
        assert_eq!(r.depth.front.fail, 1);
        assert_eq!(notes, ["a logic stencil op: keep"]);
    }

    #[test]
    fn decodes_blending_and_write_masks() {
        let (r, notes) = GraphicsRegisters::decode(&registers(&[
            // Source alpha over, separate alpha the same.
            (at::BLEND_CONTROL, 0xe504_0504),
            // Additive: source alpha plus one; alpha zero plus one.
            (at::BLEND_CONTROL + 1, 0xe100_0104),
            // Destination colour times source colour, not separate: the
            // alpha factors become alpha ones; reverse subtract.
            (at::BLEND_CONTROL + 2, 0x4000_0882),
            // Constant alpha (noted).
            (at::BLEND_CONTROL + 3, 0x4000_0113),
            // RGB to target 0, nothing to target 1, all to the rest.
            (at::TARGET_MASK, 0xffff_ff07),
        ]));
        let over = TargetBlend {
            enable: true,
            src: 5,
            dest: 6,
            op: 1,
            src_alpha: 5,
            dest_alpha: 6,
            op_alpha: 1,
        };
        assert_eq!(r.blend[0], over);
        let add = TargetBlend {
            enable: true,
            src: 5,
            dest: 2,
            op: 1,
            src_alpha: 1,
            dest_alpha: 2,
            op_alpha: 1,
        };
        assert_eq!(r.blend[1], add);
        let mul = TargetBlend {
            enable: true,
            src: 3,
            dest: 9,
            op: 3,
            src_alpha: 5,
            dest_alpha: 7,
            op_alpha: 3,
        };
        assert_eq!(r.blend[2], mul);
        assert_eq!((r.blend[3].src, r.blend[3].dest), (14, 2));
        assert_eq!(r.blend[4], TargetBlend::OFF);
        assert_eq!(r.write_masks, [7, 0, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf]);
        assert_eq!(notes, ["blend factor constant alpha: the blend factor"; 2]);
    }

    /// Registers of descriptions dumped in run 39, against the state the
    /// title's own pipelines of the same shaders had in run 45 (made by
    /// another method, with D3D12's state, as its frame listings show).
    #[test]
    fn decodes_what_the_titles_own_pipelines_have() {
        let off = StencilFace::KEEP;
        // Description 15 (vertex 0602c2384acbb3b2, pixel a78d088fa55aa495):
        // "depth false; cull 1; blend true src 2 dst 6 op 1 mask 0xf".
        let (r, notes) = GraphicsRegisters::decode(&registers(&[(at::BLEND_CONTROL, 0xe501_0501)]));
        assert!(notes.is_empty(), "{notes:?}");
        assert!(!r.depth.depth_enable && !r.depth.stencil_enable);
        assert_eq!(
            (r.depth.front, r.raster.cull, r.write_masks[0]),
            (off, 1, 0xf)
        );
        assert_eq!(
            (
                r.blend[0].enable,
                r.blend[0].src,
                r.blend[0].dest,
                r.blend[0].op
            ),
            (true, 2, 6, 1)
        );
        // Descriptions 20 and 24 (vertex f7f331c8c5b3a0d8, 80cb958cde505e15):
        // "blend true src 5 dst 6 op 1".
        let (r, _) = GraphicsRegisters::decode(&registers(&[(at::BLEND_CONTROL, 0xe504_0504)]));
        assert_eq!(
            (
                r.blend[0].enable,
                r.blend[0].src,
                r.blend[0].dest,
                r.blend[0].op
            ),
            (true, 5, 6, 1)
        );
        // Description 3 (vertex 58d926085fca4999): no depth, no culling,
        // source alpha over; and its culling twin (0x00201a42): "cull 3
        // front-ccw true; blend false src 2 dst 1".
        let (r, _) = GraphicsRegisters::decode(&registers(&[(at::BLEND_CONTROL, 0xe504_0504)]));
        assert_eq!(
            (r.depth.depth_enable, r.raster.cull, r.raster.front_ccw),
            (false, 1, true)
        );
        let (r, _) = GraphicsRegisters::decode(&registers(&[(at::SC_MODE_CNTL, 0x0020_1a42)]));
        assert_eq!((r.raster.cull, r.raster.front_ccw), (3, true));
        assert_eq!(r.blend[0], TargetBlend::OFF);
    }

    #[test]
    fn decodes_the_rasterizer() {
        let (r, _) = GraphicsRegisters::decode(&registers(&[(at::SC_MODE_CNTL, 0x0020_1a42)]));
        assert_eq!(
            r.raster,
            Raster {
                cull: 3,
                front_ccw: true,
                wireframe: false,
                depth_clip: true,
                depth_bias: 0,
                slope_bias: 0.0,
                bias_clamp: 0.0,
                conservative: false,
            }
        );
        // Front culled, clockwise front, lines, no depth clipping.
        let (r, notes) = GraphicsRegisters::decode(&registers(&[
            (at::SC_MODE_CNTL, 0x0000_012d),
            (at::CLIP_CNTL, 0x0c08_0000),
        ]));
        assert_eq!(
            r.raster,
            Raster {
                cull: 2,
                front_ccw: false,
                wireframe: true,
                depth_clip: false,
                depth_bias: 0,
                slope_bias: 0.0,
                bias_clamp: 0.0,
                conservative: false,
            }
        );
        assert!(notes.is_empty(), "{notes:?}");
        let (r, notes) = GraphicsRegisters::decode(&registers(&[(at::SC_MODE_CNTL, 3)]));
        assert_eq!(r.raster.cull, 1);
        assert_eq!(notes.len(), 1);
    }

    /// Conservative rasterization (dword 72 of run 50's fog volume and
    /// trail map passes), and its default.
    #[test]
    fn decodes_conservative_rasterization() {
        let on = registers(&[(at::CONSERVATIVE, 0x00e0_0441)]);
        assert!(GraphicsRegisters::decode(&on).0.raster.conservative);
        let off = registers(&[(at::CONSERVATIVE, 0x0010_0000)]);
        assert!(!GraphicsRegisters::decode(&off).0.raster.conservative);
        // A section ending before it (none seen): off.
        assert!(!GraphicsRegisters::decode(&on[..70]).0.raster.conservative);
    }

    /// The polygon offsets of run 50's descriptions, against the depth
    /// bias the title gives its own pipelines of the same shaders (run 61,
    /// `VICTORY_PRECOMPILED_GRAPHICS=0`): (clamp, scale, offset) of
    /// (0.003, 29, 0) for the hair strands' cascade shadow (vertex
    /// 643101624bd14159, pixel f758c2f8e6b9b517) is the title's slope
    /// 1.8125, clamp 0.003, bias 0; (0, 16, 1) for contact markers is
    /// slope 1, bias 1; (0, 16, 0) for distant lights slope 1, bias 0.
    #[test]
    fn decodes_the_depth_bias() {
        let bias = |words: [u32; 3], enable: u32| {
            let mut set = vec![(at::SC_MODE_CNTL, 0x0020_0240 | enable << 11)];
            set.extend((0..3).map(|i| (at::POLY_OFFSET + i, words[i])));
            let r = GraphicsRegisters::decode(&registers(&set)).0.raster;
            (r.depth_bias, r.slope_bias, r.bias_clamp)
        };
        assert_eq!(bias([0, 0x4180_0000, 0x3f80_0000], 1), (1, 1.0, 0.0));
        assert_eq!(bias([0, 0x4180_0000, 0], 1), (0, 1.0, 0.0));
        assert_eq!(
            bias([0x3b44_9ba6, 0x41e8_0000, 0], 1),
            (0, 1.8125, f32::from_bits(0x3b44_9ba6))
        );
        // Not enabled, or none: no bias.
        assert_eq!(bias([0, 0x4180_0000, 0x3f80_0000], 0), (0, 0.0, 0.0));
        assert_eq!(bias([0, 0, 0], 1), (0, 0.0, 0.0));
        // Not a number: none, noted.
        let mut set = registers(&[(at::POLY_OFFSET + 1, 0x7fc0_0000)]);
        set[at::SC_MODE_CNTL] |= 1 << 11;
        let (r, notes) = GraphicsRegisters::decode(&set);
        assert_eq!(r.raster.slope_bias, 0.0);
        assert_eq!(notes.len(), 1);
    }

    #[test]
    fn a_short_section_gets_console_defaults() {
        let (r, notes) = GraphicsRegisters::decode(&[0x0070_02f7; 50]);
        assert_eq!(r, GraphicsRegisters::console_defaults());
        assert_eq!(notes.len(), 1);
        // The longer sections (99 dwords) read the same.
        let mut long = registers(&[(at::DEPTH_CONTROL, 0x0070_07e2)]);
        long.extend([0x401, 0x402]);
        assert!(GraphicsRegisters::decode(&long).0.depth.depth_enable);
    }

    #[test]
    fn targets_key_what_a_pipeline_is_made_for() {
        let a = Targets::new(&[28, 10], 40, 1).drawing(4);
        assert_eq!(
            (a.count, a.formats, a.topology),
            (2, [28, 10, 0, 0, 0, 0, 0, 0], 3)
        );
        // Strips and lists, fans and quads: the same pipeline.
        assert_eq!(a, Targets::new(&[28, 10], 40, 0).drawing(5));
        assert_eq!(a, Targets::new(&[28, 10], 40, 1).drawing(19));
        // Anything else bound: another one.
        for other in [
            Targets::new(&[28], 40, 1).drawing(4),
            Targets::new(&[28, 10, 0], 40, 1).drawing(4),
            Targets::new(&[10, 28], 40, 1).drawing(4),
            Targets::new(&[28, 10], 0, 1).drawing(4),
            Targets::new(&[28, 10], 40, 4).drawing(4),
            Targets::new(&[28, 10], 40, 1).drawing(2),
            Targets::new(&[28, 10], 40, 1).drawing(17),
        ] {
            assert_ne!(a, other);
        }
        assert_eq!(Targets::new(&[1; 9], 0, 1).count, 8);
        assert_eq!(
            Targets::new(&[], 0, 1).drawing(17).topology,
            RECTANGLES_TYPE
        );
        assert_eq!(topology_type(1), 1);
        assert_eq!(topology_type(36), 4);
    }

    #[test]
    fn variants_are_made_once_per_targets() {
        let v: Variants<u32> = Variants::default();
        let a = Targets::new(&[28], 0, 1).drawing(4);
        let b = Targets::new(&[10], 0, 1).drawing(4);
        let mut made = 0;
        let mut make = || {
            made += 1;
            made
        };
        let (first, new) = v.get(&a, &mut make);
        assert!(new);
        let (again, new) = v.get(&a, &mut make);
        assert!(!new && std::sync::Arc::ptr_eq(&first, &again));
        let (other, new) = v.get(&b, &mut make);
        assert!(new);
        assert_eq!((*first, *other), (1, 2));
        assert_eq!(v.all().len(), 2);
    }
}
