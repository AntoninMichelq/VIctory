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
//!               (graphics: u32 hash, u32 0x1000, ...)
//! code          GFX10 machine code: the native program of one of the
//!               title's shader containers, followed by s_code_end
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
//! The programs are the same native programs (FSSM) the title's containers
//! in its shader pack hold, byte for byte apart from the padding; the
//! title never hands those containers to the known creation methods. So a
//! description's program is matched to its container by its first bytes,
//! among the containers of the pack ([`PackIndex`]), and the container is
//! translated as any other. What the registers, info and map sections hold
//! beyond that is not read: the container has it in its own form.

use std::collections::HashMap;

/// Device method of graphics pipelines described this way.
pub const GRAPHICS_SLOT: usize = 0x1c8;
/// Device method of compute pipelines described this way.
pub const COMPUTE_SLOT: usize = 0x218;
/// `s_code_end`, which pads programs.
const CODE_END: u32 = 0xbf9f_0000;
/// Programs are matched by this many first bytes (whole if shorter).
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

/// The key a program is matched by: a hash of its first [`HEAD`] bytes,
/// or of the whole if it is shorter, padding trimmed first (programs in
/// containers are padded too, by other amounts).
pub fn program_key(code: &[u8]) -> [u8; 32] {
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
}
