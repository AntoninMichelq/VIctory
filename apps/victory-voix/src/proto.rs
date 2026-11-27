//! Ce qui passe sur le réseau entre deux copies de `victory-voix`, et entre
//! l'émulateur et `victory-voix` sur la même machine. Tout est en UDP, petit
//! boutiste, un paquet = un message.

/// Entre amis : salut (code + nom), voix (un paquet Opus de 20 ms), position.
pub const MAGIC: [u8; 4] = *b"VXV1";
/// De l'émulateur vers `victory-voix` : où est la caméra du joueur.
pub const MAGIC_JEU: [u8; 4] = *b"VXP1";
/// De `victory-voix` vers l'émulateur : où sont les amis (pour un marqueur).
pub const MAGIC_AMIS: [u8; 4] = *b"VXA1";

const HELLO: u8 = 1;
const VOICE: u8 = 2;
const POS: u8 = 3;
const BYE: u8 = 4;

/// Où l'on est et comment on est tourné : position, vecteur « droite » de la
/// caméra (pour savoir si l'ami parle à gauche ou à droite) et vecteur haut.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub pos: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
}

#[derive(Debug, PartialEq)]
pub enum Packet<'a> {
    Hello { code: u64, name: &'a str },
    Voice { seq: u16, data: &'a [u8] },
    Pos(Option<Pose>),
    Bye,
}

/// Le code de la partie n'est jamais envoyé en clair, seulement son empreinte.
pub fn code_hash(code: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in code.trim().to_lowercase().bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn put_f32s(out: &mut Vec<u8>, v: &[f32]) {
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn f32s<const N: usize>(b: &[u8]) -> Option<[f32; N]> {
    if b.len() < N * 4 {
        return None;
    }
    let mut o = [0f32; N];
    for (i, x) in o.iter_mut().enumerate() {
        *x = f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().ok()?);
    }
    o.iter().all(|x| x.is_finite()).then_some(o)
}

fn pose_bytes(out: &mut Vec<u8>, p: &Pose) {
    put_f32s(out, &p.pos);
    put_f32s(out, &p.right);
    put_f32s(out, &p.up);
}

fn pose_from(b: &[u8]) -> Option<Pose> {
    let v = f32s::<9>(b)?;
    Some(Pose {
        pos: [v[0], v[1], v[2]],
        right: [v[3], v[4], v[5]],
        up: [v[6], v[7], v[8]],
    })
}

impl Packet<'_> {
    pub fn write(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&MAGIC);
        match self {
            Packet::Hello { code, name } => {
                out.push(HELLO);
                out.extend_from_slice(&code.to_le_bytes());
                let n = name.as_bytes();
                let n = &n[..n.len().min(32)];
                out.extend_from_slice(n);
            }
            Packet::Voice { seq, data } => {
                out.push(VOICE);
                out.extend_from_slice(&seq.to_le_bytes());
                out.extend_from_slice(data);
            }
            Packet::Pos(p) => {
                out.push(POS);
                match p {
                    Some(p) => {
                        out.push(1);
                        pose_bytes(out, p);
                    }
                    None => out.push(0),
                }
            }
            Packet::Bye => out.push(BYE),
        }
    }

    pub fn read(b: &[u8]) -> Option<Packet<'_>> {
        if b.len() < 5 || b[..4] != MAGIC {
            return None;
        }
        let body = &b[5..];
        match b[4] {
            HELLO if body.len() >= 8 => Some(Packet::Hello {
                code: u64::from_le_bytes(body[..8].try_into().ok()?),
                name: std::str::from_utf8(&body[8..]).unwrap_or("ami"),
            }),
            VOICE if body.len() >= 2 => Some(Packet::Voice {
                seq: u16::from_le_bytes([body[0], body[1]]),
                data: &body[2..],
            }),
            POS if !body.is_empty() => Some(Packet::Pos(match body[0] {
                1 => Some(pose_from(&body[1..])?),
                _ => None,
            })),
            BYE => Some(Packet::Bye),
            _ => None,
        }
    }
}

/// Le paquet que l'émulateur envoie (`VXP1` + pose).
#[cfg_attr(not(test), allow(dead_code))]
pub fn write_game_pose(out: &mut Vec<u8>, p: &Pose) {
    out.clear();
    out.extend_from_slice(&MAGIC_JEU);
    pose_bytes(out, p);
}

pub fn read_game_pose(b: &[u8]) -> Option<Pose> {
    if b.len() < 4 || b[..4] != MAGIC_JEU {
        return None;
    }
    pose_from(&b[4..])
}

/// Les amis connus, pour l'émulateur : `VXA1`, nombre, puis par ami un
/// octet « position connue » et sa position.
pub fn write_friends(out: &mut Vec<u8>, friends: &[Option<[f32; 3]>]) {
    out.clear();
    out.extend_from_slice(&MAGIC_AMIS);
    out.push(friends.len().min(255) as u8);
    for f in friends.iter().take(255) {
        match f {
            Some(p) => {
                out.push(1);
                put_f32s(out, p);
            }
            None => {
                out.push(0);
                put_f32s(out, &[0.0; 3]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let pose = Pose { pos: [1.0, -2.5, 300.0], right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0] };
        let mut buf = Vec::new();
        for p in [
            Packet::Hello { code: code_hash("Pizza"), name: "Paul" },
            Packet::Voice { seq: 65535, data: &[1, 2, 3] },
            Packet::Pos(Some(pose)),
            Packet::Pos(None),
            Packet::Bye,
        ] {
            p.write(&mut buf);
            assert_eq!(Packet::read(&buf), Some(p));
        }
        write_game_pose(&mut buf, &pose);
        assert_eq!(read_game_pose(&buf), Some(pose));
        assert_eq!(code_hash(" pizza "), code_hash("PIZZA"));
        assert_eq!(Packet::read(b"junk"), None);
    }
}
