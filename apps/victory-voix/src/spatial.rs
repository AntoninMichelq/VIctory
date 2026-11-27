//! Le volume et le côté (gauche/droite) d'un ami selon où il est par rapport
//! à notre caméra.

use crate::proto::Pose;

#[derive(Clone, Copy, Debug)]
pub struct Reglages {
    /// En dessous : volume plein (mètres du jeu).
    pub pres: f32,
    /// Au-delà : plus rien.
    pub loin: f32,
    /// Échange gauche et droite (si le jeu tourne le repère dans l'autre sens).
    pub inverser: bool,
}

/// Gains gauche/droite, distance (None : position inconnue, vocal normal)
/// et côté (-1 gauche … +1 droite).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rendu {
    pub gauche: f32,
    pub droite: f32,
    pub distance: Option<f32>,
    pub cote: f32,
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn len(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

pub fn rendu(moi: Option<&Pose>, ami: Option<[f32; 3]>, r: &Reglages) -> Rendu {
    let (Some(moi), Some(ami)) = (moi, ami) else {
        return Rendu { gauche: 1.0, droite: 1.0, distance: None, cote: 0.0 };
    };
    let v = sub(ami, moi.pos);
    let d = len(v);
    let volume = if d <= r.pres {
        1.0
    } else if d >= r.loin {
        0.0
    } else {
        // Une pente douce : on entend encore un peu de loin, fort de près.
        let t = 1.0 - (d - r.pres) / (r.loin - r.pres);
        t * t
    };
    // Le côté : la direction de l'ami, sans la hauteur, sur l'axe « droite ».
    let up_len = len(moi.up);
    let flat = if up_len > 0.5 {
        let up = moi.up.map(|x| x / up_len);
        let h = dot(v, up);
        [v[0] - up[0] * h, v[1] - up[1] * h, v[2] - up[2] * h]
    } else {
        v
    };
    let fl = len(flat);
    let rl = len(moi.right);
    let mut cote = if fl > 0.3 && rl > 0.5 { (dot(flat, moi.right) / (fl * rl)).clamp(-1.0, 1.0) } else { 0.0 };
    if r.inverser {
        cote = -cote;
    }
    // Panoramique à puissance constante, sans couper totalement une oreille.
    let cote_doux = cote * 0.8;
    let angle = (cote_doux + 1.0) * std::f32::consts::FRAC_PI_4;
    let norm = std::f32::consts::SQRT_2;
    Rendu {
        gauche: volume * angle.cos() * norm,
        droite: volume * angle.sin() * norm,
        distance: Some(d),
        cote,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: Reglages = Reglages { pres: 3.0, loin: 40.0, inverser: false };
    fn moi() -> Pose {
        Pose { pos: [0.0; 3], right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0] }
    }

    #[test]
    fn distance_et_cote() {
        let inconnu = rendu(None, Some([1.0, 0.0, 0.0]), &R);
        assert_eq!((inconnu.gauche, inconnu.droite, inconnu.distance), (1.0, 1.0, None));
        let pres = rendu(Some(&moi()), Some([0.0, 2.0, 0.0]), &R);
        assert!((pres.gauche - 1.0).abs() < 1e-5 && (pres.droite - 1.0).abs() < 1e-5);
        let loin = rendu(Some(&moi()), Some([0.0, 50.0, 0.0]), &R);
        assert_eq!((loin.gauche, loin.droite), (0.0, 0.0));
        let droite = rendu(Some(&moi()), Some([10.0, 0.0, 5.0]), &R);
        assert!(droite.droite > droite.gauche && droite.gauche > 0.0);
        assert!((droite.cote - 1.0).abs() < 1e-5);
        let milieu = rendu(Some(&moi()), Some([0.0, 20.0, 0.0]), &R);
        assert!(milieu.gauche > 0.1 && milieu.gauche < 0.6);
    }
}
