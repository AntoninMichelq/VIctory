//! La file d'attente du son d'un ami : on attend d'avoir un peu d'avance
//! avant de jouer, pour que les paquets en retard ne coupent pas la voix.

use std::collections::VecDeque;

/// 60 ms d'avance avant de commencer à jouer (à 48 kHz).
const AVANCE: usize = 2880;
/// Au-delà de 250 ms de retard on jette le plus vieux pour revenir à 80 ms.
const TROP: usize = 12_000;
const REVENIR: usize = 3840;

#[derive(Default)]
pub struct Jitter {
    q: VecDeque<f32>,
    joue: bool,
    dernier: Option<u16>,
}

impl Jitter {
    /// Combien de paquets manquent avant celui-ci (0 : il suit le précédent),
    /// ou None si le paquet est en retard / en double (à jeter).
    pub fn ordre(&mut self, seq: u16) -> Option<u16> {
        let manque = match self.dernier {
            None => 0,
            Some(d) => {
                let ecart = seq.wrapping_sub(d);
                if ecart == 0 || ecart > 30_000 {
                    return None;
                }
                // Après une longue pause (ami muet), on ne comble pas.
                if ecart > 6 { 0 } else { ecart - 1 }
            }
        };
        self.dernier = Some(seq);
        Some(manque)
    }

    pub fn pousser(&mut self, s: &[f32]) {
        self.q.extend(s.iter().copied());
        if self.q.len() > TROP {
            let jeter = self.q.len() - REVENIR;
            self.q.drain(..jeter);
        }
    }

    pub fn prendre(&mut self) -> f32 {
        if !self.joue {
            if self.q.len() < AVANCE {
                return 0.0;
            }
            self.joue = true;
        }
        match self.q.pop_front() {
            Some(s) => s,
            None => {
                self.joue = false;
                0.0
            }
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn en_attente_ms(&self) -> usize {
        self.q.len() / 48
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attend_puis_joue() {
        let mut j = Jitter::default();
        j.pousser(&[0.5; 960]);
        assert_eq!(j.prendre(), 0.0);
        j.pousser(&[0.5; 960 * 2]);
        assert_eq!(j.prendre(), 0.5);
        assert_eq!(j.ordre(10), Some(0));
        assert_eq!(j.ordre(11), Some(0));
        assert_eq!(j.ordre(13), Some(1));
        assert_eq!(j.ordre(12), None);
        assert_eq!(j.ordre(100), Some(0));
        j.pousser(&vec![0.1; 20_000]);
        assert!(j.en_attente_ms() <= 250);
    }
}
