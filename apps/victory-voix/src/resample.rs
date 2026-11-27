//! Changement de fréquence simple (linéaire) entre la carte son et les
//! 48 kHz d'Opus. Pour de la voix, c'est largement suffisant.

/// Pousse des échantillons à `de` Hz, ressort des échantillons à `vers` Hz.
pub struct Pousse {
    pas: f64,
    t: f64,
    avant: f32,
}

impl Pousse {
    pub fn new(de: u32, vers: u32) -> Self {
        Pousse { pas: f64::from(de) / f64::from(vers), t: 0.0, avant: 0.0 }
    }

    pub fn traiter(&mut self, entree: &[f32], sortie: &mut Vec<f32>) {
        for &s in entree {
            // Les sorties qui tombent entre `avant` et `s`.
            while self.t < 1.0 {
                sortie.push(self.avant + (s - self.avant) * self.t as f32);
                self.t += self.pas;
            }
            self.t -= 1.0;
            self.avant = s;
        }
    }
}

/// Tire des paires (gauche, droite) à 48 kHz, en donne à la fréquence de la
/// carte son.
pub struct Tire {
    pas: f64,
    t: f64,
    a: (f32, f32),
    b: (f32, f32),
}

impl Tire {
    pub fn new(de: u32, vers: u32) -> Self {
        Tire { pas: f64::from(de) / f64::from(vers), t: 0.0, a: (0.0, 0.0), b: (0.0, 0.0) }
    }

    pub fn suivant(&mut self, source: &mut impl FnMut() -> (f32, f32)) -> (f32, f32) {
        while self.t >= 1.0 {
            self.a = self.b;
            self.b = source();
            self.t -= 1.0;
        }
        let t = self.t as f32;
        self.t += self.pas;
        (self.a.0 + (self.b.0 - self.a.0) * t, self.a.1 + (self.b.1 - self.a.1) * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longueurs() {
        let mut p = Pousse::new(44_100, 48_000);
        let mut out = Vec::new();
        p.traiter(&vec![0.25; 44_100], &mut out);
        assert!((out.len() as i64 - 48_000).abs() <= 2, "{}", out.len());
        assert!(out[100..].iter().all(|&x| (x - 0.25).abs() < 1e-6));
        let mut t = Tire::new(48_000, 44_100);
        let mut n = 0;
        for _ in 0..44_100 {
            t.suivant(&mut || {
                n += 1;
                (1.0, 1.0)
            });
        }
        assert!((n as i64 - 48_000).abs() <= 2, "{n}");
    }
}
