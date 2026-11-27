//! Les réglages : la ligne de commande donne les valeurs de départ, puis
//! `multijoueur.ini` (à côté de l'exe, ou `VICTORY_MULTI_INI`), relu chaque
//! seconde, les remplace pendant la partie (le menu F1 de l'émulateur
//! l'écrit). Une ligne `cle=valeur` par réglage, `#` pour un commentaire.
//! En retour, `multijoueur-etat.ini` à côté dit où on en est.

use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct Conf {
    pub pseudo: String,
    /// Adresses données (IP ou IP:port), séparées par des virgules.
    pub ami: String,
    pub code: String,
    pub port: u16,
    /// Pousser pour parler (sinon micro ouvert avec un seuil).
    pub appuyer: bool,
    pub touche_parler: String,
    pub muet: bool,
    pub volume_micro: f32,
    pub volume_amis: f32,
    pub pres: f32,
    pub loin: f32,
    pub inverser: bool,
    pub seuil: f32,
    pub test_micro: bool,
}

impl Default for Conf {
    fn default() -> Self {
        Conf {
            pseudo: "joueur".into(),
            ami: String::new(),
            code: "victory".into(),
            port: 47777,
            appuyer: false,
            touche_parler: "V".into(),
            muet: false,
            volume_micro: 1.0,
            volume_amis: 1.0,
            pres: 3.0,
            loin: 40.0,
            inverser: false,
            seuil: 0.012,
            test_micro: false,
        }
    }
}

fn oui(v: &str) -> Option<bool> {
    match v.to_ascii_lowercase().as_str() {
        "1" | "oui" | "true" | "on" => Some(true),
        "0" | "non" | "false" | "off" => Some(false),
        _ => None,
    }
}

impl Conf {
    /// Applique le texte d'un `multijoueur.ini` ; les clés inconnues ou
    /// illisibles sont ignorées (la valeur d'avant reste).
    pub fn appliquer(&mut self, texte: &str) {
        for ligne in texte.lines() {
            let ligne = ligne.trim().trim_start_matches('\u{feff}');
            if ligne.starts_with('#') || ligne.starts_with(';') || ligne.starts_with('[') {
                continue;
            }
            let Some((k, v)) = ligne.split_once('=') else { continue };
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
            let pct = |v: &str| v.parse::<f32>().ok().filter(|x| x.is_finite()).map(|x| x.clamp(0.0, 200.0) / 100.0);
            let num = |v: &str| v.parse::<f32>().ok().filter(|x| x.is_finite() && *x >= 0.0);
            match k.as_str() {
                "pseudo" if !v.is_empty() => self.pseudo = v.chars().take(32).collect(),
                "ami" => self.ami = v.to_string(),
                "code" if !v.is_empty() => self.code = v.to_string(),
                "port" => {
                    if let Ok(p) = v.parse() {
                        self.port = p;
                    }
                }
                "micro" => self.appuyer = v.eq_ignore_ascii_case("appuyer"),
                "touche_parler" if !v.is_empty() => self.touche_parler = v.to_string(),
                "muet" => self.muet = oui(v).unwrap_or(self.muet),
                "volume_micro" => self.volume_micro = pct(v).unwrap_or(self.volume_micro),
                "volume_amis" => self.volume_amis = pct(v).unwrap_or(self.volume_amis),
                "pres" => self.pres = num(v).unwrap_or(self.pres),
                "loin" => self.loin = num(v).unwrap_or(self.loin),
                "inverser" => self.inverser = oui(v).unwrap_or(self.inverser),
                "seuil" => self.seuil = num(v).map(|x| x.min(1.0)).unwrap_or(self.seuil),
                "test_micro" => self.test_micro = oui(v).unwrap_or(self.test_micro),
                _ => {}
            }
        }
        if self.loin < self.pres + 1.0 {
            self.loin = self.pres + 1.0;
        }
    }

    pub fn amis(&self) -> impl Iterator<Item = &str> {
        self.ami.split([',', ';', ' ']).map(str::trim).filter(|s| !s.is_empty())
    }
}

/// Où est `multijoueur.ini`.
pub fn chemin() -> PathBuf {
    if let Some(p) = std::env::var_os("VICTORY_MULTI_INI") {
        return PathBuf::from(p);
    }
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
        .unwrap_or_default()
        .join("multijoueur.ini")
}

/// `multijoueur-etat.ini`, à côté du fichier de réglages.
pub fn chemin_etat(ini: &Path) -> PathBuf {
    ini.with_file_name("multijoueur-etat.ini")
}

/// La touche « pousser pour parler » en code de touche Windows.
pub fn touche(nom: &str) -> Option<i32> {
    let n = nom.trim().to_ascii_uppercase();
    let b = n.as_bytes();
    if b.len() == 1 && (b[0].is_ascii_uppercase() || b[0].is_ascii_digit()) {
        return Some(i32::from(b[0]));
    }
    if let Some(f) = n.strip_prefix('F').and_then(|x| x.parse::<i32>().ok())
        && (1..=24).contains(&f)
    {
        return Some(0x6F + f);
    }
    Some(match n.as_str() {
        "SPACE" | "ESPACE" => 0x20,
        "TAB" => 0x09,
        "SHIFT" | "MAJ" => 0x10,
        "LSHIFT" => 0xA0,
        "CTRL" | "CONTROL" => 0x11,
        "LCTRL" => 0xA2,
        "ALT" => 0x12,
        "CAPSLOCK" | "VERRMAJ" => 0x14,
        "MOUSE4" | "SOURIS4" | "XBUTTON1" => 0x05,
        "MOUSE5" | "SOURIS5" | "XBUTTON2" => 0x06,
        "MOUSEMIDDLE" | "SOURISMILIEU" => 0x04,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lecture() {
        let mut c = Conf::default();
        c.appliquer("\u{feff}# essai\npseudo = Paul\nami=1.2.3.4, 5.6.7.8:9000\nmicro=appuyer\nmuet=1\nvolume_amis=150\npres=5\nloin=2\nseuil=x\n");
        assert_eq!(c.pseudo, "Paul");
        assert_eq!(c.amis().collect::<Vec<_>>(), ["1.2.3.4", "5.6.7.8:9000"]);
        assert!(c.appuyer && c.muet);
        assert_eq!(c.volume_amis, 1.5);
        assert_eq!((c.pres, c.loin), (5.0, 6.0));
        assert_eq!(c.seuil, 0.012);
        assert_eq!(touche("v"), Some(0x56));
        assert_eq!(touche("F5"), Some(0x74));
        assert_eq!(touche("mouse4"), Some(0x05));
        assert_eq!(touche("??"), None);
    }
}
