//! The tabs of the F1 menu besides the keys (`keymap`): GRAPHISMES (the
//! title's own quality settings and ours, in `reglages.ini`, applied at the
//! next start by [`apply_graphics`]), MULTIJOUEUR (`multijoueur.ini`, read
//! by victory-voix every second, and its `multijoueur-etat.ini`) and
//! DEVELOPPEUR (to come).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use victory_gpu::menu::Item;

/// What a key did in a tab.
#[derive(Default)]
pub(super) struct Reply {
    pub note: Option<String>,
    /// The tab's file is to be written again.
    pub changed: bool,
}

impl Reply {
    fn note(text: impl Into<String>) -> Reply {
        Reply {
            note: Some(text.into()),
            changed: false,
        }
    }
}

const ENTER: u8 = 0x0D;
const BACK: u8 = 0x08;
const DELETE: u8 = 0x2E;
const ESCAPE: u8 = 0x1B;
const KEY_LEFT: u8 = 0x25;
const KEY_RIGHT: u8 = 0x27;

/// Our files sit beside `touches.ini` (`VICTORY_TOUCHES`), else beside
/// victory.exe; `env` names one file on its own.
fn beside(env: &str, name: &str) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(env) {
        return Some(PathBuf::from(path));
    }
    let dir = std::env::var_os("VICTORY_TOUCHES")
        .map(PathBuf::from)
        .and_then(|p| Some(p.parent()?.to_path_buf()))
        .or_else(|| Some(std::env::current_exe().ok()?.parent()?.to_path_buf()))?;
    Some(dir.join(name))
}

/// `key = value` lines (`#` and `;` start a comment), names in upper case.
fn lines(text: &str) -> Vec<(String, String)> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with(';'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_ascii_uppercase(), v.trim().to_owned()))
        .collect()
}

fn yes(value: &str) -> bool {
    matches!(
        value.to_ascii_uppercase().as_str(),
        "OUI" | "1" | "ON" | "YES" | "VRAI"
    )
}

fn number(value: &str) -> Option<f32> {
    value
        .replace(',', ".")
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
}

/// Yes or no, on screen.
fn shown(b: bool) -> &'static str {
    if b { "Oui" } else { "Non" }
}

/// A preset's name, on screen.
fn pretty(name: &str) -> &'static str {
    match name {
        "NORMALE" => "Normale",
        "LEGERE" => "Légère",
        "TRES LEGERE" => "Très légère",
        _ => "Perso",
    }
}

fn yes_no(b: bool) -> &'static str {
    if b { "OUI" } else { "NON" }
}

// ---------------------------------------------------------------- GRAPHISMES

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Graphics {
    /// A preset picked with Left/Right, applied only on Enter.
    preview: Preview,
    /// `lodScale`: how far details are kept.
    distance: f32,
    /// `grassLod`.
    grass: f32,
    /// `smallCullThreshold`: small things left out from this size up.
    small: f32,
    /// `motionBlur`.
    blur: bool,
    /// `shadowParticleShadows`.
    particle_shadows: bool,
    /// `rtDynamicShadows`.
    rt_shadows: bool,
    /// `waterReflectionSSR` and its size (512x256, else 256x128).
    water: bool,
    /// `ssroSampleCount`, 1 or 2.
    occlusion: u32,
    /// `sharpenEnabled`.
    sharpen: bool,
}

/// A preset being looked at; not part of what the settings are (two
/// settings with different previews are the same).
#[derive(Clone, Copy, Debug, Default)]
struct Preview(Option<usize>);

impl PartialEq for Preview {
    fn eq(&self, _: &Preview) -> bool {
        true
    }
}

/// The title's own settings (as a Series S writes them), then the lighter ones.
const PRESETS: [(&str, Graphics); 3] = [
    (
        "NORMALE",
        Graphics {
            preview: Preview(None),
            distance: 1.0,
            grass: 0.8,
            small: 1.0,
            blur: true,
            particle_shadows: true,
            rt_shadows: true,
            water: true,
            occlusion: 2,
            sharpen: true,
        },
    ),
    (
        "LEGERE",
        Graphics {
            preview: Preview(None),
            distance: 0.6,
            grass: 0.3,
            small: 2.0,
            blur: false,
            particle_shadows: false,
            rt_shadows: false,
            water: false,
            occlusion: 1,
            sharpen: true,
        },
    ),
    (
        "TRES LEGERE",
        Graphics {
            preview: Preview(None),
            distance: 0.4,
            grass: 0.0,
            small: 3.0,
            blur: false,
            particle_shadows: false,
            rt_shadows: false,
            water: false,
            occlusion: 1,
            sharpen: true,
        },
    ),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum GraphicsRow {
    Preset,
    Distance,
    Grass,
    Small,
    Blur,
    ParticleShadows,
    RtShadows,
    Water,
    Occlusion,
    Sharpen,
    Stats,
    Defaults,
}

impl Default for Graphics {
    fn default() -> Graphics {
        PRESETS[0].1
    }
}

/// Steps `v` by `step` (sign from the key) within `lo..=hi`, to tenths.
fn nudge(v: f32, key: u8, step: f32, lo: f32, hi: f32) -> f32 {
    let step = if key == KEY_LEFT { -step } else { step };
    ((v + step) * 10.0).round().clamp(lo * 10.0, hi * 10.0) / 10.0
}

impl Graphics {
    pub(super) fn path() -> Option<PathBuf> {
        beside("VICTORY_REGLAGES", "reglages.ini")
    }

    /// The file's settings, if there is one.
    pub(super) fn load() -> Option<Graphics> {
        Some(Graphics::parse(
            &std::fs::read_to_string(Graphics::path()?).ok()?,
        ))
    }

    fn parse(text: &str) -> Graphics {
        let mut g = Graphics::default();
        for (name, value) in lines(text) {
            let n = number(&value);
            match name.as_str() {
                "DISTANCE" => g.distance = n.map_or(g.distance, |v| v.clamp(0.2, 1.5)),
                "HERBE" => g.grass = n.map_or(g.grass, |v| v.clamp(0.0, 1.0)),
                "PETITS_OBJETS" => g.small = n.map_or(g.small, |v| v.clamp(0.5, 5.0)),
                "FLOU" => g.blur = yes(&value),
                "OMBRES_PARTICULES" => g.particle_shadows = yes(&value),
                "OMBRES_RT" => g.rt_shadows = yes(&value),
                "REFLETS_EAU" => g.water = value.eq_ignore_ascii_case("HAUT"),
                "OCCLUSION" => g.occlusion = n.map_or(g.occlusion, |v| (v as u32).clamp(1, 2)),
                "NETTETE" => g.sharpen = yes(&value),
                _ => tracing::warn!(name, "reglages.ini: unknown line"),
            }
        }
        g
    }

    fn text(&self) -> String {
        format!(
            "# Reglages graphiques de VIctory (menu F1, onglet GRAPHISMES).\n\
             # Pris en compte au prochain lancement. Effacer ce fichier remet ceux du jeu.\n\n\
             DISTANCE = {:.1}\nHERBE = {:.1}\nPETITS_OBJETS = {:.1}\nFLOU = {}\n\
             OMBRES_PARTICULES = {}\nOMBRES_RT = {}\nREFLETS_EAU = {}\nOCCLUSION = {}\nNETTETE = {}\n",
            self.distance,
            self.grass,
            self.small,
            yes_no(self.blur),
            yes_no(self.particle_shadows),
            yes_no(self.rt_shadows),
            if self.water { "HAUT" } else { "BAS" },
            self.occlusion,
            yes_no(self.sharpen),
        )
    }

    /// Writes `path` (none: nowhere); a note if that failed.
    pub(super) fn save_to(&self, path: Option<&Path>) -> Option<String> {
        let path = path?;
        // The settings before, one step back (reglages.ini.avant).
        if path.exists() {
            let _ = std::fs::copy(path, path.with_extension("ini.avant"));
        }
        match std::fs::write(path, self.text()) {
            Ok(()) => None,
            Err(e) => {
                tracing::warn!(?path, %e, "reglages.ini not written");
                Some("REGLAGES NON ENREGISTRES (FICHIER PROTEGE ?)".into())
            }
        }
    }

    fn preset(&self) -> &'static str {
        PRESETS
            .iter()
            .find(|(_, p)| p == self)
            .map_or("PERSO", |(name, _)| name)
    }

    pub(super) fn entries(&self) -> Vec<(Item, Option<GraphicsRow>)> {
        use GraphicsRow::*;
        let level = |v: f32| format!("{:.0} %", v * 100.0);
        vec![
            (Item::header("QUALITÉ"), None),
            (
                Item::setting(
                    "Préréglage",
                    match self.preview.0 {
                        Some(i) => format!("{}  ·  Entrée pour appliquer", pretty(PRESETS[i].0)),
                        None => pretty(self.preset()).to_owned(),
                    },
                ),
                Some(Preset),
            ),
            (Item::header("DISTANCE ET DÉTAILS"), None),
            (
                Item::setting("Distance des détails", level(self.distance)),
                Some(Distance),
            ),
            (Item::setting("Herbe", level(self.grass)), Some(Grass)),
            (
                Item::setting("Cacher les petits objets", format!("X {:.1}", self.small)),
                Some(Small),
            ),
            (Item::header("OMBRES ET EFFETS"), None),
            (
                Item::setting("Ombres dynamiques (RT)", shown(self.rt_shadows)),
                Some(RtShadows),
            ),
            (
                Item::setting("Ombres des particules", shown(self.particle_shadows)),
                Some(ParticleShadows),
            ),
            (
                Item::setting("Flou de mouvement", shown(self.blur)),
                Some(Blur),
            ),
            (
                Item::setting("Reflets de l'eau", if self.water { "Hauts" } else { "Bas" }),
                Some(Water),
            ),
            (
                Item::setting(
                    "Occlusion ambiante",
                    if self.occlusion >= 2 {
                        "Haute"
                    } else {
                        "Basse"
                    },
                ),
                Some(Occlusion),
            ),
            (Item::setting("Netteté", shown(self.sharpen)), Some(Sharpen)),
            (Item::header("AFFICHAGE"), None),
            (
                Item::setting(
                    "Afficher les FPS (F3)",
                    shown(victory_gpu::menu::stats_shown()),
                ),
                Some(Stats),
            ),
            (Item::button("Remettre les réglages du jeu"), Some(Defaults)),
            (
                Item::text("Changements appliqués au prochain lancement", ""),
                None,
            ),
        ]
    }

    pub(super) fn key(&mut self, row: GraphicsRow, key: u8) -> Reply {
        use GraphicsRow::*;
        if row != GraphicsRow::Preset {
            self.preview = Preview(None);
        }
        let before = *self;
        let toggle = matches!(key, ENTER | KEY_LEFT | KEY_RIGHT);
        let step = matches!(key, KEY_LEFT | KEY_RIGHT);
        match row {
            // Looking through the presets changes nothing; Enter applies the one shown.
            Preset if step => {
                let n = PRESETS.len();
                let from = self
                    .preview
                    .0
                    .or_else(|| PRESETS.iter().position(|(_, p)| p == self));
                let i = match (from, key) {
                    (Some(i), KEY_LEFT) => (i + n - 1) % n,
                    (Some(i), _) => (i + 1) % n,
                    (None, _) => 0,
                };
                self.preview = Preview(Some(i));
                return Reply::note(format!(
                    "Entrée pour appliquer le préréglage {}",
                    pretty(PRESETS[i].0)
                ));
            }
            Preset if key == ENTER => {
                let Some(i) = self.preview.0 else {
                    return Reply::note("Choisis un préréglage avec ← →, puis Entrée");
                };
                *self = PRESETS[i].1;
            }
            Distance if step => self.distance = nudge(self.distance, key, 0.1, 0.2, 1.5),
            Grass if step => self.grass = nudge(self.grass, key, 0.1, 0.0, 1.0),
            Small if step => self.small = nudge(self.small, key, 0.5, 0.5, 5.0),
            Blur if toggle => self.blur = !self.blur,
            ParticleShadows if toggle => self.particle_shadows = !self.particle_shadows,
            RtShadows if toggle => self.rt_shadows = !self.rt_shadows,
            Water if toggle => self.water = !self.water,
            Occlusion if toggle => self.occlusion = if self.occlusion >= 2 { 1 } else { 2 },
            Sharpen if toggle => self.sharpen = !self.sharpen,
            Stats if toggle => {
                let on = !victory_gpu::menu::stats_shown();
                victory_gpu::menu::ask_stats(on);
                // The panel takes it at the next picture; show it at once.
                victory_gpu::menu::set_stats_shown(on);
                return Reply::note(if on { "FPS affichés" } else { "FPS cachés" });
            }
            Defaults if key == ENTER => *self = Graphics::default(),
            _ => return Reply::default(),
        }
        if *self == before {
            return Reply::default();
        }
        Reply {
            note: Some("Enregistré : relance le jeu pour l'appliquer".into()),
            changed: true,
        }
    }

    /// The title's settings this sets in its `settings.xml`.
    fn title_settings(&self) -> Vec<(&'static str, String)> {
        let b = |v: bool| v.to_string();
        vec![
            ("lodScale", format!("{:.6}", self.distance)),
            ("grassLod", format!("{:.6}", self.grass)),
            ("smallCullThreshold", format!("{:.6}", self.small)),
            ("motionBlur", b(self.blur)),
            ("shadowParticleShadows", b(self.particle_shadows)),
            ("rtDynamicShadows", b(self.rt_shadows)),
            ("waterReflectionSSR", b(self.water)),
            (
                "waterReflectionResX",
                if self.water { "512" } else { "256" }.into(),
            ),
            (
                "waterReflectionResY",
                if self.water { "256" } else { "128" }.into(),
            ),
            ("ssroSampleCount", self.occlusion.to_string()),
            ("sharpenEnabled", b(self.sharpen)),
        ]
    }
}

/// `<name value="..." />` set to `value` in `xml`; whether it was there.
fn set_xml(xml: &mut String, name: &str, value: &str) -> bool {
    let open = format!("<{name} value=\"");
    let Some(at) = xml.find(&open) else {
        return false;
    };
    let start = at + open.len();
    let Some(len) = xml[start..].find('"') else {
        return false;
    };
    xml.replace_range(start..start + len, value);
    true
}

/// At start, before the title runs: the menu's graphics settings
/// (`reglages.ini`, if there is one) put into the title's `settings.xml`
/// under `storage`. Nothing without the file. (No output size: on Series S
/// the title's FSR4 tables take 1920x1080 only, 1280x720 stopped it, run
/// leger-L2 2026-11-27.)
pub fn apply_graphics(storage: &Path) {
    let Some(g) = Graphics::load() else { return };
    let path = storage.join("settings").join("settings.xml");
    match std::fs::read_to_string(&path) {
        Ok(mut xml) => {
            let missing: Vec<&str> = g
                .title_settings()
                .iter()
                .filter(|(n, v)| !set_xml(&mut xml, n, v))
                .map(|(n, _)| *n)
                .collect();
            match std::fs::write(&path, xml) {
                Ok(()) => println!(
                    "graphics       reglages.ini: preset {} put into settings.xml",
                    g.preset()
                ),
                Err(e) => eprintln!("VIctory: {} not written: {e}", path.display()),
            }
            if !missing.is_empty() {
                tracing::warn!(?missing, "settings.xml: settings not found");
            }
        }
        // The title writes its file at its first start; ours goes in at the next.
        Err(_) => {
            println!("graphics       no settings.xml yet: reglages.ini applies from the next start")
        }
    }
}

// ---------------------------------------------------------------- MULTIJOUEUR

/// A text being typed into a row.
#[derive(Clone, Debug, PartialEq)]
struct Typing {
    row: MultiRow,
    text: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum MultiRow {
    Name,
    Friend,
    Code,
    Port,
    Mic,
    TalkKey,
    Mute,
    MicVolume,
    FriendsVolume,
    Near,
    Far,
    Swap,
    Threshold,
    TestMic,
}

/// `multijoueur.ini`, as written by us and read by victory-voix.
#[derive(Clone, Debug, PartialEq)]
struct Voice {
    name: String,
    friend: String,
    code: String,
    port: u32,
    push_to_talk: bool,
    talk_key: u8,
    mute: bool,
    mic_volume: u32,
    friends_volume: u32,
    near: u32,
    far: u32,
    swap: bool,
    threshold: f32,
    test: bool,
}

impl Default for Voice {
    fn default() -> Voice {
        Voice {
            name: "JOUEUR".into(),
            friend: String::new(),
            code: String::new(),
            port: 47777,
            push_to_talk: false,
            talk_key: b'V',
            mute: false,
            mic_volume: 100,
            friends_volume: 100,
            near: 3,
            far: 40,
            swap: false,
            threshold: 0.012,
            test: false,
        }
    }
}

impl Voice {
    fn parse(text: &str) -> Voice {
        let mut v = Voice::default();
        for (name, value) in lines(text) {
            let n = || value.parse::<u32>().ok();
            match name.as_str() {
                "PSEUDO" => v.name = value,
                "AMI" => v.friend = value,
                "CODE" => v.code = value,
                "PORT" => v.port = n().unwrap_or(v.port),
                "MICRO" => v.push_to_talk = value.eq_ignore_ascii_case("appuyer"),
                "TOUCHE_PARLER" => v.talk_key = talk_key(&value).unwrap_or(v.talk_key),
                "MUET" => v.mute = yes(&value),
                "VOLUME_MICRO" => v.mic_volume = n().unwrap_or(v.mic_volume).min(200),
                "VOLUME_AMIS" => v.friends_volume = n().unwrap_or(v.friends_volume).min(200),
                "PRES" => v.near = n().unwrap_or(v.near),
                "LOIN" => v.far = n().unwrap_or(v.far),
                "INVERSER" => v.swap = yes(&value),
                "SEUIL" => v.threshold = number(&value).unwrap_or(v.threshold),
                "TEST_MICRO" => v.test = yes(&value),
                _ => {}
            }
        }
        v
    }

    fn text(&self) -> String {
        let b = |v: bool| if v { "1" } else { "0" };
        format!(
            "# Multijoueur et vocal de VIctory (menu F1, onglet MULTIJOUEUR). Relu chaque seconde par victory-voix.\n\
             pseudo={}\nami={}\ncode={}\nport={}\nmicro={}\ntouche_parler={}\nmuet={}\nvolume_micro={}\n\
             volume_amis={}\npres={}\nloin={}\ninverser={}\nseuil={}\ntest_micro={}\n",
            self.name,
            self.friend,
            self.code,
            self.port,
            if self.push_to_talk {
                "appuyer"
            } else {
                "ouvert"
            },
            talk_name(self.talk_key).unwrap_or("V".into()),
            b(self.mute),
            self.mic_volume,
            self.friends_volume,
            self.near,
            self.far,
            b(self.swap),
            self.threshold,
            b(self.test),
        )
    }
}

/// victory-voix's name of a push-to-talk key (its `config::touche`):
/// letters, digits, F1-F24, ESPACE, TAB, MAJ, CTRL, ALT, VERRMAJ, SOURIS4,
/// SOURIS5, SOURISMILIEU. `None`: a key it does not know.
fn talk_name(vk: u8) -> Option<String> {
    Some(match vk {
        b'A'..=b'Z' | b'0'..=b'9' => (vk as char).to_string(),
        0x70..=0x87 => format!("F{}", vk - 0x6F),
        0x20 => "ESPACE".into(),
        0x09 => "TAB".into(),
        0x10 | 0xA0 | 0xA1 => "MAJ".into(),
        0x11 | 0xA2 | 0xA3 => "CTRL".into(),
        0x12 | 0xA4 | 0xA5 => "ALT".into(),
        0x14 => "VERRMAJ".into(),
        0x05 => "SOURIS4".into(),
        0x06 => "SOURIS5".into(),
        0x04 => "SOURISMILIEU".into(),
        _ => return None,
    })
}

/// A push-to-talk key's name, victory-voix's or touches.ini's.
fn talk_key(name: &str) -> Option<u8> {
    let n = name.trim().to_ascii_uppercase();
    (0..=0xFF_u8)
        .find(|&vk| talk_name(vk).is_some_and(|t| t == n) && !matches!(vk, 0xA0..=0xA5))
        .or_else(|| super::keymap::parse_key(&n))
}

/// What victory-voix says (`multijoueur-etat.ini`).
#[derive(Clone, Debug, Default, PartialEq)]
struct VoiceState {
    state: String,
    error: String,
    friends: String,
    mic_level: u32,
    game_position: bool,
}

pub(super) struct Multi {
    voice: Voice,
    path: Option<PathBuf>,
    state_path: Option<PathBuf>,
    state: Option<VoiceState>,
    read: Option<Instant>,
    typing: Option<Typing>,
    /// Waiting for the push-to-talk key.
    waiting_key: bool,
}

impl Multi {
    /// Not tied to any file (tests).
    pub(super) fn detached() -> Multi {
        Multi {
            voice: Voice::default(),
            path: None,
            state_path: None,
            state: None,
            read: None,
            typing: None,
            waiting_key: false,
        }
    }

    pub(super) fn load() -> Multi {
        let path = beside("VICTORY_MULTI_INI", "multijoueur.ini");
        let state_path = path
            .as_ref()
            .map(|p| p.with_file_name("multijoueur-etat.ini"));
        let voice = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map_or_else(Voice::default, |t| Voice::parse(&t));
        Multi {
            voice,
            path,
            state_path,
            state: None,
            read: None,
            typing: None,
            waiting_key: false,
        }
    }

    fn save(&self) -> Option<String> {
        let path = self.path.as_ref()?;
        std::fs::write(path, self.voice.text()).err().map(|e| {
            tracing::warn!(?path, %e, "multijoueur.ini not written");
            "Réglages non enregistrés".to_owned()
        })
    }

    /// While the tab is open: victory-voix's state read again each second;
    /// whether it changed.
    pub(super) fn tick(&mut self) -> bool {
        if self
            .read
            .is_some_and(|r| r.elapsed() < Duration::from_secs(1))
        {
            return false;
        }
        self.read = Some(Instant::now());
        let state = self
            .state_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|t| {
                let mut s = VoiceState::default();
                for (k, v) in lines(&t) {
                    match k.as_str() {
                        "ÉTAT" => s.state = v,
                        "ERREUR" => s.error = v,
                        "AMIS" => s.friends = v,
                        "NIVEAU_MICRO" => s.mic_level = v.parse().unwrap_or(0),
                        "POSITION_JEU" => s.game_position = yes(&v),
                        _ => {}
                    }
                }
                s
            });
        let changed = state != self.state;
        self.state = state;
        changed
    }

    /// Keys go to the tab (a text or a key is being typed).
    pub(super) fn capturing(&self) -> bool {
        self.typing.is_some() || self.waiting_key
    }

    pub(super) fn entries(&self) -> Vec<(Item, Option<MultiRow>)> {
        use MultiRow::*;
        let v = &self.voice;
        let typed = |row: MultiRow, value: &str| match &self.typing {
            Some(t) if t.row == row => format!("{}_", t.text),
            _ if value.is_empty() => "-".to_owned(),
            _ => value.to_owned(),
        };
        let mut out = vec![(Item::header("ÉTAT"), None)];
        match &self.state {
            None => out.push((Item::text("Vocal", "Pas lancé"), None)),
            Some(s) => {
                let state = match s.state.as_str() {
                    "attente" => "En attente d'un ami".to_owned(),
                    "connecte" => "Connecté".to_owned(),
                    "erreur" => format!("Erreur : {}", s.error),
                    other => other.to_uppercase(),
                };
                out.push((Item::text("Vocal", state), None));
                for friend in s.friends.split(';').filter(|f| !f.is_empty()) {
                    let mut parts = friend.split(':');
                    let who = parts.next().unwrap_or_default().to_uppercase();
                    let rest: Vec<String> = parts.map(str::to_uppercase).collect();
                    out.push((Item::text(format!("  {who}"), rest.join("  ")), None));
                }
                let bar = "*".repeat((s.mic_level.min(100) / 10) as usize);
                out.push((
                    Item::text("Niveau du micro", format!("{bar:<10} {:>3} %", s.mic_level)),
                    None,
                ));
                out.push((
                    Item::text(
                        "Position dans le jeu",
                        if s.game_position {
                            "Reçue"
                        } else {
                            "Pas encore"
                        },
                    ),
                    None,
                ));
            }
        }
        out.extend([
            (Item::header("CONNEXION"), None),
            (Item::setting("Pseudo", typed(Name, &v.name)), Some(Name)),
            (
                Item::setting("Adresse de l'ami (IP)", typed(Friend, &v.friend)),
                Some(Friend),
            ),
            (
                Item::setting("Code de la partie", typed(Code, &v.code)),
                Some(Code),
            ),
            (
                Item::setting("Port (au démarrage)", typed(Port, &v.port.to_string())),
                Some(Port),
            ),
            (Item::header("Vocal"), None),
            (
                Item::setting(
                    "Micro",
                    if v.push_to_talk {
                        "Appuyer pour parler"
                    } else {
                        "Toujours ouvert"
                    },
                ),
                Some(Mic),
            ),
            (
                Item::setting(
                    "Touche pour parler",
                    if self.waiting_key {
                        "...".to_owned()
                    } else {
                        super::keymap::key_shown(v.talk_key)
                    },
                ),
                Some(TalkKey),
            ),
            (Item::setting("Couper mon micro", shown(v.mute)), Some(Mute)),
            (
                Item::setting("Volume du micro", format!("{} %", v.mic_volume)),
                Some(MicVolume),
            ),
            (
                Item::setting("Volume des amis", format!("{} %", v.friends_volume)),
                Some(FriendsVolume),
            ),
            (
                Item::setting("Sensibilité du micro", format!("{:.3}", v.threshold)),
                Some(Threshold),
            ),
            (Item::header("SON EN 3D"), None),
            (
                Item::setting("Volume plein jusqu'à", format!("{} m", v.near)),
                Some(Near),
            ),
            (
                Item::setting("Plus rien au-delà de", format!("{} m", v.far)),
                Some(Far),
            ),
            (
                Item::setting("Inverser gauche / droite", shown(v.swap)),
                Some(Swap),
            ),
            (
                Item::button(if v.test {
                    "Arrêter le test du micro"
                } else {
                    "Tester le micro (on s'entend)"
                }),
                Some(TestMic),
            ),
        ]);
        out
    }

    pub(super) fn key(&mut self, row: MultiRow, key: u8) -> Reply {
        use MultiRow::*;
        if self.waiting_key {
            self.waiting_key = false;
            if key == ESCAPE || key == 0x70 {
                return Reply::note("Annulé");
            }
            if talk_name(key).is_none() {
                return Reply::note(format!(
                    "{} : touche non prise par le vocal (lettre, chiffre, F1-F12, Espace, Maj, Ctrl, Alt)",
                    super::keymap::key_shown(key)
                ));
            }
            self.voice.talk_key = key;
            return self.saved(format!(
                "Touche pour parler : {}",
                super::keymap::key_shown(key)
            ));
        }
        if let Some(t) = &mut self.typing {
            match key {
                ENTER => {
                    let Some(t) = self.typing.take() else {
                        return Reply::default();
                    };
                    let v = &mut self.voice;
                    match t.row {
                        Name => v.name = t.text,
                        Friend => v.friend = t.text,
                        Code => v.code = t.text,
                        Port => match t.text.parse::<u32>() {
                            Ok(p @ 1..=65535) => v.port = p,
                            _ => return Reply::note("Port : un nombre de 1 à 65535"),
                        },
                        _ => {}
                    }
                    return self.saved("Enregistré".into());
                }
                ESCAPE => {
                    self.typing = None;
                    return Reply::note("Annulé");
                }
                BACK => {
                    t.text.pop();
                }
                _ => {
                    if let Some(c) = typed_char(key, t.row)
                        && t.text.len() < 40
                    {
                        t.text.push(c);
                    }
                }
            }
            return Reply::default();
        }
        let toggle = matches!(key, ENTER | KEY_LEFT | KEY_RIGHT);
        let step = matches!(key, KEY_LEFT | KEY_RIGHT);
        let sign: i64 = if key == KEY_LEFT { -1 } else { 1 };
        let add = |v: u32, by: i64, lo: u32, hi: u32| {
            (i64::from(v) + by * sign).clamp(i64::from(lo), i64::from(hi)) as u32
        };
        let v = &mut self.voice;
        match row {
            Name | Friend | Code | Port if key == ENTER => {
                let text = match row {
                    Name => v.name.clone(),
                    Friend => v.friend.clone(),
                    Code => v.code.clone(),
                    _ => v.port.to_string(),
                };
                self.typing = Some(Typing { row, text });
                return Reply::note("Tape puis Entrée (Échap annule, Retour efface)");
            }
            Friend | Code if matches!(key, DELETE | BACK) => {
                if row == Friend {
                    v.friend.clear()
                } else {
                    v.code.clear()
                }
            }
            Mic if toggle => v.push_to_talk = !v.push_to_talk,
            TalkKey if key == ENTER => {
                self.waiting_key = true;
                return Reply::note("Appuie sur la touche pour parler (Échap annule)");
            }
            Mute if toggle => v.mute = !v.mute,
            MicVolume if step => v.mic_volume = add(v.mic_volume, 10, 0, 200),
            FriendsVolume if step => v.friends_volume = add(v.friends_volume, 10, 0, 200),
            Near if step => v.near = add(v.near, 1, 0, v.far.saturating_sub(1)),
            Far if step => v.far = add(v.far, 5, v.near + 1, 500),
            Swap if toggle => v.swap = !v.swap,
            Threshold if step => {
                v.threshold = ((v.threshold + 0.002 * sign as f32) * 1000.0)
                    .round()
                    .clamp(1.0, 100.0)
                    / 1000.0
            }
            TestMic if key == ENTER => {
                v.test = !v.test;
                let note = if v.test {
                    "Test du micro : parle, tu dois t'entendre"
                } else {
                    "Test du micro arrêté"
                };
                return self.saved(note.into());
            }
            _ => return Reply::default(),
        }
        self.saved("Enregistré (s'applique tout de suite)".into())
    }

    fn saved(&self, note: String) -> Reply {
        Reply::note(self.save().unwrap_or(note))
    }
}

/// The character a key types into a text of `row`: letters, digits (top
/// row or keypad, whatever the layout), `.`, `:` and `-`.
fn typed_char(key: u8, row: MultiRow) -> Option<char> {
    let c = match key {
        b'A'..=b'Z' if row != MultiRow::Port => key as char,
        b'0'..=b'9' => key as char,
        0x60..=0x69 => (b'0' + key - 0x60) as char,
        0xBE | 0x6E | 0xBC if matches!(row, MultiRow::Friend) => '.',
        0xBF | 0xBA if matches!(row, MultiRow::Friend) => ':',
        0xBD | 0x6D if row != MultiRow::Port => '-',
        0x20 if row == MultiRow::Name => ' ',
        _ => return None,
    };
    Some(c)
}

// ---------------------------------------------------------------- DEVELOPPEUR

pub(super) fn dev_entries() -> Vec<(Item, Option<()>)> {
    vec![
        (Item::header("MODE DÉVELOPPEUR"), None),
        (Item::text("Bientôt ici :", ""), None),
        (Item::text("  Invincible", "À venir"), None),
        (Item::text("  Aller plus vite", "À venir"), None),
        (Item::text("  Se téléporter", "À venir"), None),
        (Item::text("  Changer l'heure et la météo", "À venir"), None),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphics_file_round_trips_and_presets_are_named() {
        for (name, p) in PRESETS {
            assert_eq!(Graphics::parse(&p.text()), p);
            assert_eq!(p.preset(), name);
        }
        let g = Graphics {
            grass: 0.5,
            ..Graphics::default()
        };
        assert_eq!(Graphics::parse(&g.text()), g);
        assert_eq!(g.preset(), "PERSO");
    }

    #[test]
    fn a_preset_is_only_applied_on_enter() {
        let mut g = Graphics {
            grass: 0.5,
            ..Graphics::default()
        };
        let mine = g;
        for _ in 0..5 {
            assert!(!g.key(GraphicsRow::Preset, KEY_RIGHT).changed);
        }
        assert_eq!(g, mine);
        let mut g = mine;
        assert!(
            !g.key(GraphicsRow::Preset, ENTER).changed,
            "Enter alone applies nothing"
        );
        assert_eq!(g, mine);
        g.key(GraphicsRow::Preset, KEY_RIGHT);
        assert!(g.key(GraphicsRow::Preset, ENTER).changed);
        assert_eq!(g, PRESETS[0].1);
    }

    #[test]
    fn the_title_settings_are_put_in_place() {
        let mut xml =
            "<a>\r\n  <lodScale value=\"1.000000\" />\r\n  <motionBlur value=\"true\" />\r\n</a>"
                .to_owned();
        assert!(set_xml(&mut xml, "lodScale", "0.600000"));
        assert!(set_xml(&mut xml, "motionBlur", "false"));
        assert!(!set_xml(&mut xml, "grassLod", "0.3"));
        assert_eq!(
            xml,
            "<a>\r\n  <lodScale value=\"0.600000\" />\r\n  <motionBlur value=\"false\" />\r\n</a>"
        );
    }

    #[test]
    fn talk_keys_use_victory_voix_names() {
        assert_eq!(talk_name(0xA0).as_deref(), Some("MAJ"));
        assert_eq!(talk_key("MAJ"), Some(0x10));
        assert_eq!(talk_key("espace"), Some(0x20));
        assert_eq!(talk_key("F5"), Some(0x74));
        assert_eq!(talk_name(0x2E), None);
    }

    #[test]
    fn voice_file_round_trips() {
        let v = Voice {
            friend: "192.168.1.20:47777".into(),
            push_to_talk: true,
            mic_volume: 150,
            ..Voice::default()
        };
        assert_eq!(Voice::parse(&v.text()), v);
    }

    #[test]
    fn an_address_is_typed() {
        let mut m = Multi::detached();
        m.key(MultiRow::Friend, ENTER);
        for k in [b'1', 0xBE, 0x62, BACK, b'2', 0xBF, b'4'] {
            m.key(MultiRow::Friend, k);
        }
        m.key(MultiRow::Friend, ENTER);
        assert_eq!(m.voice.friend, "1.2:4");
    }
}
