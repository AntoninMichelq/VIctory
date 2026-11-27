//! Keyboard and mouse as the stand-in pad, PC-game style. Keys and mouse
//! buttons are bound to pad buttons, triggers and stick directions; moving
//! the mouse pushes the right stick (the camera), with the cursor held in the
//! window. The bindings live in a text file (`touches.ini` beside
//! victory.exe, or the path in `VICTORY_TOUCHES`), written on first start and
//! on every change, and a menu over the picture changes them:
//!
//! - F1 opens and closes the menu (Escape closes it too); Tab and Page
//!   Up/Down change tabs (GRAPHISMES, CLAVIER / SOURIS, MULTIJOUEUR,
//!   DEVELOPPEUR: the other tabs are in `tabs`); arrows choose, Enter changes
//!   a binding (then press the new key or mouse button), Right adds a second
//!   one, Delete or Backspace clears it, Left/Right change numbers.
//! - F2 lets go of the mouse (and takes it again).
//!
//! Letters of the default bindings are by position on the keyboard (W A S D
//! on a QWERTY keyboard are Z Q S D on an AZERTY one). The menu's own keys
//! never change, so a bad binding can always be undone.

use std::collections::VecDeque;
use std::path::PathBuf;

use victory_gpu::menu::{self, Item, Page};

use super::tabs::{self, Graphics, GraphicsRow, Multi, MultiRow};

use super::{
    A, B, DOWN, KeyState, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

/// The mouse since the last poll.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Mouse {
    /// Pixels moved (right, down) while held in the window.
    pub dx: i32,
    pub dy: i32,
    /// Virtual keys of the buttons held (0x01, 0x02, 0x04, 0x05, 0x06).
    pub buttons: Vec<u8>,
    /// The cursor is held in the window (`dx`/`dy` mean something).
    pub held: bool,
    /// Keyboard layout (HKL) of the title's window in front; 0: none.
    pub layout: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Target {
    Button(u32),
    LeftTrigger,
    RightTrigger,
    /// A stick direction: left stick or right, x or y, sign.
    Stick {
        right: bool,
        y: bool,
        sign: f32,
    },
}

/// A default binding: a virtual key, or a letter by its place on a QWERTY
/// keyboard.
#[derive(Clone, Copy)]
enum Bind {
    Vk(u8),
    At(u8),
}
use Bind::{At, Vk};

struct Action {
    /// Name in the file.
    id: &'static str,
    /// Name in the menu.
    label: &'static str,
    target: Target,
    default: &'static [Bind],
}

const fn stick(right: bool, y: bool, sign: f32) -> Target {
    Target::Stick { right, y, sign }
}

const ACTIONS: [Action; 24] = [
    Action {
        id: "AVANCER",
        label: "Avancer",
        target: stick(false, true, 1.0),
        default: &[At(b'W')],
    },
    Action {
        id: "RECULER",
        label: "Reculer",
        target: stick(false, true, -1.0),
        default: &[At(b'S')],
    },
    Action {
        id: "GAUCHE",
        label: "Aller à gauche",
        target: stick(false, false, -1.0),
        default: &[At(b'A')],
    },
    Action {
        id: "DROITE",
        label: "Aller à droite",
        target: stick(false, false, 1.0),
        default: &[At(b'D')],
    },
    Action {
        id: "A",
        label: "Bouton A",
        target: Target::Button(A),
        default: &[Vk(0x0D), Vk(0xA0)],
    },
    Action {
        id: "B",
        label: "Bouton B",
        target: Target::Button(B),
        default: &[Vk(0x08), At(b'R')],
    },
    Action {
        id: "X",
        label: "Bouton X",
        target: Target::Button(X),
        default: &[At(b'X'), Vk(0x20)],
    },
    Action {
        id: "Y",
        label: "Bouton Y",
        target: Target::Button(Y),
        default: &[At(b'Y')],
    },
    Action {
        id: "RT",
        label: "RT (gâchette droite)",
        target: Target::RightTrigger,
        default: &[Vk(0x01), At(b'C')],
    },
    Action {
        id: "LT",
        label: "LT (gâchette gauche)",
        target: Target::LeftTrigger,
        default: &[Vk(0x02), At(b'Z')],
    },
    Action {
        id: "RB",
        label: "RB",
        target: Target::Button(RIGHT_SHOULDER),
        default: &[At(b'E')],
    },
    Action {
        id: "LB",
        label: "LB",
        target: Target::Button(LEFT_SHOULDER),
        default: &[At(b'Q')],
    },
    Action {
        id: "L3",
        label: "Clic stick gauche (L3)",
        target: Target::Button(LEFT_STICK),
        default: &[At(b'F'), Vk(0xA2)],
    },
    Action {
        id: "R3",
        label: "Clic stick droit (R3)",
        target: Target::Button(RIGHT_STICK),
        default: &[At(b'H'), Vk(0x04)],
    },
    Action {
        id: "MENU",
        label: "Menu (Start)",
        target: Target::Button(MENU),
        default: &[Vk(0x1B)],
    },
    Action {
        id: "VUE",
        label: "Vue (Select)",
        target: Target::Button(VIEW),
        default: &[Vk(0x09)],
    },
    Action {
        id: "CROIX_HAUT",
        label: "Croix haut",
        target: Target::Button(UP),
        default: &[Vk(0x26)],
    },
    Action {
        id: "CROIX_BAS",
        label: "Croix bas",
        target: Target::Button(DOWN),
        default: &[Vk(0x28)],
    },
    Action {
        id: "CROIX_GAUCHE",
        label: "Croix gauche",
        target: Target::Button(LEFT),
        default: &[Vk(0x25)],
    },
    Action {
        id: "CROIX_DROITE",
        label: "Croix droite",
        target: Target::Button(RIGHT),
        default: &[Vk(0x27)],
    },
    Action {
        id: "CAMERA_HAUT",
        label: "Caméra haut",
        target: stick(true, true, 1.0),
        default: &[At(b'I')],
    },
    Action {
        id: "CAMERA_BAS",
        label: "Caméra bas",
        target: stick(true, true, -1.0),
        default: &[At(b'K')],
    },
    Action {
        id: "CAMERA_GAUCHE",
        label: "Caméra gauche",
        target: stick(true, false, -1.0),
        default: &[At(b'J')],
    },
    Action {
        id: "CAMERA_DROITE",
        label: "Caméra droite",
        target: stick(true, false, 1.0),
        default: &[At(b'L')],
    },
];

/// Keys at most per action.
const MOST: usize = 3;
/// Mouse moves averaged over this many polls (8 ms each).
const WINDOW: usize = 6;
/// Stick per pixel moved in one poll, at sensitivity 1.
const PER_PIXEL: f32 = 0.06;

const F1: u8 = 0x70;
const F2: u8 = 0x71;
const ENTER: u8 = 0x0D;
const BACK: u8 = 0x08;
const DELETE: u8 = 0x2E;
const KEY_UP: u8 = 0x26;
const KEY_DOWN: u8 = 0x28;
const KEY_LEFT: u8 = 0x25;
const KEY_RIGHT: u8 = 0x27;
const TAB: u8 = 0x09;
const PAGE_UP: u8 = 0x21;
const PAGE_DOWN: u8 = 0x22;
const ESCAPE: u8 = 0x1B;

/// The menu's tabs.
const TABS: [&str; 4] = [
    "Graphismes",
    "Clavier / souris",
    "Multijoueur",
    "Développeur",
];
const GRAPHICS_TAB: usize = 0;
const KEYS_TAB: usize = 1;
const MULTI_TAB: usize = 2;

/// A row the cursor can be on, in any tab.
#[derive(Clone, Copy, PartialEq)]
enum Sel {
    Keys(Row),
    Graphics(GraphicsRow),
    Multi(MultiRow),
}

/// Menu rows after the actions.
#[derive(Clone, Copy, PartialEq)]
enum Row {
    Action(usize),
    Camera,
    Sensitivity,
    Invert,
    DeadZone,
    Defaults,
    Close,
}

fn rows() -> impl Iterator<Item = Row> {
    (0..ACTIONS.len()).map(Row::Action).chain([
        Row::Camera,
        Row::Sensitivity,
        Row::Invert,
        Row::DeadZone,
        Row::Defaults,
        Row::Close,
    ])
}

#[derive(Clone, Copy, PartialEq)]
enum Wait {
    Replace,
    Add,
}

struct Menu {
    tab: usize,
    /// The cursor of each tab: an index in its rows (headers included).
    cursor: [usize; TABS.len()],
    wait: Option<Wait>,
    note: String,
}

#[derive(Clone, Debug, PartialEq)]
struct Settings {
    keys: Vec<Vec<u8>>,
    camera: bool,
    sensitivity: f32,
    invert: bool,
    dead_zone: f32,
    /// The keyboard layout (HKL) the default letters were placed for; 0:
    /// not known yet.
    layout: usize,
}

impl Settings {
    fn defaults() -> Settings {
        Settings::defaults_for(0)
    }

    /// Defaults with letters placed for that keyboard layout (0: ours).
    fn defaults_for(layout: usize) -> Settings {
        Settings {
            keys: ACTIONS
                .iter()
                .map(|a| {
                    a.default
                        .iter()
                        .map(|d| match *d {
                            Vk(vk) => vk,
                            At(letter) => at(letter, layout),
                        })
                        .collect()
                })
                .collect(),
            camera: true,
            sensitivity: 1.0,
            invert: false,
            dead_zone: 0.2,
            layout,
        }
    }

    /// The file's settings; `layout` stays 0 without a `CLAVIER` line
    /// (files of the first version, whose letters may be placed for the
    /// wrong keyboard).
    fn parse(text: &str) -> Settings {
        let mut s = Settings::defaults();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }
            let Some((name, value)) = line.split_once('=') else {
                continue;
            };
            let name = name.trim().to_ascii_uppercase();
            let value = value.trim();
            let yes = || {
                matches!(
                    value.to_ascii_uppercase().as_str(),
                    "OUI" | "1" | "ON" | "YES" | "VRAI"
                )
            };
            let number = || {
                value
                    .replace(',', ".")
                    .parse::<f32>()
                    .ok()
                    .filter(|v| v.is_finite())
            };
            match name.as_str() {
                "CLAVIER" => s.layout = usize::from_str_radix(value, 16).unwrap_or(0),
                "SOURIS_CAMERA" => s.camera = yes(),
                "SOURIS_INVERSER" => s.invert = yes(),
                "SOURIS_SENSIBILITE" => {
                    if let Some(v) = number() {
                        s.sensitivity = v.clamp(0.1, 10.0);
                    }
                }
                "SOURIS_ZONE_MORTE" => {
                    if let Some(v) = number() {
                        s.dead_zone = v.clamp(0.0, 0.9);
                    }
                }
                _ => {
                    if let Some(i) = ACTIONS.iter().position(|a| a.id == name) {
                        let keys: Vec<u8> = value.split([',', ' ']).filter_map(parse_key).collect();
                        s.keys[i] = keys.into_iter().take(MOST).collect();
                    } else {
                        tracing::warn!(name, "touches.ini: unknown line");
                    }
                }
            }
        }
        s
    }

    fn text(&self) -> String {
        let yes = |b: bool| if b { "OUI" } else { "NON" };
        let mut out = String::from(
            "# Touches de VIctory. Menu en jeu : F1. Lacher / reprendre la souris : F2.\n\
             # Une ligne par action : ACTION = TOUCHE, TOUCHE (3 au plus, vide = aucune).\n\
             # Noms : lettres, chiffres, ENTREE, ESPACE, RETOUR, TAB, ECHAP, MAJ_G, MAJ_D,\n\
             # CTRL_G, CTRL_D, ALT_G, FLECHE_HAUT/BAS/GAUCHE/DROITE, F3..F12, PAVE_0..9,\n\
             # CLIC_GAUCHE, CLIC_DROIT, CLIC_MILIEU, SOURIS_4, SOURIS_5, ou VK_xx (hexa).\n\
             # Effacer ce fichier remet tout par defaut.\n\n",
        );
        for (action, keys) in ACTIONS.iter().zip(&self.keys) {
            let names: Vec<String> = keys.iter().map(|&k| key_name(k)).collect();
            out += &format!("{} = {}\n", action.id, names.join(", "));
        }
        out += &format!(
            "\nSOURIS_CAMERA = {}\nSOURIS_SENSIBILITE = {:.1}\nSOURIS_INVERSER = {}\nSOURIS_ZONE_MORTE = {:.2}\n\
             \n# Clavier pour lequel les lettres par defaut ont ete placees (ne pas toucher).\nCLAVIER = {:X}\n",
            yes(self.camera),
            self.sensitivity,
            yes(self.invert),
            self.dead_zone,
            self.layout
        );
        out
    }
}

/// Keyboard and mouse to pad, and the key menu.
pub(super) struct Keymap {
    settings: Settings,
    path: Option<PathBuf>,
    menu: Option<Menu>,
    /// The tab the menu opens on (the one shown last).
    last_tab: usize,
    graphics: Graphics,
    graphics_path: Option<PathBuf>,
    multi: Multi,
    /// F2: the mouse is let go.
    free: bool,
    before: Vec<u8>,
    moves: VecDeque<(i32, i32)>,
    /// The keyboard layout seen last (0: none yet).
    layout: usize,
}

impl Default for Keymap {
    /// The default bindings, not saved anywhere (tests).
    fn default() -> Keymap {
        Keymap {
            settings: Settings::defaults(),
            path: None,
            menu: None,
            last_tab: GRAPHICS_TAB,
            graphics: Graphics::default(),
            graphics_path: None,
            multi: Multi::detached(),
            free: false,
            before: Vec::new(),
            moves: VecDeque::with_capacity(WINDOW),
            layout: 0,
        }
    }
}

impl Keymap {
    /// The bindings of the file, written with the defaults when missing.
    pub(super) fn load() -> Keymap {
        let path = std::env::var_os("VICTORY_TOUCHES")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::current_exe()
                    .ok()
                    .and_then(|exe| Some(exe.parent()?.join("touches.ini")))
            });
        let mut map = Keymap {
            path,
            graphics: Graphics::load().unwrap_or_default(),
            graphics_path: Graphics::path(),
            multi: Multi::load(),
            ..Keymap::default()
        };
        match map.path.as_ref().map(std::fs::read_to_string) {
            Some(Ok(text)) => map.settings = Settings::parse(&text),
            _ => map.save(),
        }
        tracing::info!(path = ?map.path, "key bindings (F1: key menu, F2: free the mouse)");
        map
    }

    fn save(&self) {
        if let Some(path) = &self.path
            && let Err(e) = std::fs::write(path, self.settings.text())
        {
            tracing::warn!(?path, %e, "touches.ini not written");
        }
    }

    /// The player types on `layout` now (the title window's; it changes
    /// with Alt+Shift): each action still on its default keys gets the
    /// defaults placed for it, so letters stay where they are on the
    /// keyboard (the poll thread's own layout may be another one). Keys the
    /// player chose stay as chosen.
    fn place(&mut self, layout: usize) {
        if self.settings.layout == layout {
            return;
        }
        let before = Settings::defaults_for(self.settings.layout);
        let now = Settings::defaults_for(layout);
        let first = self.settings.layout == 0;
        for ((keys, old), new) in self.settings.keys.iter_mut().zip(before.keys).zip(now.keys) {
            // A file without a layout (the first version's): all defaults again.
            if first || *keys == old {
                *keys = new;
            }
        }
        self.settings.layout = layout;
        tracing::info!(
            layout = format_args!("{layout:X}"),
            "default keys placed for the keyboard"
        );
        self.save();
    }

    /// Whether the mouse should be held in the window now.
    pub(super) fn wants_mouse(&self) -> bool {
        self.settings.camera && !self.free && self.menu.is_none()
    }

    /// One poll: the stand-in pad, and whether the title may see the keys
    /// (not while the menu is open).
    pub(super) fn step(&mut self, keys: &[KeyState], mouse: &Mouse) -> (PadState, bool) {
        if mouse.layout != 0 && self.layout != mouse.layout {
            self.layout = mouse.layout;
            self.place(mouse.layout);
        }
        let down: Vec<u8> = keys
            .iter()
            .map(|k| k.virtual_key)
            .chain(mouse.buttons.iter().copied())
            .collect();
        let pressed: Vec<u8> = down
            .iter()
            .copied()
            .filter(|k| !self.before.contains(k))
            .collect();
        self.before = down.clone();
        if self.moves.len() == WINDOW {
            self.moves.pop_front();
        }
        self.moves.push_back(if mouse.held {
            (mouse.dx, mouse.dy)
        } else {
            (0, 0)
        });

        let waiting = self
            .menu
            .as_ref()
            .is_some_and(|m| m.wait.is_some() || (m.tab == MULTI_TAB && self.multi.capturing()));
        if pressed.contains(&F1) && !waiting {
            self.menu = match self.menu {
                Some(_) => None,
                None => {
                    let mut m = Menu {
                        tab: self.last_tab,
                        cursor: [0; TABS.len()],
                        wait: None,
                        note: String::new(),
                    };
                    for tab in 0..TABS.len() {
                        m.tab = tab;
                        m.cursor[tab] = self.selectable(&m, 0, 1);
                    }
                    m.tab = self.last_tab;
                    Some(m)
                }
            };
            self.show();
        } else if self.menu.is_some() {
            let mut again = !pressed.is_empty();
            if again {
                self.menu_keys(&pressed);
            }
            // victory-voix's state, each second while its tab is open.
            if self.menu.as_ref().is_some_and(|m| m.tab == MULTI_TAB) && self.multi.tick() {
                again = true;
            }
            if again {
                self.show();
            }
        } else if pressed.contains(&F2) {
            self.free = !self.free;
        }
        if self.menu.is_some() {
            return (PadState::default(), false);
        }
        (self.pad(&down), true)
    }

    fn pad(&self, down: &[u8]) -> PadState {
        let s = &self.settings;
        let mut p = PadState::default();
        let mut sticks = [0.0_f32; 4];
        for (action, keys) in ACTIONS.iter().zip(&s.keys) {
            if !keys.iter().any(|k| down.contains(k)) {
                continue;
            }
            match action.target {
                Target::Button(bit) => p.buttons |= bit,
                Target::LeftTrigger => p.left_trigger = 1.0,
                Target::RightTrigger => p.right_trigger = 1.0,
                Target::Stick { right, y, sign } => {
                    sticks[usize::from(right) * 2 + usize::from(y)] += sign
                }
            }
        }
        let clamp = |v: f32| v.clamp(-1.0, 1.0);
        [p.left_x, p.left_y, p.right_x, p.right_y] = sticks.map(clamp);
        if s.camera {
            let (sx, sy) = self
                .moves
                .iter()
                .fold((0, 0), |(x, y), &(dx, dy)| (x + dx, y + dy));
            let n = self.moves.len().max(1) as f32;
            let k = PER_PIXEL * s.sensitivity / n;
            let (mut x, mut y) = (sx as f32 * k, -(sy as f32) * k);
            if s.invert {
                y = -y;
            }
            let length = x.hypot(y);
            if length > 1e-3 {
                // Past the title's dead zone at once, then in proportion.
                let to = (s.dead_zone + (1.0 - s.dead_zone) * length).min(1.0);
                (x, y) = (x * to / length, y * to / length);
                if x.hypot(y) > p.right_x.hypot(p.right_y) {
                    (p.right_x, p.right_y) = (x, y);
                }
            }
        }
        p
    }

    /// The rows of the menu's tab, each with what the cursor selects there
    /// (`None`: a title or a line to read).
    fn entries(&self, m: &Menu) -> Vec<(Item, Option<Sel>)> {
        match m.tab {
            GRAPHICS_TAB => self
                .graphics
                .entries()
                .into_iter()
                .map(|(i, r)| (i, r.map(Sel::Graphics)))
                .collect(),
            KEYS_TAB => self.keys_entries(m),
            MULTI_TAB => self
                .multi
                .entries()
                .into_iter()
                .map(|(i, r)| (i, r.map(Sel::Multi)))
                .collect(),
            _ => tabs::dev_entries()
                .into_iter()
                .map(|(i, _)| (i, None))
                .collect(),
        }
    }

    /// The first row from `from` (going `by` rows, wrapping) the cursor can
    /// be on; `from` if none.
    fn selectable(&self, m: &Menu, from: usize, by: isize) -> usize {
        let entries = self.entries(m);
        let n = entries.len() as isize;
        let mut i = from as isize;
        for _ in 0..n {
            if entries.get(i as usize).is_some_and(|(_, s)| s.is_some()) {
                return i as usize;
            }
            i = (i + by).rem_euclid(n.max(1));
        }
        from
    }

    fn menu_keys(&mut self, pressed: &[u8]) {
        for &key in pressed {
            let Some(m) = self.menu.as_ref() else { return };
            let entries = self.entries(m);
            let at = m.cursor[m.tab].min(entries.len().saturating_sub(1));
            let here = entries.get(at).and_then(|(_, s)| *s);
            let tab = m.tab;
            if m.wait.is_some() {
                if let Some(Sel::Keys(row)) = here {
                    self.keys_tab_keys(row, &[key]);
                }
                continue;
            }
            if tab == MULTI_TAB && self.multi.capturing() {
                if let Some(Sel::Multi(row)) = here {
                    let reply = self.multi.key(row, key);
                    self.take(reply);
                }
                continue;
            }
            let n = entries.len();
            match key {
                TAB | PAGE_DOWN | PAGE_UP => {
                    let t = if key == PAGE_UP {
                        (tab + TABS.len() - 1) % TABS.len()
                    } else {
                        (tab + 1) % TABS.len()
                    };
                    if let Some(m) = self.menu.as_mut() {
                        m.tab = t;
                        m.note.clear();
                    }
                    self.last_tab = t;
                }
                ESCAPE => {
                    self.menu = None;
                    return;
                }
                KEY_UP | KEY_DOWN => {
                    let by = if key == KEY_UP { -1 } else { 1 };
                    let next = (at as isize + by).rem_euclid(n.max(1) as isize) as usize;
                    let Some(m) = self.menu.as_ref() else { return };
                    let next = self.selectable(m, next, by);
                    if let Some(m) = self.menu.as_mut() {
                        m.cursor[tab] = next;
                    }
                }
                _ => match here {
                    Some(Sel::Keys(row)) => {
                        self.keys_tab_keys(row, &[key]);
                        if self.menu.is_none() {
                            return;
                        }
                    }
                    Some(Sel::Graphics(row)) => {
                        let reply = self.graphics.key(row, key);
                        if reply.changed {
                            tracing::info!(?row, key, path = ?self.graphics_path, "menu: graphics settings written");
                        }
                        if reply.changed
                            && let Some(failed) =
                                self.graphics.save_to(self.graphics_path.as_deref())
                        {
                            self.take(tabs::Reply {
                                note: Some(failed),
                                changed: false,
                            });
                        } else {
                            self.take(reply);
                        }
                    }
                    Some(Sel::Multi(row)) => {
                        let reply = self.multi.key(row, key);
                        self.take(reply);
                    }
                    None => {}
                },
            }
        }
    }

    /// A tab's note shown.
    fn take(&mut self, reply: tabs::Reply) {
        if let (Some(m), Some(note)) = (self.menu.as_mut(), reply.note) {
            m.note = note;
        }
    }

    /// The CLAVIER / SOURIS tab's rows.
    fn keys_entries(&self, m: &Menu) -> Vec<(Item, Option<Sel>)> {
        let s = &self.settings;
        let yes = |b: bool| if b { "Oui" } else { "Non" };
        let mut out = vec![(Item::header("ACTIONS"), None)];
        let at = m.cursor[KEYS_TAB];
        for r in rows() {
            let i = out.len();
            let item = match r {
                Row::Action(a) => {
                    let names: Vec<String> = s.keys[a].iter().map(|&k| key_shown(k)).collect();
                    let names = if at == i && m.wait.is_some() {
                        "...".to_owned()
                    } else if names.is_empty() {
                        "-".to_owned()
                    } else {
                        names.join(", ")
                    };
                    Item::setting(ACTIONS[a].label, names)
                }
                Row::Camera => {
                    out.push((Item::header("SOURIS"), None));
                    Item::setting("La souris tourne la caméra", yes(s.camera))
                }
                Row::Sensitivity => Item::setting("Sensibilité", format!("{:.1}", s.sensitivity)),
                Row::Invert => Item::setting("Inverser haut / bas", yes(s.invert)),
                Row::DeadZone => Item::setting("Zone morte", format!("{:.2}", s.dead_zone)),
                Row::Defaults => {
                    out.push((Item::header("AUTRES"), None));
                    Item::button("Remettre les touches par défaut")
                }
                Row::Close => Item::button("Fermer le menu"),
            };
            out.push((item, Some(Sel::Keys(r))));
        }
        out
    }

    /// Keys on the CLAVIER / SOURIS tab's row `here`.
    fn keys_tab_keys(&mut self, here: Row, pressed: &[u8]) {
        let Some(menu) = self.menu.as_mut() else {
            return;
        };
        let s = &mut self.settings;
        let mut changed = false;
        if let Some(wait) = menu.wait {
            let Some(&key) = pressed.first() else { return };
            menu.wait = None;
            let Row::Action(i) = here else {
                return;
            };
            if key == F1 || key == ESCAPE {
                menu.note = "Annulé".into();
                return;
            }
            match wait {
                Wait::Replace => s.keys[i] = vec![key],
                Wait::Add if s.keys[i].contains(&key) => {}
                Wait::Add => {
                    if s.keys[i].len() == MOST {
                        s.keys[i].remove(0);
                    }
                    s.keys[i].push(key);
                }
            }
            let others: Vec<&str> = ACTIONS
                .iter()
                .zip(&s.keys)
                .enumerate()
                .filter(|&(j, (_, keys))| j != i && keys.contains(&key))
                .map(|(_, (a, _))| a.label)
                .collect();
            menu.note = if others.is_empty() {
                format!("{} = {}", ACTIONS[i].label, key_shown(key))
            } else {
                format!(
                    "Attention : {} sert aussi à {}",
                    key_shown(key),
                    others.join(", ")
                )
            };
            self.save();
            return;
        }
        for &key in pressed {
            match (key, here) {
                (ENTER, Row::Action(_)) => {
                    menu.wait = Some(Wait::Replace);
                    menu.note = "Appuie sur la nouvelle touche (Échap annule)".into();
                }
                (KEY_RIGHT, Row::Action(_)) => {
                    menu.wait = Some(Wait::Add);
                    menu.note = "Appuie sur la touche à ajouter (Échap annule)".into();
                }
                (DELETE | BACK, Row::Action(i)) => {
                    s.keys[i].clear();
                    menu.note = format!("{} : plus aucune touche", ACTIONS[i].label);
                    changed = true;
                }
                (ENTER | KEY_LEFT | KEY_RIGHT, Row::Camera) => {
                    s.camera = !s.camera;
                    changed = true;
                }
                (ENTER | KEY_LEFT | KEY_RIGHT, Row::Invert) => {
                    s.invert = !s.invert;
                    changed = true;
                }
                (KEY_LEFT | KEY_RIGHT, Row::Sensitivity) => {
                    let step = if key == KEY_RIGHT { 0.1 } else { -0.1 };
                    s.sensitivity =
                        ((s.sensitivity + step) * 10.0).round().clamp(1.0, 100.0) / 10.0;
                    changed = true;
                }
                (KEY_LEFT | KEY_RIGHT, Row::DeadZone) => {
                    let step = if key == KEY_RIGHT { 0.05 } else { -0.05 };
                    s.dead_zone = ((s.dead_zone + step) * 20.0).round().clamp(0.0, 18.0) / 20.0;
                    changed = true;
                }
                (ENTER, Row::Defaults) => {
                    *s = Settings::defaults_for(self.layout);
                    menu.note = "Touches par défaut remises".into();
                    changed = true;
                }
                (ENTER, Row::Close) => {
                    self.menu = None;
                    break;
                }
                _ => {}
            }
            if menu.wait.is_some() {
                break;
            }
        }
        if changed {
            self.save();
        }
    }

    /// Hands the menu's page to the picture (or takes it away).
    fn show(&self) {
        let Some(m) = &self.menu else {
            menu::show(None);
            return;
        };
        let entries = self.entries(m);
        let help = match m.tab {
            KEYS_TAB => "Entrée changer   → ajouter   Suppr effacer   Tab onglet   Échap fermer",
            GRAPHICS_TAB | MULTI_TAB => {
                "↑↓ choisir   ←→ ou Entrée changer   Tab onglet   Échap fermer"
            }
            _ => "Tab onglet suivant   Échap fermer",
        };
        menu::show(Some(Page {
            tabs: TABS.iter().map(|t| (*t).to_owned()).collect(),
            tab: m.tab,
            selected: m.cursor[m.tab],
            items: entries.into_iter().map(|(i, _)| i).collect(),
            help: help.into(),
            note: m.note.clone(),
        }));
    }
}

/// The virtual key of a letter's place on a QWERTY keyboard, in the host's
/// layout (QWERTY's W is Z on AZERTY).
pub(super) fn at(letter: u8, layout: usize) -> u8 {
    #[cfg(windows)]
    {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
            HKL, MAPVK_VSC_TO_VK, MapVirtualKeyExW,
        };
        const SCAN: [u8; 26] = [
            0x1E, 0x30, 0x2E, 0x20, 0x12, 0x21, 0x22, 0x23, 0x17, 0x24, 0x25, 0x26, 0x32, 0x31,
            0x18, 0x19, 0x10, 0x13, 0x1F, 0x14, 0x16, 0x2F, 0x11, 0x2D, 0x15, 0x2C,
        ];
        if letter.is_ascii_uppercase() {
            // SAFETY: a plain query.
            let vk = unsafe {
                MapVirtualKeyExW(
                    u32::from(SCAN[usize::from(letter - b'A')]),
                    MAPVK_VSC_TO_VK,
                    layout as HKL,
                )
            };
            if let Ok(vk @ 1..=0xFE) = u8::try_from(vk) {
                return vk;
            }
        }
    }
    let _ = layout;
    letter
}

const NAMES: [(u8, &str); 35] = [
    (0x01, "CLIC_GAUCHE"),
    (0x02, "CLIC_DROIT"),
    (0x04, "CLIC_MILIEU"),
    (0x05, "SOURIS_4"),
    (0x06, "SOURIS_5"),
    (0x08, "RETOUR"),
    (0x09, "TAB"),
    (0x0D, "ENTREE"),
    (0x13, "PAUSE"),
    (0x14, "VERR_MAJ"),
    (0x1B, "ECHAP"),
    (0x20, "ESPACE"),
    (0x21, "PAGE_HAUT"),
    (0x22, "PAGE_BAS"),
    (0x23, "FIN"),
    (0x24, "DEBUT"),
    (0x25, "FLECHE_GAUCHE"),
    (0x26, "FLECHE_HAUT"),
    (0x27, "FLECHE_DROITE"),
    (0x28, "FLECHE_BAS"),
    (0x2D, "INSER"),
    (0x2E, "SUPPR"),
    (0x6A, "PAVE_FOIS"),
    (0x6B, "PAVE_PLUS"),
    (0x6D, "PAVE_MOINS"),
    (0x6E, "PAVE_POINT"),
    (0x6F, "PAVE_DIVISE"),
    (0xA0, "MAJ_G"),
    (0xA1, "MAJ_D"),
    (0xA2, "CTRL_G"),
    (0xA3, "CTRL_D"),
    (0xA4, "ALT_G"),
    (0xA5, "ALT_D"),
    (0x5B, "WIN_G"),
    (0x5D, "MENU_CONTEXTE"),
];

/// A key's name on screen: `Flèche haut`, `Clic gauche`, `W`, `F5`.
pub(super) fn key_shown(vk: u8) -> String {
    let name = key_name(vk);
    if name.len() <= 3 {
        return name;
    }
    let mut out = name.replace('_', " ").to_lowercase();
    for (plain, accented) in [
        ("fleche", "flèche"),
        ("entree", "entrée"),
        ("echap", "échap"),
        ("pave", "pavé"),
        ("debut", "début"),
        ("divise", "divisé"),
    ] {
        out = out.replace(plain, accented);
    }
    let mut chars = out.chars();
    chars
        .next()
        .map_or(out.clone(), |c| c.to_uppercase().chain(chars).collect())
}

pub(super) fn key_name(vk: u8) -> String {
    if let Some((_, name)) = NAMES.iter().find(|(k, _)| *k == vk) {
        return (*name).to_owned();
    }
    match vk {
        b'0'..=b'9' | b'A'..=b'Z' => (vk as char).to_string(),
        0x60..=0x69 => format!("PAVE_{}", vk - 0x60),
        0x70..=0x87 => format!("F{}", vk - 0x6F),
        _ => format!("VK_{vk:02X}"),
    }
}

pub(super) fn parse_key(name: &str) -> Option<u8> {
    let name = name.trim().to_ascii_uppercase();
    if name.is_empty() {
        return None;
    }
    if let Some((vk, _)) = NAMES.iter().find(|(_, n)| *n == name) {
        return Some(*vk);
    }
    let aliases = [
        ("ENTER", 0x0D),
        ("SPACE", 0x20),
        ("ESC", 0x1B),
        ("BACKSPACE", 0x08),
        ("SHIFT", 0xA0),
        ("CTRL", 0xA2),
        ("ALT", 0xA4),
        ("HAUT", 0x26),
        ("BAS", 0x28),
        ("UP", 0x26),
        ("DOWN", 0x28),
        ("LEFT", 0x25),
        ("RIGHT", 0x27),
    ];
    if let Some((_, vk)) = aliases.iter().find(|(n, _)| *n == name) {
        return Some(*vk);
    }
    if let [c @ (b'0'..=b'9' | b'A'..=b'Z')] = name.as_bytes() {
        return Some(*c);
    }
    if let Some(n) = name
        .strip_prefix("PAVE_")
        .and_then(|n| n.parse::<u8>().ok())
        .filter(|n| *n <= 9)
    {
        return Some(0x60 + n);
    }
    if let Some(n) = name
        .strip_prefix('F')
        .and_then(|n| n.parse::<u8>().ok())
        .filter(|n| (1..=24).contains(n))
    {
        return Some(0x6F + n);
    }
    let hex = name
        .strip_prefix("VK_")
        .or_else(|| name.strip_prefix("0X"))?;
    u8::from_str_radix(hex, 16).ok().filter(|vk| *vk != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(vk: u8) -> KeyState {
        KeyState {
            virtual_key: vk,
            ..Default::default()
        }
    }

    #[test]
    fn file_round_trips_and_every_name_parses() {
        let mut s = Settings::defaults();
        s.keys[0] = vec![0x01, 0xA0, b'7'];
        s.sensitivity = 2.5;
        s.invert = true;
        assert_eq!(Settings::parse(&s.text()), s);
        for vk in 1..=0xFE_u8 {
            assert_eq!(parse_key(&key_name(vk)), Some(vk), "{}", key_name(vk));
        }
    }

    #[test]
    fn keys_and_mouse_make_the_pad() {
        let mut map = Keymap::default();
        let mouse = Mouse {
            buttons: vec![0x01],
            ..Default::default()
        };
        let (pad, seen) = map.step(&[key(at(b'W', 0)), key(at(b'D', 0))], &mouse);
        assert!(seen);
        assert_eq!((pad.left_x, pad.left_y, pad.right_trigger), (1.0, 1.0, 1.0));
        // The mouse to the right: the camera turns right, past the dead zone.
        let mouse = Mouse {
            dx: 30,
            held: true,
            ..Default::default()
        };
        let (pad, _) = map.step(&[], &mouse);
        assert!(pad.right_x > 0.2 && pad.right_y == 0.0, "{pad:?}");
    }

    #[test]
    fn a_new_layout_moves_only_default_keys() {
        let mut map = Keymap::default();
        map.settings.layout = 1;
        map.settings.keys[1] = vec![b'U'];
        let defaults = Settings::defaults_for(2);
        map.place(2);
        assert_eq!(map.settings.layout, 2);
        assert_eq!(map.settings.keys[0], defaults.keys[0]);
        assert_eq!(map.settings.keys[1], vec![b'U']);
    }

    #[test]
    fn the_menu_rebinds_and_hides_keys_from_the_title() {
        let mut map = Keymap::default();
        let none = Mouse::default();
        let (_, seen) = map.step(&[key(F1)], &none);
        assert!(!seen && map.menu.is_some());
        // To the keys' tab; its first row (forward): Enter, then the new key.
        map.step(&[], &none);
        map.step(&[key(TAB)], &none);
        map.step(&[], &none);
        map.step(&[key(ENTER)], &none);
        map.step(&[], &none);
        map.step(&[key(b'U')], &none);
        assert_eq!(map.settings.keys[0], vec![b'U']);
        map.step(&[], &none);
        map.step(&[key(F1)], &none);
        assert!(map.menu.is_none());
        let (pad, seen) = map.step(&[key(b'U')], &none);
        assert!(seen && pad.left_y == 1.0);
    }
}
