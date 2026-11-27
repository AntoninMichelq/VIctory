//! `victory-voix` : multijoueur léger pour VIctory.
//!
//! Chacun joue sa partie, mais on se connecte à un ami et on s'entend : la
//! voix passe en direct (UDP, Opus 20 ms), et son volume dépend de la
//! distance entre les deux caméras dans le jeu ; le côté (gauche/droite)
//! suit la direction de l'ami. L'émulateur lancé avec `VICTORY_MULTI=1`
//! envoie la position de la caméra sur 127.0.0.1:47800 ; sans elle, c'est un
//! vocal normal (volume plein).
//!
//! Programme à part (pas dans l'émulateur) : un client audio dans le
//! processus du jeu change la résolution du minuteur de ce processus, donc
//! les attentes du jeu (voir `victory-hle` `audio/capture.rs`).
//!
//! Réglages : ligne de commande, puis `multijoueur.ini` relu chaque seconde
//! (voir `config`) ; état écrit dans `multijoueur-etat.ini`.

mod config;
mod jitter;
mod proto;
mod resample;
mod spatial;

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use opus_pure::{Application, OpusDecoder, OpusEncoder};

use jitter::Jitter;
use proto::{Packet, Pose};
use config::Conf;
use spatial::Reglages;

const TRAME: usize = 960; // 20 ms à 48 kHz

#[derive(Parser, Debug)]
#[command(name = "victory-voix", about = "Vocal de proximité entre deux joueurs VIctory")]
struct Args {
    /// Adresse de l'ami (IP ou IP:port). Les deux côtés peuvent la donner ;
    /// sans elle, on attend que l'ami se connecte.
    #[arg(long = "ami")]
    amis: Vec<String>,
    /// Mot de passe de la partie : il faut le même des deux côtés.
    #[arg(long)]
    code: Option<String>,
    /// Ton nom, affiché chez l'ami.
    #[arg(long)]
    nom: Option<String>,
    /// Port UDP (à ouvrir sur la box de celui qui reçoit la connexion).
    #[arg(long)]
    port: Option<u16>,
    /// Port local où l'émulateur envoie la position de la caméra.
    #[arg(long, default_value_t = 47800)]
    port_jeu: u16,
    /// Port local où l'on renvoie la position des amis à l'émulateur.
    #[arg(long, default_value_t = 47801)]
    port_amis: u16,
    /// Distance (mètres du jeu) en dessous de laquelle le volume est plein.
    #[arg(long)]
    pres: Option<f32>,
    /// Distance au-delà de laquelle on n'entend plus l'ami.
    #[arg(long)]
    loin: Option<f32>,
    /// Seuil du micro (0 à 1) : en dessous, on n'envoie rien.
    #[arg(long)]
    seuil: Option<f32>,
    /// Échange gauche et droite.
    #[arg(long)]
    inverser: bool,
    /// Essai seul : ta voix revient dans le casque, comme si un ami tournait
    /// autour de toi (passe par Opus comme en vrai).
    #[arg(long)]
    test_micro: bool,
}

impl Args {
    /// Ce qui est donné en ligne de commande passe avant le fichier.
    fn appliquer(&self, c: &mut Conf) {
        if !self.amis.is_empty() {
            c.ami = self.amis.join(",");
        }
        if let Some(v) = &self.code {
            c.code = v.clone();
        }
        if let Some(v) = &self.nom {
            c.pseudo = v.clone();
        }
        if let Some(v) = self.port {
            c.port = v;
        }
        if let Some(v) = self.pres {
            c.pres = v;
        }
        if let Some(v) = self.loin {
            c.loin = v;
        }
        if let Some(v) = self.seuil {
            c.seuil = v;
        }
        c.inverser |= self.inverser;
        c.test_micro |= self.test_micro;
        if c.loin < c.pres + 1.0 {
            c.loin = c.pres + 1.0;
        }
    }
}

struct Ami {
    nom: String,
    vu: Instant,
    jitter: Jitter,
    decodeur: OpusDecoder,
    pos: Option<Pose>,
    gains: (f32, f32),
    recu: u64,
}

impl Ami {
    fn new(nom: &str) -> Self {
        Ami {
            nom: nom.to_string(),
            vu: Instant::now(),
            jitter: Jitter::default(),
            decodeur: OpusDecoder::new(48_000, 1).expect("décodeur Opus"),
            pos: None,
            gains: (0.0, 0.0),
            recu: 0,
        }
    }

    fn voix(&mut self, seq: u16, data: &[u8]) {
        let Some(manque) = self.jitter.ordre(seq) else { return };
        let mut pcm = [0f32; TRAME];
        for _ in 0..manque.min(3) {
            if self.decodeur.decode(&[], TRAME, &mut pcm).is_ok() {
                self.jitter.pousser(&pcm);
            }
        }
        if let Ok(n) = self.decodeur.decode(data, TRAME, &mut pcm) {
            self.jitter.pousser(&pcm[..n.min(TRAME)]);
            self.recu += 1;
        }
    }
}

struct Etat {
    amis: HashMap<SocketAddr, Ami>,
    moi: Option<(Pose, Instant)>,
    conf: Conf,
    code: u64,
}

impl Etat {
    fn new(conf: Conf) -> Self {
        Etat { amis: HashMap::new(), moi: None, code: proto::code_hash(&conf.code), conf }
    }

    fn reglages(&self) -> Reglages {
        Reglages { pres: self.conf.pres, loin: self.conf.loin, inverser: self.conf.inverser }
    }

    fn ma_pose(&self) -> Option<Pose> {
        self.moi.filter(|(_, t)| t.elapsed() < Duration::from_secs(2)).map(|(p, _)| p)
    }

    /// Un échantillon à 48 kHz du mélange de tous les amis.
    fn melange(&mut self) -> (f32, f32) {
        let moi = self.ma_pose();
        let reglages = self.reglages();
        let (mut l, mut r) = (0.0, 0.0);
        for ami in self.amis.values_mut() {
            let s = ami.jitter.prendre();
            let cible = spatial::rendu(moi.as_ref(), ami.pos.map(|p| p.pos), &reglages);
            // Les gains glissent (~20 ms) pour ne pas claquer.
            ami.gains.0 += (cible.gauche - ami.gains.0) * 0.001;
            ami.gains.1 += (cible.droite - ami.gains.1) * 0.001;
            l += s * ami.gains.0;
            r += s * ami.gains.1;
        }
        let v = self.conf.volume_amis;
        ((l * v).tanh(), (r * v).tanh())
    }
}

/// L'adresse qui sert à l'« ami » de l'essai micro (jamais sur le réseau).
fn adresse_test() -> SocketAddr {
    "127.0.0.1:1".parse().unwrap()
}

fn adresse(s: &str, port: u16) -> Option<SocketAddr> {
    let s = s.trim();
    let avec_port = if s.contains(':') { s.to_string() } else { format!("{s}:{port}") };
    let liste: Vec<SocketAddr> = avec_port.to_socket_addrs().ok()?.collect();
    liste.iter().find(|a| a.is_ipv4()).or(liste.first()).copied()
}

/// La touche « pousser pour parler » est-elle enfoncée ?
fn touche_enfoncee(vk: i32) -> bool {
    #[cfg(windows)]
    {
        // SAFETY: lit seulement l'état du clavier.
        unsafe { windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState(vk) as u16 & 0x8000 != 0 }
    }
    #[cfg(not(windows))]
    {
        let _ = vk;
        false
    }
}

const INI_DEPART: &str = "\
# Reglages du vocal multijoueur VIctory (relus chaque seconde pendant la partie).
# pseudo : ton nom ; ami : IP ou IP:port de ton ami (vide = tu attends qu'il se connecte a toi)
# code : le meme mot pour vous deux ; port : UDP (au demarrage seulement)
# micro : ouvert ou appuyer (pousser pour parler, touche_parler)
# volume_micro / volume_amis : 0 a 200 (%) ; pres / loin : metres ; seuil : sensibilite du micro
pseudo=joueur
ami=
code=victory
port=47777
micro=ouvert
touche_parler=V
muet=0
volume_micro=100
volume_amis=100
pres=3
loin=40
inverser=0
seuil=0.012
test_micro=0
";

fn main() {
    let args = Args::parse();
    if let Err(e) = lancer(args) {
        eprintln!("\nErreur : {e}");
        eprintln!("Appuie sur Entrée pour fermer.");
        let _ = std::io::stdin().lock().lines().next();
        std::process::exit(1);
    }
}

fn lire_ini(chemin: &std::path::Path) -> Option<(String, std::time::SystemTime)> {
    let quand = std::fs::metadata(chemin).and_then(|m| m.modified()).ok()?;
    Some((std::fs::read_to_string(chemin).ok()?, quand))
}

fn lancer(args: Args) -> Result<(), String> {
    let ini = config::chemin();
    let ini_etat = config::chemin_etat(&ini);
    if !ini.exists() {
        let _ = std::fs::write(&ini, INI_DEPART.replace('\n', "\r\n"));
    }
    let mut conf = Conf::default();
    let mut vu_ini = None;
    if let Some((texte, quand)) = lire_ini(&ini) {
        conf.appliquer(&texte);
        vu_ini = Some(quand);
    }
    args.appliquer(&mut conf);
    let port = conf.port;
    let etat = Arc::new(Mutex::new(Etat::new(conf.clone())));
    let niveau = Arc::new(AtomicU32::new(0));
    let fin = Arc::new(AtomicBool::new(false));

    // --- Réseau ---
    let socket = UdpSocket::bind(("0.0.0.0", port))
        .map_err(|e| format!("impossible d'ouvrir le port UDP {port} ({e}). Un autre victory-voix tourne déjà ?"))?;
    socket.set_read_timeout(Some(Duration::from_millis(200))).ok();
    let socket = Arc::new(socket);
    for a in conf.amis() {
        if adresse(a, port).is_none() {
            return Err(format!("adresse de l'ami illisible : « {a} »"));
        }
    }

    println!("=== VIctory voix : vocal de proximité ===");
    println!("Réglages : {} (relus pendant la partie)", ini.display());
    if conf.test_micro {
        println!("Essai micro : parle, tu dois t'entendre dans le casque, ta voix tourne autour de toi.");
    }
    println!("Port UDP {port} ouvert. Code de la partie : « {} ». Pseudo : {}.", conf.code.trim(), conf.pseudo);
    if conf.amis().next().is_none() {
        println!("En attente d'un ami (donne-lui ton adresse IP et le code).");
    } else {
        for a in conf.amis() {
            println!("Connexion à {a} …");
        }
    }
    println!("Commandes : m + Entrée = couper/remettre le micro, q + Entrée = quitter.\n");

    {
        let (s, e, f) = (socket.clone(), etat.clone(), fin.clone());
        std::thread::Builder::new()
            .name("voix réception".into())
            .spawn(move || reception(&s, &e, &f))
            .map_err(|e| e.to_string())?;
        // Salut régulier (ouvre aussi le passage dans les box des deux côtés),
        // et notre position.
        let (s, e, f) = (socket.clone(), etat.clone(), fin.clone());
        let port_amis = args.port_amis;
        std::thread::Builder::new()
            .name("voix salut".into())
            .spawn(move || salut(&s, &e, port, port_amis, &f))
            .map_err(|e| e.to_string())?;
    }

    // Position de la caméra, envoyée par l'émulateur.
    let mut position_jeu = true;
    match UdpSocket::bind(("127.0.0.1", args.port_jeu)) {
        Ok(jeu) => {
            jeu.set_read_timeout(Some(Duration::from_millis(500))).ok();
            let (e, f) = (etat.clone(), fin.clone());
            std::thread::Builder::new()
                .name("voix jeu".into())
                .spawn(move || {
                    let mut buf = [0u8; 256];
                    while !f.load(Relaxed) {
                        if let Ok(n) = jeu.recv(&mut buf)
                            && let Some(p) = proto::read_game_pose(&buf[..n])
                        {
                            e.lock().unwrap().moi = Some((p, Instant::now()));
                        }
                    }
                })
                .map_err(|e| e.to_string())?;
        }
        Err(e) => {
            position_jeu = false;
            println!("(position du jeu indisponible sur le port {} : {e} ; vocal normal)", args.port_jeu);
        }
    }

    // --- Audio ---
    let hote = cpal::default_host();
    let micro = hote.default_input_device().ok_or("aucun micro trouvé (branche un micro / casque-micro)")?;
    let hp = hote.default_output_device().ok_or("aucune sortie son trouvée")?;
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let entree = ouvrir_micro(&micro, tx)?;
    let sortie = ouvrir_sortie(&hp, etat.clone())?;
    entree.play().map_err(|e| format!("micro : {e}"))?;
    sortie.play().map_err(|e| format!("son : {e}"))?;
    let freq_micro = micro.default_input_config().map(|c| c.sample_rate().0).unwrap_or(48_000);
    println!(
        "Micro : {}  |  Sortie : {}",
        micro.name().unwrap_or_default(),
        hp.name().unwrap_or_default()
    );
    println!("Conseil : utilise un casque, sinon l'ami s'entendra en écho.\n");

    // Encodage du micro, envoi.
    {
        let (e, n, f, s) = (etat.clone(), niveau.clone(), fin.clone(), socket.clone());
        std::thread::Builder::new()
            .name("voix micro".into())
            .spawn(move || encodage(rx, freq_micro, &s, &e, &n, &f))
            .map_err(|e| e.to_string())?;
    }

    // Clavier (m / q).
    {
        let (e, f) = (etat.clone(), fin.clone());
        std::thread::spawn(move || {
            for ligne in std::io::stdin().lock().lines() {
                let Ok(ligne) = ligne else { break };
                match ligne.trim().to_lowercase().as_str() {
                    "m" => {
                        let mut e = e.lock().unwrap();
                        e.conf.muet = !e.conf.muet;
                        println!("\nMicro {}", if e.conf.muet { "coupé" } else { "remis" });
                    }
                    "q" => {
                        f.store(true, Relaxed);
                        break;
                    }
                    _ => {}
                }
            }
        });
    }

    // Ligne d'état et fichier d'état ; réglages relus.
    let mut connus: Vec<String> = Vec::new();
    let mut tour = 0u32;
    while !fin.load(Relaxed) {
        std::thread::sleep(Duration::from_millis(250));
        tour = tour.wrapping_add(1);
        let relire = tour.is_multiple_of(4);
        let nouveau = if relire { lire_ini(&ini).filter(|(_, q)| Some(*q) != vu_ini) } else { None };
        let mut e = etat.lock().unwrap();
        if let Some((texte, quand)) = nouveau {
            vu_ini = Some(quand);
            let avant = e.conf.clone();
            e.conf.appliquer(&texte);
            if e.conf != avant {
                println!("\nRéglages relus ({}).", ini.display());
                if e.conf.port != port {
                    println!("(le nouveau port sera pris au prochain lancement)");
                }
            }
            e.code = proto::code_hash(&e.conf.code);
        }
        if !e.conf.test_micro {
            e.amis.remove(&adresse_test());
        }
        // Les amis silencieux depuis 10 s sont partis.
        let partis: Vec<_> = e.amis.iter().filter(|(_, a)| a.vu.elapsed() > Duration::from_secs(10)).map(|(k, _)| *k).collect();
        for k in partis {
            if let Some(a) = e.amis.remove(&k) {
                println!("\n{} est parti.", a.nom);
            }
        }
        let noms: Vec<String> = e.amis.values().map(|a| a.nom.clone()).collect();
        for n in &noms {
            if !connus.contains(n) {
                println!("\n{n} est connecté !");
            }
        }
        connus = noms;
        let moi = e.ma_pose();
        let reglages = e.reglages();
        let lvl = f32::from_bits(niveau.load(Relaxed));
        let niveau_pct = ((lvl * 60.0).sqrt() * 50.0).clamp(0.0, 100.0);
        let barres = (niveau_pct / 12.5) as usize;
        let mut ligne = format!(
            "Micro {:<8}{}",
            "#".repeat(barres),
            if e.conf.muet { " (COUPÉ)" } else if e.conf.appuyer { " (pousser pour parler)" } else { "" }
        );
        ligne.push_str(if moi.is_some() { " | jeu: position OK" } else { " | jeu: pas de position" });
        let mut amis_etat = Vec::new();
        for a in e.amis.values() {
            let r = spatial::rendu(moi.as_ref(), a.pos.map(|p| p.pos), &reglages);
            let cote = if r.cote < -0.3 { "gauche" } else if r.cote > 0.3 { "droite" } else { "face" };
            match r.distance {
                Some(d) => {
                    ligne.push_str(&format!(
                        " | {} à {:.0} m, {}, volume {:.0}%",
                        a.nom,
                        d,
                        cote,
                        (r.gauche.max(r.droite) / std::f32::consts::SQRT_2 * 100.0).min(100.0)
                    ));
                    amis_etat.push(format!("{}:{:.0}m:{cote}", a.nom.replace([':', ';'], " "), d));
                }
                None => {
                    ligne.push_str(&format!(" | {} : vocal normal ({} paquets)", a.nom, a.recu));
                    amis_etat.push(format!("{}:inconnu", a.nom.replace([':', ';'], " ")));
                }
            }
        }
        let connecte = !e.amis.is_empty();
        drop(e);
        print!("\r{ligne:<150}");
        let _ = std::io::stdout().flush();
        if relire {
            let texte = format!(
                "etat={}\r\nerreur={}\r\namis={}\r\nniveau_micro={:.0}\r\nposition_jeu={}\r\n",
                if connecte { "connecte" } else { "attente" },
                if position_jeu { "" } else { "port de position du jeu occupe" },
                amis_etat.join(";"),
                niveau_pct,
                u8::from(moi.is_some())
            );
            let _ = std::fs::write(&ini_etat, texte);
        }
    }
    let mut buf = Vec::new();
    Packet::Bye.write(&mut buf);
    for a in etat.lock().unwrap().amis.keys() {
        let _ = socket.send_to(&buf, a);
    }
    let _ = std::fs::write(&ini_etat, "etat=arrete\r\n");
    println!("\nAu revoir !");
    Ok(())
}

fn reception(s: &UdpSocket, etat: &Mutex<Etat>, fin: &AtomicBool) {
    let mut buf = [0u8; 2048];
    while !fin.load(Relaxed) {
        let Ok((n, de)) = s.recv_from(&mut buf) else { continue };
        let Some(p) = Packet::read(&buf[..n]) else { continue };
        let mut e = etat.lock().unwrap();
        match p {
            Packet::Hello { code: c, name } => {
                if c != e.code {
                    continue;
                }
                let ami = e.amis.entry(de).or_insert_with(|| Ami::new(name));
                ami.vu = Instant::now();
                ami.nom = name.to_string();
            }
            Packet::Bye => {
                if let Some(a) = e.amis.remove(&de) {
                    println!("\n{} est parti.", a.nom);
                }
            }
            Packet::Voice { seq, data } => {
                if let Some(a) = e.amis.get_mut(&de) {
                    a.vu = Instant::now();
                    a.voix(seq, data);
                }
            }
            Packet::Pos(p) => {
                if let Some(a) = e.amis.get_mut(&de) {
                    a.vu = Instant::now();
                    a.pos = p;
                }
            }
        }
    }
}

fn salut(s: &UdpSocket, etat: &Mutex<Etat>, port: u16, port_amis: u16, fin: &AtomicBool) {
    let mut buf = Vec::new();
    let mut pos = Vec::new();
    let mut tour = 0u32;
    let mut cibles: Vec<SocketAddr> = Vec::new();
    let mut ami_vu = None::<String>;
    while !fin.load(Relaxed) {
        let (amis, moi, positions, code, nom, ami) = {
            let e = etat.lock().unwrap();
            let amis: Vec<SocketAddr> = e.amis.keys().copied().filter(|a| *a != adresse_test()).collect();
            let positions: Vec<Option<[f32; 3]>> = e.amis.values().map(|a| a.pos.map(|p| p.pos)).collect();
            (amis, e.ma_pose(), positions, e.code, e.conf.pseudo.clone(), e.conf.ami.clone())
        };
        if ami_vu.as_ref() != Some(&ami) {
            let c = Conf { ami: ami.clone(), ..Conf::default() };
            cibles = c.amis().filter_map(|a| adresse(a, port)).collect();
            ami_vu = Some(ami);
        }
        if tour.is_multiple_of(5) {
            Packet::Hello { code, name: &nom }.write(&mut buf);
            for a in cibles.iter().chain(amis.iter().filter(|a| !cibles.contains(a))) {
                let _ = s.send_to(&buf, a);
            }
        }
        Packet::Pos(moi).write(&mut pos);
        for a in &amis {
            let _ = s.send_to(&pos, a);
        }
        // Pour un marqueur dans l'émulateur (plus tard) : où sont les amis.
        proto::write_friends(&mut buf, &positions);
        let _ = s.send_to(&buf, ("127.0.0.1", port_amis));
        tour = tour.wrapping_add(1);
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn encodage(
    rx: mpsc::Receiver<Vec<f32>>,
    freq: u32,
    socket: &UdpSocket,
    etat: &Mutex<Etat>,
    niveau: &AtomicU32,
    fin: &AtomicBool,
) {
    let mut enc = OpusEncoder::new(48_000, 1, Application::Voip).expect("encodeur Opus");
    enc.bitrate_bps = 32_000;
    enc.use_inband_fec = true;
    enc.packet_loss_perc = 5;
    let mut rs = resample::Pousse::new(freq, 48_000);
    let mut a48 = Vec::new();
    let mut paquet = vec![0u8; 1500];
    let mut envoi = Vec::new();
    let mut seq: u16 = 0;
    let mut parle_jusqua = Instant::now();
    let debut = Instant::now();
    while !fin.load(Relaxed) {
        let Ok(bloc) = rx.recv_timeout(Duration::from_millis(200)) else { continue };
        rs.traiter(&bloc, &mut a48);
        while a48.len() >= TRAME {
            let mut trame: Vec<f32> = a48.drain(..TRAME).collect();
            let conf = etat.lock().unwrap().conf.clone();
            for x in &mut trame {
                *x *= conf.volume_micro;
            }
            let rms = (trame.iter().map(|x| x * x).sum::<f32>() / TRAME as f32).sqrt();
            niveau.store(rms.to_bits(), Relaxed);
            let parle = if conf.appuyer {
                config::touche(&conf.touche_parler).is_some_and(touche_enfoncee)
            } else {
                if rms > conf.seuil {
                    parle_jusqua = Instant::now() + Duration::from_millis(400);
                }
                Instant::now() <= parle_jusqua
            };
            if conf.muet || !parle {
                continue;
            }
            let Ok(n) = enc.encode(&trame, TRAME, &mut paquet) else { continue };
            seq = seq.wrapping_add(1);
            if conf.test_micro {
                // Un « ami » qui tourne autour de nous à 2-30 m, un tour en 12 s.
                let t = debut.elapsed().as_secs_f32();
                let angle = t / 12.0 * std::f32::consts::TAU;
                let dist = 16.0 + 14.0 * (t / 7.0).sin();
                let mut e = etat.lock().unwrap();
                e.moi = Some((
                    Pose { pos: [0.0; 3], right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0] },
                    Instant::now(),
                ));
                let a = e.amis.entry(adresse_test()).or_insert_with(|| Ami::new("écho"));
                a.vu = Instant::now();
                a.pos = Some(Pose {
                    pos: [dist * angle.cos(), dist * angle.sin(), 0.0],
                    right: [1.0, 0.0, 0.0],
                    up: [0.0, 0.0, 1.0],
                });
                a.voix(seq, &paquet[..n]);
            } else {
                Packet::Voice { seq, data: &paquet[..n] }.write(&mut envoi);
                let amis: Vec<SocketAddr> = etat.lock().unwrap().amis.keys().copied().filter(|a| *a != adresse_test()).collect();
                for a in amis {
                    let _ = socket.send_to(&envoi, a);
                }
            }
        }
    }
}

fn erreur_flux(e: cpal::StreamError) {
    eprintln!("\n(son : {e})");
}

fn ouvrir_micro(dev: &cpal::Device, tx: mpsc::Sender<Vec<f32>>) -> Result<cpal::Stream, String> {
    let cfg = dev.default_input_config().map_err(|e| format!("micro : {e}"))?;
    let canaux = usize::from(cfg.channels()).max(1);
    let format = cfg.sample_format();
    let config: cpal::StreamConfig = cfg.into();
    fn flux<T: SizedSample>(
        dev: &cpal::Device,
        config: &cpal::StreamConfig,
        canaux: usize,
        tx: mpsc::Sender<Vec<f32>>,
    ) -> Result<cpal::Stream, cpal::BuildStreamError>
    where
        f32: FromSample<T>,
    {
        dev.build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let mono: Vec<f32> = data
                    .chunks(canaux)
                    .map(|c| c.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / canaux as f32)
                    .collect();
                let _ = tx.send(mono);
            },
            erreur_flux,
            None,
        )
    }
    match format {
        cpal::SampleFormat::F32 => flux::<f32>(dev, &config, canaux, tx),
        cpal::SampleFormat::I16 => flux::<i16>(dev, &config, canaux, tx),
        cpal::SampleFormat::U16 => flux::<u16>(dev, &config, canaux, tx),
        cpal::SampleFormat::I32 => flux::<i32>(dev, &config, canaux, tx),
        f => return Err(format!("format de micro non géré : {f:?}")),
    }
    .map_err(|e| format!("micro : {e}"))
}

fn ouvrir_sortie(dev: &cpal::Device, etat: Arc<Mutex<Etat>>) -> Result<cpal::Stream, String> {
    let cfg = dev.default_output_config().map_err(|e| format!("son : {e}"))?;
    let canaux = usize::from(cfg.channels()).max(1);
    let freq = cfg.sample_rate().0;
    let format = cfg.sample_format();
    let config: cpal::StreamConfig = cfg.into();
    fn flux<T: SizedSample + FromSample<f32>>(
        dev: &cpal::Device,
        config: &cpal::StreamConfig,
        canaux: usize,
        freq: u32,
        etat: Arc<Mutex<Etat>>,
    ) -> Result<cpal::Stream, cpal::BuildStreamError> {
        let mut tire = resample::Tire::new(48_000, freq);
        dev.build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                let mut e = etat.lock().unwrap();
                for trame in data.chunks_mut(canaux) {
                    let (l, r) = tire.suivant(&mut || e.melange());
                    for (i, s) in trame.iter_mut().enumerate() {
                        let v = match (canaux, i) {
                            (1, _) => (l + r) * 0.5,
                            (_, 0) => l,
                            (_, 1) => r,
                            _ => 0.0,
                        };
                        *s = T::from_sample(v);
                    }
                }
            },
            erreur_flux,
            None,
        )
    }
    match format {
        cpal::SampleFormat::F32 => flux::<f32>(dev, &config, canaux, freq, etat),
        cpal::SampleFormat::I16 => flux::<i16>(dev, &config, canaux, freq, etat),
        cpal::SampleFormat::U16 => flux::<u16>(dev, &config, canaux, freq, etat),
        cpal::SampleFormat::I32 => flux::<i32>(dev, &config, canaux, freq, etat),
        f => return Err(format!("format de sortie non géré : {f:?}")),
    }
    .map_err(|e| format!("son : {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toute la chaîne sans carte son : encodage Opus, réseau local, file,
    /// mélange selon la distance.
    #[test]
    fn de_bout_en_bout_en_local() {
        let a = UdpSocket::bind("127.0.0.1:0").unwrap();
        let b = UdpSocket::bind("127.0.0.1:0").unwrap();
        b.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut enc = OpusEncoder::new(48_000, 1, Application::Voip).unwrap();
        let mut ami = Ami::new("a");
        let mut pkt = vec![0u8; 1500];
        let mut buf = Vec::new();
        let mut rx = [0u8; 2048];
        for i in 0..10u16 {
            let trame: Vec<f32> = (0..TRAME).map(|k| (k as f32 * 0.05).sin() * 0.3).collect();
            let n = enc.encode(&trame, TRAME, &mut pkt).unwrap();
            if i == 4 {
                continue; // un paquet perdu
            }
            Packet::Voice { seq: i, data: &pkt[..n] }.write(&mut buf);
            a.send_to(&buf, b.local_addr().unwrap()).unwrap();
            let n = b.recv(&mut rx).unwrap();
            if let Some(Packet::Voice { seq, data }) = Packet::read(&rx[..n]) {
                ami.voix(seq, data);
            }
        }
        assert_eq!(ami.recu, 9);
        assert!(ami.jitter.en_attente_ms() >= 180);
        ami.pos = Some(Pose { pos: [10.0, 0.0, 0.0], right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0] });
        let mut etat = Etat::new(Conf::default());
        etat.amis.insert(a.local_addr().unwrap(), ami);
        etat.moi = Some((Pose { pos: [0.0; 3], right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0] }, Instant::now()));
        let (mut l, mut r) = (0.0f32, 0.0f32);
        for _ in 0..TRAME * 6 {
            let (x, y) = etat.melange();
            l += x.abs();
            r += y.abs();
        }
        assert!(r > l * 2.0 && l > 0.0, "l={l} r={r}");
    }
}
