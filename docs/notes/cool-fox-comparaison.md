# cool-fox-codec vs notre fork (jouer/cumul11e) — comparaison vérifiée (2026-11-26)

Écrit pour 4nto et pour les agents T3 Code. Tout ce qui suit a été vérifié dans
l'historique git et le code de https://github.com/ComputerScape04/cool-fox-codec
(`main` = `f4b48ad`, 627 commits) et de notre branche `jouer/cumul11e` (`c3765e7`).
Lecture seule sur leur dépôt : rien n'a été porté, rien n'a été lancé en jeu.

## En bref (français simple)

- **Les deux partent du même point** : fxceless/victory-fork `4e17478`. Depuis, eux ont
  404 commits, nous 291. Ce sont deux chemins séparés, pas une version "en avance" de l'autre.
- **Eux vont plus loin dans le jeu** : leur README décrit la première mission en bateau,
  une course-poursuite en voiture, le camp et le vol. Nous sommes encore dans la prison (prologue).
- **Eux tournent plus vite, mais sur des machines bien plus grosses** : 13-22 img/s sur une
  RX 7900 XTX 24 Go + i9-12900K + 64 Go de RAM ; ~16-18 img/s sur une RTX 5090 (32 Go de VRAM).
  Nous : 8,87 img/s (cumul9b) sur une RX 7600 8 Go avec FSR. Une partie de l'écart, c'est juste le matériel.
- **Ils consomment beaucoup plus de mémoire** : 40-46 Go de RAM (64 Go conseillés) sur la 7900 XTX,
  26-34 Go sur la machine RTX 5090. Nous : ~21 Go. Ils ne sont **pas** à 8-10 Go.
- **Leur vraie avance est côté processeur** (préparer les images sur plusieurs fils en parallèle).
  C'est exactement notre point faible actuel (préparation ~72 ms + carte ~50 ms, l'un après l'autre).
- **On ne peut pas "copier" leur code tel quel** : 166 fichiers diffèrent, des fichiers entiers
  n'existent que chez eux. Il faut reprendre leurs idées une par une, dans l'ordre ci-dessous.

## 1. Ce que disait l'autre IA, vérifié

| Affirmation | Vérifié ? | Ce que dit vraiment le dépôt |
|---|---|---|
| Miroir de codec-d/cool_fox | Oui | 6 commits de `codec_d` (25/11) par-dessus avatarmanz/VIctory-7900XTX `0a43566` (README : « VIctory on an RX 7900 XTX »). Les binaires sont dans des releases privées de codec-d/cool_fox. |
| Base = avatarmanz (fxceless + DexNom + Belobug) | Oui | Crédits du README : fxceless, DexNom, Belobug (romainhedouin), ZJOU, myokex/victory-amd, Geckomacabre. Base commune avec nous : `4e17478`. |
| 25 img/s en jeu, 30 en cinématique | **Non, exagéré** | 7900 XTX (builds *diagnostics*, 1440p, Series X) : prologue intérieur 50-75 ms (13-20 img/s), bateau 46-53 ms (19-22), poursuite en rue 100-165 ms (6-10), menus/vidéos 30. RTX 5090 (bug-journal codec_d) : médiane 54-64 ms (~16-18 img/s) dans la rue d'ouverture. |
| 8 threads de traduction des draw calls | Oui | `VICTORY_THIN` (`3a99ee6`) : les appels de liste du jeu sont notés puis traduits par 8 fils. **Gain mesuré par eux : seulement 75 → 68 ms.** |
| Préparation sur un thread séparé | Oui | `VICTORY_ASYNC_SUBMIT` (`b8b30d1`, idée de fxceless `f85101b6`) : 103 → 75 ms chez eux. |
| File d'upload dédiée | Oui | `VICTORY_UPLOAD_AHEAD` (`350c686`) : textures envoyées sur une file de copie à elles. |
| 8-10 Go de RAM | **Faux** | 40-46 Go de mémoire privée (findings.md : allocateurs de listes 9 Go, fenêtres 7, uploads 4). codec_d : pic 26-34 Go de commit sur 32 Go. Nous faisons mieux (~21 Go). |
| 16 Go de VRAM | Faux en détail | 24 Go (7900 XTX) et 32 Go (RTX 5090, pic mesuré 17-19 Go). Rien n'est réglé pour 8 Go. |
| 80 % de l'histoire | **Non vérifiable** | Rien dans le dépôt ne le dit. Le README liste prologue, première mission bateau, rue, camp, vol. |
| Pas d'allègement des shaders | Oui | Ils ont mesuré les maths de shaders comme « < 1 ms » sur 7900 XTX (findings.md). Sur notre RX 7600, `LEAN_MATH` a donné −8 à −18 ms : notre carte est plus petite, ça compte plus chez nous. |
| Nos correctifs 8 Go absents chez eux | Oui | Aucun de `TEXTURE_HALF`, `VRAM_HYSTERESIS`, `EVICT_REQUEUE_FRAMES`, `EVICT_LESS`, `HOIST_CLAMPS`, `LEAN_MATH`, FSR hôte chez eux. |

## 2. Qui est le plus avancé ?

- **Progression dans le jeu et maturité du moteur côté CPU : eux.** Ils ont résolu des choses qu'on
  n'a pas encore rencontrées (bateaux qui coulent = lecture des textures par le CPU,
  `VICTORY_TEXTURE_BACK`, on par défaut ; perte du device toutes les ~16 min = `VICTORY_SHOW_AFTER`).
- **Tenir sur une carte 8 Go et 32 Go de RAM : nous.** Leur build ne vise ni 8 Go de VRAM ni
  moins de 32 Go de RAM ; sur la RX 7600 il saturerait la VRAM.
- **Vitesse à matériel égal : inconnue.** Personne n'a lancé leur build sur la RX 7600. Leur
  carte fait ~41 ms par image sur une 7900 XTX (~2,5-3× une RX 7600) ; la nôtre ~51 ms avec FSR 0,42.
  Côté CPU, leur thread de préparation fait 48 ms sur un i9-12900K ; le nôtre ~72 ms.

## 3. Où va une image, chez eux et chez nous

| | eux (7900 XTX, bateau, 55 ms) | nous (RX 7600, cumul9b, 112,8 ms) |
|---|---|---|
| Préparation CPU | 48 ms, **en parallèle** de la carte | ~72 ms, **puis** la carte |
| Carte | ~41 ms | ~50-55 ms |
| Appels de liste | 36 000/image, 230 ms de travail sur 8 fils | sur les fils du jeu |

Notre priorité est donc claire : **faire travailler CPU et carte en même temps** et éclater la
préparation sur plusieurs fils. C'est ce qu'ils ont de mieux.

## 4. Ce qu'il faut porter, classé (idées à réécrire sur notre arbre, pas des cherry-picks)

Chaque ligne : leur commit, ce que ça fait, leur gain mesuré, la difficulté chez nous.

1. **Préparation et envoi sur un thread à nous** — `b8b30d1` (`VICTORY_ASYNC_SUBMIT`, idée fxceless
   `f85101b6`), puis `7c607d6`/`37332f4` (`VICTORY_JOIN_PARTS=8`). Gain chez eux 103 → 75 ms.
   Chez nous : la version fxceless est déjà fusionnée dans `port/fx-d510ee4` (`23e5a09`), **jamais
   lancée**. Le plus rapide : rebaser cette partie sur cumul9b/cumul11 et l'essayer. Difficulté moyenne.
   Ne pas combiner sans test avec notre `LATE_WAIT=3`.
2. **Enregistrements des tas de descripteurs sur plusieurs fils** — `1f1a433` (enregistrements des
   draws laissés aux draws : 280 → 124 ms, leur plus gros gain), `53c5256` (`HEAP_SYNC_ASIDE`,
   `QUICK_MIRROR`), `6c17ce9` (`VIEW_THREADS`), `c3e643e`, `44629de` (copie en octets, pages
   comparées sur plusieurs fils, slots vides ignorés), `e8d3d91` (`SAME_RECORDS`). Leur
   `descriptor/host.rs` fait 5 941 lignes contre 3 395 chez nous. Notre `port/fx-heap-sync`
   (`a898679`, empreinte fxceless `9376fd6`) va dans le même sens, jamais lancé. Difficulté élevée.
3. **Comparaisons AVX2 et travail en avance sur leurs propres fils** — `b620284`, `9b90ad1`,
   `1a411ae` (`ASK_EARLY` : demander la write watch avant la préparation). 14 → 4-7 ms. Moyen.
4. **Copies faites par la carte au lieu d'attendre** — `cc03ecb`, `007fe55` (`VICTORY_CARD_STEPS=2`).
   Partie de « 68 → 63 ms ». Moyen.
5. **Traduction des appels de liste sur 8 fils** — `3a99ee6` (`VICTORY_THIN`). Seulement 75 → 68 ms
   chez eux, et c'est le plus gros chantier (nouveau dossier `list/`). À faire **après** 1 à 4.
6. **Textures : file de copie dédiée + afficher un slot seulement après son upload** — `350c686`
   (`UPLOAD_AHEAD`), `3bab23c` (`SHOW_AFTER`), `0322b39` (`WAIT_RESIDENT`). Chez eux `SHOW_AFTER`
   a fait disparaître une perte du device toutes les ~16 min (page fault dans une texture créée
   dans le même dixième de seconde). **À regarder pour la chasse au crash de cumul11** et pour nos
   taches chrome/noires (slots montrés avant que la texture soit là). Moyen.
7. **Petits gains CPU faciles** (venus de Belobug et ZJOU) — `ca1d519` (`BIND_CACHE` : ne pas
   remettre les mêmes render targets), `ab86ef7` (pipeline re-lié non renommé), `c96525d` (constant
   buffer non aligné copié une fois par liste), `b32fb99` (`DECLARE_ONCE`). Faibles risques, petits gains.
8. **Pour plus tard dans le jeu** — `23925a4` (`TEXTURE_BACK` : sans ça, les bateaux coulent à la
   première mission ; vient de fxceless `f5c03f3b` / Geckomacabre), `449830f` (`INDIRECT_REST=0`,
   scintillement dehors), découpage des dispatch indirects > 65 535 groupes (blocs).
9. **Garde-fous mémoire de codec_d (utiles avec nos 32 Go de RAM)** — `ab75ac3` (une pool qui se voit
   refuser de la mémoire attend 500 ms avant de regrossir ; uploads différés quand le commit Windows
   est bas, plus de E_OUTOFMEMORY), `18bdf79` (staging spéculatif limité à 16-32 Mo/image sous
   pression), libération des slots dynamiques à la fin du GPU (`Detached::release_tables`, dans
   `f4b48ad`). Nouveau fichier `residency.rs` (201 lignes). Moyen.
10. **Mode Series S** — `1ad6e67` (écran noir Series S : `RetailPlane` lu au pas de 72 octets).
    Chez nous Series S bloque au chargement (`patterns/series-s-boot.txt`), symptôme différent, mais
    c'est un essai pas cher : Series S = image plus petite, utile pour 8 Go.

## 5. À ne pas prendre

- `VICTORY_RDNA_FLOAT`, `SHADER_OPT=127`, tailles de wave : ils les ont mesurés sans gain notable.
  Nos `LEAN_MATH` / `HOIST_CLAMPS` sont meilleurs pour la RX 7600 : les garder.
- `MAP_THREADS` > 1 : plus lent chez eux (`4e99544`).
- Le passage Wave64 → Wave32 de codec_d (`8cc31ab`) : spécifique NVIDIA.
- Leurs réglages du lanceur `Play-VIctory.cmd` (16-32 Go de VRAM supposés) : ne pas les copier.
- Aucun de leurs binaires : releases privées, et jamais de fichiers du jeu.

## 6. Ordre conseillé

1. Remettre `port/fx-d510ee4` (ASYNC_SUBMIT + JOIN_PARTS de fxceless) sur la ligne cumul et le lancer
   (essai F1 de `docs/notes/fx-d510ee4-essai.md`), en gardant nos réglages 8 Go.
2. En parallèle, réécrire `SHOW_AFTER` + `WAIT_RESIDENT` (petits) : test contre le crash cumul11 et le chrome.
3. Puis enregistrements des tas sur plusieurs fils (point 2), puis AVX2 / ASK_EARLY / CARD_STEPS.
4. `VICTORY_THIN` en dernier.

Objectif réaliste : préparation sous la carte (~50 ms) et les deux en même temps → ~15-18 img/s
sur la RX 7600, pas 30. Le reste viendra de la carte (FSR, LEAN_MATH, clamps).

## Méthode (pour refaire la comparaison)

```sh
git clone --filter=blob:none https://github.com/ComputerScape04/cool-fox-codec cfc
cd cfc
git remote add fork https://github.com/AntoninMichelq/VIctory-forkAMD
git fetch --filter=blob:none fork jouer/cumul11e
git merge-base origin/main fork/jouer/cumul11e   # 4e17478
git log --author=local --oneline                # avatarmanz' commits
git log --author=codec_d --oneline              # codec_d's 6 commits
```

Leurs mesures : `README.md`, `docs/findings.md`, `docs/bug-journal.md` (codec_d) dans leur dépôt.
