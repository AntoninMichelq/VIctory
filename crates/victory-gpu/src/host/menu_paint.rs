//! The F1 menu's picture (`crate::menu::Page`), painted smooth at the
//! screen's own size: shapes with soft edges, text from a real font
//! ([`Fonts`]: GDI on Windows, `menu_gdi`), a neon night over the sea.
//! Painted only when the page changes.

use crate::menu::{Kind, Page};

/// Sets text in a font: the coverage (0 to 255) of `text`, `px` pixels
/// tall, as (width, height, rows). `title`: the name's own face.
pub(crate) trait Fonts {
    fn raster(&mut self, text: &str, px: u32, weight: Weight) -> (usize, usize, Vec<u8>);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Weight {
    Regular,
    Bold,
    /// The big name at the top.
    Title,
}

type Rgb = [f32; 3];

const fn rgb(r: u8, g: u8, b: u8) -> Rgb {
    [r as f32, g as f32, b as f32]
}

const NIGHT_TOP: Rgb = rgb(40, 14, 66);
const NIGHT_MID: Rgb = rgb(20, 8, 40);
const NIGHT_LOW: Rgb = rgb(8, 5, 20);
const PINK: Rgb = rgb(255, 52, 156);
const DEEP_PINK: Rgb = rgb(176, 26, 112);
const CYAN: Rgb = rgb(64, 228, 250);
const SUN_TOP: Rgb = rgb(255, 226, 110);
const SUN_LOW: Rgb = rgb(255, 70, 146);
const LAVENDER: Rgb = rgb(200, 180, 232);
const DUSK: Rgb = rgb(104, 66, 150);
const SAND: Rgb = rgb(255, 176, 86);
const PALE_PINK: Rgb = rgb(255, 204, 232);
const GOLD: Rgb = rgb(255, 230, 120);
const WHITE: Rgb = rgb(255, 246, 252);

/// The menu's size at 1080p; it grows with the screen.
pub(crate) const BASE: (f32, f32) = (920.0, 680.0);

/// The menu's size in pixels for a screen `height` pixels tall.
pub(crate) fn size(height: u32) -> (u32, u32, f32) {
    let s = (height as f32 / 1080.0).clamp(0.67, 2.0);
    ((BASE.0 * s).round() as u32, (BASE.1 * s).round() as u32, s)
}

fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// A picture being painted.
pub(crate) struct Img {
    pub w: usize,
    pub h: usize,
    pub px: Vec<Rgb>,
}

impl Img {
    fn new(w: u32, h: u32) -> Img {
        Img {
            w: w as usize,
            h: h as usize,
            px: vec![NIGHT_LOW; (w * h) as usize],
        }
    }

    /// `color` over the pixel by `alpha` (0 to 1).
    fn blend(&mut self, x: i64, y: i64, color: Rgb, alpha: f32) {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h || alpha <= 0.0 {
            return;
        }
        let p = &mut self.px[y as usize * self.w + x as usize];
        *p = mix(*p, color, alpha);
    }

    /// `color` added (light), by `alpha`.
    fn glow_add(&mut self, x: i64, y: i64, color: Rgb, alpha: f32) {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h || alpha <= 0.0 {
            return;
        }
        let p = &mut self.px[y as usize * self.w + x as usize];
        for i in 0..3 {
            p[i] = (p[i] + color[i] * alpha).min(255.0);
        }
    }

    /// The whole picture shading from `top` through `mid` to `low`.
    fn sky(&mut self, top: Rgb, mid: Rgb, low: Rgb) {
        for y in 0..self.h {
            let t = y as f32 / self.h as f32;
            let c = if t < 0.35 {
                mix(top, mid, t / 0.35)
            } else {
                mix(mid, low, (t - 0.35) / 0.65)
            };
            self.px[y * self.w..(y + 1) * self.w].fill(c);
        }
    }

    /// A box with round corners (radius `r`), soft-edged, shading from
    /// `left` to `right`, by `alpha`.
    #[allow(clippy::too_many_arguments)]
    fn round_box(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        r: f32,
        left: Rgb,
        right: Rgb,
        alpha: f32,
    ) {
        let r = r.min(w / 2.0).min(h / 2.0);
        for py in (y.floor() as i64)..((y + h).ceil() as i64) {
            for px in (x.floor() as i64)..((x + w).ceil() as i64) {
                let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
                let d = round_distance(cx - x, cy - y, w, h, r);
                let a = (0.5 - d).clamp(0.0, 1.0) * alpha;
                if a > 0.0 {
                    self.blend(px, py, mix(left, right, (cx - x) / w), a);
                }
            }
        }
    }

    /// The edge of a box with round corners, `width` wide.
    #[allow(clippy::too_many_arguments)]
    fn round_frame(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        r: f32,
        width: f32,
        color: Rgb,
        alpha: f32,
    ) {
        let r = r.min(w / 2.0).min(h / 2.0);
        for py in ((y - 1.0).floor() as i64)..((y + h + 1.0).ceil() as i64) {
            for px in ((x - 1.0).floor() as i64)..((x + w + 1.0).ceil() as i64) {
                let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
                let d = round_distance(cx - x, cy - y, w, h, r).abs();
                let a = (width / 2.0 + 0.5 - d).clamp(0.0, 1.0) * alpha;
                self.blend(px, py, color, a);
            }
        }
    }

    /// Soft light around a box's edge (a neon tube's glow).
    #[allow(clippy::too_many_arguments)]
    fn round_glow(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        r: f32,
        reach: f32,
        color: Rgb,
        alpha: f32,
    ) {
        let r = r.min(w / 2.0).min(h / 2.0);
        for py in ((y - reach).floor() as i64)..((y + h + reach).ceil() as i64) {
            for px in ((x - reach).floor() as i64)..((x + w + reach).ceil() as i64) {
                let (cx, cy) = (px as f32 + 0.5, py as f32 + 0.5);
                let d = round_distance(cx - x, cy - y, w, h, r).abs();
                if d < reach {
                    let f = 1.0 - d / reach;
                    self.glow_add(px, py, color, f * f * alpha);
                }
            }
        }
    }

    /// A line across, from `x0` to `x1`, `width` tall, shading.
    #[allow(clippy::too_many_arguments)]
    fn hline(&mut self, x0: f32, x1: f32, y: f32, width: f32, from: Rgb, to: Rgb, alpha: f32) {
        self.round_box(
            x0,
            y - width / 2.0,
            x1 - x0,
            width,
            width / 2.0,
            from,
            to,
            alpha,
        );
    }

    /// Text set at `(x, y)` (its top left), `px` tall; its width.
    #[allow(clippy::too_many_arguments)]
    fn text(
        &mut self,
        fonts: &mut dyn Fonts,
        x: f32,
        y: f32,
        px: f32,
        weight: Weight,
        color: Rgb,
        glow: Option<(Rgb, f32)>,
        text: &str,
    ) -> f32 {
        if text.is_empty() {
            return 0.0;
        }
        let (w, h, cover) = fonts.raster(text, px.round().max(6.0) as u32, weight);
        let (x0, y0) = (x.round() as i64, y.round() as i64);
        if let Some((color, reach)) = glow {
            let blurred = blur(&cover, w, h, reach.round().max(1.0) as usize);
            let pad = reach.round().max(1.0) as i64 * 2;
            let bw = w + 2 * pad as usize;
            for (i, a) in blurred.iter().enumerate() {
                let (bx, by) = ((i % bw) as i64, (i / bw) as i64);
                self.glow_add(x0 + bx - pad, y0 + by - pad, color, a * 0.9);
            }
        }
        for row in 0..h {
            for col in 0..w {
                let a = f32::from(cover[row * w + col]) / 255.0;
                self.blend(x0 + col as i64, y0 + row as i64, color, a);
            }
        }
        w as f32
    }

    /// The picture in bytes, red, green, blue.
    pub(crate) fn bytes(&self) -> Vec<[u8; 3]> {
        self.px
            .iter()
            .map(|p| p.map(|c| c.round().clamp(0.0, 255.0) as u8))
            .collect()
    }
}

/// How far `(x, y)` is outside a `w`x`h` box with corners of radius `r`
/// (negative: inside).
fn round_distance(x: f32, y: f32, w: f32, h: f32, r: f32) -> f32 {
    let qx = (x - w / 2.0).abs() - (w / 2.0 - r);
    let qy = (y - h / 2.0).abs() - (h / 2.0 - r);
    let outside = qx.max(0.0).hypot(qy.max(0.0));
    outside + qx.max(qy).min(0.0) - r
}

/// `cover` (w x h) spread by a box blur of radius `r`, twice, on a canvas
/// `2r` wider on each side; 0 to 1.
fn blur(cover: &[u8], w: usize, h: usize, r: usize) -> Vec<f32> {
    let pad = 2 * r;
    let (bw, bh) = (w + 2 * pad, h + 2 * pad);
    let mut a = vec![0.0_f32; bw * bh];
    for y in 0..h {
        for x in 0..w {
            a[(y + pad) * bw + x + pad] = f32::from(cover[y * w + x]) / 255.0;
        }
    }
    let n = (2 * r + 1) as f32;
    for _ in 0..2 {
        let mut b = vec![0.0_f32; bw * bh];
        for y in 0..bh {
            let mut sum = 0.0;
            for x in 0..bw + r {
                if x < bw {
                    sum += a[y * bw + x];
                }
                if x > 2 * r {
                    sum -= a[y * bw + x - 2 * r - 1];
                }
                if x >= r && x - r < bw {
                    b[y * bw + x - r] = sum / n;
                }
            }
        }
        let mut c = vec![0.0_f32; bw * bh];
        for x in 0..bw {
            let mut sum = 0.0;
            for y in 0..bh + r {
                if y < bh {
                    sum += b[y * bw + x];
                }
                if y > 2 * r {
                    sum -= b[(y - 2 * r - 1) * bw + x];
                }
                if y >= r && y - r < bh {
                    c[(y - r) * bw + x] = sum / n;
                }
            }
        }
        a = c;
    }
    a
}

/// At most `max` characters, `...` when cut.
fn fit(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// The menu for a screen `height` pixels tall.
pub(crate) fn paint(page: &Page, height: u32, fonts: &mut dyn Fonts) -> Img {
    let (w, h, s) = size(height);
    let (wf, hf) = (w as f32, h as f32);
    let mut c = Img::new(w, h);
    c.sky(NIGHT_TOP, NIGHT_MID, NIGHT_LOW);

    // A sun going down behind the title, cut by the sea's lines.
    let (sx, sy, sr) = (wf - 170.0 * s, 128.0 * s, 74.0 * s);
    let horizon = 116.0 * s;
    for py in ((sy - sr).floor() as i64).max(0)..(horizon.ceil() as i64) {
        let y = py as f32 + 0.5;
        // Gaps that widen towards the sea.
        let below = y - (sy - 0.45 * sr);
        if below > 0.0 {
            let band = 10.0 * s;
            let gap = (below / band).floor() * 1.1 * s + 1.5 * s;
            if below % band < gap {
                continue;
            }
        }
        let color = mix(SUN_TOP, SUN_LOW, (y - (sy - sr)) / sr);
        let fade = ((horizon - y) / (2.0 * s)).clamp(0.0, 1.0);
        for px in ((sx - sr - 1.0).floor() as i64)..((sx + sr + 1.0).ceil() as i64) {
            let d = (px as f32 + 0.5 - sx).hypot(y - sy) - sr;
            c.blend(px, py, color, (0.5 - d).clamp(0.0, 1.0) * fade);
        }
    }
    // The sun's light on the sky.
    for py in 0..(horizon as i64) {
        for px in ((sx - 3.0 * sr) as i64).max(0)..((sx + 3.0 * sr) as i64).min(w as i64) {
            let d = (px as f32 - sx).hypot(py as f32 - sy) / (2.6 * sr);
            if d < 1.0 {
                c.glow_add(px, py, SUN_LOW, (1.0 - d) * (1.0 - d) * 0.10);
            }
        }
    }
    // Stars.
    for (x, y, b) in [
        (360.0, 24.0, 1.0),
        (440.0, 66.0, 0.6),
        (548.0, 18.0, 0.8),
        (620.0, 52.0, 0.5),
        (860.0, 30.0, 0.7),
        (712.0, 16.0, 0.9),
        (494.0, 40.0, 0.5),
        (660.0, 86.0, 0.4),
    ] {
        c.round_box(x * s, y * s, 2.0 * s, 2.0 * s, s, WHITE, WHITE, b);
    }
    // The sea's horizon.
    c.hline(0.0, wf, horizon, 2.0 * s, PINK, CYAN, 1.0);
    c.round_glow(0.0, horizon - s, wf, 2.0 * s, s, 10.0 * s, PINK, 0.35);

    // The name, a neon sign.
    let tx = 34.0 * s;
    c.text(
        fonts,
        tx,
        2.0 * s,
        60.0 * s,
        Weight::Title,
        WHITE,
        Some((PINK, 7.0 * s)),
        "VIctory",
    );
    c.text(
        fonts,
        tx + 4.0 * s,
        80.0 * s,
        15.0 * s,
        Weight::Bold,
        CYAN,
        Some((CYAN, 3.0 * s)),
        "MENU DE L'ÉMULATEUR  ·  F1",
    );

    // Tabs.
    let (tab_y, tab_h, tab_px) = (134.0 * s, 36.0 * s, 18.0 * s);
    let mut x = 26.0 * s;
    for (i, tab) in page.tabs.iter().enumerate() {
        let (tw, _, _) = fonts.raster(tab, tab_px.round() as u32, Weight::Bold);
        let bw = tw as f32 + 36.0 * s;
        if i == page.tab {
            c.round_glow(x, tab_y, bw, tab_h, tab_h / 2.0, 12.0 * s, PINK, 0.45);
            c.round_box(x, tab_y, bw, tab_h, tab_h / 2.0, PINK, DEEP_PINK, 1.0);
            c.text(
                fonts,
                x + 18.0 * s,
                tab_y + 7.0 * s,
                tab_px,
                Weight::Bold,
                WHITE,
                None,
                tab,
            );
        } else {
            c.round_box(
                x,
                tab_y,
                bw,
                tab_h,
                tab_h / 2.0,
                rgb(30, 14, 54),
                rgb(30, 14, 54),
                0.9,
            );
            c.round_frame(x, tab_y, bw, tab_h, tab_h / 2.0, 1.2 * s, DUSK, 1.0);
            c.text(
                fonts,
                x + 18.0 * s,
                tab_y + 7.0 * s,
                tab_px,
                Weight::Bold,
                LAVENDER,
                None,
                tab,
            );
        }
        x += bw + 10.0 * s;
    }

    // Rows, scrolled to keep the cursor in sight.
    let (top, row_h, px) = (190.0 * s, 28.0 * s, 19.0 * s);
    let bottom = hf - 80.0 * s;
    let seen = (((bottom - top) / row_h) as usize).max(1);
    let first = page
        .selected
        .saturating_sub(seen.saturating_sub(2))
        .min(page.items.len().saturating_sub(seen));
    let (left, right) = (44.0 * s, wf - 48.0 * s);
    let chars = ((right - left) / (px * 0.5)) as usize;
    for (n, item) in page.items.iter().enumerate().skip(first).take(seen) {
        let y = top + (n - first) as f32 * row_h;
        let selected = n == page.selected && item.kind != Kind::Header;
        if selected {
            c.round_glow(
                28.0 * s,
                y,
                wf - 64.0 * s,
                row_h - 2.0 * s,
                8.0 * s,
                8.0 * s,
                PINK,
                0.25,
            );
            c.round_box(
                28.0 * s,
                y,
                wf - 64.0 * s,
                row_h - 2.0 * s,
                8.0 * s,
                rgb(196, 34, 128),
                rgb(60, 16, 70),
                0.95,
            );
            c.round_box(
                28.0 * s,
                y + 4.0 * s,
                4.0 * s,
                row_h - 10.0 * s,
                2.0 * s,
                CYAN,
                CYAN,
                1.0,
            );
        }
        let ty = y + (row_h - 2.0 * s - px * 1.2) / 2.0;
        let value = match (item.kind, selected) {
            (Kind::Setting, true) if !item.value.is_empty() => format!("‹  {}  ›", item.value),
            _ => item.value.clone(),
        };
        let value = fit(&value, chars / 2);
        let label = fit(&item.label, chars.saturating_sub(value.chars().count() + 4));
        match item.kind {
            Kind::Header => {
                let lw = c.text(
                    fonts,
                    left,
                    ty + 3.0 * s,
                    15.0 * s,
                    Weight::Bold,
                    SAND,
                    None,
                    &label,
                );
                let lx = left + lw + 14.0 * s;
                if lx < right {
                    c.hline(lx, right, ty + 13.0 * s, 1.2 * s, SAND, NIGHT_MID, 0.8);
                }
            }
            kind => {
                let (lc, vc) = match (kind, selected) {
                    (_, true) => (WHITE, GOLD),
                    (Kind::Button, false) => (PALE_PINK, CYAN),
                    (Kind::Text, false) => (LAVENDER, WHITE),
                    _ => (WHITE, CYAN),
                };
                let mut lx = left;
                if kind == Kind::Button {
                    c.round_box(
                        left,
                        ty + 6.0 * s,
                        9.0 * s,
                        9.0 * s,
                        4.5 * s,
                        if selected { GOLD } else { PINK },
                        PINK,
                        1.0,
                    );
                    lx += 20.0 * s;
                }
                c.text(fonts, lx, ty, px, Weight::Regular, lc, None, &label);
                if !value.is_empty() {
                    let (vw, _, _) = fonts.raster(&value, px.round() as u32, Weight::Bold);
                    c.text(
                        fonts,
                        right - vw as f32,
                        ty,
                        px,
                        Weight::Bold,
                        vc,
                        None,
                        &value,
                    );
                }
            }
        }
    }
    // Where the rows shown are in all of them.
    if page.items.len() > seen {
        let track = seen as f32 * row_h;
        let bar = (track * seen as f32 / page.items.len() as f32).max(16.0 * s);
        let at = top + (track - bar) * first as f32 / (page.items.len() - seen) as f32;
        c.round_box(
            wf - 22.0 * s,
            top,
            4.0 * s,
            track,
            2.0 * s,
            rgb(48, 26, 76),
            rgb(48, 26, 76),
            1.0,
        );
        c.round_box(wf - 22.0 * s, at, 4.0 * s, bar, 2.0 * s, PINK, PINK, 1.0);
    }

    // The last change and the keys.
    c.hline(
        26.0 * s,
        wf - 26.0 * s,
        hf - 72.0 * s,
        1.2 * s,
        DUSK,
        PINK,
        1.0,
    );
    let chars = ((wf - 64.0 * s) / (17.0 * s * 0.5)) as usize;
    if !page.note.is_empty() {
        c.text(
            fonts,
            34.0 * s,
            hf - 62.0 * s,
            17.0 * s,
            Weight::Bold,
            GOLD,
            None,
            &fit(&page.note, chars),
        );
    }
    c.text(
        fonts,
        34.0 * s,
        hf - 34.0 * s,
        16.0 * s,
        Weight::Regular,
        LAVENDER,
        None,
        &fit(&page.help, chars),
    );

    // A neon frame.
    c.round_glow(
        3.0 * s,
        3.0 * s,
        wf - 6.0 * s,
        hf - 6.0 * s,
        16.0 * s,
        9.0 * s,
        PINK,
        0.5,
    );
    c.round_frame(
        3.0 * s,
        3.0 * s,
        wf - 6.0 * s,
        hf - 6.0 * s,
        16.0 * s,
        2.0 * s,
        PINK,
        1.0,
    );
    c.round_frame(
        8.0 * s,
        8.0 * s,
        wf - 16.0 * s,
        hf - 16.0 * s,
        12.0 * s,
        1.0 * s,
        CYAN,
        0.45,
    );
    // Outside the round corners: dark, as the picture behind is not seen.
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every character is a filled box as wide as half its height.
    struct Boxes;

    impl Fonts for Boxes {
        fn raster(&mut self, text: &str, px: u32, _: Weight) -> (usize, usize, Vec<u8>) {
            let (w, h) = (text.chars().count() * px as usize / 2, px as usize);
            (w, h, vec![255; w * h])
        }
    }

    #[test]
    fn a_menu_is_painted_at_the_screens_size() {
        let page = Page {
            tabs: vec!["GRAPHISMES".into(), "CLAVIER".into()],
            tab: 0,
            items: (0..40)
                .map(|i| crate::menu::Item::setting(format!("LIGNE {i}"), "OUI"))
                .collect(),
            selected: 30,
            help: "AIDE".into(),
            note: "NOTE".into(),
        };
        for height in [720, 1080, 2160] {
            let img = paint(&page, height, &mut Boxes);
            let (w, h, _) = size(height);
            assert_eq!((img.w, img.h), (w as usize, h as usize));
            assert!(
                img.px
                    .iter()
                    .all(|p| p.iter().all(|c| (0.0..=255.0).contains(c)))
            );
        }
    }

    #[test]
    fn round_boxes_have_soft_edges() {
        assert!(round_distance(5.0, 5.0, 10.0, 10.0, 2.0) < 0.0);
        assert!((round_distance(0.0, 5.0, 10.0, 10.0, 2.0)).abs() < 1e-4);
        assert!(round_distance(0.0, 0.0, 10.0, 10.0, 3.0) > 0.0);
    }
}
