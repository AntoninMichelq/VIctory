//! Text for the F1 menu (`menu_paint::Fonts`) set by GDI in Windows' own
//! Segoe UI, smoothed (grey anti-aliasing), white on black: the coverage.

use std::collections::HashMap;

use windows::Win32::Foundation::{COLORREF, SIZE};
use windows::Win32::Graphics::Gdi::*;
use windows::core::PCWSTR;

use super::menu_paint::{Fonts, Weight};

pub(super) struct Gdi {
    dc: HDC,
    fonts: HashMap<(u32, Weight), HFONT>,
}

impl Gdi {
    pub(super) fn new() -> Option<Gdi> {
        // SAFETY: a memory DC of our own, deleted on drop.
        let dc = unsafe { CreateCompatibleDC(None) };
        (!dc.is_invalid()).then(|| Gdi {
            dc,
            fonts: HashMap::new(),
        })
    }

    fn font(&mut self, px: u32, weight: Weight) -> HFONT {
        *self.fonts.entry((px, weight)).or_insert_with(|| {
            let (face, bold, italic) = match weight {
                Weight::Regular => ("Segoe UI", 400, false),
                Weight::Bold => ("Segoe UI Semibold", 600, false),
                Weight::Title => ("Segoe UI Black", 900, true),
            };
            let face: Vec<u16> = face.encode_utf16().chain([0]).collect();
            // SAFETY: a plain font creation; the name lives through the call.
            unsafe {
                CreateFontW(
                    -(px as i32),
                    0,
                    0,
                    0,
                    bold,
                    u32::from(italic),
                    0,
                    0,
                    DEFAULT_CHARSET,
                    OUT_TT_PRECIS,
                    CLIP_DEFAULT_PRECIS,
                    ANTIALIASED_QUALITY,
                    (DEFAULT_PITCH.0 | FF_SWISS.0) as u32,
                    PCWSTR(face.as_ptr()),
                )
            }
        })
    }
}

impl Drop for Gdi {
    fn drop(&mut self) {
        // SAFETY: ours, no longer selected anywhere once the DC goes.
        unsafe {
            let _ = DeleteDC(self.dc);
            for font in self.fonts.values() {
                let _ = DeleteObject((*font).into());
            }
        }
    }
}

impl Fonts for Gdi {
    fn raster(&mut self, text: &str, px: u32, weight: Weight) -> (usize, usize, Vec<u8>) {
        let font = self.font(px, weight);
        let wide: Vec<u16> = text.encode_utf16().collect();
        let dc = self.dc;
        // SAFETY: our DC, a DIB of our own for the time of the call, read
        // within its size.
        unsafe {
            let old_font = SelectObject(dc, font.into());
            let mut size = SIZE::default();
            let _ = GetTextExtentPoint32W(dc, &wide, &mut size);
            // Slanted faces lean out of their box: room on the right.
            let lean = if weight == Weight::Title {
                px as i32 / 4
            } else {
                2
            };
            let (w, h) = ((size.cx + lean).max(1), size.cy.max(1));
            let info = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: w,
                    biHeight: -h,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut bits = std::ptr::null_mut();
            let Ok(dib) = CreateDIBSection(Some(dc), &info, DIB_RGB_COLORS, &mut bits, None, 0)
            else {
                SelectObject(dc, old_font);
                return (0, 0, Vec::new());
            };
            let old_bitmap = SelectObject(dc, dib.into());
            // A fresh DIB is black.
            SetBkMode(dc, TRANSPARENT);
            SetTextColor(dc, COLORREF(0x00FF_FFFF));
            let _ = TextOutW(dc, 0, 0, &wide);
            let _ = GdiFlush();
            let n = (w * h) as usize;
            let pixels = std::slice::from_raw_parts(bits as *const [u8; 4], n);
            let cover = pixels.iter().map(|p| p[0].max(p[1]).max(p[2])).collect();
            SelectObject(dc, old_bitmap);
            SelectObject(dc, old_font);
            let _ = DeleteObject(dib.into());
            (w as usize, h as usize, cover)
        }
    }
}
