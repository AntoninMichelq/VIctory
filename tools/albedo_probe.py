"""Probe a packed G-buffer dump (VICTORY_DUMP_FULL=1 PNGs, 8-bit RGB/RGBA,
filter 0 rows as victory writes them) for how its colour is stored.

Usage: python albedo_probe.py target.png [more.png ...]
Prints per-channel stats, per-phase means (x%2, y%2), neighbour
correlations, raw 4x4 blocks at a few spots, and ranks YCoCg-style decodes
(luma channel, chroma channel, interleave pattern, Co/Cg order) by how smooth
the decoded colours are. Also writes <name>-best.png (the best decode) next
to the input, for a person to look at. Standard library only.
"""

import struct
import sys
import zlib
from itertools import product


def read_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, idat, w = 8, b"", None
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, colour = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and colour in (2, 6), "8-bit RGB/RGBA only"
            ch = 4 if colour == 6 else 3
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    stride = w * ch
    rows, prev = [], bytearray(stride)
    for y in range(h):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - ch] if i >= ch else 0
            b = prev[i]
            c = prev[i - ch] if i >= ch else 0
            if f == 1:
                line[i] = (line[i] + a) & 255
            elif f == 2:
                line[i] = (line[i] + b) & 255
            elif f == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        rows.append(bytes(line))
        prev = line
    return w, h, ch, rows


def write_png(path, w, h, pixels):
    raw = b"".join(b"\x00" + bytes(pixels[y * w * 3:(y + 1) * w * 3]) for y in range(h))

    def chunk(kind, body):
        c = struct.pack(">I", len(body)) + kind + body
        return c + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)


def probe(path):
    w, h, ch, rows = read_png(path)
    print(f"== {path}: {w}x{h}, {ch} channels")

    def px(x, y, c):
        return rows[y][x * ch + c]

    # Sample a grid (every 3rd pixel pair) to keep it fast.
    ys = range(0, h - 2, 3)
    xs = range(0, w - 2, 3)
    alternating = []
    for c in range(ch):
        vals = [px(x, y, c) for y in ys for x in xs]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        hist = [0] * 8
        for v in vals:
            hist[v >> 5] += 1
        phase = {}
        for py, pxp in product((0, 1), (0, 1)):
            pv = [px(x + pxp, y + py, c) for y in range(0, h - 2, 6) for x in range(0, w - 2, 6)]
            phase[(pxp, py)] = sum(pv) / len(pv)
        dx = sum(abs(px(x, y, c) - px(x + 1, y, c)) for y in ys for x in xs) / len(vals)
        dy = sum(abs(px(x, y, c) - px(x, y + 1, c)) for y in ys for x in xs) / len(vals)
        d2 = sum(abs(px(x, y, c) - px(x + 2, y, c)) for y in ys for x in xs) / len(vals)
        print(f" ch{c}: mean {mean:6.1f} std {std:5.1f} hist/32 {hist}")
        print(f"      phase means (x%2,y%2) {({k: round(v, 1) for k, v in phase.items()})}")
        print(f"      mean |diff| right {dx:5.1f} down {dy:5.1f} right2 {d2:5.1f}")
        # A channel whose neighbours differ much more than pixels two apart
        # interleaves two quantities.
        if dx > 1.5 * d2 + 2 or dy > 1.5 * d2 + 2:
            alternating.append(c)
    print(f" interleaved channels: {alternating}")
    for sx, sy in ((w // 2, h // 2), (w // 3, 2 * h // 3), (2 * w // 3, h // 2)):
        print(f" raw 4x4 at ({sx},{sy}):")
        for y in range(sy, sy + 4):
            print("   " + " ".join("/".join(f"{px(x, y, c):3d}" for c in range(ch)) for x in range(sx, sx + 4)))

    # YCoCg hypotheses: luma channel l, chroma channel k, interleave
    # pattern, which chroma comes first.
    patterns = {
        "checker": lambda x, y: (x + y) & 1,
        "columns": lambda x, y: x & 1,
        "rows": lambda x, y: y & 1,
        "quads": lambda x, y: ((x >> 1) + (y >> 1)) & 1,
    }

    def decode(l, k, pat, order, x, y):
        f = patterns[pat]
        here = px(x, y, k) / 255 - 0.5
        if pat == "rows":
            other = px(x, y + 1, k) / 255 - 0.5
        elif pat == "quads":
            other = px(x + 2, y, k) / 255 - 0.5
        else:
            other = px(x + 1, y, k) / 255 - 0.5
        first = (f(x, y) == 0) == (order == 0)
        co, cg = (here, other) if first else (other, here)
        yy = px(x, y, l) / 255
        r, g, b = yy + co - cg, yy + cg, yy - co - cg
        return [min(255, max(0, int(v * 255))) for v in (r, g, b)]

    results = []
    sx = range(4, w - 6, 29)
    sy = range(4, h - 6, 29)
    for l, k, pat, order in product(range(min(ch, 3)), range(ch), patterns, (0, 1)):
        if l == k or (alternating and k not in alternating) or l in alternating:
            continue
        tv = 0
        n = 0
        seen = []
        for y in sy:
            for x in sx:
                a = decode(l, k, pat, order, x, y)
                b = decode(l, k, pat, order, x + 1, y)
                c = decode(l, k, pat, order, x, y + 1)
                tv += sum(abs(a[i] - b[i]) + abs(a[i] - c[i]) for i in range(3))
                seen.append(sum(a))
                n += 1
        # Smooth relative to how much the colours vary: a flat decode
        # (a constant channel) is no answer.
        mean = sum(seen) / n
        spread = (sum((v - mean) ** 2 for v in seen) / n) ** 0.5
        if spread < 15:
            continue
        results.append((tv / n / spread, l, k, pat, order))
    results.sort()
    # Baseline: the raw RGB's own smoothness.
    base = 0
    n = 0
    for y in sy:
        for x in sx:
            base += sum(abs(px(x, y, i) - px(x + 1, y, i)) + abs(px(x, y, i) - px(x, y + 1, i)) for i in range(3))
            n += 1
    print(f" raw RGB roughness {base / n:.1f}; best YCoCg decodes (roughness per spread, luma ch, chroma ch, pattern, order):")
    for r in results[:6]:
        print(f"   {r[0]:6.3f}  luma ch{r[1]} chroma ch{r[2]} {r[3]} order {r[4]}")
    if not results:
        print(" no YCoCg decode varies enough")
        return
    _, l, k, pat, order = results[0]
    out = []
    for y in range(h):
        for x in range(w):
            out += decode(l, k, pat, order, min(x, w - 3), min(y, h - 3))
    best = path.rsplit(".", 1)[0] + "-best.png"
    write_png(best, w, h, out)
    print(f" wrote {best}")


for p in sys.argv[1:]:
    probe(p)
