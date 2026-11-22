#!/usr/bin/env python3
"""Oracle: compare the Parallax JPEG decoder against PIL (libjpeg-turbo) pixel by pixel.

Test-only tooling: Python never runs inside Parallax. Run from the project root with -I:

    python3 -I tools/oracle/jpeg_vs_pil.py --dump build-img/bin/plx_decode_dump \
        --work /tmp/scratch/img [--lfw data/lfw] [--synth data/lfw/X/X_0001.jpg] [--limit N]

--lfw    decodes every *.jpg under the directory in one plx_decode_dump run and compares.
--synth  derives a set of non-LFW test JPEGs from one source image (progressive, grayscale,
         4:4:4 / 4:2:2 / 4:4:0 / 4:1:1 / 3x1, restart intervals, odd sizes, Adobe RGB, 16-bit DQT,
         SOF1, fill bytes, split/duplicated tables, missing DHT, non-interleaved scans) and
         compares each one. Exit status is non-zero if any image differs.
--png    also has plx_decode_dump write every synthetic image (and the first --png-lfw LFW
         images) as PNG: native channels, +alpha and 16-bit gray. PIL must read each back
         exactly; the LFW subset also reports the compression ratio.
"""
import argparse
import io
import os
import subprocess
import sys

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------------

def read_plxt(path):
    with open(path, "rb") as f:
        hdr = f.read(64)
        assert hdr[:4] == b"PLXT", "bad magic"
        version, dtype, ndim = np.frombuffer(hdr[4:16], "<u4")
        assert version == 1 and dtype == 0, "expected a u8 tensor"
        dims = [int(d) for d in np.frombuffer(hdr[16:64], "<i8")[:ndim]]
        data = np.frombuffer(f.read(), np.uint8)
    return data.reshape(dims)


def pil_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def run_dump(dump, paths, work, tag, png_dir=None):
    lst = os.path.join(work, tag + ".txt")
    out = os.path.join(work, tag + ".plxt")
    with open(lst, "w") as f:
        f.write("\n".join(paths) + "\n")
    cmd = [dump, "--list", lst, "--out", out] + (["--png", png_dir] if png_dir else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip()
    return read_plxt(out), r.stderr.strip()


def compare(mine, paths):
    """Returns (differing images, max abs diff, mean abs diff over all samples, worst path)."""
    ndiff, maxd, total, count, worst = 0, 0, 0.0, 0, None
    for i, p in enumerate(paths):
        ref = pil_rgb(p).astype(np.int16)
        d = np.abs(mine[i].astype(np.int16) - ref)
        m = int(d.max())
        if m:
            ndiff += 1
        if m > maxd:
            maxd, worst = m, p
        total += float(d.sum())
        count += d.size
    return ndiff, maxd, total / max(count, 1), worst


def check_pngs(png_dir, mine, paths):
    """mine: [N, H, W, 3] from the dump; returns (bad files, png bytes, raw bytes)."""
    bad, png_bytes, raw_bytes = 0, 0, 0
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            gray = im.mode == "L"
        rgb = mine[i].astype(np.int32)
        native = rgb[..., :1] if gray else rgb
        want = {
            f"{i:06d}.png": native,
            f"{i:06d}_a.png": np.concatenate([native, native[..., :1] ^ 0x5A], axis=2),
            f"{i:06d}_16.png": (native[..., :1] << 8) | native[..., -1:],
        }
        for name, ref in want.items():
            path = os.path.join(png_dir, name)
            png_bytes += os.path.getsize(path)
            raw_bytes += ref.size * (2 if name.endswith("_16.png") else 1)
            with Image.open(path) as im:
                im.load()
                got = np.asarray(im).astype(np.int32)
            got = got.reshape(ref.shape) if got.size == ref.size else None
            if got is None or not np.array_equal(got, ref):
                print(f"  PNG mismatch: {path}")
                bad += 1
    return bad, png_bytes, raw_bytes


# ---------------------------------------------------------------------------------------------
# A minimal baseline encoder, for stream structures libjpeg's encoder will not emit
# (non-interleaved sequential scans, arbitrary sampling factors with explicit planes, ...).
# ---------------------------------------------------------------------------------------------

ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34,
          27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44,
          51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]

DC_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
DC_VALS = list(range(12))
AC_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d]
AC_VALS = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa]


def huff_codes(bits, vals):
    codes, code, k = {}, 0, 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            codes[vals[k]] = (code, length)
            code += 1
            k += 1
        code <<= 1
    return codes


DC_CODES, AC_CODES = huff_codes(DC_BITS, DC_VALS), huff_codes(AC_BITS, AC_VALS)
_n = np.arange(8)
DCT_M = np.sqrt(2 / 8) * np.cos((2 * _n[None, :] + 1) * _n[:, None] * np.pi / 16)
DCT_M[0] /= np.sqrt(2)


class BitWriter:
    def __init__(self):
        self.out, self.acc, self.n = bytearray(), 0, 0

    def put(self, v, n):
        for i in range(n - 1, -1, -1):
            self.acc = (self.acc << 1) | ((v >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF:
                    self.out.append(0)
                self.acc, self.n = 0, 0

    def flush(self):
        if self.n:
            self.put((1 << (8 - self.n)) - 1, 8 - self.n)


def magnitude(v):
    s = int(abs(v)).bit_length()
    return s, (v if v >= 0 else v + (1 << s) - 1)


def encode_block(bw, coefs, pred):
    diff = int(coefs[0]) - pred
    s, bitsv = magnitude(diff)
    bw.put(*DC_CODES[s])
    bw.put(bitsv, s)
    run = 0
    for k in range(1, 64):
        v = int(coefs[ZIGZAG[k]])
        if v == 0:
            run += 1
            continue
        while run > 15:
            bw.put(*AC_CODES[0xF0])
            run -= 16
        s, bitsv = magnitude(v)
        bw.put(*AC_CODES[(run << 4) | s])
        bw.put(bitsv, s)
        run = 0
    if run:
        bw.put(*AC_CODES[0x00])
    return int(coefs[0])


def seg(marker, payload):
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def encode_planes(planes, factors, ids, w, h, q16=False, interleaved=True, restart=0,
                  adobe=None, sof=0xC0, qscale=1):
    """planes: per-component uint8 arrays already at their sampled size."""
    hmax, vmax = max(f[0] for f in factors), max(f[1] for f in factors)
    mcux, mcuy = -(-w // (8 * hmax)), -(-h // (8 * vmax))
    base = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24,
                     40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103,
                     77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72,
                     92, 95, 98, 112, 100, 103, 99])
    qt = np.maximum(1, (base * qscale) // 4).astype(np.int64)
    blocks = []
    for plane, (fh, fv) in zip(planes, factors):
        bw, bh = mcux * fh, mcuy * fv
        p = np.pad(plane.astype(np.float64) - 128,
                   ((0, bh * 8 - plane.shape[0]), (0, bw * 8 - plane.shape[1])), mode="edge")
        b = p.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
        coef = np.einsum("ux,ijxy,vy->ijuv", DCT_M, b, DCT_M)
        blocks.append(np.round(coef.reshape(bh, bw, 64) / qt).astype(np.int64))

    out = bytearray(b"\xFF\xD8")
    if adobe is not None:
        out += seg(0xEE, b"Adobe" + bytes([0, 100, 0, 0, 0, 0, adobe]))
    if q16:
        out += seg(0xDB, bytes([0x10]) + b"".join(int(qt[ZIGZAG[k]]).to_bytes(2, "big") for k in range(64)))
    else:
        out += seg(0xDB, bytes([0x00]) + bytes(int(qt[ZIGZAG[k]]) for k in range(64)))
    sofp = bytes([8]) + h.to_bytes(2, "big") + w.to_bytes(2, "big") + bytes([len(planes)])
    for cid, (fh, fv) in zip(ids, factors):
        sofp += bytes([cid, (fh << 4) | fv, 0])
    out += seg(sof, sofp)
    out += seg(0xC4, bytes([0x00]) + bytes(DC_BITS) + bytes(DC_VALS))
    out += seg(0xC4, bytes([0x10]) + bytes(AC_BITS) + bytes(AC_VALS))
    if restart:
        out += seg(0xDD, restart.to_bytes(2, "big"))

    groups = [list(range(len(planes)))] if interleaved else [[i] for i in range(len(planes))]
    for grp in groups:
        sos = bytes([len(grp)]) + b"".join(bytes([ids[i], 0x00]) for i in grp) + bytes([0, 63, 0])
        out += seg(0xDA, sos)
        bw_ = BitWriter()
        preds = [0] * len(planes)
        if len(grp) == 1:
            i = grp[0]
            fh, fv = factors[i]
            cw, ch = -(-w * fh // hmax), -(-h * fv // vmax)
            mcus = [[(i, by, bx)] for by in range(-(-ch // 8)) for bx in range(-(-cw // 8))]
        else:
            mcus = [[(i, my * factors[i][1] + y, mx * factors[i][0] + x)
                     for i in grp for y in range(factors[i][1]) for x in range(factors[i][0])]
                    for my in range(mcuy) for mx in range(mcux)]
        rst = 0
        for n, mcu in enumerate(mcus):
            if restart and n and n % restart == 0:
                bw_.flush()
                bw_.out += bytes([0xFF, 0xD0 + rst])
                rst = (rst + 1) & 7
                preds = [0] * len(planes)
            for i, by, bx in mcu:
                preds[i] = encode_block(bw_, blocks[i][by, bx], preds[i])
        bw_.flush()
        out += bw_.out
    out += b"\xFF\xD9"
    return bytes(out)


def rgb_to_ycc(rgb):
    r, g, b = [rgb[..., k].astype(np.float64) for k in range(3)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return [np.clip(np.round(c), 0, 255).astype(np.uint8) for c in (y, cb, cr)]


def subsample(plane, fh, fv, hmax, vmax, w, h):
    cw, ch = -(-w * fh // hmax), -(-h * fv // vmax)
    ys = (np.arange(ch) * vmax) // fv
    xs = (np.arange(cw) * hmax) // fh
    return plane[np.ix_(ys, xs)]


def custom_jpeg(rgb, factors, **kw):
    h, w = rgb.shape[:2]
    hmax, vmax = max(f[0] for f in factors), max(f[1] for f in factors)
    if kw.get("adobe") == 0:
        comps, ids = [rgb[..., k] for k in range(3)], [ord("R"), ord("G"), ord("B")]
    else:
        comps, ids = rgb_to_ycc(rgb), [1, 2, 3]
    planes = [subsample(c, fh, fv, hmax, vmax, w, h) for c, (fh, fv) in zip(comps, factors)]
    return encode_planes(planes, factors, ids, w, h, **kw)


# ---------------------------------------------------------------------------------------------
# Byte-level surgery on libjpeg output
# ---------------------------------------------------------------------------------------------

def segments(data):
    """Splits a JPEG into (marker, payload) up to SOS and returns (segments, scan_and_rest)."""
    pos, segs = 2, []
    while True:
        assert data[pos] == 0xFF
        m = data[pos + 1]
        n = int.from_bytes(data[pos + 2:pos + 4], "big")
        if m == 0xDA:
            return segs, data[pos:]
        segs.append((m, data[pos + 4:pos + 2 + n]))
        pos += 2 + n


def assemble(segs, rest):
    return b"\xFF\xD8" + b"".join(seg(m, p) for m, p in segs) + rest


def split_dht(data):
    segs, rest = segments(data)
    out = []
    for m, p in segs:
        if m != 0xC4:
            out.append((m, p))
            continue
        while p:
            n = 17 + sum(p[1:17])
            out.append((0xC4, p[:n]))
            p = p[n:]
    return assemble(out, rest)


def merge_dqt_with_decoy(data):
    """One DQT segment holding all tables, preceded by a decoy definition that must be replaced."""
    segs, rest = segments(data)
    dqt = b"".join(p for m, p in segs if m == 0xDB)
    decoy = bytes([0x00]) + bytes([255] * 64)
    out = [(0xDB, decoy)] + [(m, p) for m, p in segs if m != 0xDB]
    out.insert(1, (0xDB, dqt))
    return assemble(out, rest)


def add_fill_bytes(data):
    """Inserts 0xFF fill bytes before every marker after SOI, including inside scans before RSTn."""
    out, i = bytearray(data[:2]), 2
    while i < len(data):
        b = data[i]
        if b == 0xFF and i + 1 < len(data) and data[i + 1] not in (0x00, 0xFF):
            out += b"\xFF\xFF\xFF"
            out.append(data[i + 1])
            i += 2
            continue
        out.append(b)
        i += 1
    return bytes(out)


def strip_dht(data):
    segs, rest = segments(data)
    return assemble([(m, p) for m, p in segs if m != 0xC4], rest)


def to_sof1(data):
    segs, rest = segments(data)
    return assemble([(0xC1 if m == 0xC0 else m, p) for m, p in segs], rest)


# ---------------------------------------------------------------------------------------------
# Synthetic set
# ---------------------------------------------------------------------------------------------

def pil_jpeg(im, **kw):
    buf = io.BytesIO()
    im.save(buf, "JPEG", **kw)
    return buf.getvalue()


def imagemagick_jpeg(png_path, out_path, *args):
    r = subprocess.run(["convert", png_path, *args, out_path], capture_output=True)
    return r.returncode == 0


def build_synth(src, work):
    os.makedirs(work, exist_ok=True)
    with Image.open(src) as im:
        base = im.convert("RGB")
    cases = {}
    sizes = [(1, 1), (2, 2), (3, 5), (17, 33), (33, 17), (250, 1), (1, 250), (129, 65), (250, 250)]
    for w, h in sizes:
        im = base.resize((w, h), Image.BICUBIC) if (w, h) != base.size else base
        tag = f"{w}x{h}"
        for ss, name in ((0, "444"), (1, "422"), (2, "420")):
            cases[f"base_{name}_{tag}"] = pil_jpeg(im, quality=90, subsampling=ss)
            cases[f"prog_{name}_{tag}"] = pil_jpeg(im, quality=90, subsampling=ss, progressive=True)
        cases[f"gray_{tag}"] = pil_jpeg(im.convert("L"), quality=85)
        cases[f"gray_prog_{tag}"] = pil_jpeg(im.convert("L"), quality=85, progressive=True)
        cases[f"rst1_420_{tag}"] = pil_jpeg(im, quality=80, restart_marker_blocks=1)
        cases[f"rstrow_prog_{tag}"] = pil_jpeg(im, quality=80, progressive=True, restart_marker_rows=1)
    im = base
    cases["q100_444"] = pil_jpeg(im, quality=100, subsampling=0)
    cases["q100_prog"] = pil_jpeg(im, quality=100, progressive=True)
    cases["q3_420"] = pil_jpeg(im, quality=3)
    cases["optimize_420"] = pil_jpeg(im, quality=75, optimize=True)
    cases["optimize_prog_rst7"] = pil_jpeg(im, quality=75, optimize=True, progressive=True,
                                           restart_marker_blocks=7)
    cases["rst5_444"] = pil_jpeg(im, quality=90, subsampling=0, restart_marker_blocks=5)
    cases["keep_rgb_adobe"] = pil_jpeg(im, quality=90, keep_rgb=True, subsampling=0)
    cases["q16bit_tables"] = pil_jpeg(im, qtables=[[300 + k for k in range(64)], [700] * 64])
    plain = pil_jpeg(im, quality=88)
    cases["split_dht"] = split_dht(plain)
    cases["dqt_merged_decoy"] = merge_dqt_with_decoy(plain)
    cases["fill_bytes_rst"] = add_fill_bytes(pil_jpeg(im, quality=88, restart_marker_blocks=3))
    cases["fill_bytes_prog"] = add_fill_bytes(pil_jpeg(im, quality=88, progressive=True))
    cases["no_dht_mjpeg"] = strip_dht(plain)
    cases["sof1"] = to_sof1(plain)
    cases["wide_30000x3"] = pil_jpeg(base.resize((30000, 3), Image.BICUBIC), quality=90)
    cases["tall_2x3000_gray"] = pil_jpeg(base.resize((2, 3000), Image.BICUBIC).convert("L"))

    png = os.path.join(work, "src.png")
    base.resize((131, 77), Image.BICUBIC).save(png)
    for sf, name in (("1x2", "440"), ("4x1", "411"), ("3x1", "311"), ("2x1", "422im"),
                     ("4x2", "4x2"), ("1x4", "1x4"), ("2x2,1x1,2x2", "mixed")):
        for prog in (False, True):
            out = os.path.join(work, f"im_{name}{'_prog' if prog else ''}.jpg")
            args = ["-quality", "90", "-sampling-factor", sf] + (["-interlace", "Plane"] if prog else [])
            if imagemagick_jpeg(png, out, *args):
                with open(out, "rb") as f:
                    cases[os.path.basename(out)[:-4]] = f.read()

    arr = np.asarray(base.resize((61, 45), Image.BICUBIC))
    cases["own_noninterleaved_420"] = custom_jpeg(arr, [(2, 2), (1, 1), (1, 1)], interleaved=False)
    cases["own_noninterleaved_rst"] = custom_jpeg(arr, [(2, 1), (1, 1), (1, 1)], interleaved=False, restart=3)
    cases["own_interleaved_rst2"] = custom_jpeg(arr, [(1, 2), (1, 1), (1, 1)], restart=2)
    cases["own_y1x1_c2x2"] = custom_jpeg(arr, [(1, 1), (2, 2), (2, 2)])
    cases["own_q16_sof1"] = custom_jpeg(arr, [(2, 2), (1, 1), (1, 1)], q16=True, sof=0xC1, qscale=3)
    cases["own_adobe_rgb"] = custom_jpeg(arr, [(1, 1), (1, 1), (1, 1)], adobe=0)
    cases["own_adobe_ycc_noninter"] = custom_jpeg(arr, [(2, 2), (1, 2), (2, 1)], adobe=1, interleaved=False)

    paths = []
    for name, data in sorted(cases.items()):
        p = os.path.join(work, name + ".jpg")
        with open(p, "wb") as f:
            f.write(data)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--lfw")
    ap.add_argument("--synth")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--png-lfw", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    failed = False

    if args.lfw:
        paths = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.lfw) for f in fs
                       if f.lower().endswith(".jpg"))
        if args.limit:
            paths = paths[:args.limit]
        mine, log = run_dump(args.dump, paths, args.work, "lfw")
        print(log)
        if mine is None:
            print("plx_decode_dump failed")
            return 1
        ndiff, maxd, mean, worst = compare(mine, paths)
        print(f"LFW: {len(paths)} images, {ndiff} differ, max abs diff {maxd}, mean abs diff {mean:.3g}"
              + (f" (worst {worst})" if worst else ""))
        failed |= ndiff > 0
        if args.png:
            sub = paths[:args.png_lfw]
            png_dir = os.path.join(args.work, "png_lfw")
            mine, _ = run_dump(args.dump, sub, args.work, "lfw_png", png_dir)
            bad, pb, rb = check_pngs(png_dir, mine, sub)
            print(f"PNG on {len(sub)} LFW images x 3 variants: {bad} mismatches; {pb} bytes for {rb} raw bytes"
                  f" (ratio {rb / pb:.2f}:1)")
            failed |= bad > 0

    if args.synth:
        paths = build_synth(args.synth, os.path.join(args.work, "synth"))
        bad = png_bad = 0
        for p in paths:
            try:
                ref = pil_rgb(p)
            except Exception as e:  # PIL refusing a file means the case itself is invalid
                print(f"  skip {os.path.basename(p)}: PIL cannot decode ({e})")
                continue
            name = os.path.basename(p)
            png_dir = os.path.join(args.work, "png_synth", name[:-4]) if args.png else None
            mine, log = run_dump(args.dump, [p], args.work, "one", png_dir)
            if mine is None:
                print(f"  FAIL {name}: {log}")
                bad += 1
                continue
            d = np.abs(mine[0].astype(np.int16) - ref.astype(np.int16))
            status = "ok  " if d.max() == 0 else "DIFF"
            bad += d.max() != 0
            if png_dir:
                png_bad += check_pngs(png_dir, mine, [p])[0]
            print(f"  {status} {name:32s} {ref.shape[1]}x{ref.shape[0]}  differing px {int((d.max(axis=2) > 0).sum())}"
                  f"  max {int(d.max())}  mean {d.mean():.3g}")
        print(f"synthetic: {len(paths)} files, {bad} differ or fail")
        if args.png:
            print(f"PNG on synthetic set x 3 variants: {png_bad} mismatches")
        failed |= bad > 0 or png_bad > 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
