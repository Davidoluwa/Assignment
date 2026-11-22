#!/usr/bin/env python3
"""Reader/writer for the Parallax .plxt tensor format, plus a PNG viewer.

Format (src/core/core.h): 64-byte header = magic b"PLXT", u32 version=1, u32 dtype,
u32 ndim, i64 dim[6]; then raw little-endian row-major data. dtype: 0 u8, 1 i32, 2 f32, 3 f64.

Test oracle / inspection tool only; never part of the system.

CLI:
  plxt.py info  FILE.plxt
  plxt.py topng FILE.plxt OUT.png [--layout auto|hw|hwc|chw] [--range lo hi | --auto]
      [H,W] -> grey, [H,W,3] / [3,H,W] -> RGB, [1,H,W] -> grey.
      Default range for float data is [0, 1] (the plx_img convention); --auto stretches the
      1..99% percentiles of the finite values. NaN is drawn black. u8 is written as is.
"""
import struct
import sys

import numpy as np

_DTYPES = {0: np.uint8, 1: np.int32, 2: np.float32, 3: np.float64}
_CODES = {np.dtype(v): k for k, v in _DTYPES.items()}


def read(path):
    with open(path, "rb") as f:
        head = f.read(64)
        if len(head) != 64 or head[:4] != b"PLXT":
            raise ValueError(f"{path}: not a .plxt file")
        version, dtype, ndim = struct.unpack_from("<III", head, 4)
        dims = struct.unpack_from("<6q", head, 16)
        if version != 1 or dtype not in _DTYPES or not 1 <= ndim <= 6:
            raise ValueError(f"{path}: bad header (version {version}, dtype {dtype}, ndim {ndim})")
        shape = tuple(dims[:ndim])
        data = np.frombuffer(f.read(), dtype=np.dtype(_DTYPES[dtype]).newbyteorder("<"))
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: {data.size} elements for shape {shape}")
    return data.reshape(shape).copy()


def write(path, arr):
    arr = np.ascontiguousarray(arr)
    if arr.dtype not in _CODES:
        raise ValueError(f"unsupported dtype {arr.dtype}")
    if not 1 <= arr.ndim <= 6:
        raise ValueError("ndim must be 1..6")
    dims = list(arr.shape) + [0] * (6 - arr.ndim)
    head = b"PLXT" + struct.pack("<III", 1, _CODES[arr.dtype], arr.ndim) + struct.pack("<6q", *dims)
    with open(path, "wb") as f:
        f.write(head)
        f.write(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())


def to_hwc(a, layout="auto"):
    """Returns an [H, W] or [H, W, 3] array."""
    if layout == "auto":
        if a.ndim == 2:
            layout = "hw"
        elif a.ndim == 3 and a.shape[0] in (1, 3):  # plx_img planar layout first
            layout = "chw"
        elif a.ndim == 3 and a.shape[2] in (1, 3):
            layout = "hwc"
        else:
            raise ValueError(f"cannot view shape {a.shape} as an image")
    if layout == "chw":
        a = np.moveaxis(a, 0, -1)
    elif layout == "hw" and a.ndim != 2:
        raise ValueError("--layout hw needs a 2-D tensor")
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return a


def to_png(arr, out, layout="auto", rng=None, auto=False):
    from PIL import Image

    a = to_hwc(arr, layout)
    if a.dtype == np.uint8 and rng is None and not auto:
        u8 = a
    else:
        a = a.astype(np.float64)
        finite = np.isfinite(a)
        if auto:
            vals = a[finite]
            lo, hi = (np.percentile(vals, [1, 99]) if vals.size else (0.0, 1.0))
        else:
            lo, hi = rng if rng is not None else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        t = np.where(finite, (a - lo) / (hi - lo), 0.0)
        u8 = np.round(np.clip(t, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(u8, "L" if u8.ndim == 2 else "RGB").save(out)


def _main(argv):
    if len(argv) >= 2 and argv[0] == "info":
        a = read(argv[1])
        finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
        print(f"{argv[1]}: dtype {a.dtype} shape {a.shape}", end="")
        if finite.size:
            print(f" min {finite.min():.6g} max {finite.max():.6g} mean {finite.mean():.6g}", end="")
        print(f" non-finite {a.size - finite.size}")
        return 0
    if len(argv) >= 3 and argv[0] == "topng":
        layout, rng, auto, i = "auto", None, False, 3
        while i < len(argv):
            if argv[i] == "--layout":
                layout, i = argv[i + 1], i + 2
            elif argv[i] == "--range":
                rng, i = (float(argv[i + 1]), float(argv[i + 2])), i + 3
            elif argv[i] == "--auto":
                auto, i = True, i + 1
            else:
                raise SystemExit(f"unknown option {argv[i]}")
        to_png(read(argv[1]), argv[2], layout, rng, auto)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
