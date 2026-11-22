#!/usr/bin/env python3
"""numpy/LAPACK cross-check for src/la (test oracle only; never part of the system).

Writes deterministic problems and numpy's answers as .plxt files into DIR; tests/test_la.c then
compares against them when run with PLX_LA_ORACLE=DIR:

    python3 tools/oracle/la_oracle.py /tmp/la_oracle
    PLX_LA_ORACLE=/tmp/la_oracle build-la/bin/test_la

Only sign/basis-invariant quantities are compared (values, residual-free solutions, projectors),
so the two implementations do not need to agree on eigenvector signs or degenerate bases.
"""
import os
import struct
import sys

import numpy as np

DTYPES = {np.dtype(np.float32): 2, np.dtype(np.float64): 3}


def save(path, arr):
    """.plxt: 'PLXT', u32 version, u32 dtype, u32 ndim, i64 dim[6], then raw little-endian data."""
    arr = np.ascontiguousarray(arr)
    dims = list(arr.shape) + [0] * (6 - arr.ndim)
    header = b"PLXT" + struct.pack("<III", 1, DTYPES[arr.dtype], arr.ndim) + struct.pack("<6q", *dims)
    assert len(header) == 64
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "la_oracle"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20240607)
    w = lambda name, a: save(os.path.join(out, name + ".plxt"), a)

    # products
    A = rng.standard_normal((37, 41)); B = rng.standard_normal((41, 53))
    w("gemm_a", A); w("gemm_b", B); w("gemm_c", A @ B)

    # SPD solve and log-determinant
    M = rng.standard_normal((60, 60)); S = M @ M.T + 60 * np.eye(60); b = rng.standard_normal((60, 2))
    w("spd_a", S); w("spd_b", b); w("spd_x", np.linalg.solve(S, b))
    w("spd_logdet", np.array([np.linalg.slogdet(S)[1]]))

    # general solve and determinant
    G = rng.standard_normal((45, 45)); g = rng.standard_normal((45, 3))
    w("lu_a", G); w("lu_b", g); w("lu_x", np.linalg.solve(G, g)); w("lu_det", np.array([np.linalg.det(G)]))

    # rank-deficient least squares: numpy's lstsq is the SVD pseudo-inverse solution
    L = rng.standard_normal((80, 20)); L[:, 7] = L[:, 3] - 2 * L[:, 11]; L[:, 15] = 0
    lb = rng.standard_normal((80, 2))
    w("ls_a", L); w("ls_b", lb); w("ls_x", np.linalg.lstsq(L, lb, rcond=None)[0])

    # symmetric eigenvalues, and the spectral projector onto the top-5 eigenvectors (sign-free)
    E = rng.standard_normal((70, 70)); E = (E + E.T) / 2
    ev, V = np.linalg.eigh(E)
    w("eig_a", E); w("eig_w", ev[::-1].copy()); w("eig_p5", V[:, -5:] @ V[:, -5:].T)

    # generalised symmetric-definite eigenvalues via Cholesky reduction in numpy
    Bg = rng.standard_normal((50, 50)); Bg = Bg @ Bg.T + 5 * np.eye(50)
    Ag = rng.standard_normal((50, 50)); Ag = (Ag + Ag.T) / 2
    Lc = np.linalg.cholesky(Bg); Li = np.linalg.inv(Lc)
    w("geig_a", Ag); w("geig_b", Bg); w("geig_w", np.sort(np.linalg.eigvalsh(Li @ Ag @ Li.T))[::-1].copy())

    # SVD values of tall and wide matrices
    T = rng.standard_normal((90, 33))
    w("svd_a", T); w("svd_s", np.linalg.svd(T, compute_uv=False))
    Wd = rng.standard_normal((12, 70))
    w("svdw_a", Wd); w("svdw_s", np.linalg.svd(Wd, compute_uv=False))

    # PCA of float data: singular values of the centred matrix and the top-8 axis projector
    X = (rng.standard_normal((300, 25)) @ rng.standard_normal((25, 120)) + 5.0).astype(np.float32)
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    w("pca_x", X); w("pca_s", s[:8].copy()); w("pca_p8", Vt[:8].T @ Vt[:8])
    print("wrote oracle data to", out)


if __name__ == "__main__":
    main()
