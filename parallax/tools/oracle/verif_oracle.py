#!/usr/bin/env python3
"""Brute-force oracle for src/eval/verif.c (test use only).

usage: verif_oracle.py IN.plxt OUT.plxt
  IN  f64 [n, 3]: score, same (0/1), fold
  OUT f64 vector: auc, eer, tar@1e-1, tar@1e-2, tar@1e-3, mean_acc, sem, n_folds,
                  then per fold: held-out acc, threshold, train acc

Every quantity is computed the slow, obvious way (all-pairs comparisons, explicit counting at
each candidate threshold) so that it shares no code path with the C sweep implementation.
The decision rules are the ones documented in src/eval/verif.h.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plxt  # noqa: E402


def clean(s):
    s = s.copy()
    s[np.isnan(s)] = -np.inf
    return s


def auc(s, y):
    a, b = s[y == 1], s[y == 0]
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return (gt + 0.5 * eq) / (a.size * b.size)


def roc(s, y):
    ns, nd = (y == 1).sum(), (y == 0).sum()
    far, tar = [0.0], [0.0]
    for v in np.unique(s)[::-1]:
        pred = s >= v
        tar.append((pred & (y == 1)).sum() / ns)
        far.append((pred & (y == 0)).sum() / nd)
    return np.array(far), np.array(tar)


def tar_at(far, tar, f):
    at = tar[far == f]
    if at.size:
        return at.max()
    for i in range(far.size - 1):
        if far[i] < f < far[i + 1]:
            return tar[i] + (f - far[i]) / (far[i + 1] - far[i]) * (tar[i + 1] - tar[i])
    return tar[-1]


def eer(far, tar):
    for i in range(far.size - 1):
        d0, d1 = far[i] + tar[i] - 1, far[i + 1] + tar[i + 1] - 1
        if d0 < 0 <= d1:
            lam = -d0 / (d1 - d0)
            return far[i] + lam * (far[i + 1] - far[i])
    return 1.0


def best_threshold(s, y):
    v = np.unique(s)
    acc = lambda t: ((s > t) == (y == 1)).mean()  # noqa: E731
    # interval 0: t < v[0] (skipped when v[0] = -inf), interval j: [v[j-1], v[j])
    cands = []
    if v[0] > -np.inf:
        cands.append((acc(np.nextafter(v[0], -np.inf)), -np.inf, v[0], True, False))
    for j in range(v.size):
        hi = v[j + 1] if j + 1 < v.size else np.inf
        cands.append((acc(v[j]), v[j], hi, False, j + 1 == v.size))
    best = max(c[0] for c in cands)
    runs, cur = [], None
    for c in cands:
        if c[0] == best:
            cur = [c[1], c[2], c[3], c[4]] if cur is None else [cur[0], c[2], cur[2], c[4]]
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    bounded = [r for r in runs if not r[2] and not r[3]]
    if bounded:
        r = max(bounded, key=lambda r: r[1] - r[0])  # max() keeps the first (lowest) on ties
        lo, hi = r[0], r[1]
        if lo == -np.inf:
            t = -np.inf
        elif hi == np.inf:
            t = np.finfo(np.float64).max
        else:
            t = 0.5 * lo + 0.5 * hi
            if not t < hi:
                t = lo
    else:
        r = runs[0]
        t = np.nextafter(v[0], -np.inf) if r[2] else v[-1]
    return t, acc(t)


def main(inp, out):
    a = plxt.read(inp)
    s, y, fold = clean(a[:, 0]), a[:, 1].astype(int), a[:, 2].astype(int)
    far, tar = roc(s, y)
    res = [auc(s, y), eer(far, tar)] + [tar_at(far, tar, f) for f in (1e-1, 1e-2, 1e-3)]
    nf = fold.max() + 1
    accs, per = [], []
    for k in range(nf):
        tr, te = fold != k, fold == k
        t, tracc = best_threshold(s[tr], y[tr])
        acc = ((s[te] > t) == (y[te] == 1)).mean()
        accs.append(acc)
        per += [acc, t, tracc]
    accs = np.array(accs)
    res += [accs.mean(), accs.std(ddof=1) / np.sqrt(nf), nf] + per
    plxt.write(out, np.array(res, dtype=np.float64))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
