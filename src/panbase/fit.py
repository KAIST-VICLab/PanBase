# SPDX-License-Identifier: Apache-2.0
"""Calibrating PanBase coefficients on reduced-resolution training pairs (closed form, CPU).

Because Pi is affine and the gains are spatially constant, ``s = Pi(L) + sum_k g_k Q(r_k)`` (before clipping), with Q the
null-space projector of D and ``r_k`` the 9 band-split detail forms.  Per MS band b the gains solve the ridge problem

    min_g  sum_i (1 / n_i) || e_i,b - sum_k g_k a_i,b,k ||^2  +  gamma_rel * mean(diag(A_b)) * ||g||^2
    e_i = GT_i - Pi(L_i; m_i),   a_i,k = Q(r_i,k),   A_b = sum_i (1 / n_i) <a_i,b,k, a_i,b,l>

(n_i = pixels of sample i), i.e. ``g_b = (A_b + gamma_rel mean diag(A_b) I)^{-1} c_b`` with
``c_b = sum_i (1 / n_i) <a_i,b,k, e_i,b>``.  The paper's coefficients use gamma_rel = 1e-2 for every dataset and 6,000
training patches drawn with ``numpy.random.default_rng(20261007).choice(n, 6000, replace=False)`` (sorted) from the
PanCollection ``train_<dataset>.h5``.

    python -m panbase.fit --train <PanCollection root>/WV3/train_wv3.h5 --dataset wv3 --geometry pan --out wv3_pan.json
    python -m panbase.fit --train my_train.h5 --gnyq 0.3,0.3,0.3,0.3 --geometry ms --out my_ms.json      # new sensor

The h5 file must hold ``gt``, ``lms``, ``ms`` and ``pan`` (PanCollection layout, N x C x H x W, any consistent unit).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .base import PanBase
from .mtf import GNYQ

SEED = 20261007
N_TRAIN = 6000
GAMMA_REL = 1e-2


def sample_indices(n_total: int, n: int = N_TRAIN, seed: int = SEED) -> np.ndarray:
    """The paper's training subset: all patches if there are at most `n`, else `n` drawn without replacement (sorted)."""
    if n is None or n_total <= n:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


def default_batch_size(C: int, H: int, W: int) -> int:
    """Batch size of the paper's statistics pass (it fixes the batch-shared floors and the s_ms stopping rule)."""
    return max(1, min(96, int(4e7 // (C * 9 * 8 * H * W))))


@torch.no_grad()
def statistics(model: PanBase, gt, lms, ms, pan, batch_size: int | None = None, log=None) -> dict:
    """Sufficient statistics A (C, 9, 9) and c (C, 9) (float64, weights 1 / n_i) of the 9 regressors (form-major)."""
    to = lambda a: torch.as_tensor(np.asarray(a), dtype=model.dtype)
    N, C, H, W = gt.shape
    bs = batch_size or default_batch_size(C, H, W)
    A = torch.zeros(C, 9, 9, dtype=torch.float64)
    c = torch.zeros(C, 9, dtype=torch.float64)
    s = torch.zeros(C, dtype=torch.float64)
    obs, forms = model.obs, model.forms
    t0 = time.time()
    for i in range(0, N, bs):
        G, L, m, P = (to(x[i:i + bs]).to(model.device) for x in (gt, lms, ms, pan))
        n = L.shape[0]
        e = (G - obs.project(L, m)).double()
        regs, _ = forms.regressors(L, m, P)
        QR = []
        for reg in regs:
            parts = forms.bands.split(reg)                                             # (n, C or 1, 3, H, W)
            for j in range(parts.shape[2]):
                QR.append(obs.null(parts[:, :, j].expand(n, C, H, W).contiguous()).double())
        QR = torch.stack(QR, 2)                                                        # (n, C, 9, H, W)
        w = 1.0 / float(H * W)
        A += w * torch.einsum("nckhw,nclhw->ckl", QR, QR).cpu()
        c += w * torch.einsum("nckhw,nchw->ck", QR, e).cpu()
        s += w * (e * e).sum((0, 2, 3)).cpu()
        if log and (i // bs) % 25 == 0:
            log(f"  {min(i + bs, N):6d}/{N} patches  {time.time() - t0:7.1f} s")
    return dict(A=A, c=c, s=s, n=N, batch_size=bs)


def ridge(A: torch.Tensor, c: torch.Tensor, gamma_rel: float = GAMMA_REL) -> np.ndarray:
    """(C, 3, 3) gains from the statistics: per band (A_b + gamma_rel * mean(diag A_b) * I)^{-1} c_b."""
    A, c = A.numpy(), c.numpy()
    C, K = c.shape
    g = np.zeros((C, K))
    for b in range(C):
        M = A[b]
        sc = np.mean(np.diag(M)) if np.mean(np.diag(M)) > 0 else 1.0
        g[b] = np.linalg.solve(M + gamma_rel * sc * np.eye(K), c[b])
    return g.reshape(C, 3, 3)


def fit_h5(train_h5: str, gnyq, geometry: str = "pan", ratio: int = 4, offset: int | None = None, n: int = N_TRAIN,
           seed: int = SEED, gamma_rel: float = GAMMA_REL, batch_size: int | None = None, log=print):
    """Fit the gains of one geometry on a PanCollection-style training file.  Returns (PanBase, info dict)."""
    import h5py
    with h5py.File(train_h5, "r") as f:
        n_total = f["gt"].shape[0]
        idx = sample_indices(n_total, n, seed)
        data = {k: np.asarray(f[k][idx], dtype=np.float32) for k in ("gt", "lms", "ms", "pan")}
    C = data["gt"].shape[1]
    model = PanBase(np.zeros((C, 3, 3)), gnyq, geometry, ratio, offset)
    t0 = time.time()
    log(f"fit {geometry}: {len(idx)} of {n_total} patches, shape {tuple(data['gt'].shape[1:])}")
    st = statistics(model, data["gt"], data["lms"], data["ms"], data["pan"], batch_size, log)
    g = ridge(st["A"], st["c"], gamma_rel)
    model = PanBase(g, gnyq, geometry, ratio, offset)
    info = dict(n_train=int(len(idx)), n_total=int(n_total), seed=seed, gamma_rel=gamma_rel, batch_size=st["batch_size"],
                seconds=round(time.time() - t0, 1), train_mse_projected_lms=(st["s"] / st["n"]).tolist())
    return model, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit PanBase coefficients on a PanCollection-style training h5.")
    ap.add_argument("--train", required=True, help="training h5 with gt / lms / ms / pan")
    ap.add_argument("--geometry", choices=("pan", "ms"), default="pan", help="pan = s_pan, ms = s_ms")
    ap.add_argument("--dataset", choices=sorted(GNYQ), help="PanCollection sensor (sets the Nyquist gains)")
    ap.add_argument("--gnyq", help="comma-separated MS Nyquist gains (new sensor; overrides --dataset)")
    ap.add_argument("--ratio", type=int, default=4)
    ap.add_argument("--offset", type=int, default=None, help="decimation offset of D (default ratio // 2)")
    ap.add_argument("--n", type=int, default=N_TRAIN, help="training patches (default 6000; <= 0: all)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gamma", type=float, default=GAMMA_REL, help="relative ridge (default 1e-2)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", required=True, help="output coefficient file (.json)")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.gnyq:
        gnyq = [float(v) for v in a.gnyq.split(",")]
    elif a.dataset:
        gnyq = GNYQ[a.dataset]
    else:
        ap.error("pass --dataset or --gnyq")
    model, info = fit_h5(a.train, gnyq, a.geometry, a.ratio, a.offset, a.n if a.n > 0 else None, a.seed, a.gamma,
                         a.batch_size)
    model.save(a.out, dataset=a.dataset or "custom", fit=info)
    print(json.dumps(dict(out=a.out, **info)))


if __name__ == "__main__":
    main()
