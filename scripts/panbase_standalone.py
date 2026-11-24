#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compute the PanBase base s (s_pan or s_ms) for a PanCollection split, save it and print the metrics.

    python scripts/panbase_standalone.py --dataset wv3 --geometry pan --data-root <PanCollection root> \
        --coeffs <panbase coefficient folder> --out-dir outputs/panbase [--split reduced|full|both|train|valid]

Inputs (``--data-root`` may also come from the environment variable PANCOLLECTION_ROOT):
    reduced  <root>/<WV3|QB|GF2|WV2>/reduced_examples_h5/test_<ds>_multiExm1.h5        (RR test, 20 x 256^2, with gt)
    full     <root>/<WV3|QB|GF2|WV2>/full_examples_h5/test_<ds>_OrigScale_multiExm1.h5   (FR test, 20 x 512^2)
    train    <root>/<WV3|QB|GF2>/train_<ds>.h5,   valid  <root>/<WV3|QB|GF2>/valid_<ds>.h5   (64^2 training patches)
The coefficient folder is the released one (config.json + panbase_<ds>_<geometry>.json); WV2 uses the WV3 coefficients
with the WV2 operator.

Outputs (``--format``), always in the units of the h5 files (digital numbers, clipped to [0, 2047] or [0, 1023]; divide
by that range for the usual [0, 1] network inputs):
    npy      <out-dir>/<dataset>/<split>/s_<geometry>/<index>.npy, one float32 (C, H, W) array per image
             (default for the test splits)
    sidecar  <out-dir>/<S_PAN|S_MS>/<the PanCollection relative path>, an h5 file with one dataset 'lms' (N, C, H, W)
             float32 -- the layout of the paper's precomputed bases, read by a trainer in place of the original 'lms'
             (default for train / valid)
    none     metrics only
plus <out-dir>/<dataset>/<split>/s_<geometry>/metrics.json.

Metrics, with the conventions of the paper's tables:
    reduced  arrays / data range, 30-pixel border cut, per-image PSNR (peak 1), SSIM (11 x 11 Gaussian, sigma 1.5),
             SAM (degrees), ERGAS (ratio 4) [this file, Apache-2.0]; Q2n (32 x 32 blocks) [toolbox]
    full     HQNR, D_lambda, D_s (block 32) against the PanCollection lms [toolbox]
    train / valid   PSNR (no border cut)
    all      consistency ||D s - ms|| / ||ms|| before clipping, seconds per image
[toolbox] = G. Vivone's Python assessment toolbox (nonprofit licence), imported from --toolbox-module (default: the copy
vendored in panbase.classical.third_party.vivone_assessment); these indices are skipped when it cannot be imported.

Batches: by default 256 patches at 64 x 64, 8 images at 256 x 256 and 2 at 512 x 512, the batches of the paper's bases
(s_ms depends on the batch composition at the 1e-5 level, see panbase.align).
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(os.path.join(_SRC, "panbase")) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from panbase import DATA_RANGE, PanBase  # noqa: E402

DIRS = {"wv3": "WV3", "qb": "QB", "gf2": "GF2", "wv2": "WV2"}
FILES = {"reduced": "{D}/reduced_examples_h5/test_{ds}_multiExm1.h5",
         "full": "{D}/full_examples_h5/test_{ds}_OrigScale_multiExm1.h5",
         "train": "{D}/train_{ds}.h5", "valid": "{D}/valid_{ds}.h5"}
HQNR_SENSOR = {"wv3": "WV3", "qb": "QB", "gf2": "none", "wv2": "WV3"}     # WV2 is scored with the WV3 MTF gains
PAPER_BATCH = {64: 256, 256: 8, 512: 2}
RR_CUT = 30
RATIO = 4


# ---------------------------------------------------------------------------------------------- RR indices (normalised)
def rr_crop(x, cut=RR_CUT):
    return x[..., cut - 1:x.shape[-2] - cut, cut - 1:x.shape[-1] - cut]


def psnr(x, y):
    mse = ((x - y) ** 2).mean((1, 2, 3)).clamp_min(1e-20)
    return -10 * torch.log10(mse)


def sam(x, y, eps=1e-12):
    x, y = x.double(), y.double()
    cos = (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0))).flatten(1).mean(1)


def ergas(x, y, ratio=RATIO, eps=1e-8):
    mse = ((x - y) ** 2).mean((2, 3))
    mu = y.mean((2, 3)).abs().clamp_min(eps)
    return 100.0 / ratio * torch.sqrt((mse / mu ** 2).mean(1))


def ssim(x, y, w=11, sigma=1.5):
    c = x.shape[1]
    g = torch.arange(w, dtype=x.dtype) - (w - 1) / 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g[:, None] @ g[None, :]).expand(c, 1, w, w).contiguous()
    blur = lambda z: F.conv2d(z, k, groups=c)                                           # noqa: E731
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))).flatten(1).mean(1)


def load_toolbox(pkg):
    """(HQNR, q2n) functions of the assessment toolbox package `pkg` (submodules HQNR.py / q2n.py, or re-exports)."""
    out = []
    for name in ("HQNR", "q2n"):
        try:
            try:
                fn = getattr(importlib.import_module(f"{pkg}.{name}"), name)
            except ModuleNotFoundError:
                fn = getattr(importlib.import_module(pkg), name)
            out.append(fn if callable(fn) else None)
        except Exception as e:                                                           # noqa: BLE001
            print(f"[panbase] {name} skipped: cannot import it from {pkg} ({type(e).__name__}: {e})", flush=True)
            out.append(None)
    return tuple(out)


def run_split(pb, a, split, HQNR, q2n):
    import h5py
    ds, geo, div = a.dataset, a.geometry, DATA_RANGE[a.dataset]
    rel = FILES[split].format(D=DIRS[ds], ds=ds)
    test = split in ("reduced", "full")
    fmt = "none" if a.no_save else (a.format or ("npy" if test else "sidecar"))
    out = os.path.join(a.out_dir, ds, split, f"s_{geo}")
    os.makedirs(out, exist_ok=True)
    keep, cons, taus, psnr_tr = [], [], [], []
    with h5py.File(os.path.join(a.data_root, rel), "r") as f:
        n, C, H, W = f["lms"].shape
        bs = a.batch_size or PAPER_BATCH.get(H, max(1, 524288 // (H * W)))
        side = None
        if fmt == "sidecar":
            sp = os.path.join(a.out_dir, "S_PAN" if geo == "pan" else "S_MS", rel)
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            side = h5py.File(sp + ".tmp", "w")
            dset = side.create_dataset("lms", (n, C, H, W), dtype="float32")
        t0 = time.perf_counter()
        for i in range(0, n, bs):
            j = min(i + bs, n)
            ms, pan, lms = (np.asarray(f[k][i:j], dtype=np.float32) for k in ("ms", "pan", "lms"))
            s, info = pb(ms, pan, lms, max_value=div, return_info=True)                  # one batch
            cons.append(info["consistency"].numpy())
            if info["tau"] is not None:
                taus.append(info["tau"].numpy())
            if side is not None:
                dset[i:j] = s
            if fmt == "npy":
                for k in range(j - i):
                    np.save(os.path.join(out, f"{i + k:05d}.npy" if n > 100 else f"{i + k:02d}.npy"), s[k])
            if test:
                keep.append(s)
            elif "gt" in f:
                g = torch.from_numpy(np.asarray(f["gt"][i:j], dtype=np.float32) / div)
                psnr_tr.append(psnr(torch.from_numpy(s / div), g).numpy())
        sec = (time.perf_counter() - t0) / n
        if side is not None:
            side.attrs.update(geometry=geo, dataset=ds, coefficients=pb.meta.get("source_file", ""), clip=f"[0, {div}]",
                              units="h5 values (DN)", batch=bs)
            side.close()
            os.replace(sp + ".tmp", sp)
        d = {k: np.asarray(f[k], dtype=np.float32) for k in (("gt",) if split == "reduced" else ("ms", "lms", "pan"))} \
            if test else {}
    cons = np.concatenate(cons)
    res = dict(dataset=ds, geometry=geo, split=split, n=int(n), shape=[int(C), int(H), int(W)], batch_size=bs,
               seconds_per_image=round(sec, 4), threads=torch.get_num_threads(), consistency_median=float(np.median(cons)),
               consistency_max=float(cons.max()), units=f"h5 values (DN), clipped to [0, {div:g}]", output_format=fmt)
    if taus and n <= 100:
        res["tau_px"] = np.concatenate(taus).round(4).tolist()
    per = {}
    if split == "reduced":
        X, G = torch.from_numpy(np.concatenate(keep) / div), torch.from_numpy(d["gt"] / div)
        Xc, Gc = rr_crop(X), rr_crop(G)
        per = dict(psnr=psnr(Xc, Gc), ssim=ssim(Xc, Gc), sam=sam(Xc, Gc), ergas=ergas(Xc, Gc))
        if q2n is not None:
            g64, p64 = Gc.double().numpy() * div, Xc.double().numpy() * div
            per["q2n"] = [q2n(g64[i].transpose(1, 2, 0), p64[i].transpose(1, 2, 0), 32, 32)[0] for i in range(n)]
    elif split == "full" and HQNR is not None:
        s = np.concatenate(keep).astype(np.float64)
        r = [HQNR(s[i].transpose(1, 2, 0), np.array(d["ms"][i].astype(np.float64).transpose(1, 2, 0)),
                  d["lms"][i].astype(np.float64).transpose(1, 2, 0), d["pan"][i, 0].astype(np.float64), 32,
                  HQNR_SENSOR[ds], RATIO) for i in range(n)]
        per = dict(hqnr=[x[0] for x in r], d_lambda=[x[1] for x in r], d_s=[x[2] for x in r])
    elif psnr_tr:
        per = dict(psnr=np.concatenate(psnr_tr))
    if n <= 100:
        res["per_image"] = {k: [float(v) for v in np.asarray(x, dtype=np.float64)] for k, x in per.items()}
    res["mean"] = {k: float(np.mean(np.asarray(v, dtype=np.float64))) for k, v in per.items()}
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    line = "  ".join(f"{k.upper() if k != 'd_lambda' else 'D_lambda'} {v:.4f}" for k, v in res["mean"].items())
    print(f"[panbase] {ds} {split:7s} s_{geo}: {line}  | consistency {res['consistency_median']:.1e}  "
          f"{sec:.2f} s/image ({res['threads']} threads, batch {bs})", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="PanBase s_pan / s_ms on a PanCollection split.")
    ap.add_argument("--dataset", required=True, choices=sorted(DIRS))
    ap.add_argument("--geometry", required=True, choices=("pan", "ms"), help="pan = s_pan, ms = s_ms")
    ap.add_argument("--data-root", default=os.environ.get("PANCOLLECTION_ROOT"),
                    help="PanCollection root (folders WV3, QB, GF2, WV2); default $PANCOLLECTION_ROOT")
    ap.add_argument("--coeffs", required=True, help="released coefficient folder (config.json) or a coefficient file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="both", choices=("reduced", "full", "both", "train", "valid"))
    ap.add_argument("--format", choices=("npy", "sidecar", "none"), default=None,
                    help="npy (default for the test splits) | sidecar (default for train / valid) | none")
    ap.add_argument("--batch-size", type=int, default=0, help="images per batch (0: 256 / 8 / 2 at 64 / 256 / 512 px)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: torch default)")
    ap.add_argument("--no-save", action="store_true", help="same as --format none")
    ap.add_argument("--toolbox-module", default="panbase.classical.third_party.vivone_assessment",
                    help="importable module exposing the toolbox's HQNR and q2n (for Q2n / HQNR)")
    a = ap.parse_args(argv)
    if not a.data_root:
        ap.error("pass --data-root or set PANCOLLECTION_ROOT")
    if a.dataset == "wv2" and a.split in ("train", "valid"):
        ap.error("WV2 has no training split (its bases use the WV3 coefficients on the test sets)")
    if a.threads:
        torch.set_num_threads(a.threads)
    pb = PanBase.from_pretrained(a.coeffs, dataset=a.dataset, geometry=a.geometry)
    splits = ("reduced", "full") if a.split == "both" else (a.split,)
    HQNR, q2n = load_toolbox(a.toolbox_module) if any(s in ("reduced", "full") for s in splits) else (None, None)
    print(f"[panbase] {pb}  coefficients {pb.meta.get('source_file')}", flush=True)
    for split in splits:
        run_split(pb, a, split, HQNR, q2n)


if __name__ == "__main__":
    main()
