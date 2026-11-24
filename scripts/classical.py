#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run the classical pansharpening methods of the PanBase paper (Table 2) on a PanCollection test set.

Examples:
    python scripts/classical.py --method all --dataset wv3 --data-root /path/to/PanCollection --out-dir out/classical
    python scripts/classical.py --method MTF-GLP-HPM,GSA --dataset gf2 --split full --data-root ... --out-dir ...
    python scripts/classical.py --method LRTCFPan --dataset qb --images 0-3 --threads 4 --data-root ... --out-dir ...

For every method this writes ``<out-dir>/<dataset>_<split>/<METHOD>.npz`` with
    hrms     (N, C, H, W) float32, fused images normalised like PanCollection (DN / 2047; GF2: DN / 1023), in [0, 1]
    indices  (N,) test-image numbers;  seconds  (N,) run time per image
and ``<out-dir>/<dataset>_<split>/metrics.json``, and prints the mean metrics:
    --split reduced   PSNR, SAM, ERGAS, SSIM, Q2n   (30-pixel border removed, PanCollection protocol)
    --split full      HQNR, D_lambda, D_s
When all 20 images are processed the stored Table-2 value and the difference are printed next to each mean.

--data-root is the PanCollection folder (searched for test_<ds>_multiExm1.h5 / test_<ds>_OrigScale_multiExm1.h5) or
the .h5 file itself; it defaults to the environment variable PANCOLLECTION_ROOT.  Everything runs on CPU in float64.
--threads N sets OMP/MKL/OpenBLAS threads and torch's intra-op threads (default: library defaults).
"""
import argparse
import json
import os
import sys
import time


def parse_images(s):
    if not s:
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, help="method name, comma-separated list, or 'all'")
    ap.add_argument("--dataset", required=True, choices=["wv3", "qb", "gf2", "wv2"])
    ap.add_argument("--data-root", default=os.environ.get("PANCOLLECTION_ROOT"),
                    help="PanCollection root folder or test .h5 file (default: $PANCOLLECTION_ROOT)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="reduced", choices=["reduced", "full"])
    ap.add_argument("--images", default="", help="subset of test images, e.g. '0,1' or '0-4' (default: all)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-metrics", action="store_true", help="only write the fused images")
    ap.add_argument("--no-q2n", action="store_true", help="skip Q2n (reduced resolution)")
    ap.add_argument("--no-save", action="store_true", help="do not write the fused images")
    ap.add_argument("--resume", action="store_true", help="reuse complete <METHOD>.npz files of an earlier run")
    a = ap.parse_args(argv)
    if a.data_root is None:
        ap.error("--data-root is required (or set PANCOLLECTION_ROOT)")
    if a.threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[k] = str(a.threads)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import numpy as np
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))
    from panbase import classical as C
    from panbase.classical.reference import reference

    if a.threads:
        torch.set_num_threads(a.threads)
    methods = C.list_methods() if a.method.lower() == "all" else [C.canonical_name(m) for m in a.method.split(",")]
    data = C.load_test_set(a.data_root, a.dataset, a.split, parse_images(a.images))
    idx = data["indices"]
    full_set = idx.size == 20 and a.images == ""
    odir = os.path.join(a.out_dir, f"{a.dataset}_{a.split}")
    os.makedirs(odir, exist_ok=True)
    mpath = os.path.join(odir, "metrics.json")
    try:
        with open(mpath) as fh:
            allm = json.load(fh)
    except (OSError, ValueError):
        allm = {}
    print(f"[classical] {a.dataset} {a.split}: {idx.size} images from {data['path']}; methods: {', '.join(methods)}",
          flush=True)
    lms, pan, ms = (torch.from_numpy(data[k]) for k in ("lms", "pan", "ms"))
    keys = ("psnr", "sam", "ergas", "ssim", "q2n") if a.split == "reduced" else ("hqnr", "d_lambda", "d_s")
    if a.split == "reduced" and a.no_q2n:
        keys = keys[:-1]
    rows = []
    for m in methods:
        fam = C.METHODS[m][0]
        npz = os.path.join(odir, f"{m}.npz")
        Y, secs = None, None
        if a.resume and os.path.exists(npz):
            try:
                z = np.load(npz)
                if np.array_equal(z["indices"], idx) and z["hrms"].shape == tuple(lms.shape) and np.isfinite(z["hrms"]).all():
                    Y, secs = z["hrms"], z["seconds"]
                    print(f"[classical] {m}: reusing {npz}", flush=True)
            except Exception as e:                                   # noqa: BLE001 -- unreadable: recompute
                print(f"[classical] {m}: cannot reuse {npz} ({type(e).__name__}); recomputing", flush=True)
        if Y is None:
            out, secs = [], []
            for j in range(idx.size):
                t0 = time.perf_counter()
                out.append(C.run(m, lms[j], pan[j], ms[j], a.dataset, resolution=a.split))
                secs.append(time.perf_counter() - t0)
                print(f"[classical] {m} {a.dataset} {a.split} image {int(idx[j])} ({j + 1}/{idx.size}) {secs[-1]:.2f} s",
                      flush=True)
            Y, secs = torch.stack(out).numpy(), np.asarray(secs)
            if not a.no_save:
                tmp = npz + ".part.npz"
                np.savez_compressed(tmp, hrms=Y, indices=idx, seconds=secs)
                os.replace(tmp, npz)
        rec = dict(family=fam, indices=idx.tolist(), sec_per_image_median=float(np.median(secs)))
        if not a.no_metrics:
            Yt = torch.from_numpy(Y)
            if a.split == "reduced":
                sc = C.rr_metrics(Yt, torch.from_numpy(data["gt"]), a.dataset, with_q2n=not a.no_q2n)
            else:
                sc = C.fr_metrics(Yt, ms, lms, pan, a.dataset)
            rec["per_image"] = {k: [float(v) for v in sc[k]] for k in keys}
            rec["mean"] = {k: float(np.mean(sc[k])) for k in keys}
            ref = reference(a.dataset, m)
            if full_set and ref is not None:
                rec["table2"] = {k: ref[k] for k in ("psnr", "sam", "ergas", "q2n", "hqnr", "d_lambda", "d_s") if k in keys}
        allm[m] = rec
        rows.append((m, rec))
        tmpj = mpath + ".part"
        with open(tmpj, "w") as fh:
            json.dump(dict(allm, _meta=dict(dataset=a.dataset, split=a.split, data_file=os.path.basename(data["path"]),
                                            torch=torch.__version__, numpy=np.__version__)), fh, indent=1)
        os.replace(tmpj, mpath)
    if a.no_metrics:
        return 0
    nd = {"psnr": 4, "sam": 4, "ergas": 4, "ssim": 4, "q2n": 4, "hqnr": 4, "d_lambda": 4, "d_s": 4}
    print(f"\n{a.dataset} {a.split}, {idx.size} images" + ("  (table = stored Table-2 value, d = here - table)" if full_set else ""))
    print(f"{'method':12s} {'fam':4s} " + " ".join(f"{k:>10s}" for k in keys) + f" {'s/img':>7s}")
    for m, rec in rows:
        print(f"{m:12s} {rec['family']:4s} " + " ".join(f"{rec['mean'][k]:10.{nd[k]}f}" for k in keys)
              + f" {rec['sec_per_image_median']:7.2f}")
        if "table2" in rec:
            print(f"{'  table':12s} {'':4s} " + " ".join(f"{rec['table2'][k]:10.{nd[k]}f}" if k in rec["table2"] else f"{'':>10s}"
                                                       for k in keys))
            print(f"{'  d':12s} {'':4s} " + " ".join(f"{rec['mean'][k] - rec['table2'][k]:+10.{nd[k]}f}" if k in rec["table2"]
                                                   else f"{'':>10s}" for k in keys))
    print(f"\nwrote {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
