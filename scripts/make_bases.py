#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Write the PanBase bases (s_pan and/or s_ms) of PanCollection splits in the layout the trainer and infer.py read.

    python scripts/make_bases.py --dataset wv3 --split reduced                       # test bases for infer.py
    python scripts/make_bases.py --dataset wv3 --split train,valid,reduced,full      # everything a PanBase arm needs
    python scripts/make_bases.py --dataset wv2 --split reduced,full                  # zero-shot WV2 (WV3 coefficients)

Each (geometry, split) pair is one call of scripts/panbase_standalone.py with --format sidecar, which writes
<out-dir>/<S_PAN|S_MS>/<the PanCollection relative path>: an h5 file whose dataset 'lms' holds the base in digital
numbers.  lib/pancoll.py reads it in place of the original 'lms' when a run uses --skip S_PAN / S_MS (infer.py: --arm
s_pan / s_ms).  The default --out-dir is that reader's folder (<repository>/data_skip, or $PANBASE_ROOT/data_skip).

Options
  --geometry     pan,ms (default: both) -- pan = s_pan (arm s_pan), ms = s_ms (arm s_ms)
  --coeffs       released coefficient folder (default weights/PanBase/panbase, i.e. a local snapshot of the Hugging Face
                 model repository: hf download <hf-repo> --include "panbase/*" --local-dir weights/PanBase)
  --data-root    PanCollection root (default $PANCOLLECTION_ROOT)
Other options (--batch-size, --threads, --toolbox-module) are passed through to panbase_standalone.py.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLITS = ("train", "valid", "reduced", "full")


def main(argv=None):
    ap = argparse.ArgumentParser(description="PanBase bases of PanCollection splits in the trainer's sidecar layout.")
    ap.add_argument("--dataset", required=True, choices=("wv3", "qb", "gf2", "wv2"))
    ap.add_argument("--split", default="reduced,full", help="comma-separated subset of train,valid,reduced,full")
    ap.add_argument("--geometry", default="pan,ms", help="pan, ms or pan,ms")
    ap.add_argument("--coeffs", default=str(ROOT / "weights" / "PanBase" / "panbase"))
    ap.add_argument("--data-root", default=os.environ.get("PANCOLLECTION_ROOT"))
    ap.add_argument("--out-dir", default=str(Path(os.environ.get("PANBASE_ROOT") or ROOT) / "data_skip"))
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--toolbox-module", default=None)
    a = ap.parse_args(argv)
    splits = [s.strip() for s in a.split.split(",") if s.strip()]
    geos = [g.strip() for g in a.geometry.split(",") if g.strip()]
    bad = [s for s in splits if s not in SPLITS] + [g for g in geos if g not in ("pan", "ms")]
    if bad:
        ap.error(f"unknown split / geometry: {bad}")
    if a.dataset == "wv2" and set(splits) & {"train", "valid"}:
        ap.error("WV2 has no training split: its bases use the WV3 coefficients on the test sets (--split reduced,full)")
    if not a.data_root:
        ap.error("pass --data-root or set PANCOLLECTION_ROOT")
    if not Path(a.coeffs).exists():
        ap.error(f"no coefficients at {a.coeffs}: download them first, e.g. "
                 f'hf download <hf-repo> --include "panbase/*" --local-dir weights/PanBase')

    sys.path.insert(0, str(HERE))
    import panbase_standalone as PS
    for geo in geos:
        for split in splits:
            args = ["--dataset", a.dataset, "--geometry", geo, "--split", split, "--format", "sidecar",
                    "--data-root", a.data_root, "--coeffs", a.coeffs, "--out-dir", a.out_dir]
            if a.batch_size:
                args += ["--batch-size", str(a.batch_size)]
            if a.threads:
                args += ["--threads", str(a.threads)]
            if a.toolbox_module:
                args += ["--toolbox-module", a.toolbox_module]
            print(f"[make_bases] {a.dataset} {split} s_{geo} -> {a.out_dir}/S_{geo.upper()}/", flush=True)
            PS.main(args)


if __name__ == "__main__":
    main()
