#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Run one released checkpoint on a PanCollection test set and score it.

    python scripts/infer.py --method PINO --arm s_pan --train-set wv3 --split reduced \
        --weights weights/PanBase --out-dir outputs/PINO/s_pan/wv3

    # zero-shot WorldView-2 with the WorldView-3 model, full resolution
    python scripts/infer.py --method PINO --arm s_pan --train-set wv3 --eval-set wv2 --split full \
        --weights weights/PanBase --out-dir outputs/PINO/s_pan/wv2_full

Arguments
  --method     one of the 15 backbones (see MODEL_ZOO.md)
  --arm        original | s_pan | s_ms   (original: the backbone with its own interpolated-LRMS base;
               s_pan / s_ms: the backbone with PanBase as the global residual base and an extra input)
  --train-set  wv3 | qb | gf2           (the checkpoint's training set)
  --eval-set   default: the training set; wv2 is allowed for wv3 checkpoints (zero-shot)
  --split      reduced (RR: reference available, 20 images of 256x256) | full (FR: 20 images of 512x512)
  --weights    local snapshot of the Hugging Face model repository, e.g.
               hf download <hf-repo> --include "backbones/PINO/*" --local-dir weights/PanBase

Before the first run
  bash scripts/fetch_repos.sh                       # upstream architectures at pinned commits + our patches
  python scripts/make_bases.py --dataset wv3 --split reduced   # s_pan / s_ms arms only: PanBase base of each image

Outputs: <out-dir>/output.npz (`y`: N x C x H x W, float32 in [0, 1] = DN / divisor) and <out-dir>/metrics.json.
The metrics are computed with the same functions as the paper tables (lib/pancoll.py): RR after a 30-pixel border cut,
FR with the Vivone toolbox HQNR (fetched with scripts/fetch_repos.sh).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "lib"), str(ROOT), str(ROOT / "src")]

ARMS = {"original": None, "s_pan": "S_PAN", "s_ms": "S_MS"}
CPU_ONLY = {"MSDDN": "its GPU output differs from the CPU result by 1-10 dB PSNR even with TF32 off; "
                     "the table values are CPU float32"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--train-set", required=True, choices=["wv3", "qb", "gf2"])
    ap.add_argument("--eval-set", default=None, choices=["wv3", "qb", "gf2", "wv2"])
    ap.add_argument("--split", default="reduced", choices=["reduced", "full"])
    ap.add_argument("--weights", required=True, help="local snapshot of the Hugging Face model repository")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--no-save", action="store_true", help="only print the metrics")
    a = ap.parse_args(argv)

    cdir = Path(a.weights) / "backbones" / a.method / a.arm / a.train_set
    if not (cdir / "config.json").is_file():
        raise SystemExit(f"no checkpoint at {cdir} (download it first; see MODEL_ZOO.md for the available cells)")
    cfg = json.loads((cdir / "config.json").read_text())
    ev = a.eval_set or a.train_set
    if ev != a.train_set and not (a.train_set == "wv3" and ev == "wv2"):
        raise SystemExit("cross-dataset evaluation is defined only for wv3 checkpoints on wv2 (zero-shot)")

    import torch
    dev = a.device
    if dev == "auto":
        dev = "cpu" if (a.method in CPU_ONLY or not torch.cuda.is_available()) else "cuda"
    if a.method in CPU_ONLY and dev != "cpu":
        print(f"[warn] {a.method}: {CPU_ONLY[a.method]} -- use --device cpu to reproduce the table", flush=True)
    torch.set_flush_denormal(True)                    # some checkpoints hold denormal weights (50x slower CPU convs)
    if dev.startswith("cuda"):
        torch.backends.cudnn.allow_tf32 = False       # TF32 changes the outputs of the FFT-based networks
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.Tensor.cuda = lambda self, *x, **k: self          # upstream code with hard-coded .cuda()
        torch.nn.Module.cuda = lambda self, *x, **k: self
    from safetensors.torch import load_file
    import pancoll as PC
    import registry as R
    import harness as H

    spec_name = cfg["registry_spec"]
    spec = R.SPECS[spec_name]
    sk = ARMS[a.arm]
    PC.SKIP = sk or "lrms"
    PC.LMS_STACK = bool(spec.get("lms_stack"))
    try:
        e = PC.load_examples(ev, a.split)
    except (FileNotFoundError, OSError) as err:
        hint = (f"; for the {a.arm} arm run: python scripts/make_bases.py --dataset {ev} --split {a.split}" if sk else "")
        raise SystemExit(f"cannot read the {ev} {a.split} test set ({err}){hint}")
    finally:
        PC.SKIP, PC.LMS_STACK = "lrms", False
    if "lpan" not in e and a.method == "PAN-Crafter":
        import lpan_synth as LS                        # WorldView-2 has no author lpan: synthesised as in the tables
        e["lpan"] = LS.lpan_from_pan(e["pan"])

    model, _ = R.build(spec_name, ms_ch=cfg["ms_bands"], ratio=cfg["ratio"])
    sd = load_file(str(cdir / "model.safetensors"))
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[warn] state dict: {len(miss)} missing, {len(unexp)} unexpected keys "
              f"(e.g. {(list(miss) + list(unexp))[:3]})", flush=True)
    model = model.to(dev).eval()
    fwd = H.eval_forward(spec, ratio=cfg["ratio"])
    ctx0 = {"step": cfg["selected_step"], "total": cfg["selected_step"]}
    n = e["pan"].shape[0]
    outs, t0 = [], time.time()
    with torch.no_grad():
        for i in range(n):
            torch.manual_seed(0)                       # the inference seed rule of the tables (stochastic methods)
            ctx = dict(ctx0, lpan=e["lpan"][i:i + 1].to(dev) if "lpan" in e else None)
            y = fwd(model, e["lms"][i:i + 1].to(dev), e["pan"][i:i + 1].to(dev), e["ms"][i:i + 1].to(dev), ctx)
            outs.append((y[0] if isinstance(y, tuple) else y).clamp(0, 1).float().cpu())
    Y = torch.cat(outs)
    sec = time.time() - t0

    div = PC.SPEC[ev][3]
    res = {"method": a.method, "arm": a.arm, "train_set": a.train_set, "eval_set": ev, "split": a.split, "device": dev,
           "n_images": n, "seconds": round(sec, 1)}
    if a.split == "reduced":
        m = PC.all_metrics(PC.rr_cut(Y), PC.rr_cut(e["gt"]), ratio=cfg["ratio"], divisor=div)
        res["metrics"] = {k: float(torch.as_tensor(v).double().mean()) for k, v in m.items()}
        ref = ((cfg.get("table_metrics") or {}).get(ev) or {}).get("rr") or {}
    else:
        hq, dl, ds = PC.hqnr(Y, e["ms"], e.get("lms_orig", e["lms"]), e["pan"], ev, ratio=cfg["ratio"])
        res["metrics"] = {"hqnr": float(hq.mean()), "d_lambda": float(dl.mean()), "d_s": float(ds.mean())}
        ref = ((cfg.get("table_metrics") or {}).get(ev) or {}).get("fr") or {}
    res["table"] = {k: ref.get(k) for k in res["metrics"]}
    for k, v in res["metrics"].items():
        t = ref.get(k)
        print(f"  {k:9s} {v:10.4f}" + (f"   table {t:10.4f}   diff {v - t:+.4f}" if t is not None else ""), flush=True)
    print(f"[infer] {a.method} {a.arm} {a.train_set}->{ev} {a.split}: {n} images in {sec:.1f} s on {dev}", flush=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    if not a.no_save:
        import numpy as np
        np.savez_compressed(out / "output.npz", y=Y.numpy())
    return res


if __name__ == "__main__":
    main()
