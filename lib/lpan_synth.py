"""PAN-Crafter 의 `lpan`(MS 해상도 저역 PAN)을 PAN 에서 **합성**한다 -- 저자 파일이 없는 데이터셋용.

왜 필요한가.  저자는 lpan 을 PanCollection wv3/qb/gf2 의 12개 h5 로만
배포했다 (`data_lpan/`).  그래서 WV2 zero-shot · 코퍼스 zero-shot 에서 PAN-Crafter 는 `--` 다.
합성 lpan 으로 그 칸을 채우려면 "합성 lpan 이 저자 lpan 과 같은 수를 낸다"를 **먼저** 보여야 한다 --
그 관문이 `validate_pancrafter_lpan.py` 이고, 통과 전에는 아무 표에도 싣지 않는다.

**규칙 하나 (`RULE` = mtf0.318+dec4) -- 저자 파일을 0.001% 로 재현한다.**

    lpan = decimate_r( correlate_nearest( pan, h ) ),   r 4,  위상 r//2 (= 2)
    h    Vivone 툴박스 genMTF 의 한 밴드 필터 (N=41 가우시안 주파수 응답 + Kaiser(0.5) 창,
         fir_filter_wind) 에 GNyq = 0.318.  `MTF.py` 처럼 scipy.ndimage.correlate(mode='nearest').
    decimate  `resize_images.py` · `fr_metrics.D_lambda_K` 의 `[r//2::r]` 표본 추출.

    툴박스의 Wald 열화(MS 경로)와 같은 모양이고 모르는 값은 GNyq 하나다.  GNyq 는 **train split 의 저자
    lpan 만으로** 맞췄다 (표본 500, 격자 0.3160~0.3200 / 0.0002): wv3 · qb · gf2 모두 0.318.  맞추는 데
    쓰지 않은 valid · reduced · full 에서 rel_l1 = mean|합성 - 저자| / mean|저자| 가 0.001% 대다
    (float32 로 저장된 파일의 반올림 수준).  테두리까지 맞으므로 저자도 표본(패치) 단위로 만들었다.

**스펙 머리말의 'gauss(1.7)+4x4 평균 0.637%' 는 무엇이었나** (`RULE_HEADER`, 비교용으로만 남긴다).
원래 스크립트는 없다.  머리말의 세 수가 모두

    nmae_range = mean|합성 - 저자| / (max(저자) - min(저자))        (gf2 reduced 20장 전체)

로 재현된다: 4x4 평균 1.4175 · bicubic 3.0922 · gauss1.7(scipy 기본 테두리)+4x4 평균 0.6373 (%).
범위로 나눈 값이라 작아 보였을 뿐, 평균으로 나누면 gauss1.7+avg4 는 1.6~4.3% 어긋난다 (sigma 1.7 은
그 계열 안에서는 최적이다).  원인은 표본 추출이다 -- 저자는 4x4 평균이 아니라 한 점을 뽑는다.

어디서 부르나: `validate_pancrafter_lpan.py` (관문 · WV2 zero-shot), `redo_zeroshot_qa.py`
(코퍼스 zero-shot, 관문 통과 시에만).  **학습에는 쓰지 않는다** -- PanCollection 학습은 계속 저자 파일이다.
"""
from __future__ import annotations
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RATIO = 4
GNYQ = 0.318              # train split 저자 lpan 으로 맞춘 값 (wv3 · qb · gf2 같음)
MTF_N = 41                # genMTF.py:35
KAISER_BETA = 0.5         # genMTF.py:59
RULE_MTF = "mtf0.318+dec4"
RULE = RULE_MTF             # 관문 · zero-shot 이 쓰는 규칙은 이것 하나 (PROTOCOL 과 함께 바꾼다)
#: 조각 · 결과에 적는 이름.
LABEL = f"synthetic {RULE}"
#: 규약에 싣는 정의.  이것이 바뀌면 합성 lpan 으로 낸 조각 · 관문은 모두 낡는다.
PROTOCOL = {"rule": RULE, "label": LABEL, "ratio": RATIO, "gnyq": GNYQ, "mtf_n": MTF_N,
            "kaiser_beta": KAISER_BETA,
            "filter": "Vivone toolbox genMTF single-band FIR (gaussian2d/kaiser2d/fir_filter_wind), "
                      "scipy.ndimage.correlate mode='nearest'",
            "decimate": "[r//2::r] (toolbox resize_images / D_lambda_K phase)",
            "gnyq_fit": "train split author lpan only (500 patches per dataset, seed 1)",
            "input": "the normalised PAN the method receives (RR: RR-scale PAN -> LRMS size; "
                     "FR: native PAN -> native MS size)", "version": 1}
#: 스펙 머리말의 옛 근사 (비교표 전용).  관문 · zero-shot 에는 쓰지 않는다.
RULE_HEADER = "gauss1.7+avg4"
TOOLBOX = (_GB + "/repos/AWFLN")      # pancoll.q2n · fr_metrics 와 같은 사본

# ------------------------------------------------------------------ 관문 (사전등록)
#: 관문 파일.  `validate_pancrafter_lpan.py` 가 **모든 계산이 끝났을 때만** 쓴다.
GATE_PATH = Path((_GB + "/output/pancrafter_lpan_gate.json"))
GATE_DATASETS = ("wv3", "qb", "gf2")
#: 사전등록 문턱 (결과를 보기 전).  같은 가중치 · 같은 평가 경로에서 저자 lpan 대 합성 lpan.
PREREG = {"rr_psnr_abs_db": 0.05, "hqnr_abs": 0.005, "datasets": list(GATE_DATASETS),
          "rule": "pass only if on ALL of wv3/qb/gf2 |dRR PSNR| <= 0.05 dB and |dHQNR| <= 0.005 "
                  "(author vs synthetic lpan, same weights); otherwise PAN-Crafter stays '--' in "
                  "zero-shot tables"}
#: 관문을 못 넘으면 zero-shot 표 캡션.
FAIL_CAPTION = "requires author-supplied lpan"
CKPT = (_GB + "/runs_pancoll/PAN-Crafter_{ds}/best.pt")


# ------------------------------------------------------------------ 합성
_KERNEL = {}


def mtf_kernel(gnyq: float = GNYQ, ratio: int = RATIO, N: int = MTF_N) -> np.ndarray:
    """genMTF.py:55-60 의 한 밴드 몸통을 툴박스 함수 그대로 부른다 (GNyq 만 인자)."""
    key = (float(gnyq), int(ratio), int(N))
    if key not in _KERNEL:
        if TOOLBOX not in sys.path:
            sys.path.insert(0, TOOLBOX)
        from Pansharpening_Toolbox_Assessment_Python.tools import fir_filter_wind, gaussian2d, kaiser2d
        fcut = 1 / ratio
        alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(gnyq)))
        H = gaussian2d(N, alpha)
        _KERNEL[key] = np.real(fir_filter_wind(H / np.max(H), kaiser2d(N, KAISER_BETA)))
    return _KERNEL[key]


def lpan_from_pan(pan: torch.Tensor, ratio: int = RATIO, rule: str = RULE, gnyq: float = GNYQ,
                  sigma: float = 1.7) -> torch.Tensor:
    """(B,C,H,W) PAN -> (B,C,H/ratio,W/ratio) lpan.  float64 로 계산하고 입력 dtype · 장치로 돌려준다.

    표본(2D 영상)마다 따로 거른다 -- 배치를 어떻게 묶어도 같은 값이다.  선형이라 정규화 스칼라와
    순서를 바꿔도 같다 (lpan(pan/div) = lpan(pan)/div).  `rule` · `gnyq` · `sigma` 는 측정표용이다 --
    관문과 zero-shot 은 기본값(RULE)만 쓴다."""
    import scipy.ndimage as nd
    if pan.dim() != 4:
        raise ValueError(f"(B,C,H,W) 가 필요하다: {tuple(pan.shape)}")
    B, C, H, W = pan.shape
    if H % ratio or W % ratio:
        raise ValueError(f"PAN {H}x{W} 가 배율 {ratio} 로 나뉘지 않는다")
    x = pan.detach().to("cpu", torch.float64).numpy()
    if rule == RULE_MTF:
        h, p = mtf_kernel(gnyq, ratio), ratio // 2
        y = np.stack([[nd.correlate(x[b, c], h, mode="nearest")[p::ratio, p::ratio] for c in range(C)]
                      for b in range(B)])
        y = torch.from_numpy(y)
    elif rule == RULE_HEADER:
        # scipy.ndimage.gaussian_filter 기본값 (반표본 대칭 테두리, truncate 4) + 겹치지 않는 r x r 평균.
        y = np.stack([[nd.gaussian_filter(x[b, c], sigma) for c in range(C)] for b in range(B)])
        y = F.avg_pool2d(torch.from_numpy(y), ratio)
    else:
        raise ValueError(f"모르는 규칙 {rule!r}")
    return y.to(device=pan.device, dtype=pan.dtype)


def fidelity(syn: torch.Tensor, ref: torch.Tensor) -> dict:
    """합성 lpan 대 저자 lpan.  `nmae_range` 는 스펙 머리말(0.637%)의 정의다 -- 위 머리말 참고."""
    s, r = syn.double(), ref.double()
    d = (s - r).abs()
    return {"nmae_range_pct": 100 * float(d.mean() / (r.max() - r.min())),
            "rel_l1_pct": 100 * float(d.mean() / r.abs().mean()),
            "rel_l2_pct": 100 * float(((s - r) ** 2).sum().sqrt() / (r ** 2).sum().sqrt()),
            "n": int(s.shape[0])}


# ------------------------------------------------------------------ 저자 파일 대비 측정 (CPU)
def author_pairs(ds: str, split: str, n_max: int | None = None, seed: int = 0):
    """(pan, 저자 lpan) 을 [0,1] 로.  split: train | valid | reduced | full.  크면 n_max 개 표본
    (고정 seed, 정렬된 인덱스).  pan · lpan 만 읽는다 (train 의 gt · lms 는 수 GB 다)."""
    import h5py
    import pancoll as PC
    d, pre, _, div = PC.SPEC[ds]
    if split in PC.EX:
        sub, pat = PC.EX[split]
        fn = pat.format(p=pre)
        src = f"{PC.ROOT}/{d}/{sub}/{fn}"
        lf = f"{PC.LPAN_ROOT}/{fn.replace('.h5', '_pan.h5')}"
    else:
        src = f"{PC.ROOT}/{d}/{split}_{pre}.h5"
        lf = f"{PC.LPAN_ROOT}/{split}_{pre}_pan.h5"
    with h5py.File(src, "r") as f, h5py.File(lf, "r") as g:
        n = f["pan"].shape[0]
        assert g["lpan"].shape[0] == n, (src, lf)
        if n_max and n > n_max:
            idx = np.sort(np.random.default_rng(seed).choice(n, n_max, replace=False))
            pan, lp = f["pan"][idx], g["lpan"][idx]
        else:
            idx = None
            pan, lp = f["pan"][:], g["lpan"][:]
    to = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64) / div)   # noqa: E731
    return to(pan), to(lp), {"n_total": int(n), "n_used": int(pan.shape[0]),
                             "subsample_seed": seed if idx is not None else None}


def measure(dss=GATE_DATASETS, splits=("train", "valid", "reduced", "full"), n_max=1000,
            sigmas=(1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)):
    """{ds: {split: {규칙 · 머리말 근사 · 4x4 평균 · bicubic 의 fidelity, gauss sigma 주사}}}."""
    out = {}
    for ds in dss:
        out[ds] = {}
        for sp in splits:
            pan, lp, meta = author_pairs(ds, sp, n_max=n_max)
            row = {**meta, "rule": fidelity(lpan_from_pan(pan), lp),
                   "header_gauss1.7_avg4": fidelity(lpan_from_pan(pan, rule=RULE_HEADER), lp),
                   "avg4": fidelity(F.avg_pool2d(pan, RATIO), lp),
                   "bicubic": fidelity(F.interpolate(pan, scale_factor=1 / RATIO, mode="bicubic",
                                                     align_corners=False), lp),
                   "gauss_sweep_rel_l1": {s: fidelity(lpan_from_pan(pan, rule=RULE_HEADER, sigma=s),
                                                      lp)["rel_l1_pct"] for s in sigmas}}
            sw = row["gauss_sweep_rel_l1"]
            row["gauss_best_sigma"] = min(sw, key=sw.get)
            out[ds][sp] = row
            print(f"  {ds:4s} {sp:8s} n {meta['n_used']:>5}/{meta['n_total']:<5}  {RULE} rel_l1 "
                  f"{row['rule']['rel_l1_pct']:.4f}% (nmae_range {row['rule']['nmae_range_pct']:.4f}%)  | "
                  f"gauss1.7+avg4 {row['header_gauss1.7_avg4']['rel_l1_pct']:.3f}% "
                  f"(nmae_range {row['header_gauss1.7_avg4']['nmae_range_pct']:.3f}%)  avg4 "
                  f"{row['avg4']['rel_l1_pct']:.3f}%  bicubic {row['bicubic']['rel_l1_pct']:.3f}%  "
                  f"| gauss best sigma {row['gauss_best_sigma']}", flush=True)
    return out


# ------------------------------------------------------------------ 판정 · 관문 읽기 (하나)
def verdict(per_ds: dict):
    """사전등록 규칙 그대로.  ('pass' | 'fail', 걸린 항목 목록).  NaN 은 실패다."""
    fails = []
    for d in GATE_DATASETS:
        if d not in per_ds:
            fails.append(f"{d}: 없음")
            continue
        dp = per_ds[d]["rr"]["delta"]["psnr"]
        dh = per_ds[d]["fr"]["delta"]["hqnr"]
        if not abs(dp) <= PREREG["rr_psnr_abs_db"]:
            fails.append(f"{d}: |dRR PSNR| {abs(dp):.4f} > {PREREG['rr_psnr_abs_db']}")
        if not abs(dh) <= PREREG["hqnr_abs"]:
            fails.append(f"{d}: |dHQNR| {abs(dh):.5f} > {PREREG['hqnr_abs']}")
    return ("pass" if not fails else "fail"), fails


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def gate_check(path=None, ckpt=CKPT):
    """관문 파일이 **지금의 가중치에 대해** pass 라고 말하는가.  (ok, 사유, 요약).

    판정은 기록된 `verdict` 필드 하나다 ([[one-gate-not-two]]) -- 여기서 문턱을 다시 계산하지 않는다.
    대신 파일이 그 판정을 내릴 자격이 있는지를 본다:
      - 경로 시험(path_test)이 아니다 · 규칙(RULE, PROTOCOL) · 사전등록 문턱(PREREG)이 지금 것과 같다
      - wv3/qb/gf2 셋이 모두 있다 (판정이 셋의 논리곱이다)
      - 셋 모두 기록된 best.pt sha256 이 **지금의** best.pt 와 같다 -- 관문 뒤에 다시 학습됐으면 낡은 관문이다.
    """
    p = Path(path) if path is not None else GATE_PATH
    if not p.exists():
        return False, f"관문 파일 {p} 없음", {"path": str(p)}
    try:
        g = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return False, f"관문 파일을 못 읽는다: {type(e).__name__}", {"path": str(p)}
    info = {"path": str(p), "sha256": sha256(p), "verdict": g.get("verdict"), "rule": g.get("rule")}
    if g.get("path_test") or g.get("limit"):
        return False, "경로 시험(path test) 파일이다 -- 결과가 아니다", info
    if g.get("rule") != RULE or g.get("lpan_protocol") != PROTOCOL:
        return False, f"규칙이 다르다 ({g.get('rule')!r} != {RULE!r})", info
    if g.get("preregistered") != PREREG:
        return False, "사전등록 문턱이 지금 것과 다르다", info
    per = g.get("per_ds") or {}
    miss = [d for d in GATE_DATASETS if d not in per]
    if miss:
        return False, f"데이터셋 {miss} 이 관문에 없다", info
    for d in GATE_DATASETS:
        want = (per[d].get("ckpt") or {}).get("sha256")
        ck = Path(ckpt.format(ds=d))
        if not want or not ck.exists():
            return False, f"{d}: 체크포인트 기록 · 파일이 없다", info
        if sha256(ck) != want:
            return False, f"{d}: best.pt 가 관문을 잰 가중치와 다르다 (관문 뒤에 다시 학습됨) -- 관문을 다시 돈다", info
        info[f"ckpt_{d}"] = want
    if g.get("verdict") != "pass":
        return False, f"관문 판정 {g.get('verdict')!r} -- {FAIL_CAPTION}", info
    return True, "pass", info


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description="합성 lpan 대 저자 lpan 측정표 (CPU)")
    ap.add_argument("--ds", nargs="+", default=list(GATE_DATASETS))
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "reduced", "full"])
    ap.add_argument("--n_max", type=int, default=1000, help="표본이 이보다 많으면 부분표본 (seed 0)")
    ap.add_argument("--json", default=None, help="측정표를 여기에 쓴다 (선택)")
    a = ap.parse_args()
    res = measure(a.ds, a.splits, a.n_max)
    if a.json:
        Path(a.json).write_text(json.dumps({"rule": RULE, "protocol": PROTOCOL, "table": res}, indent=1))
