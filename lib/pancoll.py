"""PanCollection 공식 split 로더 + 표준 RR 지표.

PanCollection 은 이 분야의 정본이고 **우리 사전학습 코퍼스에 한 번도 들어간 적이 없다**
(`_corpus/metadata/eval_protocols.yaml` 의 `external_blind_pancollection`).
저장소들이 기대하는 계약 그대로다: gt / lms(업샘플 MS) / ms(저해상) / pan.

정규화 나눗수는 추측하지 않고 데이터에서 확인했다 -- WV3/QB 는 11비트(2047),
GF2 는 10비트(1023).  저장소들이 쓰는 값과 같다.
"""
from __future__ import annotations
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_PCR = _os.environ.get("PANCOLLECTION_ROOT") or _os.path.join(_GB, "data", "pancollection")
import numpy as np, torch

ROOT = _PCR
SPEC = {  # (디렉터리, 파일접두, 밴드, 나눗수)
    "wv3": ("WV3", "wv3", 8, 2047.0),
    "qb":  ("QB",  "qb",  4, 2047.0),
    "gf2": ("GF2", "gf2", 4, 1023.0),
    # WV2 는 examples 만 있고 train/valid 가 **없다**.  그래서 학습 대상이 아니라
    # **zero-shot 대상**이다: WV3(8밴드, 같은 나눗수)에서 학습한 모델을 그대로 태운다.
    "wv2": ("WV2", "wv2", 8, 2047.0),
}
ZEROSHOT = {"wv3": ["wv2"]}          # 학습 데이터셋 -> 추가로 평가할 봉인 데이터셋


# PAN-Crafter(ICCV 2025)만 쓰는 추가 입력 `lpan`(MS 해상도의 저역 PAN).
#
# 어디서 왔나.  PanCollection 의 h5 에는 없다.  저자들이 저장소에 직접 동봉했다
# (`repos/PAN-Crafter/pan_h5/pan_h5.zip`, 12개 파일, 개수가 PanCollection 과 정확히
# 일치).  형제 파일 `<split>_pan.h5` 로 읽게 되어 있다.
#
# 왜 우리가 다시 만들지 않는가.  만들 수는 있지만 **bicubic 이 아니다.**  gf2 reduced
# 20장에서 재보면 상대오차가
#     gauss(sigma=1.7) + 4x4 평균   0.637%      <- 사실상 이것 (Wald 규약의 MTF 열화)
#     4x4 평균 / area               1.418%
#     bicubic resize                3.092%
# 이다.  저자 파일이 있는데 근사치를 쓰면 그 방법만 다른 입력으로 학습시키는 셈이다.
#
# 코퍼스(`$PAN_DATASETS`)는 읽기 전용이므로 여기 따로 둔다.
LPAN_ROOT = (_GB + "/data_lpan")


def _lpan(fname: str, div: float):
    """`data_lpan/<fname>` 이 있으면 정규화해 돌려주고, 없으면 None."""
    import os
    import h5py
    p = f"{LPAN_ROOT}/{fname}"
    if not os.path.exists(p):
        return None
    with h5py.File(p, "r") as f:
        return torch.from_numpy(np.asarray(f["lpan"], dtype=np.float32) / div)


# Global-skip 변형 연구 (2026-10-09).  "lrms" 가 아니면 `lms` 자리에 Stage-1 base S 를 넣는다 -- 모델이 받는
# `lms`(= global skip 으로 더하는 업샘플 MS)만 바뀌고 gt · ms · pan 은 그대로다.  S 는 미리 계산한 sidecar h5
# (`baselines/stage1/make_skip_data.py`, PanCollection 과 같은 상대 경로, `lms` 하나)에서 읽는다.
# load_examples 는 지표용 원래 lms 를 `lms_orig` 로 남긴다 (FR 의 HQNR 은 원래 lms 로 잰다).
SKIP = "lrms"
SKIP_ROOT = (_GB + "/data_skip")
SKIPS = ("lrms", "S_PAN", "S_MS")
# base 사다리 (2026-10-16 ablation, stage1/make_ladder_data.py): 같은 sidecar 형식의 다른 base -- Π(EXP) (측정 일관성만),
# GSA (고전 CS), MTF-GLP-HPM (고전 MRA), U_CBD (매개변수 0 물리 base).  "base 가 좋을수록 최종이 좋은가" 를 같은 in 팔로 잰다.
SKIPS_LADDER = ("PiEXP", "GSA", "GLPHPM", "UCBD",
                # 2026-10-18 VO 단 (stage1/make_ladder_vo.py): 표의 고전 · VO 행 가운데 RR 최고인 LRTCFPan 을 base 로.  로컬 전용
                # (원격 pancoll.py 에는 없다 -- 그 칸은 로컬 큐에서만 돈다)
                "LRTCFPan")
SKIPS = SKIPS + SKIPS_LADDER
# 2026-10-20 QB 위상 보정 ablation (stage1/make_skip_qbfix.py): QB 학습 · 검증 쌍 2/3 가 열화 뒤에 뒤집은 사본이라 ms 의 분해 위치가
# 한 축 1 이다.  패치마다 위치를 바로잡아 계산한 같은 S_PAN / S_MS (test 파일은 기존과 같음).  QB 전용 · 로컬 전용.
SKIPS = SKIPS + ("S_PANqbfix", "S_MSqbfix")
# in 팔 (입력에도 S, lib/skipvar.py): `lms` = cat[S, 원래 lms] (2C 채널).  트레이너가 스펙의 lms_stack 을 보고 켠다.
LMS_STACK = False
# 학습 · 검증 GT 의 기하 (2026-10-13 파일럿, stage1/make_gt_geo.py).  "ms" = 원래 GT (MS 기하).  "pan_detail" = 저역은 그대로, MS Nyquist
# 위의 디테일만 PAN 기하로 옮긴 GT (sidecar).  load_examples (test 의 RR · FR) 는 바꾸지 않는다 -- 표의 RR 은 원래 GT 로 잰다.
GT_GEO = "ms"
GT_GEO_ROOT = (_GB + "/data_gtgeo")


def _gt_geo(rel: str, n_expect: int, div: float):
    """GT_GEO 의 sidecar GT/div.  없으면 멈춘다 -- 조용히 원래 GT 로 떨어지지 않는다."""
    import h5py
    p = f"{GT_GEO_ROOT}/{GT_GEO}/{rel}"
    with h5py.File(p, "r") as f:
        x = torch.from_numpy(np.asarray(f["gt"], dtype=np.float32) / div)
    assert x.shape[0] == n_expect, (p, x.shape, n_expect)
    return x


def _skip_lms(rel: str, n_expect: int, div: float):
    """SKIP 의 sidecar 에서 S/div (float32 텐서).  없으면 멈춘다 -- 조용히 원래 lms 로 떨어지지 않는다."""
    import h5py
    assert SKIP in SKIPS and SKIP != "lrms", SKIP
    p = f"{SKIP_ROOT}/{SKIP}/{rel}"
    with h5py.File(p, "r") as f:
        x = torch.from_numpy(np.asarray(f["lms"], dtype=np.float32) / div)
    assert x.shape[0] == n_expect, (p, x.shape, n_expect)
    return x


def load(dataset: str, split: str):
    """split 은 'train' 또는 'valid'.  float32 정규화 텐서를 메모리에 올린다."""
    import h5py
    d, pre, nb, div = SPEC[dataset]
    with h5py.File(f"{ROOT}/{d}/{split}_{pre}.h5", "r") as f:
        out = {k: torch.from_numpy(np.asarray(f[k], dtype=np.float32) / div)
               for k in ("gt", "lms", "ms", "pan")}
    assert out["gt"].shape[1] == nb, (out["gt"].shape, nb)
    if GT_GEO != "ms":
        g = _gt_geo(f"{d}/{split}_{pre}.h5", out["gt"].shape[0], div)
        assert g.shape == out["gt"].shape, (g.shape, out["gt"].shape)
        out["gt"] = g
    assert not (LMS_STACK and SKIP == "lrms"), "LMS_STACK 는 --skip S_PAN|S_MS 런에서만 쓴다"
    if SKIP != "lrms":
        S = _skip_lms(f"{d}/{split}_{pre}.h5", out["lms"].shape[0], div)
        assert S.shape == out["lms"].shape, (S.shape, out["lms"].shape)
        out["lms"] = torch.cat([S, out["lms"]], 1) if LMS_STACK else S
    lp = _lpan(f"{split}_{pre}_pan.h5", div)
    if lp is not None:
        assert lp.shape[0] == out["gt"].shape[0], (lp.shape, out["gt"].shape)
        assert lp.shape[-1] == out["ms"].shape[-1], ("lpan 은 MS 해상도여야 한다",
                                                     lp.shape, out["ms"].shape)
        out["lpan"] = lp
    return out


# ------------------------------------------------------------------ 지표
def psnr(x, y, data_range=1.0):
    mse = torch.mean((x - y) ** 2, dim=(1, 2, 3)).clamp_min(1e-20)
    return (20 * np.log10(data_range) - 10 * torch.log10(mse))


def sam(x, y, eps=1e-12):
    """평균 스펙트럴 각 (도).  밴드 축 기준.

    float64 로 계산한다.  float32 로는 동일한 두 영상에서도 num/den 이 1 을 살짝 밑돌아
    0.009 도의 계통 바닥이 생긴다 -- 작지만 0 이어야 할 자리에 0 이 아닌 수가 남는다."""
    x, y = x.double(), y.double()
    num = (x * y).sum(1)
    den = x.norm(dim=1) * y.norm(dim=1) + eps
    a = torch.arccos((num / den).clamp(-1.0, 1.0))
    return torch.rad2deg(a).flatten(1).mean(1)


def ergas(x, y, ratio=4, eps=1e-8):
    """Wald 의 ERGAS.  밴드별 RMSE 를 그 밴드의 평균으로 정규화한다."""
    mse = ((x - y) ** 2).mean(dim=(2, 3))                 # B,C
    mu = y.mean(dim=(2, 3)).abs().clamp_min(eps)          # B,C
    return 100.0 / ratio * torch.sqrt((mse / mu ** 2).mean(1))


def ssim(x, y, data_range=1.0, w=11, sigma=1.5):
    """밴드별 SSIM 의 평균.  가우시안 창, 표준 상수."""
    import torch.nn.functional as F
    c = x.shape[1]
    g = torch.arange(w, dtype=x.dtype, device=x.device) - (w - 1) / 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2)); g = (g / g.sum())
    k = (g[:, None] @ g[None, :]).expand(c, 1, w, w).contiguous()
    mu_x = F.conv2d(x, k, groups=c); mu_y = F.conv2d(y, k, groups=c)
    xx = F.conv2d(x * x, k, groups=c) - mu_x ** 2
    yy = F.conv2d(y * y, k, groups=c) - mu_y ** 2
    xy = F.conv2d(x * y, k, groups=c) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (xx + yy + c2))
    return s.flatten(1).mean(1)


_Q2N = {}


def _skimage_shim():
    """`D_s` 는 skimage 의 SSIM 을 `data_range` 없이 부른다.

    이 툴박스가 쓰이던 시절의 skimage 는 float 입력에 `dtype_range[float64] = (-1, 1)`,
    즉 **data_range = 2** 를 적용했다.  최신 skimage 는 이를 오류로 막는다.  공표된
    D_s / HQNR 수치가 그 동작으로 산출됐으므로 **같은 값을 주입해 재현한다** -- 여기서
    다른 값을 쓰면 우리 표가 문헌과 비교 불가능해진다.
    """
    import skimage.metrics as _sm
    if getattr(_sm, "_panbase_patched", False):
        return
    _orig = _sm.structural_similarity
    def _ssim(a, b, **kw):
        kw.setdefault("data_range", 2.0)
        return _orig(a, b, **kw)
    _sm.structural_similarity = _ssim
    _sm._panbase_patched = True
    import sys
    for m in list(sys.modules.values()):                 # 이미 import 한 곳도 갈아끼운다
        if m is not None and getattr(m, "__name__", "").startswith(("D_s", "Q")):
            if hasattr(m, "ssim"):
                m.ssim = _ssim


def _np_shim():
    """툴박스가 NumPy 1.24 에서 제거된 `np.int` 등의 별칭을 쓴다.  저장소 코드를 고치는
    대신 별칭을 되살린다 -- 동작은 동일하고(둘 다 builtin 으로 해석됨) 원본은 그대로 남는다."""
    import numpy as _np
    for a, b in (("int", int), ("float", float), ("bool", bool), ("object", object)):
        if not hasattr(_np, a):
            setattr(_np, a, b)


def q2n(pred, gt, divisor):
    """Q2n -- Vivone et al. Pansharpening Toolbox 의 공식 Python 이식본.

    팬샤프닝 문헌의 정본이라 이것을 쓴다 (`repos/AWFLN/Pansharpening_Toolbox_Assessment_Python`).
    HS-Pan-Toolbox(TGRS'25)의 독립 구현과 4.3e-04 까지 일치함을 확인했다 -- 두 공식
    구현이 서로를 뒷받침하므로 이 수치는 믿을 수 있다.

    블록 32 / shift 32.  **DN 을 넣어야 한다**; [0,1] 을 그대로 주면 값이 무의미해진다.
    """
    import sys
    import numpy as np
    ROOT = (_GB + "/repos/AWFLN")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    _np_shim()
    if "f" not in _Q2N:
        from Pansharpening_Toolbox_Assessment_Python.q2n import q2n as _f
        _Q2N["f"] = _f
    f = _Q2N["f"]
    g = gt.detach().double().cpu().numpy() * divisor
    p = pred.detach().double().cpu().numpy() * divisor
    out = [f(g[i].transpose(1, 2, 0), p[i].transpose(1, 2, 0), 32, 32)[0]
           for i in range(g.shape[0])]
    return torch.as_tensor(out, dtype=torch.float64)


#: PanCollection 의 RR 평가는 테두리를 잘라낸 뒤 지표를 잰다.
#: repos/ARConv/MetricCode/Demo1_Reduced_Resolution_MultiExm_wv3.m:47-48 이
#:     flag_cut_bounds = 1;  dim_cut = 30;  % PanCollection: 30, DLPan: 21
#: 이고, 툴박스의 indexes_evaluation.py:44-46 이 그 값으로
#:     I[dim_cut-1 : shape-dim_cut]
#: 를 GT 와 예측 **양쪽에** 적용한다.  256x256 이면 197x197 만 평가에 쓰인다.
#: 공개된 PanCollection RR 수치는 전부 이 규약으로 계산된 것이라, 자르지 않으면
#: 우리 표의 절대값이 문헌과 비교 불가능해진다(테두리가 포함되어 낮게 나온다).
#: **FR 은 자르지 않는다** -- indexes_evaluation_FS 에는 dim_cut 인자 자체가 없고,
#: FR 데모의 dim_cut=21 은 그림 출력용(showImage_zoomin)이다.
RR_DIM_CUT = 30


def eval_protocol(spec=None) -> str:
    """결과 JSON 의 `recipe.eval_protocol` 에 적는 평가 규약 문자열.  **이 함수 하나**가 정한다.

    학습기가 적는 값(train_pancoll)과 게이트가 대조하는 값(make_tables._recipe_now)이 같은
    함수에서 나와야 한다 -- 예전에는 둘이 각자 "rr_dim_cut30" 을 들고 있었다.
    스펙에 `eval_tile` 이 있으면(UTeRM) 추론을 그 크기 타일로 나눠 하므로 뒤에
    `+tile<N>` 이 붙는다 -- 같은 가중치라도 다른 수가 나오는 규약이다.  `spec=None` 은 우리
    행의 규약이다."""
    base = f"rr_dim_cut{RR_DIM_CUT}"
    t = (spec or {}).get("eval_tile")
    return f"{base}+tile{int(t)}" if t else base


def rr_cut(t, dim_cut: int = RR_DIM_CUT):
    """RR 평가용 테두리 잘라내기.  64x64 검증 패치에는 쓰지 않는다(5픽셀만 남는다)."""
    h, w = t.shape[-2], t.shape[-1]
    if h <= 2 * dim_cut or w <= 2 * dim_cut:
        raise ValueError(f"dim_cut {dim_cut} 이 크기 {h}x{w} 에 비해 너무 크다")
    return t[..., dim_cut - 1:h - dim_cut, dim_cut - 1:w - dim_cut]


def all_metrics(pred, gt, ratio=4, divisor=None):
    out = dict(psnr=psnr(pred, gt), ssim=ssim(pred, gt),
               sam=sam(pred, gt), ergas=ergas(pred, gt, ratio))
    if divisor is not None:
        out["q2n"] = q2n(pred, gt, divisor)
    return out


# ------------------------------------------------- 문헌이 보고하는 test 집합
# `valid_*.h5` 는 모델 선택용이다.  논문 표는 아래 두 집합으로 보고한다.
EX = {"reduced": ("reduced_examples_h5", "test_{p}_multiExm1.h5"),
      "full":    ("full_examples_h5",    "test_{p}_OrigScale_multiExm1.h5")}


def load_examples(dataset: str, kind: str = "reduced"):
    """reduced = RR (gt 있음, 256x256).  full = FR (gt 없음, 512x512)."""
    import h5py
    d, pre, nb, div = SPEC[dataset]
    sub, pat = EX[kind]
    fn = pat.format(p=pre)
    with h5py.File(f"{ROOT}/{d}/{sub}/{fn}", "r") as f:
        out = {k: torch.from_numpy(np.asarray(f[k], dtype=np.float32) / div)
               for k in f.keys()}
    assert not (LMS_STACK and SKIP == "lrms"), "LMS_STACK 는 --skip S_PAN|S_MS 런에서만 쓴다"
    if SKIP != "lrms":
        S = _skip_lms(f"{d}/{sub}/{fn}", out["lms"].shape[0], div)
        assert S.shape == out["lms"].shape, (S.shape, out["lms"].shape)
        out["lms_orig"], out["lms"] = out["lms"], (torch.cat([S, out["lms"]], 1) if LMS_STACK else S)
    lp = _lpan(fn.replace(".h5", "_pan.h5"), div)
    if lp is not None:
        assert lp.shape[0] == out["pan"].shape[0], (lp.shape, out["pan"].shape)
        out["lpan"] = lp
    return out


# 툴박스의 MTF 표에 있는 이름.  GF2 는 표에 없어 기본 GNyq 0.3 으로 떨어지는데,
# 이 분야 문헌이 같은 툴박스로 GF2 를 보고할 때와 동일한 처리다.
# **WV2 는 WV3 의 MTF 게인으로 잰다** (사용자 결정 2026-09-28).  툴박스 파이썬 이식본의
# genMTF 'WV2' 가지(`GNyq = [0.35*np.ones(nbands), 0.27]`)는 어떤 밴드 수에서도 깨져 HQNR 을 못 낸다.  WV2 와 WV3 는
# 같은 8밴드 배치(Coastal .. NIR2) · 같은 배율 4 의 형제 센서이고, 센서 이름이 바꾸는 것은 D_lambda (Khan) 의 MS MTF
# 필터뿐이다 (D_s 와 RR 지표는 센서와 무관).  원래(MATLAB) 툴박스의 WV2 값 [0.35 x 7, 0.27] 로 잰 민감도는.
FR_SENSOR = {"wv3": "WV3", "qb": "QB", "gf2": "none", "wv2": "WV3"}


def hqnr(fused, ms, lms, pan, dataset, ratio=4, S=32):
    """full-resolution 지표.  반환: (HQNR, D_lambda, D_s) 각각 표본별 텐서.

    **계산은 `ps_metrics.fr_metrics.hqnr_dn` 하나만 한다.**  여기는 단위 변환 껍데기다.
    전에는 Vivone 툴박스를 부르는 코드가 이 파일에만 있었고, 사전학습 스크리닝에는
    FR 지표가 아예 없었다.  FR 을 그쪽에도 붙이면서 같은 계산을 복사하면 두 표면이
    조용히 갈라지므로 (`one-gate-not-two` 와 같은 이유), 핵심을 옮기고 이쪽이 부른다.
    """
    from ps_metrics.fr_metrics import hqnr_dn
    div = SPEC[dataset][3]
    to = lambda x: x.detach().double().cpu().numpy() * div
    o = hqnr_dn(to(fused), to(ms), to(lms), to(pan),
                sensor=FR_SENSOR[dataset], ratio=ratio, S=S)
    return tuple(torch.as_tensor(x, dtype=torch.float64) for x in o)
