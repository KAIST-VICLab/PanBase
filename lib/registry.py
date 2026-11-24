"""베이스라인 모델 어댑터.

원칙: **저장소 소스를 고치지 않는다.**  밴드 수가 하드코딩된 경우에만 서브클래스로 그 한
줄을 일반화하고 무엇을 바꿨는지 `diff` 에 남긴다.  손실·옵티마이저·학습률·배치는 각 저장소가
쓰는 값 그대로 -- "우리가 불리하게 학습시켰다"는 반박을 막기 위해서다.

계약:  build(ms_ch, ratio) -> nn.Module
       forward(model, ms_up, pan, lrms, ctx) -> pred
       ms_up/lrms/pred 는 모델 정규화 공간, pan 은 (B,1,H,W).
       ctx = {"step", "total"} -- 학습 진행도가 필요한 모델(ARConv)만 쓴다.
"""
from __future__ import annotations
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
import sys
from pathlib import Path
import importlib.util
import torch, torch.nn as nn, torch.nn.functional as F

REPOS = Path((_GB + "/repos"))
SPECS: dict[str, dict] = {}

# 스펙의 **함수**들에 대한 지문.
#
# 왜 필요한가: `_RECIPE_KEYS` 는 batch/epochs/opt/sched 같은 값만 비교한다.  그런데
# 학습을 실제로 수행하는 것은 `train_step`(13개 스펙), `forward`(31개), `make_opt`(7개),
# `on_epoch`/`on_train_start` 같은 **함수**들이고, 이것들은 비교 대상이 아니었다.
# 즉 그중 하나를 고치면 이전 결과가 **조용히 유효한 채로 남는다**.
#
# 실제로 그럴 뻔했다: DCPNet 의 `forward` 에서 `[0]` 을 떼어 보조 손실이 살아나게 한
# 수정은 레시피 스냅샷을 전혀 바꾸지 않는다.  수정 전 결과가 게이트를 그대로 통과했을
# 것이고(목적함수의 43% 가 빠진 채로), 우리는 그것을 측정으로만 확인할 수 있었다.
#
# 소스 텍스트가 아니라 **바이트코드**를 해싱한다 -- 주석이나 줄바꿈을 고쳐도 지문이
# 바뀌지 않아야 하고, 파일 이름과 줄 번호에 의존해서도 안 된다(위에 주석 한 줄만 넣어도
# 모든 칸이 낡음으로 뒤집힌다).  `co_code`/`co_names`/`co_varnames`/`co_consts` 만 쓴다.
_CODE_FIELDS = ("build", "forward", "make_opt", "train_step", "on_train_start", "on_epoch")


def _code_fingerprint(fn):
    c = getattr(fn, "__code__", None)
    if c is None:
        return repr(fn)

    def norm(co):
        return (bytes(co.co_code), co.co_names, co.co_varnames,
                tuple(norm(k) if hasattr(k, "co_code") else
                      (k if isinstance(k, (int, float, str, bytes, bool, type(None)))
                       else repr(k))
                      for k in co.co_consts))
    return repr(norm(c))


def spec_code_digest(spec):
    """스펙의 함수들을 한 줄 지문으로 만든다.  없는 항목은 빈 문자열로 자리를 지킨다."""
    import hashlib
    parts = [f"{k}={_code_fingerprint(v) if callable(v := spec.get(k)) else ''}"
             for k in _CODE_FIELDS]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]



def _load(path: Path, name: str):
    """파일 경로로 직접 적재한다.  저장소들이 `model.py` / `models.py` 같은 흔한 이름을
    쓰기 때문에 sys.path 에 얹으면 서로를 가린다 (LAGConv/model.py 가 PGCU/model/ 을 가렸다)."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m



def _pkg(root: Path, name: str = "model"):
    """`model` 이라는 흔한 패키지 이름을 저장소마다 갈아 끼운다.

    sys.path 순서로는 해결되지 않는다 -- 파이썬은 __init__.py 가 있는 정규 패키지를
    네임스페이스 패키지보다 우선하므로, HFIN/model 이 한 번 잡히면 UTeRM/model 은
    경로를 앞에 넣어도 가려진다.  그래서 패키지 객체를 직접 만들어 __path__ 를 박는다."""
    import types
    for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        sys.modules.pop(k)
    m = types.ModuleType(name); m.__path__ = [str(root)]
    sys.modules[name] = m
    return m

def _hp(x, k=5):
    """PanNet 의 고주파 성분 -- 밴드별 **미끄러지는** 5x5 평균을 뺀 것.

    PanCollection/UDL 표준 `get_edge` 와 같고, PanNet 논문이 말하는 고주파다.

    주의: repos/PGCU/utils/funcation.py:6-8 도 `x - cv2.boxFilter(x,-1,(5,5))` 한
    줄이지만, 호출 지점(utils/dataset.py:15,17)이 **이미 (C,H,W) 로 transpose 된**
    배열을 넘긴다.  cv2 는 3차원 배열을 행 x 열 x 채널로 읽으므로, MS 에서는
    (밴드, 행) 평면의 5x5 -- 즉 인접 **밴드를 섞는** 필터가 되고, (1,H,W) 인 PAN 에서는
    높이축 1차원 5-tap 으로 퇴화한다.  저장소의 레이아웃 버그이지 PanNet 의 정의가
    아니므로 재현하지 않는다.  그 결과 우리 신호는 저장소가 실제로 계산하는 것과
    다르다 (wv3 valid 기준 hlrms relL2 0.85 / r 0.57, hpan relL2 0.69 / r 0.85).
    버그를 재현하는 쪽이 엄격히 더 나쁘므로, 이 차이는 숨기지 않고 여기 적어 둔다.

    전에는 `x - nearest(avg_pool2d(x, 4))`, 즉 겹치지 않는 4x4 블록 평균을 썼다.
    차단 주파수가 다를 뿐 아니라 블록 경계 계단이 신호에 섞인다.  두 고주파
    신호는 상대 표준편차로 24% 다르다.  PanNet 의 기여가 "고주파 도메인에서
    학습한다" 는 것이므로, 이건 입력을 바꾸는 것이고 곧 다른 방법을 학습하는 것이다.

    테두리는 cv2 기본값 BORDER_REFLECT_101 과 같은 torch `mode="reflect"` 를 쓴다."""
    p = k // 2
    return x - F.avg_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), k, stride=1)


# ------------------------------------------------------------------ LAGConv
def _lagconv(ms_ch=4, ratio=4):
    M = _load(REPOS / "LAGConv" / "model.py", "lagconv_model")
    class LACNET_n(M.LACNET):
        def __init__(self, c):
            super().__init__()
            self.head_conv[0] = M.LAConv2D(1 + c, 32, 3, 1, 1, use_bias=True)
            self.tail_conv = M.LAConv2D(32, c, 3, 1, 1, use_bias=True)
    return LACNET_n(ms_ch)

SPECS["LAGConv"] = dict(
    tier="confirmed",
    build=_lagconv, venue="AAAI 2022", loss="mse", batch=32, epochs=500,
    opt=dict(kind="adam", lr=3e-4, betas=(0.9, 0.999)),
    forward=lambda m, ms_up, pan, lrms, ctx: m(pan, ms_up),
    diff="LACNET 이 밴드를 9=1+8, tail 8 로 하드코딩한다.  그 두 상수만 (1+C), C 로 바꾼 "
         "서브클래스.  나머지는 원본 그대로.")


# ------------------------------------------------------------------- ARConv
def _arconv(ms_ch=4, ratio=4):
    _load(REPOS / "ARConv" / "models" / "ARConv.py", "ARConv")   # models.py 가 이 이름으로 import 한다
    MM = _load(REPOS / "ARConv" / "models" / "models.py", "arconv_models")
    return MM.ARNet(pan_channels=1, lms_channels=ms_ch)      # 밴드가 이미 인자다

def _arconv_fwd(m, ms_up, pan, lrms, ctx):
    # 셋째 인자는 **절대 에폭**이다.  models/ARConv.py:107-126 은 `epoch == 100` 에서
    # 학습된 커널 기하(N_X, N_Y)를 reserved_NXY 로 **영구 고정**한다 -- 논문의 기여
    # 그 자체다.  전에는 `1 + int(600*step/total)` 로 항상 600 에폭을 사상해서,
    # 에폭이 데이터셋마다 달라진 뒤로 QB(200 에폭)에서는 실제 33 에폭째에 얼어붙었다.
    # 저자는 100/200, 즉 예산의 절반을 탐색에 준다.  벽시계 에폭을 그대로 넘긴다.
    # ARConv 는 적응 커널 범위를 epoch 으로 스케줄한다.  저장소 기본 600 epoch 을
    # 우리 스텝 예산에 선형 사상해, 그들이 의도한 스케줄을 그대로 밟게 한다.
    ep = ctx.get("epoch")
    if ep is None:
        # 평가 ctx 에는 epoch 이 없다.  고정 이후 값을 줘서 latch 된 reserved_NXY 를 쓴다.
        ep = 10 ** 6
    return m(pan, ms_up, ep, hw_range=[0, 18])

SPECS["ARConv"] = dict(
    tier="confirmed",
    build=_arconv, venue="CVPR 2025", loss="l1", batch=16, epochs=600,
    opt=dict(kind="adam", lr=6e-4, betas=(0.9, 0.999)),
    # trainer.py:74 `StepLR(optimizer, step_size=200, gamma=0.8)`, trainer.py:106 에서
    # 에폭마다 step.  전에는 스케줄러가 없어 학습률이 끝까지 6e-4 로 고정돼 있었다.
    # QB 는 에폭이 200 이라 첫 경계가 마지막 에폭에 걸려 사실상 무영향이다.
    sched=dict(kind="step", step_size=200, gamma=0.8),
    # 에폭 수가 데이터셋마다 다르다 -- scripts/train_{wv3,gf2,qb}.sh:3 이 각각
    # 600 / 630 / 200 을 넘긴다.  하나로 뭉개면 두 데이터셋이 저자 예산과 달라진다.
    # eval_after_frac 은 커널 기하가 얼어붙기 전 가중치가 best.pt 로 저장되는 것을
    # 막는 장치다.  고정 지점이 100/epochs 라 데이터셋마다 다르다:
    # wv3 16.7% · gf2 15.9% · qb **50%**.  QB 만 0.18 로 두면 33~99 에폭 구간이
    # 선택될 수 있고, 검증 순전파가 epoch 100 안에 들어가면 reserved_NXY 를
    # 검증 배치로 덮어쓴다 (ARConv.py:119-123).
    per_dataset={"gf2": dict(epochs=630),
                 "qb": dict(epochs=200, eval_after_frac=0.51)},
    forward=_arconv_fwd, eval_after_frac=0.18,
    diff="없음 -- ARNet 이 lms_channels 를 인자로 받는다.  hw_range 는 저장소 기본값 [0,18], "
         "epoch 은 저장소 기본 600 을 우리 스텝 예산에 선형 사상.  ARConv 는 epoch 100 에서 "
         "적응 커널 크기를 reserved_NXY 로 **동결**하므로, 그 이전 구간(예산의 18%)은 "
         "체크포인트 선택에서 제외한다 -- 동결 전 가중치와 동결 후 구조가 섞이면 안 된다.")


# ------------------------------------------------------------------- PanNet
def _pannet(ms_ch=4, ratio=4):
    # PanNet.py 는 상대 import 를 쓰므로 패키지로 적재해야 한다.  `model` 이름이
    # 비어 있는지 확인한 뒤 PGCU 루트를 맨 앞에 둔다.
    sys.modules.pop("model", None)
    sys.path.insert(0, str(REPOS / "PGCU"))
    from model.PanNet import PanNet
    m = PanNet(channel=ms_ch)                              # 밴드가 이미 인자다
    if ratio != 4:                                         # ConvTrans 가 x4 고정이다
        raise ValueError("PanNet 의 ConvTrans 는 x4 전용이다 (그룹 P 만).")
    return m

SPECS["PanNet"] = dict(
    tier="confirmed",
    build=_pannet, venue="ICCV 2017 (PGCU/CVPR2023 저장소 구현)", loss="mse", batch=32, epochs=500,
    opt=dict(kind="adam", lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-5),
    forward=lambda m, ms_up, pan, lrms, ctx: m(pan, lrms, _hp(lrms), _hp(pan)),
    diff="없음 -- channel 이 인자다.  고주파 입력(hms/hpan)은 저장소 데이터로더가 주던 "
         "것이므로 PanNet 원논문 정의(원본 - 평균필터)로 우리가 만든다.")


# --------------------------------------------------------------------- PGCU
def _pgcu(ms_ch=4, ratio=4):
    sys.modules.pop("model", None)
    sys.path.insert(0, str(REPOS / "PGCU"))
    from model.PanNet import PanNet_PGCU
    from model.PGCU import PGCU as _PGCU
    class PanNet_PGCU_n(PanNet_PGCU):
        def __init__(self, c, veclen):
            super().__init__(channel=c, veclen=veclen)
            self.PGCU = _PGCU(c, veclen)      # 원본은 이 자리에 4 가 박혀 있다
    return PanNet_PGCU_n(ms_ch, 128)          # veclen 은 저장소 기본값

SPECS["PGCU"] = dict(
    # **표에서 제외한다.**  PanCollection 의 64x64 학습 패치에서 논문의 기여 그 자체인
    # PGCU 모듈이 구조적으로 퇴화한다.  DownSamplingBlock 은 stride-2 conv 와
    # MaxPool(2,2) 를 둘 다 쓰므로 블록당 1/4 이고(주석은 1/2 로 가정한다),
    # NumberBlocks=3 이면 1/64 이다.  직접 측정한 G 격자:
    #     H=64(학습)  1x1  -> 어텐션 key 1개, softmax 가 항상 상수 1
    #     H=256(RR)   4x4  -> 16
    #     H=512(FR)   8x8  -> 64
    # 즉 학습 내내 전역 교차어텐션이 무력화된 채 돌다가 평가에서만 살아난다.
    # gf2 PSNR 23.910 / HQNR 0.5071 이 그 증상이었다.  이 숫자를 싣는 것은 PGCU 에
    # 불공정하다 -- 방법이 아니라 구조와 데이터의 불일치를 재는 것이기 때문이다.
    # 다시 돌려도 같은 퇴화가 재현되므로 재큐하지 않는다.  NumberBlocks 를 줄이면
    # 피할 수 있지만 그것은 저자 설정이 아니다(공개 체크포인트도 DSBlock 3개다).
    tier="excluded",
    build=_pgcu, venue="CVPR 2023", loss="mse", batch=32, epochs=500, aux_loss=True,
    opt=dict(kind="adam", lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-5),
    forward=lambda m, ms_up, pan, lrms, ctx: m(pan, lrms, _hp(pan)),
    diff="PanNet_PGCU 의 생성자가 내부 PGCU 모듈을 `PGCU(4, veclen)` 로 밴드 4 에 고정한다. "
         "그 한 줄만 c 로 바꾼 서브클래스.  veclen 128, lr/wd 는 저장소 train_pannet.py 값. "
         "PGCU 가 업샘플을 스스로 하므로 저해상 MS 를 그대로 준다.  공식 루프가 "
         "loss_1 + loss_2 (중간 up_ms 에도 같은 MSE) 를 쓰므로 aux_loss 로 재현한다.")


# --------------------------------------------------------------------- WFANet
def _wfanet(ms_ch=4, ratio=4):
    # patches/WFANet.patch -- 어텐션 점수를 명시적으로 만들지 않고 SDPA 로 계산한다.
    # 이 어텐션은 공간 전체를 평탄화한 전역 어텐션(net_torch.py:249-251 의
    # `reshape(B, -1, C)`)이라 점수 행렬이 (HW)^2 다.  학습 크기 H=64 는 0.04 GB 지만
    # H 가 두 배가 될 때마다 16 배로 늘어 H=256 이 8.06 GB, H=512 는 **한 장만 64 GiB**
    # 를 요구해 48 GB 카드에서도 못 돈다.  실제로 원격 3칸이 학습을 마치고 최종 평가
    # 단계에서 OOM 으로 죽었다(valid 44.80 dB 까지 가 놓고).
    # 마스크도 편향도 드롭아웃도 없는 표준 softmax(QK^T/scale)V 이므로 SDPA 가 같은
    # 함수를 O(N) 메모리로 계산한다.  등가성은 측정으로 확인했다
    # (baselines/tests/test_wfanet_sdpa.py): 같은 가중치로 저장소 원본 두 줄과 대조해
    # float64 에서 2.2e-16, float32 에서 1.2e-7, **입력과 모든 파라미터의 기울기**까지
    # 5.7e-14 로 일치한다 -- 학습 경로도 바뀌지 않았다는 뜻이다.
    # 메모리: 8밴드 batch 4 에서 256x256 이 0.43 GiB, 512x512(FR) 가 1.71 GiB.
    # 원본은 같은 자리에서 80 GiB 를 요구했다.
    W = _load(REPOS / "WFANet" / "net_torch.py", "wfanet_net")
    # 전부 저장소 super_para.yml 의 값이다
    return W.HWViT(L_up_channel=ms_ch, pan_channel=1, ms_target_channel=32,
                   pan_target_channel=32, head_channel=8, dropout=0.085)


def _wfanet_opt(model, o):
    """저장소는 bias 에는 weight_decay 를 걸지 않는다.  그 구분을 그대로 살린다."""
    return torch.optim.Adam(
        [{"params": [p for n, p in model.named_parameters() if "bias" not in n],
          "weight_decay": 1e-8},
         {"params": [p for n, p in model.named_parameters() if "bias" in n]}], lr=o["lr"])


SPECS["WFANet"] = dict(
    tier="confirmed",
    build=_wfanet, venue="AAAI 2025", loss="l1", batch=32, epochs=360,
    opt=dict(kind="adam", lr=9e-4), make_opt=_wfanet_opt,
    forward=lambda m, ms_up, pan, lrms, ctx: m(pan, lrms, ms_up),
    eval_bs=4,          # 어텐션은 이제 O(N) 이지만 활성값은 배치에 선형이다
    # train_and_test.py:83-84 `lr_max * exp(-log(2)/90 * epoch)` 를 :157-158 에서
    # 에폭마다 모든 파라미터 그룹에 건다 -- 90 에폭마다 반감이다.  StepLR 을
    # step_size=1 로 두면 lr0*gamma^epoch 이라 그 닫힌형과 대수적으로 같다.
    sched=dict(kind="step", step_size=1, gamma=2 ** (-1 / 90)),
    diff="**어텐션 두 줄을 SDPA 로 바꿨다** (patches/WFANet.patch). 저장소는 점수 행렬을 "
         "명시적으로 만드는데(`((q@k)/scale).softmax(-1) @ v`), 이 어텐션은 공간 전체를 "
         "평탄화한 전역 어텐션이라 행렬이 (HW)^2 이다. 학습 크기 64x64 에서는 돌지만 "
         "전해상도 평가(256x256, FR 512x512)에서 80 GiB 를 요구해 48 GB 카드에서 죽는다 "
         "-- 실제로 세 데이터셋이 학습을 마치고 최종 평가에서 OOM 으로 통째로 날아갔다. "
         "마스크도 편향도 드롭아웃도 없는 표준 softmax(QK^T/scale)V 라 SDPA 가 **같은 "
         "함수**를 O(N) 메모리로 계산한다: float64 2.2e-16, float32 1.2e-7, 입력·파라미터 "
         "기울기 5.7e-14 일치(baselines/tests/test_wfanet_sdpa.py). 기여인 웨이블릿 4성분 교차 "
         "어텐션 구조는 한 글자도 건드리지 않았다. "
         "나머지는 없음 -- L_up_channel 이 밴드 인자다. lr 9e-4 / epochs 360 / batch 32 / "
         "wd 1e-8 / 구조 인자(32,32,8,0.085) 전부 저장소 super_para.yml 값. "
         "forward(pan, ms, lms) 에서 ms 는 저해상, lms 는 업샘플 (학습 코드의 키워드 호출로 확인).")


# ----------------------------------------------------------------------- ADWM
def _adwm(ms_ch=4, ratio=4):
    B = _load(REPOS / "ADWM" / "model" / "bwnet_fusionnet.py", "adwm_bwnet")
    if ratio != 4:
        raise ValueError("BWNet 은 forward 안에서 scale_factor=4 로 업샘플한다 (x4 전용).")
    return B.BWNet(pan_dim=1, ms_dim=ms_ch)      # channel/num_blocks 는 저장소 기본값

SPECS["ADWM"] = dict(
    tier="confirmed",
    build=_adwm, venue="CVPR 2025", loss="l1", batch=64, epochs=500,
    opt=dict(kind="adam", lr=2e-3),
    # main_train.py:86 `StepLR(optimizer, step_size=config["step_size"], gamma=0.5)`
    # 와 :179 `step_size: 150`.  :121 에서 에폭마다 step 한다.  없이 돌리면 학습률이
    # 끝까지 2e-3 이라 저자가 의도한 후반 미세조정 구간이 통째로 사라진다.
    sched=dict(kind="step", step_size=150, gamma=0.5),
    forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, pan),
    diff="없음 -- ms_dim 이 밴드 인자다.  lr 2e-3 / epochs 500 / batch 64 는 저장소 "
         "main_train.py 의 config.  forward(ms, pan) 의 ms 는 저해상이다 -- 학습 코드가 "
         "batch[4](=ms)를 넘기고 forward 가 내부에서 x4 업샘플한다.")


# ---------------------------------------------------------------------- MDCUN
def _mdcun(ms_ch=4, ratio=4):
    _pkg(REPOS / "MDCUN" / "model")
    from model.pan_unfolding_v4 import pan_unfolding
    if ratio != 4:
        raise ValueError("MDCUN 의 up_factor 는 4 고정이다.")
    # patches/MDCUN.patch: 밴드가 박혀 있던 다섯 자리만 ms_ch 로 열었다 -- conv_u 의
    # 입출력, Conv_up/Conv_down 의 c_in, blockNL 의 channels, 그리고 blockNL.forward
    # 의 view(b, 4, ...).  전부 잔차 덧셈이나 lms 와의 뺄셈이 채널 수를 밴드 수로
    # 강제하는 자리다.  **T=4 는 건드리지 않았다** -- 밴드 수와 우연히 같지만 언폴딩
    # 반복 횟수, 즉 논문의 기여다.  att_spatial 의 `2` 도 [밴드 한 장, pan] 이라
    # 밴드와 무관하다(4->8 파라미터 증분 42,560 이 나머지 다섯 자리 이론값과 정확히
    # 일치해 산술로 확인된다).
    # 4밴드: 98,347 params, state_dict 와 고정시드 forward 가 원본과 완전 동일.
    # T(언폴딩 단계 수) = 6.  전에는 넘기지 않아 클래스 시그니처 기본값 4 가 쓰였는데,
    # 그건 저장소가 학습하는 값이 아니다 -- 유일한 설정 파일 option_wv2.yml:12 가
    # `stage: 6` 이고 solver/solver.py:39 가 `T = self.cfg['stage']` 로 넘긴다.
    # 공개 체크포인트 경로 이름부터 `pan_unfolding_v4_4_..._6stage/best.pth` 다.
    # T 는 논문의 기여(언폴딩 깊이)라 값이 틀리면 다른 모델을 학습한 것이 된다.
    # T=4 98,347 params -> T=6 128,443.
    return pan_unfolding(ms_ch=ms_ch, T=6)   # mid_channels 64 는 저장소 기본값

SPECS["MDCUN"] = dict(
    tier="confirmed",
    # blockNL 의 F.unfold(fs=15) 는 표본마다 큰 텐서를 만든다 -- 8밴드 H=256 에서
    # 1.4 GB/표본이라, 기본 배치 64 로는 reduced_examples 20장이 한 번에 들어가
    # 28 GB 를 요구해 24 GB 카드에서 터진다.  지표는 표본별 평균이라 배치를 나눠도
    # 수치가 바뀌지 않는다.
    eval_bs=2,
    build=_mdcun, venue="CVPR 2022", loss="l1_sum_over_2b", batch=4, epochs=500,
    opt=dict(kind="adam", lr=5e-4, betas=(0.9, 0.999)),
    forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, ms_up, pan),
    diff="저장소는 밴드 4 고정이다 -- patches/MDCUN.patch 로 밴드 자리 다섯 곳만 열어 WV3(8밴드)도 "
         "돌렸다(저자가 돌린 적 없는 구성-5). lr 5e-4 / batch 4 / L1 은 "
         "저장소 option_wv2.yml 값.  손실은 저장소가 `L1(reduction='sum')/(batch*2)` 를 쓰므로 "
         "그대로 재현한다 -- mean 으로 바꾸면 유효 학습률이 수천 배 작아져 불리해진다. "
         "forward(lms, b_ms, pan) 의 lms 는 저해상, b_ms 는 업샘플 (data/dataset.py 의 "
         "get_patch 에서 lms 가 ip, bms 가 ip*scale 로 잘리는 것으로 확인). "
         "**기울기 클립**: 저장소 gclip 0.04 (option_wv2.yml:61, solver/solver.py:88-91), 하네스는 기본 10.0 "
         "-- 값은 다르지만 **둘 다 매 step 걸린다**: 10.0 이 매 step 걸렸고 (grad_clip_hits = 도달 step, 세 "
         "데이터셋) 그러면 매 step ‖g‖ > 10 > 0.04 다 (CPU 로 잰 qb 전체 노름도 초기 가중치 최소 7,542 · "
         "이어 돌리는 중인 구간의 best.pt(step 706,000 -- 표 칸의 가중치 best_step 552,975 가 아니다) 최소 74, "
         "무작위 256 배치씩).  매 step 잘린 기울기는 c·g/‖g‖ 라 두 상한의 기울기 열은 "
         "상수배(250배)만 다르고, Adam(weight decay 없음) 갱신은 eps 몫을 빼면 같다 (0.04 로 잘린 기울기의 "
         "파라미터당 RMS 0.04/sqrt(128,443) = 1.1e-4 는 eps 1e-8 의 1만 배다) -- 재실행하지 않는다 (-4).")


# ---------------------------------------------------------------------- MSDDN
def _msddn(ms_ch=4, ratio=4):
    for k in [k for k in sys.modules if k == "model" or k.startswith("model.")]:
        del sys.modules[k]
    sys.path.insert(0, str(REPOS / "MSDDN"))
    from model.msddn import Net              # option.yml 의 `algorithm: msddn`
    # patches/MSDDN.patch: Net.__init__ 의 밴드 4 두 곳(msconv 입력 / conv_out 출력)을
    # 이미 있던 num_channels 로 바꿨다.  4밴드에서 state_dict 와 forward 가 원본과 동일.
    #
    # base_filter 는 **32** 다.  전에는 64 를 썼는데 근거가 틀렸다 --
    # counter_flops_params.py:22 는 `model_name = 'unet_panv2'`, 즉 다른 모델을 짓는다.
    # MSDDN 을 짓는 유일한 학습 경로 solver/solver.py:34 와 testsolver.py:29 는 둘 다
    # 32 이고, 64 가 남은 testsolver_o.py 는 커밋 834bfb9 "Update testsolver.py" 로
    # 대체된 구판이다.  64 로 지으면 2,668,372 params 로 저자 모델(670,124)의 3.98배가
    # 되어, 저자가 학습한 적 없는 네트워크를 MSDDN 이라는 이름으로 싣게 된다.
    return Net(num_channels=ms_ch, base_filter=32, args=None)   # args 는 __init__ 에서 쓰이지 않는다

SPECS["MSDDN"] = dict(
    tier="confirmed",
    build=_msddn, venue="TGRS 2023", loss="l1_sum_over_2b", batch=4, epochs=500,
    opt=dict(kind="adam", lr=5e-4, betas=(0.9, 0.999)),
    # solver/solver.py:91 이 에폭 끝마다 step 한다.  500 에폭 동안 200/400 에서 10배씩
    # 떨어진다.  평평하게 두면 후반 내내 5e-4 라 수렴이 막히는데, 그 손해는 전부
    # MSDDN 쪽이다 -- 우리에게 유리한 방향의 불공정이다.
    sched=dict(kind="step", step_size=200, gamma=0.1),
    forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, ms_up, pan),
    diff="저장소는 밴드 4 고정이다 -- patches/MSDDN.patch 로 밴드 자리 두 곳만 열어 WV3(8밴드)도 "
         "돌렸다(저자가 돌린 적 없는 구성-5). lr 5e-4 / batch 4 / L1 / 500 epoch 은 저장소 option.yml 값. "
         "손실 스케일은 MDCUN 과 같은 solver 계열이라 동일하게 재현한다. "
         "forward(l_ms, bms, pan) 계약은 counter_flops_params.py 의 더미 텐서 "
         "(4,32,32)/(4,128,128)/(1,128,128) 로 확정했다. "
         "**기울기 클립**: 저장소 gclip 40 (option.yml:65, solver/solver.py:85-88), 하네스는 기본 10.0 -- "
         "값이 다르다.  하네스 클립은 **매 step** 걸렸다 (grad_clip_hits = 도달 step, 세 데이터셋).  40 과 같은 "
         "갱신이 되려면 ‖g‖ > 40 도 매 step 이어야 한다 -- CPU 로 잰 전체 노름(학습기의 손실 경로, 배치 4, 무작위 "
         "256 배치)은 초기 가중치와 best.pt 모두에서 **40 이하가 한 번도 없었다**: best.pt 최소 51 / 78 / 105, "
         "중앙값 156 / 148 / 164 (qb / gf2 / wv3), 초기 가중치 최소 4,488.  두 상한이 모두 매 step 걸리면 잘린 "
         "기울기 열은 상수배(4배)만 다르고 Adam(weight decay 없음) 갱신은 eps 몫을 빼면 같다 -- 재실행하지 "
         "않는다 (-4).  재지 않은 것: 초기와 best 사이 구간 -- 10 < ‖g‖ <= 40 인 step 이 있었다면 그 step 만 "
         "다르다.")


# =====================================================================
# 2026-09-19 추가분.  아래 아홉은 저장소가 밴드 수나 입력 크기를 하드코딩해 두어
# 앞서 보류했던 것들이다.  사용자 승인 하에 **논문의 기여를 건드리지 않는 최소 수정**만
# 가했고, 바꾼 줄은 전부 patches/*.patch 와 각 spec 의 diff 에 남긴다.
# =====================================================================

# --------------------------------------------------------------------- S2DBPN
def _s2dbpn(ms_ch=4, ratio=4):
    M = _load(REPOS / "S2DBPN" / "modelnew.py", "s2dbpn_model")
    m = M.Net()
    if ms_ch != 8:
        # 저장소는 8밴드(`4*2`) 전용으로 두 층을 박아 두었다.  그 두 층만 밴드 수에 맞춰
        # 다시 만든다.  나머지 백본(반복 back-projection)은 손대지 않는다.
        m.LRconv1 = M.ConvBlock(ms_ch, 128, 3, 1)
        m.finall[6] = torch.nn.Conv2d(32, ms_ch, 1, 1, 0, bias=True)
    return m

SPECS["S2DBPN"] = dict(
    tier="confirmed",
    build=_s2dbpn, venue="TGRS 2023", loss="l1", batch=1, epochs=1000,
    opt=dict(kind="adam", lr=3e-5, betas=(0.5, 0.999)),
    val_cap=512,
    forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, pan),
    diff="`Net()` 이 인자를 받지 않고 `ConvBlock(4*2,...)` / `Conv2d(32,4*2,...)` 로 "
         "밴드 8 을 박아 두었다.  그 두 층만 밴드 수에 맞춰 교체한다(8밴드에서는 원본과 동일). "
         "forward(ms, pan) 의 ms 는 dataloader 가 SimulatedRawMS 를 그대로 넘기고 "
         "내부 UpBlock(kernel 8, stride 4) 이 x4 업샘플하므로 **저해상 MS** 다. "
         "batch 1 / lr 3e-5 / beta1 0.5 / L1 은 저장소 main.py 값.")


# --------------------------------------------------------------------- HFIN
def _hfin(ms_ch=4, ratio=4):
    _pkg(REPOS / "HFIN" / "model")
    from model.HFIN import Net
    return Net(ms_ch)

SPECS["HFIN"] = dict(
    tier="confirmed",
    build=_hfin, venue="CVPR 2024", loss="l1", batch=4, epochs=1000,
    opt=dict(kind="adam", lr=1e-3, weight_decay=0.0),
    # 스케줄러 없음.  저장소는 만들기만 하고 `.step()` 을 부르지 않아 실제로는
    # lr 1e-3 고정이다.  우리가 넣었던 StepLR(200, 0.5) 는 저자가 돌리지 않은 설정이었다.
    forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, pan),
    diff="두 군데를 고쳤다(patches/HFIN.patch). (1) STFreprocess 가 s1=43,s2=42 로 "
         "128x128 전용 창 분할을 박아 두어 다른 크기에서 죽었다 -> 같은 겹침 비율을 "
         "유지한 채 s1=round(H*43/128), s2=H-2*s1 로 입력 크기에 비례시켰다. H=128 이면 "
         "원본과 비트 단위로 같다.  겹치는 4창 단시간 푸리에 분해라는 기여 자체는 그대로다. "
         "(2) conv_p 입력과 refine 출력의 밴드 4 고정을 channels 로 풀었다 -- 저장소 "
         "train.py 가 Net(4) 로 부르므로 원본에서 feature width 와 밴드 수가 같았고, "
         "공식 체크포인트 conv_p.weight (4,4,3,3) 로 확인했다.  4밴드에서는 원본과 동일. "
         "forward(ms, pan) 의 ms 는 utils.PSH5Datasetfu 가 h5 의 'ms'(저해상)를 넘긴다. "
         "**평가 배치에 민감하다**: 출력이 평가 배치 크기에 따라 달라진다 -- 한 장씩(배치 1) 추론하면 "
         "RR PSNR 이 wv3 +0.24 / qb -0.30 dB 다르다 (주파수 분기의 torch.angle 이 진폭 0 근처에서 FFT 반올림을 "
         "키우는 것으로 보인다; 같은 배치 크기면 결정적).  표의 값은 학습기의 평가 배치(RR 은 reduced 20장 한 "
         "배치, FR 은 4장씩) 값이다.  **기울기 클립**: 저장소는 클립하지 않는다; 하네스 10.0 은 거의 걸리지 "
         "않았다 (grad_clip_hits wv3 0 · qb 84).")


# --------------------------------------------------------------------- UTeRM
def _uterm(ms_ch=4, ratio=4):
    # UTeRM_MRA.py 가 `from model.net_modules import ...` 로 패키지 상대 임포트를 한다.
    _pkg(REPOS / "UTeRM" / "model")
    from model.UTeRM_MRA import LRTC_Net
    net = LRTC_Net(HSI_channels=ms_ch, N_iter=10)

    class PerSample(torch.nn.Module):
        """언롤링 대수가 배치 차원 없이 쓰여 있다 -- `torch.squeeze(C_k).permute(2,0,1)` 은
        B=1 에서만 성립하고, 저장소 train.py 도 batch_size=1 로 학습한다.  모델은
        본질적으로 표본별 연산이므로, 수식은 한 글자도 건드리지 않고 배치를 돌며 호출한다.
        B=1 이면 원본과 완전히 동일하고, 평가에서 배치를 키워도 결과가 바뀌지 않는다."""
        def __init__(self, net): super().__init__(); self.net = net
        def forward(self, ms_up, pan):
            cs, hr = zip(*(self.net(ms_up[i:i+1], pan[i:i+1]) for i in range(ms_up.shape[0])))
            return torch.cat(cs), torch.cat(hr)
    return PerSample(net)

def _uterm_step(model, b, ctx, lossfn):
    """공식 루프는 2단계다 (train.py:105-118).

    1~90 에폭:  `L1(hrhs, ref) + 0.25*L1(cs_comp, ref)`
    91~100:     `L1(hrhs, ref)` 만 (README 의 `--finetune`)

    `cs_comp` 는 논문의 기여인 generalized detail injection 헤드
    (UTeRM_MRA.py:78-80) 의 출력이고, 0.25 항이 그 헤드에 걸리는 **유일한 직접
    기울기**다.  전에는 spec 의 forward 람다가 `[1]` 로 첨자를 떼어 버려 트레이너가
    튜플을 못 봤고, 그래서 aux_weights=[0.25] 가 한 번도 발동하지 않았다.
    즉 그 헤드를 감독 없이 학습시키고 있었다."""
    cs, hr = model(b["lms"], b["pan"])
    loss = lossfn(hr, b["gt"])
    if ctx["epoch"] <= 90:
        loss = loss + 0.25 * lossfn(cs, b["gt"])
    return hr, loss


SPECS["UTeRM"] = dict(
    tier="confirmed",
    build=_uterm, venue="TGRS 2024", loss="l1", batch=1, epochs=100,
    opt=dict(kind="adam", lr=1e-5),
    val_cap=256,                 # 표본별 루프라 valid 전수(2,201행)가 너무 느리다
    aug="d4",                    # 저장소는 9배 오프라인 기하 증강을 H5 에 구워 둔다
    grad_clip=2.0,               # train.py:122  clip_grad_norm_(..., 2)

    # 평가는 **학습 크기 64x64 타일**로 나눠 추론하고 이어 붙인다 (`lib/harness.eval_forward`).
    # 저랭크 시작(UTeRM_MRA.py:85-86)이 W 축 FFT 의 t-product 라 네트워크가 학습 폭에 묶인다 --
    # 256(RR · WV2) · 512(FR) 를 한 번에 넣으면 wv3 RR 이 15.61 dB 로 무너지고, 겹치지 않는 64 타일은
    # 35.96 dB 다 (진단 2026-09-24).  저자도 학습 크기(256)로 잘라 추론한다 (test_full_res.py:64-87).
    # 학습 경로 · forward · code_digest 는 그대로이고, 결과의 eval_protocol 만 "+tile64" 가 붙는다.
    eval_tile=64,

    # 평가 경로.  UTeRM_MRA.py:92 가 (cs_comp, hrhs) 를 돌려주므로 [1] 이 최종 출력이다.
    forward=lambda m, ms_up, pan, lrms, ctx: m(ms_up, pan)[1],
    train_step=_uterm_step,
    diff="수식은 그대로 두고 배치를 도는 래퍼만 씌웠다(저장소도 batch_size=1). "
         "손실은 train_step 이 공식 2단계를 그대로 돈다 -- 1~90 에폭 "
         "`L1(hrhs,ref) + 0.25*L1(cs_comp,ref)`, 91~100 에폭은 `L1(hrhs,ref)` 만. "
         "upms_image 는 이름 그대로 업샘플 MS 다 (model 내부에서 pan 과 같은 격자에서 "
         "더해진다).")


# --------------------------------------------------------------------- AWFLN
def _awfln(ms_ch=4, ratio=4):
    # 저장소는 밴드 수를 생성자가 아니라 **파일**로 고른다 (model_4b.py / model_8b.py).
    f = "model_4b.py" if ms_ch == 4 else "model_8b.py"
    if ms_ch not in (4, 8):
        raise ValueError("AWFLN 은 4밴드/8밴드 파일만 제공한다.")
    sys.path.insert(0, str(REPOS / "AWFLN"))     # model_4b.py 가 models_others 를 평면 임포트한다
    A = _load(REPOS / "AWFLN" / f, f"awfln_{ms_ch}b")
    return A.AWFLN(1)            # 인자는 scale_factor 인데 실제로 쓰이지 않는다 (train.py 도 1)

SPECS["AWFLN"] = dict(
    tier="confirmed",
    build=_awfln, venue="TGRS 2023", loss="l1", batch=20, epochs=300,
    opt=dict(kind="adam", lr=5e-4, betas=(0.9, 0.999)),
    sched=dict(kind="step", step_size=100, gamma=0.5, min_lr=5e-5),
    forward=lambda m, ms_up, pan, lrms, ctx: m(ms_up, lrms, pan),
    diff="코드 수정 없음.  밴드는 import 하는 파일로 고른다.  forward(ms_up, ms_org, pan) "
         "에서 ms_org 는 data_utils_RS.py:295-297 이 1/4 로 줄인 **저해상** MS 이고 "
         "(모델이 PixelShuffle 로 x4 업샘플한다), ms_up 은 잔차의 기준이다 "
         "(model_4b.py:62 `out_f = out8 + ms_up`).")


# --------------------------------------------------------------------- PanTCR
def _pantcr(ms_ch=4, ratio=4):
    sys.path.insert(0, str(REPOS / "PanTCR-GF2"))
    N = _load(REPOS / "PanTCR-GF2" / "net.py", "pantcr_net")
    return N.CloudPan(ms_ch, 16)          # dim 16 은 train.py:48 값 (net.py 기본 32 아님)

SPECS["PanTCR"] = dict(
    tier="confirmed",
    build=_pantcr, venue="AAAI 2026", loss="l1", batch=4, epochs=700,
    opt=dict(kind="adam", lr=4.5e-4, betas=(0.9, 0.999)),
    sched=dict(kind="step", step_size=500, gamma=0.5),
    aug="d4",                    # dataset.py:35-54 -- 플래그 없이 항상 켜져 있다
    forward=lambda m, ms_up, pan, lrms, ctx: m(ms_up, pan),
    diff="코드 수정 없음.  밴드는 생성자 인자다.  forward(lms, pan) 의 lms 는 PAN 격자의 "
         "업샘플 MS 다(저해상 lrms 가 아니다).  저장소는 scheduler.step() 을 에폭당 두 번 "
         "부르는데(train.py:104,143) 그건 버그로 보여 한 번만 부른다 -- **공개 코드 대비** 학습 "
         "레시피 이탈은 이것 하나이고, 그대로 두면 학습률이 의도의 두 배 속도로 줄어든다 "
         "(하네스 클립 10.0 은 걸리지 않았다, wv3 grad_clip_hits 0).  **논문과 공개 코드가 다르다**: "
         "논문은 Adam + cosine 3e-3 -> 1e-6 / 500 epoch 이고 증강을 말하지 않는다; 공개 코드는 Adam 4.5e-4 "
         "(train.py:35) · StepLR(500, 0.5) (train.py:56) · 700 epoch (train.py:36) · D4 증강 항상 켬 "
         "(dataset.py:35-54).  우리는 공개 코드를 따른다.  **체크포인트 선택**: 저장소는 best.pth 를 "
         "**test PSNR** 으로 고른다 (train.py:118-131 이 test 로더로 t_psnr 를 재고 153-155 가 그것으로 저장) "
         "-- 우리는 valid 로 고르고 test 는 선택에 쓰지 않는다.")


# --------------------------------------------------------------------- MutInf
def _mutinf(ms_ch=4, ratio=4):
    sys.path.insert(0, str(REPOS / "Mutual-Information-driven-Pan-sharpening"))
    G = _load(REPOS / "Mutual-Information-driven-Pan-sharpening" / "models" / "GPPNN.py",
              "mutinf_gppnn")
    G.feature_save = lambda *a, **k: None        # 저장소의 디버그 덤프를 무력화한다
    net = G.GPPNN(ms_channels=ms_ch, pan_channels=1, n_feat=8)

    class Wrap(torch.nn.Module):
        """`Mutual_info_reg` 는 공식 루프에서 **옵티마이저에 들어가지 않는다**
        (train_GPPNN.py:72 는 net.parameters() 만 넘긴다).  즉 1000 에폭 내내 랜덤 초기화
        상태로 남는다.  그 성질을 그대로 살리려고 리스트에 담아 nn.Module 등록을 피한다."""
        def __init__(self, net, mutual):
            super().__init__(); self.net = net; self._mutual = [mutual]
        @property
        def mutual(self): return self._mutual[0]
        def forward(self, lrms, pan):
            return self.net(lrms, 0, pan)        # 가운데는 디버그 인덱스이지 데이터가 아니다
    w = Wrap(net, G.Mutual_info_reg(4, 4))
    return w

def _mutinf_step(model, b, ctx, lossfn):
    pred, panf, mHRf = model(b["ms"], b["pan"])
    model.mutual.to(pred.device)
    ramp = min(ctx["epoch_frac"], 1.0)           # adjust(0,1,epoch,N) = epoch/N
    latent = torch.clip(model.mutual(panf, mHRf), -1, 1)
    return pred, 300.0 * lossfn(pred, b["gt"]) + 0.2 * ramp * latent

SPECS["MutInf"] = dict(
    tier="confirmed",
    build=_mutinf, venue="CVPR 2022", loss="l1", batch=4, epochs=1000,
    opt=dict(kind="adam", lr=1e-3, weight_decay=0.0),
    forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, pan)[0],
    train_step=_mutinf_step,
    # NaN 가드 (선택 B 사전 등록) -- 학습기가 읽는 opt-in 표시다 (`lib/nan_guard.py`).  레시피 키도
    # 호출 가능한 필드도 아니라서 레시피 스냅샷 · code_digest · 게이트 판정이 바뀌지 않는다.
    nan_guard=True,
    diff="코드 수정 없음(디버그 함수 feature_save 만 no-op 으로 덮는다). 공식 손실은 "
         "`300*L1(pred,gt) + 0.2*(epoch/N)*clip(MI(panf,mHRf),-1,1)` 이고, MI 모듈은 "
         "옵티마이저에 없어 랜덤 초기화로 고정된다 -- 그 이상한 성질까지 그대로 재현한다. "
         "forward 의 가운데 인자는 디버그 인덱스이지 데이터가 아니다. "
         "ms_up 은 쓰이지 않는다(모델이 lrms 를 내부에서 bicubic 업샘플해 잔차 기준으로 쓴다). "
         "**기울기 클립**: 저장소는 클립하지 않는다 (training/train_GPPNN.py 에 clip_grad 없음).  하네스 10.0 이 "
         "걸린 step: wv3 377,429 / 1,122,266 (33.6%).  qb 의 27,274 회는 warm 으로 이어 돌린 마지막 구간(step "
         "1,308,001~1,508,000, 200,000 step)만의 값이다 (13.6%) -- 앞 구간(step 1,322,000 에서 NaN 으로 죽은 런, "
         ")의 기울기 통계는 남지 않았다.  클립 없는 런과의 차이는 재지 않았다. "
         "**NaN 가드** (저장소에 없는 장치 선택 B · 2026-09-25 부터 시작하는 학습 구간): 학습기가 step 마다 "
         "가중치 · 버퍼 · 옵티마이저 상태를 떠 두고, 그 step 이 비유한 값(GPPNN.py:410 의 Normal(loc=NaN) 예외 · 비유한 "
         "손실 · 갱신 뒤 비유한 상태)을 내면 정확히 되돌리고 그 batch 를 건너뛴다 -- step 번호는 나아가고(batch 는 "
         "소비됨), 건너뛴 횟수 · 위치는 결과 JSON 의 `nan_guard` 에 있으며, 최근 1만 step 안에 100번을 넘으면 "
         "`nan_guard_exhausted` 로 멈추고 그때까지의 best.pt 를 싣는다.")


# --------------------------------------------------------------------- PanFlowNet
def _panflow(ms_ch=4, ratio=4):
    _pkg(REPOS / "PanFlowNet" / "model")
    from model.pan_inn9 import pan_inn
    return pan_inn(mid_channels=64, T=4, bands=ms_ch)

def _panflow_step(model, b, ctx, lossfn):
    # 공식 solver 는 스텝마다 두 번 forward 한다.
    #   rev=False : GT -> latent.  두 번째 반환이 흐름 NLL(jac_loss) 이다.
    #   rev=True  : 잡음 -> 예측.  두 번째 반환은 버린다.
    # jac_loss 를 빼면 가역망이 흐름으로 학습되지 않는다 -- 반드시 살려야 한다.
    _, jac = model(b["ms"], b["gt"], b["pan"])
    y, _ = model(b["ms"], b["gt"], b["pan"], rev=True)
    if ctx["epoch_frac"] >= 1000.0 / 1100.0:
        return y, jac          # solver.py:70-76  마지막 100 에폭은 흐름 항만 쓴다
    l1 = torch.nn.functional.l1_loss(y, b["gt"], reduction="sum") / (b["gt"].shape[0] * 2)
    return y, l1 + jac

SPECS["PanFlowNet"] = dict(
    tier="confirmed",
    build=_panflow, venue="ICCV 2023", loss="l1", batch=4, epochs=1100,
    opt=dict(kind="adam", lr=5e-4, betas=(0.9, 0.999)),
    grad_clip=0.04,                        # cfg schedule.gclip -- 저장소가 매 스텝 건다
    # 추론이 가우시안 잡음에서 출발한다 -- 같은 가중치 · 같은 입력이라도 부를 때마다 다른 출력.
    # 표의 값은 표본 하나다 (qb 에서 다시 뽑으면 36.16~36.23, 표 36.244).  그림 · 재평가가 읽는다.
    stochastic=True,
    forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, ms_up, pan, rev=True)[0],
    train_step=_panflow_step,
    diff="한 곳을 고쳤다(patches/PanFlowNet.patch): subnet_coupling_layer 가 dims_in 을 "
         "받아 놓고 무시한 채 channels=4 를 박아 두었다.  인자를 실제로 쓰게만 바꾸고 "
         "pan_inn 에 bands 인자를 더했다(기본 4 -> 저장소 호출은 그대로). "
         "손실은 `L1_sum(y,gt)/(B*2) + jac_loss` 이고 마지막 100 에폭은 jac_loss 만 쓴다. "
         "**출력이 확률적이다** -- rev=True 가 가우시안 잡음에서 출발한다.  평가도 저장소와 "
         "같이 표본 하나를 쓴다.  T=4 단계가 같은 결합층을 공유하는 것도 저장소 그대로다.")


# =====================================================================
# 2026-09-19 2차 추가분.  저장소 55개 전수 검토(서브에이전트 28개)에서 나온 것들.
# 전부 4/8밴드 forward 와 H=64/256 실행을 확인한 뒤에만 등록했다.
# =====================================================================

# --------------------------------------------------------------------- RAMSF
def _ramsf(ms_ch=4, ratio=4):
    sys.path.insert(0, str(REPOS / "RAMSF"))
    M = _load(REPOS / "RAMSF" / "models" / "MSIP" / "RAMSF.py", "ramsf_model")
    return M.Network(args=None, n_bands=ms_ch, n_select_bands=1, scale_ratio=ratio)

SPECS["RAMSF"] = dict(
    tier="confirmed",
    build=_ramsf, venue="TGRS 2025", loss="l1", batch=32, epochs=600,
    opt=dict(kind="adam", lr=3e-4, betas=(0.9, 0.999)),
    # solver/solver.py:54-55 는 MultiStepLR 을 쓰고 option.yml:44 의 `decay: '100'`
    # 이 milestones=[100] 로 파싱된다.  StepLR(step_size=100) 이면 100 에폭마다
    # 반복해서 줄어 600 에폭 동안 0.9^6 이 되지만, 실제로는 100 에폭에 **한 번만**
    # 줄어 0.9^1 이다.
    sched=dict(kind="multistep", milestones=[100], gamma=0.9),
    forward=lambda m, ms_up, pan, lrms, ctx: m(pan, ms_up, lrms),
    diff="patches/RAMSF.patch -- 공간 크기 고정을 풀고 밴드 수를 생성자 인자로 뺐다. "
         "기여(LHSDR 웨이블릿 detail 가지 + CSFPA 좌표-모달 질의 MLP)는 손대지 않았다. "
         "손실은 단일 mean-L1 (solver/solver.py:87-96).  출력은 `lms + sr` 잔차다.")


# --------------------------------------------------------------------- MMNet
def _mmnet(ms_ch=4, ratio=4):
    sys.path.insert(0, str(REPOS / "MMNet"))       # model.py 가 nonlocal_block 을 평면 임포트한다
    M = _load(REPOS / "MMNet" / "model.py", "mmnet_model")
    # patches/MMNet.patch: __init__ 이 밴드 4 를 리터럴로 박아 두고(12 = 4+8 같은 합성
    # 포함) 선언된 num_channels 를 한 번도 쓰지 않던 것을, 실제로 쓰게 했다.
    # 4밴드: 51,822 params 로 원본과 동일.  8밴드: 105,198.
    net = M.Net(num_channels=ms_ch, base_filter=None, args=None)

    class Wrap(torch.nn.Module):
        """ConvLSTMCell 이 `(128,128)` 로 공간 크기를 생성자에 박아 두어 init_hidden 이
        항상 128x128 은닉 상태를 만든다.  우리는 64(학습)/256/512(평가)에서 돌려야 한다.
        기억 흐름 자체는 건드리지 않고, forward 직전에 각 셀의 height/width 를 실제
        입력 크기로 맞춰 준다 -- 128 입력에서는 원본과 완전히 동일하다."""
        def __init__(self, net): super().__init__(); self.net = net
        def forward(self, lrms, ms_up, pan):
            h, w = pan.shape[-2:]
            for c in (self.net.cellU, self.net.cellHl, self.net.cellV, self.net.cellHp):
                c.height, c.width = h, w
            return self.net(lrms, ms_up, pan)
    return Wrap(net)

SPECS["MMNet"] = dict(
    tier="confirmed",
    # H=512 에서 표본 하나가 16.2 GB 다 (ConvLSTM 이 모든 시점의 은닉 상태를 들고 있다).
    # FR 루프의 기본 배치 4 로는 65 GB 를 요구해 터진다.  지표는 표본별 평균이라
    # 배치를 나눠도 수치는 같다.
    eval_bs=1,
    # 논문 값: lr 8e-4, 200 에폭에서 x0.5, 총 1000 에폭.  전에는 lr 이 1.6배 낮고
    # 감쇠가 0.1 이라 200 에폭 이후 저자 대비 학습률이 한 자릿수 작았다.
    build=_mmnet, venue="ECCV 2022", loss="l1_sum_over_2b", batch=4, epochs=1000,
    opt=dict(kind="adam", lr=8e-4, betas=(0.9, 0.999), eps=1e-8),
    sched=dict(kind="step", step_size=200, gamma=0.5),
    # 클립 없음 -- 저장소에 학습 코드가 없어 따를 클립이 없고, 사전 규칙 [D] 가 탈락했다 (2026-09-25).
    grad_clip=0.0,
    forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, ms_up, pan),
    diff="ConvLSTMCell 의 `(128,128)` 고정 때문에 다른 크기에서 죽는다.  forward 직전에 "
         "각 셀의 height/width 를 실제 입력 크기로 맞추는 래퍼만 씌웠다(128 입력에서는 "
         "원본과 동일).  밴드는 저장소가 4 를 리터럴로 박아 두었다(선언된 num_channels 인자를 "
         "본문에서 쓰지 않는다) -- patches/MMNet.patch 로 그 인자를 쓰게 해 WV3(8밴드)도 돌렸다(저자가 "
         "돌린 적 없는 구성-5; 4밴드는 원본과 같은 51,822 params). "
         "손실은 MDCUN/MSDDN 과 같은 solver 계열의 `L1_sum/(B*2)`. "
         "**기울기 클립 없음** (-4 ·): 공개 저장소에 학습 코드가 없다 (README 'The code "
         "is being prepared and uploaded', model.py 뿐) -- 따를 클립 값도, 클립한다는 근거도 없다.  처음에는 하네스 "
         "기본 10.0 으로 돌렸고 **매 step** 걸렸다 (grad_clip_hits = 도달 step, 세 데이터셋; CPU 로 잰 best.pt 의 "
         "전체 노름 중앙값 255 / 106 / 268 (gf2 / qb / wv3)).  사전 규칙 -- 클립 없는 민감도 런(gf2, 같은 seed "
         "20260919, 같은 정지 규칙)과 비교해 abs(ΔRR PSNR) <= 0.15 dB 이고 abs(ΔHQNR) <= 0.017 이면 클립 결과를 "
         "싣는다 -- 에서 ΔRR **+0.221 dB** (41.715 -> 41.936) · ΔHQNR +0.0132 (0.9438 -> 0.9570) 로 탈락해, 세 칸을 "
         "클립 없이 처음부터 다시 돌렸다.  클립 결과는 공개용으로 results.bak_20260926_s40d 에 둔다 (n=1 짝 비교, "
         "valid 는 43.100 대 43.087 로 거의 같다).")


# --------------------------------------------------------------------- MMMamba
def _mmmamba(ms_ch=4, ratio=4):
    # mmmamba.py 는 `from .refine import Refine` 로 상대 임포트를 한다.  파일 단위로
    # 읽으면 부모 패키지가 없어 실패하므로, 다른 `model` 저장소들과 같은 방식으로
    # 패키지 자체를 갈아 끼운다.
    sys.path.insert(0, str(REPOS / "MMMamba"))
    _pkg(REPOS / "MMMamba" / "model")
    from model.mmmamba import Net
    return Net(num_channels=ms_ch, base_filter=32, args=None)

SPECS["MMMamba"] = dict(
    tier="confirmed",
    # utils/utils.py:51 은 reduction='mean' 이다.  sum 으로 두면 기울기 노름이
    # 0.32 대신 5,250 이라 우리 기본 클립에 매 스텝 걸린다.  gclip 은 option.yml:65 값.
    build=_mmmamba, venue="AAAI 2026", loss="l1_mean_over_2b", batch=4, epochs=500,
    grad_clip=4.0,
    opt=dict(kind="adam", lr=5e-4, betas=(0.9, 0.999), eps=1e-8),
    sched=dict(kind="cosine", T_max=500, eta_min=5e-8),
    forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, ms_up, pan),
    diff="patches/MMMamba.patch -- Net.__init__ 의 하드코딩된 밴드 4 두 곳을 "
         "이미 있던 num_channels 인자로 바꿨다.  forward 의 둘째 인자는 이름이 "
         "literally `_` 이고 본문에서 참조되지 않는다(ms_up 미사용).  "
         "**이름과 달리, 공개된 `Net` 은 Mamba 를 쓰지 않는다.** solver.py:124-132 가 "
         "`model.mmmamba.Net` 을 그대로 불러 학습하는데, 그 `Net` 이 쌓는 `MMMamba` "
         "블록(mmmamba.py:374)은 `Attention(dim, window_size=4)` + FeedForward 다.  "
         "같은 파일의 `SingleMambaBlock`/`TokenSwapMamba`/`CrossMamba` 는 `Net` 이 "
         "한 번도 만들지 않는 죽은 코드다.  `mamba_ssm` 은 파일 상단 import(8행)를 "
         "통과시키는 데만 필요하다.  우리는 저장소가 학습하는 바로 그 클래스를 쓰므로 "
         "배선은 충실하지만, 표에서 이 행을 Mamba 계열로 읽으면 안 된다.  "
         "solver.py:127 은 `n_colors == 4` 를 단언한다 -- 네트워크 자체는 밴드에 "
         "독립이라 8밴드도 만들어지지만(332,768 params), 저자가 돌린 적 없는 설정이다.")


# --------------------------------------------------------------------- DCPNet
def _dcpnet(ms_ch=4, ratio=4):
    sys.path.insert(0, str(REPOS / "DCPNet"))
    M = _load(REPOS / "DCPNet" / "UDL" / "pansharpening" / "models" / "UPNet" / "model_upnet.py",
              "dcpnet_upnet")
    return M.UPNet(ms_ch, criterion=None)

SPECS["DCPNet"] = dict(
    tier="confirmed",
    build=_dcpnet, venue="TGRS 2024", loss="l1", batch=16, epochs=200,
    opt=dict(kind="adam", lr=2e-4, weight_decay=0.0),
    aux_weights=[0.5],           # upnet_main -- L1(sr,gt) + 0.5*L1(msup,gt)
    out_index=0,                 # (sr, msup) 중 첫째가 최종 출력
    # `[0]` 을 떼어 튜플을 그대로 넘긴다.  그래야 aux_weights=[0.5] 가 살아
    # `L1(sr,gt) + 0.5*L1(msup,gt)` 가 된다 (model_upnet.py:397-400).
    # 논문 제목이 "Dual-task" 이고 그 둘째 과제가 바로 이 MS 초해상 항이다 --
    # 빼면 목적함수의 43% 가 사라지고 MS_net.conv5 에 걸리는 기울기가 없어진다.
    forward=lambda m, ms_up, pan, lrms, ctx: m(pan, lrms),
    diff="patches/DCPNet.patch -- 패키지 __init__ 한 줄만 고쳤다(모델/손실/연산 불변). "
         "저장소가 'UPNet' 이름으로 배포했으나 README 가 DCPNet 코드임을 명시한다. "
         "공식 손실은 `L1(sr,gt) + 0.5*L1(msup,gt)` 로 보조 MS 초해상 헤드를 감독한다 "
         "-- 이 항을 빠뜨리면 이중 과제 설계가 통째로 사라진다.")


# --------------------------------------------------------------------- specs/
# 추가 방법은 lib/specs/*.py 한 파일씩으로 등록한다.  registry.py 를 동시에 고치지
# 않아도 되므로 여러 방법을 병렬로 붙일 수 있고, 하나가 깨져도 나머지는 산다.
_SPEC_ERRORS = {}
for _f in sorted((Path(__file__).parent / "specs").glob("*.py")):
    if _f.name.startswith("_"):
        continue
    try:
        _m = _load(_f, f"panbase_spec_{_f.stem}")
        _m.register(SPECS, REPOS, _load, _pkg)
    except Exception as _e:                       # 하나가 깨져도 나머지는 등록된다
        _SPEC_ERRORS[_f.stem] = f"{type(_e).__name__}: {_e}"


def build(name, ms_ch=4, ratio=4):
    """저장소 코드를 import 하는 유일한 지점이다.  그래서 여기서 환경을 지킨다.

    저장소 일부가 import 중에 `CUDA_VISIBLE_DEVICES` 를 덮어쓴다 (PanTCR-GF2/net.py:11).
    `gpu_pin.guard_env` 가 되돌리고 로그에 남긴다.  장치 자체는 호출자가 이 함수 전에
    `gpu_pin.pin()` 으로 고정해야 한다 -- 여기서는 이미 늦을 수 있다.
    """
    from gpu_pin import guard_env
    s = SPECS[name]
    with guard_env(f"{name} 저장소"):
        m = s["build"](ms_ch=ms_ch, ratio=ratio)
    return m, s


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
