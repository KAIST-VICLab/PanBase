"""Global-skip 변형 연구의 in 팔 (lib/skipvar.py) -- UTeRM (TGRS 2024).  분류: YES (하네스 `lms` 가 base 다).

UTeRM 은 하네스 `lms` 를 그대로 받는다: registry.py:500 forward `m(ms_up, pan)[1]`, registry.py:477 train_step
`model(b["lms"], b["pan"])` -> repos/UTeRM/model/UTeRM_MRA.py:72 `forward(interp_ms, pan_image)`.  저장소는 저해상 MS 를
읽지 않는다 (업샘플러가 없다 -> INT 가 아니다).  interp_ms 가 쓰이는 곳은 셋뿐이다:
  UTeRM_MRA.py:76  Gi = att_module(cat[interp_ms, pan, PL])     본체가 lms 를 읽는 유일한 conv (att_module.att1)
  UTeRM_MRA.py:78  cs_comp = interp_ms + Gi * (pan - PL)         저자가 업샘플 MS 를 **더하는** 자리 (generalized detail injection)
  UTeRM_MRA.py:81  C = interp_ms                                 반복 초기값이지만 값은 읽히지 않는다 -- LRTC_Block.forward(:33-38)
                                                                 는 C 인자를 쓰지 않고, :82-86 은 C 의 모양만 읽는다.
출력은 펼친 반복의 마지막 C (:89-92) 이고, 블록마다 C_k = (L*R + lamb*cs_comp + alpha*G - Lg) / (1 + lamb + alpha) (:36-38) --
cs_comp 가 모든 블록의 충실도 기준이다.  맨 끝의 `+ interp_ms` 는 없다: UTeRM 의 skip 은 cs_comp 를 거친 **연한** skip 이다
(학습된 base best.pt 의 마지막 블록 lamb/(1+lamb+alpha) = 0.55 / 0.52 / 0.68 (wv3 / qb / gf2); cs_comp 의 base 만 LRMS -> S_PAN
으로 바꾸면 (att_module 입력은 그대로) 출력 변화의 (S_PAN - LRMS) 회귀 기울기가 0.13 / 0.58 / 1.10 이다 (valid 24장) -- 덧셈 skip
이면 정확히 1).  그래도 UTeRM 에서 업샘플 MS 가 base 로
들어가는 길은 하네스 lms 하나뿐이고 `--skip` 이 그 세 자리를 모두 S 로 바꾼다 -> sk 팔은 다른 YES 방법처럼 데이터만 바꾼 본래
스펙이다 (스펙 코드 없음).

in 팔 `UTeRM-in`: sk 위에서 lms 를 처음 (그리고 유일하게) 읽는 PAN 해상도 conv `net.att_module.att1` 이 LRMS 를 추가 채널로 더
받는다 (skv_w 0 초기화 -> 같은 seed 에서 in 의 초기 함수 = sk 의 초기 함수).
  att_module.att1 = Conv2d(C+2, C+2, 3, padding=1, bias=True), groups 1  (net_modules.py:8, :13; UTeRM_MRA.py:61, :76)
  다른 conv 는 lms 를 읽지 않는다: PL_conv1/2 (:62-63, :75) 는 PAN 만, 펼친 블록의 Proximal RDB (:11, :44) 는 C_k + Lg/alpha 만.
  LRMS 는 [0,1] 그대로다 (본래 스펙도 lms 를 정규화 없이 넣는다, registry.py:500).
기본 make_in 감싸개는 맞지 않는다: PerSample (registry.py:455-463) 이 표본마다 net 을 배치 1 로 부르므로 추가 입력도 표본마다
잘라 걸어야 한다 (배치 전체를 걸면 hook 의 배치 검사가 멈춘다).  그래서 base 의 forward · train_step 을 **그대로** 부르고, 그
`model(ms_up, pan)` 자리에 `_InModel` 을 넣는다: PerSample 처럼 표본을 돌며 표본 i 의 호출에 L[i] 를 건다.  손실(공식 2단계:
1~90 에폭 L1(hrhs) + 0.25 L1(cs_comp), 그 뒤 L1(hrhs), registry.py:466-481)과 평가 출력([1], registry.py:500)은 base 코드 그대로다.
eval_tile 64 (registry.py:497) 의 타일러는 stack 된 lms 를 S 와 함께 자르고 타일을 배치로 넣는다 -- 타일 하나가 표본 하나다.
D4 증강(aug="d4")은 학습기가 stack 된 lms 에 걸어 S 와 L 이 같은 변환을 받는다.  EMA 사본 · on_epoch 훅은 없다.
본래 스펙 · 저장소 코드는 건드리지 않는다 (UTeRM code_digest 불변).  파일 이름이 zz_ 인 까닭은 zz_skv_core.py 와 같다.
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys

import torch

sys.path.insert(0, (_GB + "/lib"))
import skipvar as SV                                                          # noqa: E402

#: lms 를 읽는 유일한 conv -- PerSample(registry.py:460) 의 `net` 아래 LRTC_Net.att_module.att1 (UTeRM_MRA.py:61, :76).
_SITE = "net.att_module.att1"


class _InModel:
    """base forward / train_step 의 `model(ms_up, pan)` 자리에 끼우는 대리 객체.

    base 는 model 을 부르기만 한다 (registry.py:477 `model(b["lms"], b["pan"])`, :500 `m(ms_up, pan)[1]`) -- 속성을 읽으면
    여기서 AttributeError 로 멈춘다.  부르면 PerSample.forward(registry.py:461-463) 와 같은 순서로 표본마다 net 을 부르고
    (배치 1 의 PerSample 은 net 을 한 번 부른다), 표본 i 의 호출에는 L[i:i+1] 을 추가 입력으로 건다.  base 가 S 가 아닌 텐서로
    부르면 L 과 짝이 어긋나므로 멈춘다."""
    __slots__ = ("m", "S", "L")

    def __init__(self, m, S, L):
        self.m, self.S, self.L = m, S, L

    def __call__(self, ms_up, pan):
        if ms_up is not self.S:
            raise RuntimeError("UTeRM-in: base 가 S 가 아닌 텐서로 모델을 불렀다 -- 추가 입력 L 과 표본 짝이 어긋난다")
        cs, hr = [], []
        for i in range(ms_up.shape[0]):
            with SV.extra(self.m, self.L[i:i + 1]):
                c, h = self.m(ms_up[i:i + 1], pan[i:i + 1])
            cs.append(c)
            hr.append(h)
        return torch.cat(cs), torch.cat(hr)


def register(SPECS, REPOS, _load, _pkg):
    B = SPECS["UTeRM"]
    base_fwd, base_ts = B["forward"], B["train_step"]

    def forward_in(m, ms_up, pan, lrms, ctx):
        S, L = SV.split(ms_up)
        return base_fwd(_InModel(m, S, L), S, pan, lrms, ctx)

    def train_step_in(m, b, ctx, lossfn):
        S, L = SV.split(b["lms"])
        return base_ts(_InModel(m, S, L), dict(b, lms=S), ctx, lossfn)

    SV.make_in(SPECS, "UTeRM", "UTeRM-in", [_SITE], "L", forward=forward_in, train_step=train_step_in,
               note="자리: UTeRM_MRA.py:61/:76 att_module.att1(cat[interp_ms, pan, PL]) (net_modules.py:8/:13) -- lms 를 읽는 "
                    "유일한 conv.  base = 하네스 lms: cs_comp = interp_ms + Gi*(pan-PL) (:78) 이 펼친 블록마다 충실도 기준"
                    "(:36-38)이고 출력은 마지막 C (:92) -- 맨 끝 덧셈 skip 은 없다 (연한 skip).  추가 LRMS 는 [0,1] 그대로.  "
                    "PerSample(registry.py:455-463)이 표본마다 net 을 부르므로 base forward/train_step 의 model 자리에 표본마다 "
                    "L[i] 를 거는 대리 객체를 넣는다 (손실 · 평가 코드는 base 그대로).")
