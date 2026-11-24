"""Global-skip 변형 연구 (lib/skipvar.py) -- PanFlowNet (ICCV 2023): 저자 코드에 global skip 이 **없는** (NONE) 방법의 sk · in 팔.

분류 NONE -- PanFlowNet 은 HRMS 를 직접 생성하는 조건부 정규화 흐름이다.  출력에 더해지는 업샘플 MS 가 어디에도 없다.
  repos/PanFlowNet/model/pan_inn9.py (patches/PanFlowNet.patch 적용본):
    :227-228  hms = interpolate(lms, x4, 'bilinear', align_corners=False);  x = hms        <- 저자 업샘플 = 흐름의 **조건**
    :248-252  rev=True (추론): out = 잡음 N(mean(hms), std(hms)) (:249, 배치 전체의 스칼라 둘 -- 공간 정보 없음),
              결합층을 거꾸로 (:251-252 `out = op(out, x, pan, rev)`)
    :171-195  결합층: 부망 s1 · s2 가 cat[x 절반, hms, pan] 을 읽어 (s, t) 를 내고 다른 절반을 아핀 변환한다
              (앞방향 :176 s2 -> :180 s1, 역방향 :186 s1 -> :190 s2).  hms 는 조건으로만 들어간다 -- 출력으로 가는 항등 경로가 없다.
    :259      return out (흐름의 역상), jac_loss
  registry.py:635  forward = m(lrms, ms_up, pan, rev=True)[0] -- ms_up 은 b_ms 자리로 가지만 rev=True 에서는 읽히지 않는다
              (b_ms 는 :240 rev=False 의 `out = b_ms` 에서만 읽힌다).  하네스 ms_up 은 skip 이 아니다 -> YES 아님.
  registry.py:615-625  train_step: 앞방향 model(ms, gt, pan) 의 NLL(jac, :620) + 역방향 model(.., rev=True) (:621) 의
              L1_sum/(2B) (:624).  epoch_frac >= 1000/1100 이면 NLL 만 (:622-623).

최소 · 원리적 형태 = 재중심 잔차 흐름 (저자 망 · 잠재 사전분포 · 손실 · 결합층 공유는 그대로):
  sk   x̂ = S + (F⁻¹(ε; hms, pan) − hms).  학습에서 흐름이 다루는 데이터 변수를 gt 에서 y' = gt − D 로 옮긴다 (D = S − hms;
       앞방향 NLL · 역방향 L1 모두).  역방향 L1 은 |F⁻¹ − y'| = |x̂ − gt| 라 예측 x̂ 에 건 L1 과 같다.
       - S = hms 이면 저자 모델과 같다 (D = 0: 같은 망 · 같은 잡음 · 같은 손실).  skv_ref_up = 저자 bilinear (:227) -- 검사가 증명한다.
       - 조건(hms)만으로 정해지는 평행이동은 야코비안 1 인 전단사다.  그래서 저자의 흐름 NLL 항 (:239-247, :258 -- 층마다
         0.5||z_k||² − log|det_k| 의 합) 은 잔차 r = gt − S 에 흐름 F̃(r) = F(r + hms) 를 건 것과 정확히 같다 (평행이동의
         log|det| = 0).  즉 "흐름이 gt − S 의 분포를 배운다" 이고, 그 좌표를 저자 기준선 hms 로 재중심했을 뿐이다.  원래
         PanFlowNet 도 같은 항등식으로 "hms + 잔차 흐름" 으로 읽힌다 (D = 0).
       - 재중심하는 까닭: 저자의 잠재 사전분포 N(mean(hms), std(hms)) (:249) 는 HRMS 값 범위에 맞춰져 있다.  y' 는
         mean(y') ≈ mean(hms) 라 (S 는 LRMS 와 일관이라 gt − S 의 평균 ≈ 0) 그 사전분포가 그대로 맞는다.  순수 잔차 흐름
         (데이터 = gt − S, 평균 0) 은 사전분포를 바꿔야 해서 방법이 바뀐다 -- 고르지 않는다.
       - 주의: 저자 코드에 덧셈 skip 이 없으므로 이것은 "있는 skip 을 갈아 끼우기" 가 아니라 위 재매개화다 (보고서 · 표 각주).
  in   sk 위에서 흐름의 조건을 (hms, pan) -> (hms, pan, S) 로 넓힌다.  조건을 읽는 것은 결합층의 두 부망 s1 · s2
       (HIN_CA_Block -> HinResBlock, :112-121 · :7-26) 뿐이고, 각 부망은 입력 cat[x 절반, hms, pan] 을 두 곳에서 읽는다:
       conv1 (:11, :18) 과 밀집 연결 conv3 (:14, :24-25 conv3(cat(x, residual))).  네 곳 모두에 S 채널(skv_w, 0 초기화)을
       더한다 = HIN_CA_Block(channel_in + C) 로, 즉 condition_length = 2C + 1 로 다시 지은 망과 정확히 같다 (LAGConv-in 과 같은
       원칙: 층의 입력 채널이 들어가는 모든 곳).  T=4 단계가 같은 결합층 객체를 공유하므로 (:218-224, 저장소 그대로) 자리 넷이
       모든 단계 · 양방향(학습 NLL 의 앞방향, 추론의 역방향)에 걸린다 -- build 가 공유를 확인한다.
       S 는 [0,1] 그대로 (PanFlowNet 은 정규화 없이 hms · pan 을 넣는다, option_wv2.yml normalize: False).
       train_step 의 두 모델 호출(:620-621)과 평가 forward 는 모두 lms 와 같은 배치 · 크기 -> 기본 make_in 감싸개가 맞다.
CPU: pan_inn9.py:249 가 torch.cuda.FloatTensor 를 박아 두어 rev=True 가 CPU 에서 돌지 않는다 (본래 스펙도 같다).  검사는 GPU, 또는
     그 생성자 하나만 CPU 로 바꾼 시험 프로세스에서 한다.
원래 스펙 · 저장소 코드는 건드리지 않는다 (base code_digest 불변).  파일 이름이 zz_ 인 까닭은 zz_skv_core.py 와 같다.
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys

import torch.nn.functional as F

sys.path.insert(0, (_GB + "/lib"))
import skipvar as SV                                                          # noqa: E402

#: 흐름의 조건 (hms, pan) 을 읽는 모든 conv -- 두 부망(s1 · s2) 의 HinResBlock conv1 (:11/:18) 과 conv3 (:14/:24-25).
#: operations[0..3] 은 같은 객체라 (pan_inn9.py:218-224) 'operations.0' 의 자리가 네 단계 모두다.
_IN_SITES = ["operations.0.s1.hin.conv1", "operations.0.s1.hin.conv3",
             "operations.0.s2.hin.conv1", "operations.0.s2.hin.conv3"]


def _hms(lrms, up=4):
    """pan_inn9.py:227 그대로 -- 흐름의 조건 hms (이 재매개화의 기준선)."""
    return F.interpolate(lrms, scale_factor=up, mode="bilinear", align_corners=False)


def register(SPECS, REPOS, _load, _pkg):
    B = SPECS["PanFlowNet"]
    base_fwd, base_ts = B["forward"], B["train_step"]              # registry.py:635 / :615-625 -- 같은 함수 객체

    # ------------------------------------------------------------------------------------------- sk
    def forward_sk(m, ms_up, pan, lrms, ctx):
        """x̂ = S + (F⁻¹(ε; hms, pan) − hms).  저자 역방향(registry.py:635)을 그대로 부르므로 잡음 난수열도 base 와 같다."""
        y = base_fwd(m, ms_up, pan, lrms, ctx)                     # ms_up 은 rev=True 에서 읽히지 않는다 (:239-240)
        return y - _hms(lrms, m.up_factor) + ms_up

    def train_step_sk(m, b, ctx, lossfn):
        """registry.py:615-625 그대로 -- 흐름의 데이터 변수만 gt -> gt − D (D = S − hms).  예측은 F⁻¹ + D."""
        D = b["lms"] - _hms(b["ms"], m.up_factor)
        y, loss = base_ts(m, dict(b, gt=b["gt"] - D), ctx, lossfn)
        return y + D, loss

    SV.make_sk(SPECS, "PanFlowNet", "PanFlowNet-sk", forward_sk, train_step_sk,
               ref_up=lambda lrms: _hms(lrms, 4),
               note="NONE 방법 -- 저자 코드에 덧셈 skip 이 없다 (pan_inn9.py:248-259: 출력 = 잡음 N(mean(hms), std(hms)) 의 흐름 역상, "
                    "hms(:227) 는 결합층 조건으로만).  재중심 잔차 흐름: x̂ = S + (F⁻¹(ε; hms, pan) − hms), 학습 데이터 변수 "
                    "gt − (S − hms) (NLL · L1 모두).  S = hms 이면 저자 모델과 같다.  조건만으로 정해지는 평행이동은 야코비안 1 이라 "
                    "저자 NLL 항 = 잔차 gt − S 에 F(· + hms) 를 건 NLL; 저자 잠재 사전분포(:249)가 그대로 맞도록 hms 로 재중심했다.  "
                    "저자 코드에 덧셈 skip 이 없으므로 '있는 skip 갈아 끼우기' 가 아니라 이 재매개화다.")

    # ------------------------------------------------------------------------------------------- in
    d = SV.make_in(SPECS, "PanFlowNet-sk", "PanFlowNet-in", _IN_SITES, "S",
                   note="자리: 결합층 부망 s1 · s2 의 HinResBlock conv1 (pan_inn9.py:11/:18) · conv3 (:14/:24-25, cat(x, residual) 의 "
                        "밀집 연결) -- 조건 cat[x 절반, hms, pan] (:176/:180/:186/:190) 이 들어가는 모든 곳.  = condition_length 2C+1 로 "
                        "지은 HIN_CA_Block 과 같다.  T=4 단계 공유(:218-224)라 네 자리가 모든 단계 · 양방향에 걸린다.  S 는 [0,1] 그대로.")
    _in_build = d["build"]

    def build_in(ms_ch=4, ratio=4):
        m = _in_build(ms_ch, ratio)
        ops = list(m.operations)
        if not all(op is ops[0] for op in ops):
            raise RuntimeError("PanFlowNet-in: 결합층이 단계마다 공유되지 않는다 (pan_inn9.py:218-224 가 바뀌었다) -- "
                               "자리를 단계마다 달아야 한다")
        for s in ("s1", "s2"):
            h = SV.module_at(m, f"operations.0.{s}.hin")
            # conv3 는 cat(x, residual) 을 읽는다 (:14, :24): 입력 = conv1 의 입력(x, 조건 포함) + conv1 특징
            if h.conv3.in_channels != h.conv1.out_channels + h.conv1.in_channels:
                raise RuntimeError(f"PanFlowNet-in: {s}.hin.conv3 입력 채널이 cat(x, residual) 이 아니다 (pan_inn9.py:14/:24)")
        return m

    d["build"] = build_in
