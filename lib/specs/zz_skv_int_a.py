"""Global-skip 변형 연구 (lib/skipvar.py) -- registry.py 에 있는 INT 방법 다섯: PanNet · PGCU · ADWM · MDCUN · MSDDN.

다섯 모두 저장소가 저해상 MS 를 **스스로** 업샘플해 global skip 으로 더한다 -- 하네스 `ms_up`(lms) 은 skip 이 아니다
(PanNet · PGCU · ADWM 은 forward 가 ms_up 을 받지도 않고, MDCUN · MSDDN 은 받아서 버린다: 아래 인용).  그래서
  `<M>-sk`  내부 업샘플 skip 항만 하네스 ms_up (= --skip 의 S) 로 바꾼다 (본체 입력은 그대로).  `skv_ref_up` 은 저자 업샘플.
  `<M>-in`  `<M>-sk` 위에 본체에서 PAN (또는 PAN+MS 융합) 을 처음 읽는 PAN 해상도 conv 가 S 를 추가 채널로 더 받는다.
짧은 forward (PanNet · PGCU · ADWM) 는 저자 forward 를 줄 그대로 옮기고 마지막 덧셈만 바꿨다 (ms_up = 저자 업샘플이면
비트 단위로 같다).  긴 forward (MDCUN · MSDDN) 는 저자 forward 를 그대로 부르고 skip 을 갈아 끼운다:
out - up(lrms) + ms_up -- 기울기는 같고 차이는 float 반올림(~1e-7)뿐이다.
원래 스펙 · 저장소 코드는 건드리지 않는다 (base code_digest 불변).  파일 이름이 zz_ 인 까닭은 zz_skv_core.py 와 같다.
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, (_GB + "/lib"))
import skipvar as SV                                                          # noqa: E402


def register(SPECS, REPOS, _load, _pkg):
    # 본래 PanNet/PGCU 스펙이 쓰는 고주파 함수 그 자체 (registry._hp).  registry 를 다시 import 하지 않고
    # 본래 forward 의 전역에서 꺼낸다 -- 같은 함수 객체라 정의가 갈라질 수 없다.
    _hp = SPECS["PanNet"]["forward"].__globals__["_hp"]

    # ------------------------------------------------------------------------------------------- PanNet
    # repos/PGCU/model/PanNet.py:47-54 (class PanNet):
    #   :48 x_ms = interpolate(ms, scale_factor=(4,4), mode='bicubic')   <- global skip (저자 업샘플)
    #   :49 up_ms = ConvTrans(hms);  :50 x = cat[hpan, up_ms];  :51 y = relu(Conv1(x))   <- 본체 첫 PAN 해상도 conv
    #   :52-53 ResidualBlocks, Conv2;  :54 return y + x_ms
    # (ConvTrans(:36-37) 는 ConvTranspose2d 이고 hms 만 읽는다 -- PAN 을 읽는 첫 Conv2d 는 Conv1 이다.)
    def pannet_ref_up(lrms):
        return F.interpolate(lrms, scale_factor=(4, 4), mode="bicubic")             # PanNet.py:48 그대로

    def pannet_sk(m, ms_up, pan, lrms, ctx):
        up_ms = m.ConvTrans(_hp(lrms))                                               # :49 (hms = _hp(lrms), 본래 스펙)
        y = F.relu(m.Conv1(torch.cat([_hp(pan), up_ms], dim=1)))                    # :50-51
        y = m.ResidualBlocks(y)                                                      # :52
        y = m.Conv2(y)                                                               # :53
        return y + ms_up                                                             # :54 의 x_ms -> ms_up

    SV.make_sk(SPECS, "PanNet", "PanNet-sk", pannet_sk, ref_up=pannet_ref_up,
               note="skip: PanNet.py:48/:54 의 bicubic(lrms) -> ms_up.  본체 입력(hpan, ConvTrans(hms))은 그대로.")

    # in: Conv1 (PanNet.py:38 정의, :51 사용) 이 cat[hpan, up_ms] 옆에 S 를 더 받는다.
    # 정규화: PanNet 은 본체의 **모든** 입력을 고주파 영역으로 준다 (hms = _hp(lrms), hpan = _hp(pan); 본래 스펙의
    # forward).  같은 conv 의 이웃(hpan, ConvTrans(hms))과 같은 영역이 되도록 E = _hp(S) 다 -- 날 S 를 주면 PanNet 이
    # 설계상 보지 않는 저주파 채널이 본체에 들어간다.  그래서 기본 감싸개 대신 forward 를 직접 준다.
    def pannet_in(m, ms_up, pan, lrms, ctx):
        S, _L = SV.split(ms_up)
        with SV.extra(m, _hp(S)):
            return pannet_sk(m, S, pan, lrms, ctx)

    SV.make_in(SPECS, "PanNet-sk", "PanNet-in", ["Conv1"], "S", forward=pannet_in,
               note="자리: PanNet.py:51 Conv1(cat[hpan, up_ms]).  추가 입력은 _hp(S) (PanNet 의 고주파 입력 영역).")

    # --------------------------------------------------------------------------------------------- PGCU
    # repos/PGCU/model/PanNet.py:22-28 (class PanNet_PGCU):
    #   :23 up_ms = PGCU(pan, ms)          <- **학습되는** PAN 유도 업샘플러 (PGCU.py:58-103: nearest x4 :59 + 교차어텐션)
    #   :24 x = cat[hpan, up_ms];  :25 y = relu(Conv1(x));  :26-27 ResidualBlocks, Conv2
    #   :28 return y + up_ms, up_ms         <- global skip = up_ms, 그리고 up_ms 에 보조 손실 (train_pannet.py:71-75)
    # skip 이 lrms 의 결정적 함수가 아니다 (학습되는 PGCU 의 출력).  가장 충실한 sk: **맨 끝 덧셈의 up_ms 만** ms_up 으로
    # 바꾼다.  PGCU 모듈은 그대로 본체 입력(:24)을 만들고, 보조 손실(두 번째 출력 up_ms, aux_loss)도 그대로 받는다 --
    # 논문의 기여(PGCU 업샘플러)를 지우지 않는다.  ref_up 은 둘 수 없다 (모델이 필요하다): 검사기의 대조 대신
    # ms_up = m.PGCU(pan, lrms) 를 넣어 본래 forward 와 같은지를 따로 확인한다.
    def pgcu_sk(m, ms_up, pan, lrms, ctx):
        up_ms = m.PGCU(pan, lrms)                                                    # :23
        y = F.relu(m.Conv1(torch.cat([_hp(pan), up_ms], dim=1)))                    # :24-25 (hpan = _hp(pan), 본래 스펙)
        y = m.ResidualBlocks(y)                                                      # :26
        y = m.Conv2(y)                                                               # :27
        return y + ms_up, up_ms                                                      # :28 의 skip up_ms -> ms_up

    SV.make_sk(SPECS, "PGCU", "PGCU-sk", pgcu_sk,
               note="skip: PanNet.py:28 의 up_ms(학습되는 PGCU 업샘플러 출력) -> ms_up.  PGCU 는 본체 입력(:24)과 보조 손실 "
                    "(train_pannet.py:71-75, 두 번째 출력)에 그대로 남는다.  저자 skip 이 lrms 의 결정적 함수가 아니라 "
                    "skv_ref_up 이 없다.  본래 PGCU 는 tier=excluded (64 학습 패치에서 PGCU 어텐션 퇴화) -- 그 사정은 같다.")

    # in: Conv1 (PanNet.py:14 정의, :25 사용) 이 cat[hpan, up_ms] 옆에 S 를 더 받는다.  PGCU 모듈 안의 FPConv(PGCU.py:61)
    # 도 PAN 을 읽지만 그것은 업샘플러이지 본체가 아니다.  정규화: 이웃 up_ms 가 날 [0,1] MS 영역이라 날 S (기본 감싸개).
    SV.make_in(SPECS, "PGCU-sk", "PGCU-in", ["Conv1"], "S",
               note="자리: PanNet.py:25 Conv1(cat[hpan, up_ms]).  추가 입력은 날 S (이웃 up_ms 와 같은 영역).")

    # --------------------------------------------------------------------------------------------- ADWM
    # repos/ADWM/model/bwnet_fusionnet.py:110-123 (BWNet.forward(ms, pan)):
    #   :111 ms = interpolate(ms, scale_factor=4, mode='bicubic')   <- 저자 업샘플 (본체 입력이자 skip)
    #   :112 pan = pan.repeat(1, ms_dim, 1, 1);  :113 input = pan - ms
    #   :115 x = raise_dim(input)            <- 본체 첫 PAN 해상도 conv (raise_dim.0, :98-101)
    #   :118-121 Resblock 들, bw_output;  :123 return to_output(output) + ms
    # sk: :123 의 + ms 만 ms_up 으로.  본체 입력 pan - bicubic(lrms) (:113) 은 그대로다.
    def adwm_ref_up(lrms):
        return F.interpolate(lrms, scale_factor=4, mode="bicubic")                  # bwnet_fusionnet.py:111 그대로

    def adwm_sk(m, ms_up, pan, lrms, ctx):
        ms = F.interpolate(lrms, scale_factor=4, mode="bicubic")                    # :111
        x = m.raise_dim(pan.repeat(1, m.ms_dim, 1, 1) - ms)                          # :112-115
        feature_list = []
        for layer in m.layers:                                                       # :118-120
            x = layer(x)
            feature_list.append(x)
        output = m.bw_output(feature_list)                                           # :121
        return m.to_output(output) + ms_up                                           # :123 의 ms -> ms_up

    SV.make_sk(SPECS, "ADWM", "ADWM-sk", adwm_sk, ref_up=adwm_ref_up,
               note="skip: bwnet_fusionnet.py:123 의 bicubic(lrms) -> ms_up.  본체 입력 pan - bicubic(lrms) (:113) 은 그대로.")

    # in: raise_dim.0 (bwnet_fusionnet.py:99, :115) 가 pan - bicubic(lrms) 옆에 S 를 더 받는다.  ADWM 은 날 [0,1] 을
    # 먹이므로 날 S (기본 감싸개).  pan - S 로 주지 않는다: 이웃의 차분은 두 입력의 융합이지 입력별 정규화가 아니다.
    SV.make_in(SPECS, "ADWM-sk", "ADWM-in", ["raise_dim.0"], "S",
               note="자리: bwnet_fusionnet.py:115 raise_dim.0(pan - bicubic(lrms)).  추가 입력은 날 S.")

    # -------------------------------------------------------------------------------------------- MDCUN
    # repos/MDCUN/model/pan_unfolding_v4.py:344-402 (pan_unfolding.forward(lms, b_ms, pan)):
    #   :344 b_ms(= 하네스 ms_up) 는 forward 어디에서도 읽히지 않는다 -> YES 가 아니라 INT 다.
    #   :355-356 hms = interpolate(lms, x4, 'bilinear', align_corners=False); x = hms   <- 펼친 반복의 초기값 x^(0)
    #   :398 x = x - delta[i] * (...)  (T=6 번);  :402 return outs_list[-1]
    # 명시적인 덧셈 skip 은 없다: 출력은 x^(T) = x^(0) + sum_k Delta_k 이고 x^(0) 가 저자 업샘플이다.  sk 는 정의대로
    # (본체 입력은 그대로) 반복 궤적은 x^(0) = bilinear(lrms) 에서 그대로 돌리고, 쌓인 갱신 sum_k Delta_k 를 x^(0) 대신
    # ms_up 위에 얹는다: out = x^(T) - x^(0) + ms_up.  (대안 -- 반복을 x^(0) = S 에서 시작 -- 은 본체가 S 를 읽게 되어
    # sk 정의가 아니다; 보고서에 적는다.)
    def mdcun_ref_up(lrms):
        return F.interpolate(lrms, scale_factor=4, mode="bilinear", align_corners=False)   # :355 그대로 (up_factor=4)

    def mdcun_sk(m, ms_up, pan, lrms, ctx):
        xT = m(lrms, ms_up, pan)                                                     # 저자 forward 그대로 (b_ms 는 무시된다)
        return xT - mdcun_ref_up(lrms) + ms_up                                       # x^(0) -> ms_up

    SV.make_sk(SPECS, "MDCUN", "MDCUN-sk", mdcun_sk, ref_up=mdcun_ref_up,
               note="skip: 펼친 반복의 초기값 x^(0) = bilinear(lrms) (pan_unfolding_v4.py:355-356) 가 skip 역할이다 "
                    "(출력 = x^(0) + 쌓인 갱신, :398/:402).  궤적은 그대로 두고 out = x^(T) - x^(0) + ms_up.  "
                    "b_ms(하네스 ms_up) 는 저장소 forward 가 읽지 않는다 (:344).")

    # in: 본체(펼친 반복 :364-400)에서 PAN 을 처음 읽는 conv = rm1 의 첫 conv (att_spatial.block[0] = ConvBlock(2, 32, 3),
    # :178-179, conv :226).  :375 · :391 에서 cat[uk/vk 의 밴드 하나, pan] 을 읽는다 -- PAN+MS 융합 자리이고, 밴드 ·
    # u/v 갈래 · 단계마다 (2*C*T 번) 불린다; 매번 같은 S 를 본다 (배치 · 크기가 같다).  실행 순서로는 hf_pan (:342, :352;
    # 1x1, 3 -> 1 채널, PAN 고주파 셋을 섞는 전처리) 이 먼저지만 그 자리의 추가 가중치는 C 개뿐(공간 문맥 없음)이라 in 팔이
    # sk 와 거의 같아진다 -- 본체가 S 를 실제로 볼 수 있는 첫 PAN 융합 conv 를 고른다.  날 S (MDCUN 은 날 [0,1]).
    SV.make_in(SPECS, "MDCUN-sk", "MDCUN-in", ["rm1.block.0.conv"], "S",
               note="자리: pan_unfolding_v4.py:375/:391 rm1(cat[밴드, pan]) 의 첫 conv (att_spatial.block[0].conv, :179/:226), "
                    "밴드 · 갈래 · 단계마다 2*C*T 번 불린다.  추가 입력은 날 S.")

    # -------------------------------------------------------------------------------------------- MSDDN
    # repos/MSDDN/model/msddn.py:135-192 (Net.forward(ms, _, pan)):
    #   :135 둘째 인자 `_`(= 하네스 ms_up) 는 버려진다 (:142 에서 `_` 가 다시 묶인다) -> INT 다.
    #   :145-146 mHR = upsample(ms, M, N); ms = mHR   (upsample = bicubic, align_corners=True, :101-102)  <- 본체 입력이자 skip
    #   :147 msf = msconv(ms) (MS 만);  :148 panf = panconv(pan)   <- 본체에서 PAN 을 읽는 첫 conv
    #   :191 HR = t + ms
    # sk: :191 의 + ms 만 ms_up 으로.  본체 입력 msconv(bicubic(lrms)) (:147) 은 그대로다.
    def msddn_ref_up(lrms):
        h, w = lrms.shape[-2:]
        return F.interpolate(lrms, size=[4 * h, 4 * w], mode="bicubic", align_corners=True)   # :101-102, :145 (M,N = 4h,4w)

    def msddn_sk(m, ms_up, pan, lrms, ctx):
        HR = m(lrms, ms_up, pan)                                                     # 저자 forward 그대로 (둘째 인자는 버려진다)
        mHR = F.interpolate(lrms, size=list(pan.shape[-2:]), mode="bicubic", align_corners=True)   # :145 그대로
        return HR - mHR + ms_up                                                      # :191 의 ms -> ms_up

    SV.make_sk(SPECS, "MSDDN", "MSDDN-sk", msddn_sk, ref_up=msddn_ref_up,
               note="skip: msddn.py:191 의 bicubic(lrms, align_corners=True) -> ms_up.  본체 입력 msconv(bicubic(lrms)) "
                    "(:147) 은 그대로.  둘째 인자(하네스 ms_up)는 저장소 forward 가 버린다 (:135/:142).")

    # in: panconv (msddn.py:117 정의, :148 사용) 가 pan 옆에 S 를 더 받는다 -- 본체에서 PAN 을 읽는 첫 PAN 해상도 conv
    # (:147 msconv 는 MS 만 읽는다).  날 S (MSDDN 은 날 [0,1]).
    SV.make_in(SPECS, "MSDDN-sk", "MSDDN-in", ["panconv"], "S",
               note="자리: msddn.py:148 panconv(pan).  추가 입력은 날 S.")
