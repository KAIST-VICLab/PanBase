"""Global-skip 변형 연구 (lib/skipvar.py) -- INT 방법 넷: HFIN · MutInf · MMMamba · FAME-Net.

넷 모두 저자 코드가 저해상 MS 를 **모델 안에서** 업샘플해 global skip 으로 더한다 (하네스 `ms_up` 은 skip 이 아니다):
  HFIN      HFIN.py:480-482        mHR = bicubic(ms, align_corners=True);  HR = refine(process(mHR, pan)) + mHR
  MutInf    GPPNN.py:262-272       mHR = bicubic(ms, align_corners=True);  HR = refine(interact(...)) + mHR
  MMMamba   mmmamba.py:624-631     ms_bic = interpolate(ms, x4)  -- **nearest** (mode 생략 = 'nearest');  hrms = ms_head(...) + ms_bic
            mmmamba.py:623 의 둘째 인자 이름이 `_` 이고 본문에서 읽지 않는다 -> 하네스 ms_up 은 쓰이지 않는다 (YES 아님).
  FAME-Net  moe.py:342-352         mHR = bicubic(ms, align_corners=True);  HR = refine(dec) + mHR
            moe.py:332 의 둘째 인자 `_` 도 본문에서 읽지 않는다 (YES 아님).

sk 팔 `<M>-sk`: 본체(내부 업샘플을 읽는 encoder 들)는 그대로 두고 맨 끝 덧셈의 skip 항만 하네스 `ms_up`(= --skip 의 S)으로 바꾼다.
  Net.forward 를 저장소 줄 그대로 스펙 안에서 다시 쓴다 (저장소 클래스 · 모듈은 건드리지 않는다).  `skv_ref_up` 은 저자 업샘플과
  같은 호출이고, 검사가 ms_up = ref_up(lrms) 에서 sk forward == 본래 forward 를 증명한다.
in 팔 `<M>-in`: `<M>-sk` 위에서 PAN 을 처음 읽는 PAN 해상도 conv 가 S 를 추가 채널로 더 받는다 (skv_w 0 초기화).
  넷 모두 입력은 [0,1] 그대로 모델에 들어가므로(정규화 없음) S 도 [0,1] 그대로다 -- 기본 make_in 감싸개가 맞다.
  넷 모두 train_step 이 증강 · 크롭 · 복제 없이 배치를 그대로 모델에 넣으므로 기본 감싸개의 크기 · 배치가 맞다.
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, (_GB + "/lib"))
import skipvar as SV                                                          # noqa: E402


def _bicubic_ac(x, h, w):
    """HFIN.py:459-460 / GPPNN.py:233-234 / moe.py:287-288 의 `upsample` 와 같은 호출."""
    return F.interpolate(x, size=[h, w], mode="bicubic", align_corners=True)


def _ref_bicubic_ac(lrms):
    return _bicubic_ac(lrms, 4 * lrms.shape[-2], 4 * lrms.shape[-1])


def _ref_nearest4(lrms):
    """mmmamba.py:624 `F.interpolate(ms, scale_factor=4)` -- mode 기본값 'nearest'."""
    return F.interpolate(lrms, scale_factor=4)


def register(SPECS, REPOS, _load, _pkg):

    # ======================================================================== HFIN
    # 본래: registry.py SPECS["HFIN"] forward = m(lrms, pan) -> Net.forward (HFIN.py:470-484)
    #   :480  mHR = upsample(ms, M, N)          (bicubic, align_corners=True, :459-460)
    #   :481  HRf = self.process(mHR, pan)      <- 본체는 내부 업샘플을 읽는다 (sk 에서도 그대로)
    #   :482  HR = self.refine(HRf) + mHR       <- 이 `+ mHR` 만 `+ ms_up` 으로 바꾼다
    def hfin_forward_sk(m, ms_up, pan, lrms, ctx):
        mHR = _bicubic_ac(lrms, pan.shape[-2], pan.shape[-1])               # HFIN.py:477-480
        return m.refine(m.process(mHR, pan)) + ms_up                         # HFIN.py:481-482, skip -> ms_up

    SV.make_sk(SPECS, "HFIN", "HFIN-sk", hfin_forward_sk, ref_up=_ref_bicubic_ac,
               note="skip: HFIN.py:482 `refine(HRf) + mHR` 의 mHR(= HFIN.py:480 bicubic align_corners=True) -> ms_up.  "
                    "본체 FeatureProcess 는 여전히 mHR 를 읽는다.")
    # in 자리: FeatureProcess.conv_p1 = Conv2d(1, C, 3, 1, 1) (HFIN.py:426), HFIN.py:444 `panf = self.conv_p1(pan)` --
    # 본체에서 PAN 을 처음 읽는 PAN 해상도 conv (같은 해상도의 MS 갈래 conv_p(:443)는 mHR 를 읽는다).
    SV.make_in(SPECS, "HFIN-sk", "HFIN-in", ["process.conv_p1"], "S",
               note="자리: HFIN.py:426/444 FeatureProcess.conv_p1(pan) -> conv_p1(cat[pan, S]).  입력 [0,1] 그대로.")

    # ======================================================================== MutInf
    # 본래: registry.py `_mutinf` Wrap.forward(lrms, pan) -> GPPNN.forward(ms, 0, pan) (GPPNN.py:252-274)
    #   :262  mHR = upsample(ms, M, N)          (bicubic, align_corners=True, :233-234)
    #   :264  panf = self.extract_pan(pan)
    #   :265  mHRf = self.extract_ms(mHR)       <- 본체는 내부 업샘플을 읽는다 (sk 에서도 그대로)
    #   :267-268 feature_save(...)              <- registry 가 no-op 으로 덮어 둔 디버그 덤프 -- 여기서는 부르지 않는다
    #   :270-271 fmid = interact(cat[panf, mHRf])
    #   :272  HR = self.refine(fmid) + mHR      <- 이 `+ mHR` 만 skip 으로 바꾼다
    #   :274  return HR, panf, mHRf             (panf · mHRf 는 MI 정칙화 항이 읽는다)
    # 주의 (저장소 성질, 본래 스펙에도 같다): models/utils/CDC.py:83 `self.HP_branch = Parameter(torch.FloatTensor(1))` 는
    # **초기화되지 않은 메모리**다 (extract_pan/extract_ms 의 block1/2 에 넷).  같은 seed 라도 빌드마다 값이 다르고
    # (CPU 실측 -5.6e34 · -3.8e30 · 381.8 · 0 · 7e-15 ...), 학습된 best.pt 에도 wv3 extract_ms -1.2e30 / 6.9e27,
    # qb 넷 모두 -4.9e17 ~ -1.7e12 로 sigmoid 가 포화한 채 남아 있다.  그래서 MutInf 는 sk · in 팔이 init 에서 정확히
    # 짝지어지지 않는다 (그 4 스칼라만).  본래 칸과 같은 방법을 유지하려고 여기서 고치지 않는다.
    # 또 CDC.py:128,162 · GPPNN.py:373 이 `.cuda()` / torch.cuda.FloatTensor 를 박아 두어 CPU 에서 돌지 않는다.
    def _mutinf_sk(m, lrms, skip, pan):
        net = m.net
        mHR = _bicubic_ac(lrms, pan.shape[-2], pan.shape[-1])               # GPPNN.py:259-262
        panf = net.extract_pan(pan)                                         # GPPNN.py:264
        mHRf = net.extract_ms(mHR)                                          # GPPNN.py:265
        fmid = net.interact(torch.cat([panf, mHRf], dim=1))                 # GPPNN.py:270-271
        return net.refine(fmid) + skip, panf, mHRf                          # GPPNN.py:272-274, skip -> ms_up

    def mutinf_forward_sk(m, ms_up, pan, lrms, ctx):
        return _mutinf_sk(m, lrms, ms_up, pan)[0]

    def mutinf_train_step_sk(model, b, ctx, lossfn):
        # registry.py `_mutinf_step` 와 한 줄씩 같다 -- model(ms, pan) 자리만 skip 을 b["lms"](= S) 로 바꾼 _mutinf_sk 다.
        pred, panf, mHRf = _mutinf_sk(model, b["ms"], b["lms"], b["pan"])
        model.mutual.to(pred.device)
        ramp = min(ctx["epoch_frac"], 1.0)
        latent = torch.clip(model.mutual(panf, mHRf), -1, 1)
        return pred, 300.0 * lossfn(pred, b["gt"]) + 0.2 * ramp * latent

    SV.make_sk(SPECS, "MutInf", "MutInf-sk", mutinf_forward_sk, mutinf_train_step_sk, ref_up=_ref_bicubic_ac,
               note="skip: GPPNN.py:272 `refine(fmid) + mHR` 의 mHR(= GPPNN.py:262 bicubic align_corners=True) -> ms_up.  "
                    "본체 extract_ms 는 여전히 mHR 를 읽고, 손실(300·L1 + 0.2·ramp·clip(MI))은 본래 train_step 그대로다.  "
                    "CDC.py:83 HP_branch(4개)가 초기화되지 않은 메모리라 같은 seed 라도 팔 사이 init 이 그 4 스칼라만 다르다.")
    # in 자리: GPPNN.extract_pan.conv = Conv2d(1, n_feat//2, 1, 1, 0) (GPPNN.py:243 + FeatureExtract.conv :317),
    # GPPNN.py:264 `panf = self.extract_pan(pan)` -> FeatureExtract.forward :322 `xf = self.conv(x)` -- PAN 을 처음 읽는 conv.
    SV.make_in(SPECS, "MutInf-sk", "MutInf-in", ["net.extract_pan.conv"], "S",
               note="자리: GPPNN.py:243/264/317/322 extract_pan.conv(pan) -> conv(cat[pan, S]).  입력 [0,1] 그대로.  "
                    "panf 가 바뀌므로 MI 항(mutual(panf, mHRf))도 S 를 본 panf 를 읽는다.")

    # ======================================================================== MMMamba
    # 본래: registry.py SPECS["MMMamba"] forward = m(lrms, ms_up, pan) -> Net.forward(ms, _, pan) (mmmamba.py:623-632)
    #   :623  둘째 인자 `_` (하네스 ms_up) 는 본문에서 쓰이지 않는다 -> YES 가 아니라 INT
    #   :624  ms_bic = F.interpolate(ms, scale_factor=4)   <- mode 생략 = **nearest** (bicubic 이 아니다)
    #   :625  ms_f = self.ms_encoder(ms_bic)               <- 본체는 내부 업샘플을 읽는다 (sk 에서도 그대로)
    #   :627  pan_f = self.pan_encoder(pan)
    #   :628-630 mm_mamba([ms_f, pan_f]) -> simple_fusion(cat)
    #   :631  hrms = self.ms_head(ms_f) + ms_bic          <- 이 `+ ms_bic` 만 `+ ms_up` 으로 바꾼다
    # 주의: mm_mamba 의 `Attention`(mmmamba.py:25) 은 이름과 달리 selective scan 이다 -- :106 `self.selective_scan =
    # selective_scan_fn` (:21 mamba_ssm CUDA 커널), :309 forward_core 에서 부른다 (:338).  CPU 에서는 mamba_ssm import 부터
    # (triton "0 active drivers") 실패하므로 GPU 에서만 지을 수 있다.
    def mmmamba_forward_sk(m, ms_up, pan, lrms, ctx):
        ms_bic = F.interpolate(lrms, scale_factor=4)                        # mmmamba.py:624 (nearest)
        ms_f = m.ms_encoder(ms_bic)                                         # mmmamba.py:625
        pan_f = m.pan_encoder(pan)                                          # mmmamba.py:627
        ms_f, pan_f = m.mm_mamba([ms_f, pan_f])                             # mmmamba.py:628-629
        ms_f = m.simple_fusion(torch.concat([ms_f, pan_f], dim=1))          # mmmamba.py:630
        return m.ms_head(ms_f) + ms_up                                      # mmmamba.py:631, skip -> ms_up

    SV.make_sk(SPECS, "MMMamba", "MMMamba-sk", mmmamba_forward_sk, ref_up=_ref_nearest4,
               note="skip: mmmamba.py:631 `ms_head(ms_f) + ms_bic` 의 ms_bic(= mmmamba.py:624 nearest x4) -> ms_up.  "
                    "본체 ms_encoder 는 여전히 nearest ms_bic 를 읽는다.  하네스 ms_up 은 본래 Net 에서 쓰이지 않는다(:623 `_`).")
    # in 자리: Net.pan_encoder[0] = Conv2d(1, 32, 3, 1, 1) (mmmamba.py:618), mmmamba.py:627 `pan_f = self.pan_encoder(pan)`.
    SV.make_in(SPECS, "MMMamba-sk", "MMMamba-in", ["pan_encoder.0"], "S",
               note="자리: mmmamba.py:618/627 pan_encoder[0](pan) -> pan_encoder[0](cat[pan, S]).  입력 [0,1] 그대로.")

    # ======================================================================== FAME-Net
    # 본래: specs/fame_net.py forward = m(lrms, ms_up, pan)[0], train_step `_step` -> Net.forward(ms, _, pan) (moe.py:332-353)
    #   :332  둘째 인자 `_` (하네스 ms_up) 는 본문에서 쓰이지 않는다 -> YES 가 아니라 INT
    #   :342  mHR = upsample(ms, M, N)          (bicubic, align_corners=True, :287-288)
    #   :343  msf = self.msconv(mHR)            <- 본체는 내부 업샘플을 읽는다 (sk 에서도 그대로)
    #   :344  panf = self.pconv(pan)
    #   :345-351 encoders · maskp(gumbel, 확률적) · moeInstance · decoder
    #   :352  HR = self.refine(dec) + mHR       <- 이 `+ mHR` 만 skip 으로 바꾼다
    #   :353  return HR, mask, [lf_gate, hf_gate, dec_gate]
    def _fame_sk(m, lrms, skip, pan):
        mHR = _bicubic_ac(lrms, pan.shape[-2], pan.shape[-1])               # moe.py:339-342
        msf = m.msconv(mHR)                                                 # moe.py:343
        panf = m.pconv(pan)                                                 # moe.py:344
        msf = m.msencoder(msf)                                              # moe.py:345
        panf = m.panencoder(panf)                                           # moe.py:346
        mask = m.maskp(torch.cat([msf, panf], dim=1))                       # moe.py:347
        high_fre_mask = (mask[:, 0, ...]).unsqueeze(1)                      # moe.py:348
        low_fre_mask = (mask[:, 1, ...]).unsqueeze(1)                       # moe.py:349
        lf, hf, lf_gate, hf_gate = m.moeInstance(panf, msf, high_fre_mask, low_fre_mask)   # moe.py:350
        dec, dec_gate = m.decoder(torch.cat([msf, hf, panf, lf], dim=1))   # moe.py:351
        return m.refine(dec) + skip, mask, [lf_gate, hf_gate, dec_gate]     # moe.py:352-353, skip -> ms_up

    def fame_forward_sk(m, ms_up, pan, lrms, ctx):
        return _fame_sk(m, lrms, ms_up, pan)[0]

    fame_epochs = 1000                                                      # fame_net.py:11 OFFICIAL_EPOCHS = 1000

    def _cv(z):                                                             # fame_net.py:29-36 (moesolver.py:87-96)
        return torch.mean((torch.std(z, dim=1) / torch.mean(z, dim=1)) ** 2)

    def fame_train_step_sk(model, b, ctx, lossfn):
        # specs/fame_net.py `_step` (:38-67) 와 한 줄씩 같다 -- model(ms, None, pan) 자리만 skip 을 b["lms"](= S) 로 바꾼
        # _fame_sk 다.  fame_net.py:16 MASK_TARGET_FN 은 None 이므로 마스크 항은 b["mask_gt"] 가 있을 때만 걸린다 (하네스 배치에는
        # 없다 -- 본래 스펙과 같이 그 항은 0).
        y, mask, (lf_gate, hf_gate, dec_gate) = _fame_sk(model, b["ms"], b["lms"], b["pan"])
        gl = _cv(lf_gate) + _cv(hf_gate) + _cv(dec_gate)
        loss = lossfn(y, b["gt"]) + gl
        ep = min(fame_epochs, int(ctx.get("epoch_frac", 0.0) * fame_epochs) + 1)
        para = 0.99 ** (ep - 1)
        mgt = b.get("mask_gt")
        if mgt is not None:
            loss = loss + para * lossfn(mgt, mask)
        return y, loss

    SV.make_sk(SPECS, "FAME-Net", "FAME-Net-sk", fame_forward_sk, fame_train_step_sk, ref_up=_ref_bicubic_ac,
               note="skip: moe.py:352 `refine(dec) + mHR` 의 mHR(= moe.py:342 bicubic align_corners=True) -> ms_up.  "
                    "본체 msconv 는 여전히 mHR 를 읽고, 손실(L1 + CV 게이트 3항 [+ 마스크 항, 라벨 없어 0])은 본래 그대로다.")
    # in 자리: Net.pconv = Conv2d(1, 32, 3, 1, 1) (moe.py:323), moe.py:344 `panf = self.pconv(pan)`.
    SV.make_in(SPECS, "FAME-Net-sk", "FAME-Net-in", ["pconv"], "S",
               note="자리: moe.py:323/344 pconv(pan) -> pconv(cat[pan, S]).  입력 [0,1] 그대로.")
