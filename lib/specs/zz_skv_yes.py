"""Global-skip 변형 연구의 in 팔 (lib/skipvar.py) -- YES 방법 여섯: LAGConv · RAMSF · BNNPan · CANConv · PanDiT · TMDiff.

YES 방법은 하네스 `lms` 를 global skip 으로 더하고 본체도 같은 `lms` 를 읽는다.  sk 팔은 데이터만 바꾼 본래 스펙이다
(`--skip`, lms = S).  in 팔은 그 위에서 본체가 lms 를 처음 읽는 PAN 해상도 conv 가 LRMS(= 원래 하네스 lms, 업샘플 MS)를
추가 채널로 더 받는다 (가중치 `skv_w` 0 초기화 -> 같은 seed 에서 in 의 초기 함수 = sk 의 초기 함수).

skipvar.add_extra_input 이 그대로 맞지 않는 곳은 이 파일 안에서 같은 원리(conv(cat[x, E]) = conv_x(x) + conv_E(E),
`_skv_box` 를 나눠 써서 `SV.extra(model, E)` 가 그대로 E 를 건다)로 hook 을 단다.  저장소 클래스 · 모듈은 건드리지 않는다
-- 모든 hook 은 in 팔 build 가 지은 **인스턴스**에만 붙는다.
  LAGConv  첫 층이 nn.Conv2d 가 아니라 LAConv2D (국소 적응 가중치 x unfold x weight + 전역 적응 bias) 다.
           LAConv2D(1+2C) 를 cat[pan, S, L] 에 건 것과 같도록 세 곳(attention1 첫 conv, attention3 1x1 conv, 본 weight)에
           모두 L 채널을 더한다.
  PanDiT   EMA 사본(ema_net)이 평가를 맡고 파라미터를 zip 으로 갱신한다(pandit.py:173-177).  net 과 ema_net 의 같은 자리에
           같은 구조의 skv_w 를 단다 (conv 안에서 weight, bias 다음 -> zip 짝 그대로, EMA 가 skv_w 도 평균한다).  ema_net 의
           skv_w 는 requires_grad=False (pandit.py:112 의 ema_net 과 같게).  평가 크기별 사본(H != 64)도 같은 자리를 달고
           _share 가 ema_net 의 skv_w 를 꽂는다.  일반 hook (skipvar._hook) 을 쓴다.
           자리 conv 앞에 ReplicationPad2d 가 있어 E 도 같은 패딩을 거친다.  샘플러가 배치를 청크로 쪼개므로
           (pandit.py:213-217) forward 는 pandit.py 의 sample 을 그대로 옮기고 청크마다 E 를 건다.
  TMDiff   본체에 nn.Conv2d 가 없다 -- 스펙트럼 축을 depth 로 쓰는 Conv3d(채널 1) 다.  조건은 to3D(PAN - MS) 이고
           첫 conv 는 Conv3d(1, 32, 1) 이다.  E = PAN - L 을 같은 to3D 로 두 번째 '채널' 로 붙인다(정규화 = 조건과 같은 꼴).
           샘플러가 픽셀 예산으로 청크를 쪼개므로(tmdiff.py:175, 184-185) 청크마다 E 를 건다.
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import contextlib
import hashlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, (_GB + "/lib"))
import skipvar as SV                                                          # noqa: E402


class _Slot:
    """모듈 트리 밖의 값 하나 (등록되지 않는다, deepcopy 는 따라간다)."""
    __slots__ = ("v",)

    def __init__(self, v=None):
        self.v = v


def _fp(fn):
    """함수 바이트코드 지문 (주석 · 줄 번호 무관).  옮겨 쓴 저장소/스펙 코드가 바뀌면 in 팔이 조용히 어긋나지 않게 멈춘다."""
    fn = getattr(fn, "__wrapped__", fn)

    def norm(co):
        return (bytes(co.co_code), co.co_names, co.co_varnames,
                tuple(norm(k) if hasattr(k, "co_code") else
                      (k if isinstance(k, (int, float, str, bytes, bool, type(None))) else repr(k))
                      for k in co.co_consts))
    return hashlib.sha256(repr(norm(fn.__code__)).encode()).hexdigest()[:16]


def _check_E(name, E, x, hw=True):
    if E is None:
        raise RuntimeError(f"skipvar: '{name}' 의 추가 입력이 비어 있다 -- extra(model, E) 안에서만 부른다")
    if E.shape[0] != x.shape[0] or (hw and tuple(E.shape[-2:]) != tuple(x.shape[-2:])):
        raise RuntimeError(f"skipvar: '{name}' 입력 {tuple(x.shape)} 와 추가 입력 {tuple(E.shape)} 의 배치/크기가 다르다")


# ============================================================== LAGConv: LAConv2D 세 갈래
# repos/LAGConv/model.py:47-84  LAConv2D.forward(x):
#   atw = attention1(x)                       :53   (attention1[0] = Conv2d(n, k*k, k, s, p), :21-28)
#   y   = fold( (atw 를 n 채널로 반복) * unfold(x) @ weight )   :56-74  (weight = Conv2d(n, m, k).weight, :43-44)
#   y  += attention3(x)                       :76-80 (attention3 = [AdaptiveAvgPool2d(1), Conv2d(n, m, 1), ReLU, Conv2d], :36-41)
# 입력 채널 n 이 들어가는 곳은 이 셋뿐이다.  cat[x, E] 를 넣은 LAConv2D(n+C) 는 정확히
#   attention1[0](x) + conv(E; skv_w_a1)  ·  본 항 + Σ atw * unfold(E) * skv_w  ·  attention3[1](pool x) + conv(pool E; skv_w_a3)
# 이다 (unfold 의 채널 우선 순서 c*k*k+q 와 atw 의 반복 순서가 같다, :57-58 · :66).
def _lag_pooled_hook(mod, inp, out):
    x = inp[0]
    E = mod._skv_box.E
    _check_E("LAGConv attention3.1", E, x, hw=False)
    Ep = F.adaptive_avg_pool2d(E.to(dtype=x.dtype), tuple(x.shape[-2:]))     # attention3[0] = AdaptiveAvgPool2d(1)
    return out + F.conv2d(Ep, mod.skv_w, None, mod.stride, mod.padding, mod.dilation)


def _lag_capture_hook(mod, inp, out):
    mod._skv_slot.v = out                       # atw (b, k*k, nH, nW) -- E 가 반영된 attention1 출력


def _lag_main_hook(mod, inp, out):
    x = inp[0]
    E = mod._skv_box.E
    _check_E("LAGConv head_conv.0", E, x)
    s = mod.attention1._skv_slot
    atw, s.v = s.v, None
    if atw is None:
        raise RuntimeError("skipvar: LAGConv head_conv.0 의 attention1 출력이 잡히지 않았다")
    b, C = E.shape[:2]
    k, m = mod.kernel_size, mod.out_planes
    U = F.unfold(E.to(dtype=x.dtype), kernel_size=k, stride=mod.stride, padding=mod.padding)   # (b, C*k*k, L) :66
    U = U.view(b, C, k * k, -1) * atw.reshape(b, 1, k * k, -1)                                 # :56-58, :67
    y = torch.einsum("bcql,mcq->bml", U, mod.skv_w.view(m, C, k * k))                          # :70-75
    return out + y.view(b, m, out.shape[-2], out.shape[-1])                                     # :82-83


def _lag_install(model, C):
    box = SV._Box()
    SV.add_extra_input(model, ["head_conv.0.attention1.0"], C, box=box)       # 일반 hook (Conv2d, groups 1)
    lac = SV.module_at(model, "head_conv.0")
    a3 = SV.module_at(model, "head_conv.0.attention3.1")
    assert type(lac).__name__ == "LAConv2D" and lac.bias is True, type(lac).__name__
    assert isinstance(a3, nn.Conv2d) and a3.groups == 1 and a3.kernel_size == (1, 1)
    assert isinstance(lac.attention3[0], nn.AdaptiveAvgPool2d)
    dev, dt = lac.weight.device, lac.weight.dtype
    a3.register_parameter("skv_w", nn.Parameter(torch.zeros(a3.out_channels, C, 1, 1, device=dev, dtype=dt)))
    a3._skv_box = box
    a3.register_forward_hook(_lag_pooled_hook)
    k = lac.kernel_size
    lac.register_parameter("skv_w", nn.Parameter(torch.zeros(lac.out_planes, C, k, k, device=dev, dtype=dt)))
    lac._skv_box = box
    lac.attention1._skv_slot = _Slot()
    lac.attention1.register_forward_hook(_lag_capture_hook)
    lac.register_forward_hook(_lag_main_hook)
    model._skv_sites = list(model._skv_sites) + ["head_conv.0.attention3.1", "head_conv.0"]
    return model


# ============================================================== PanDiT
# 자리: dit.layers.0 (PYRAMID, PanVIT_modules.py:709-722) 의 cond_inj.ms_conv.body.1.
#   PanVIT.py:72-82 -- 디노이저 입력 conv prev_conv(:44, :77) 은 잡음 잔차 x (+ self-cond) 만 읽는다.  조건 cond 를 처음 읽는
#   것은 dit.layers[0].cond_inj(cond) (:80 -> PanVIT_modules.py:548-550) 이고, 그 안의 PyramidalSpatialBlock.forward 가
#   ms = ms_conv(cond[:, :C]) (:471) -- cond[:, :C] 는 lms 그 자체([0,1], pandit.py:54-62 _cond).  ms_conv = ConvLayer(:410-417)
#   의 body = [ReplicationPad2d(k//2), Conv2d(C, 2h, k), ...] (:176-182) 라 첫 Conv2d 는 패딩된 입력을 받는다 -> E 도 같은
#   ReplicationPad2d 를 거쳐 건다 (패딩은 채널별이라 conv(pad(cat[S, L])) 와 정확히 같다).
#   나머지 12 개 cond 주입(dit.layers.1-11 · final_layer, :470-471 · :239 · :682)은 sk 와 같이 S 만 읽는다.
_PD_SITE = "dit.layers.0.cond_inj.ms_conv.body.1"
_PD_PAD = "dit.layers.0.cond_inj.ms_conv.body.0"
# lib/specs/pandit.py 의 _PanDiT.sample (:192-227) 을 아래 _pd_sample 로 옮겼다.  원본이 바뀌면 멈춘다.
_PD_SAMPLE_FP = "a3844920ddadc2ea"


def _pd_install_site(net, C, box, frozen):
    """PanDiT 한 벌(net / ema_net / 크기별 사본)의 자리에 일반 추가 입력을 단다.  E 는 미리 ReplicationPad2d 를 거친 것."""
    c = SV.module_at(net, _PD_SITE)
    p = SV.module_at(net, _PD_PAD)
    assert isinstance(c, nn.Conv2d) and c.groups == 1 and c.padding_mode == "zeros" and tuple(c.padding) == (0, 0), c
    assert isinstance(p, nn.ReplicationPad2d), p
    SV.add_extra_input(net, [_PD_SITE], C, box=box)
    if frozen:
        c.skv_w.requires_grad_(False)
    return net


# ============================================================== TMDiff
# 자리: net.denoise_fn.conv1.conv20 -- Hyper_unet_general.py:600-611:
#   cond = to3D(PAN.repeat(1, c) - MS)  (:606-608, MS = 하네스 lms [0,1], tmdiff.py:139-144)
#   h0_0 = self.conv1(cond, ...)        (:611)   conv1 = AdaptionModulateBEST(1, 32, ...) (:533)
#   AdaptionModulateBEST.forward: h = self.conv20(h) (:169), conv20 = Conv3d(1, 32, kernel 1) (:161-162)
# 모델에 nn.Conv2d 가 없다 (tmdiff.py:217) -- 스펙트럼이 Conv3d 의 depth, 채널 축은 1 이다.  그래서 추가 입력은
# 두 번째 '채널' 이다: conv20(cat[to3D(PAN - S), to3D(PAN - L)], dim=1).  E 는 조건과 같은 꼴(PAN - MS, [-1,1])로 만든다.
_TM_SITE = "net.denoise_fn.conv1.conv20"
# tmdiff.py 의 TMDiff.sample (:146-202).  아래 forward 는 그것을 청크마다 부른다 -- 청크 규칙 · 난수 규칙이 바뀌면 멈춘다.
_TM_SAMPLE_FP = "e7503286e35a0ad1"


def _tm_hook(mod, inp, out):
    x = inp[0]                                  # (B, 1, C, H, W)
    E = mod._skv_box.E                          # (B, C, H, W)  = PAN - L
    if E is None:
        raise RuntimeError(f"skipvar: '{_TM_SITE}' 의 추가 입력이 비어 있다 -- extra(model, E) 안에서만 부른다")
    E3 = E.unsqueeze(1).to(dtype=x.dtype)       # to3D (Hyper_unet_general.py:120-122)
    if E3.shape[0] != x.shape[0] or tuple(E3.shape[2:]) != tuple(x.shape[2:]):
        raise RuntimeError(f"skipvar: '{_TM_SITE}' 입력 {tuple(x.shape)} 와 추가 입력 {tuple(E3.shape)} 의 배치/크기가 다르다")
    return out + F.conv3d(E3, mod.skv_w, None, mod.stride, mod.padding, mod.dilation)


def _tm_install(model, box=None):
    c = SV.module_at(model, _TM_SITE)
    assert isinstance(c, nn.Conv3d) and c.groups == 1 and c.in_channels == 1 and c.padding_mode == "zeros", c
    assert not hasattr(c, "skv_w")
    c.register_parameter("skv_w", nn.Parameter(torch.zeros_like(c.weight)))   # (32, 1, 1, 1, 1): 추가 채널 1 개
    c._skv_box = box or SV._Box()
    c.register_forward_hook(_tm_hook)
    model._skv_sites = list(getattr(model, "_skv_sites", [])) + [_TM_SITE]
    return model


def _tm_E(pan, L):
    return pan.repeat(1, L.shape[1], 1, 1) - L                  # Hyper_unet_general.py:606-607 와 같은 꼴


def register(SPECS, REPOS, _load, _pkg):
    # ---------------------------------------------------------------- LAGConv
    # registry.py:111-116 (LACNET_n: head_conv[0] = LAConv2D(1+C, 32, 3, 1, 1)), :119-124 forward m(pan, ms_up);
    # repos/LAGConv/model.py:121-123  x = head_conv(cat[pan, lms]);  skip :130 sr = lms + x.
    lag = SV.make_in(SPECS, "LAGConv", "LAGConv-in",
                     ["head_conv.0.attention1.0", "head_conv.0.attention3.1", "head_conv.0"], "L",
                     note="자리: model.py:121-123 head_conv[0] = LAConv2D(cat[pan, lms]) -- nn.Conv2d 가 아니다.  "
                          "LAConv2D(1+2C) 를 cat[pan, S, L] 에 건 것과 같게 입력 채널이 들어가는 세 곳 -- attention1[0] "
                          "(국소 적응 가중치, :21-28/:53), 본 weight (unfold x atw, :43-44/:56-74), attention3[1] (전역 적응 bias, "
                          ":36-41/:76-80, 풀링된 L) -- 에 L 채널(0 초기화)을 더했다 (zz_skv_yes.py 의 hook).")
    _lag_base_build = SPECS["LAGConv"]["build"]

    def _lag_build(ms_ch=4, ratio=4):
        return _lag_install(_lag_base_build(ms_ch, ratio), ms_ch)
    lag["build"] = _lag_build

    # ---------------------------------------------------------------- RAMSF
    # repos/RAMSF/models/MSIP/RAMSF.py:147-150  blk_8_32 = [Conv2d(n_bands, 64, 3, padding=1), PReLU];
    # :281  m_* = dwt_lms(DE_lms(blk_8_32(lms)));  skip :304 output = lms + sr.  registry.py:666 forward m(pan, ms_up, lrms).
    SV.make_in(SPECS, "RAMSF", "RAMSF-in", ["blk_8_32.0"], "L",
               note="자리: RAMSF.py:147-150/:281 blk_8_32[0](lms).  주의: RAMSF 본체는 원래도 저해상 LRMS 를 "
                    "spectral_encoder(ms) 로 읽는다(:298) -- sk 팔에서도 LRMS 를 본다.  in 팔이 더하는 것은 PAN 해상도의 "
                    "업샘플 LRMS 다.")

    # ---------------------------------------------------------------- BNNPan
    # repos/BNNPan/models/BiWaveDiffSNTBBFDN.py:297  Body.conv_head = Conv2d(1 + C, 64, 3, 1, 1);
    # :306  x = conv_head(cat(lms, pan));  skip :367/:394/:400 out + lms.
    # 기본 감싸개로 충분하다: train_step(bnnpan.py:94-119)의 세 단계가 모두 lms 와 같은 배치/크기로 net 을 부르고
    # (S2 는 net 을 부르지 않는다 -- hook 이 불리지 않을 뿐), 평가 forward(:136)도 한 번 부른다.
    SV.make_in(SPECS, "BNNPan", "BNNPan-in", ["net.conv_head"], "L",
               note="자리: BiWaveDiffSNTBBFDN.py:297/:306 net.conv_head(cat(lms, pan)).  DM 의 조건 인코더 "
                    "CE(cat(pan, lms), :14/:28-29 -- 64차원 prior 로 풀링된다)는 sk 와 같이 S 만 읽는다.  S2 단계"
                    "(DM 증류)에서는 net 이 그래프에 없어 skv_w 도 갱신되지 않는다 (본래 net 과 같다).")

    # ---------------------------------------------------------------- CANConv
    # repos/CANConv/canconv/models/cannet/model.py:74  head_conv = Conv2d(spectral_num + 1, 32, 3, 1, 1);
    # :87-88  x1 = head_conv(cat[pan, lms]);  skip :103 lms + x5.
    # k-means 캐시(canconv/layers/kmeans/cache.py:61-100)는 head_conv 뒤의 CANConv 층 특징(rb1.conv1 이후)만 군집한다 --
    # head_conv 를 따로 부르는 길이 없다.  on_train_start/on_epoch(canconv.py:23-33)은 캐시 상태만 바꾸고 모델을 부르지
    # 않는다.  기본 감싸개가 canconv.py:35-55 의 _forward(sample_idx 캐시 처리 포함)를 lms=S 로 부른다.
    SV.make_in(SPECS, "CANConv", "CANConv-in", ["head_conv"], "L",
               note="자리: cannet/model.py:74/:87-88 head_conv(cat[pan, lms]).  k-means 군집은 head_conv 출력 뒤의 "
                    "특징에서 하므로 군집 배정도 L 을 반영한다 (캐시 경로가 head_conv 를 따로 부르지 않는다).")

    # ---------------------------------------------------------------- PanDiT
    P = sys.modules["panbase_spec_pandit"]       # 본래 스펙 모듈 (registry 가 specs/pandit.py 를 이 이름으로 적재)
    _pd_base = SPECS["PanDiT"]

    def _pd_build(ms_ch=4, ratio=4):
        fp = _fp(P._PanDiT.sample)
        if fp != _PD_SAMPLE_FP:
            raise RuntimeError(f"PanDiT-in: pandit.py 의 _PanDiT.sample 이 바뀌었다 ({fp} != {_PD_SAMPLE_FP}) -- "
                               "zz_skv_yes.py 의 _pd_sample 을 다시 옮겨야 한다")
        m = _pd_base["build"](ms_ch, ratio)
        box = SV._Box()                         # net · ema_net · 크기별 사본이 한 상자를 나눈다
        _pd_install_site(m.net, ms_ch, box, frozen=False)
        _pd_install_site(m.ema_net, ms_ch, box, frozen=True)      # ema_net.requires_grad_(False) (pandit.py:112) 와 같게
        # ema_update(pandit.py:173) 의 zip(net.parameters(), ema_net.parameters()) 짝: 두 벌의 이름 순서가 같아야 한다
        # (skv_w 는 자리 conv 의 weight, bias 다음에 붙는다 -> 같은 자리).  EMA 가 skv_w 도 평균한다.
        assert [n for n, _ in m.net.named_parameters()] == [n for n, _ in m.ema_net.named_parameters()]
        m._skv_sites = ["net." + _PD_SITE, "ema_net." + _PD_SITE]
        # 평가 크기별 사본(pandit.py:143-158, H != 64)도 같은 자리를 단다 -- 그러면 _share 의 키 집합이 ema_net 과 같고
        # ema_net 의 skv_w 가 꽂힌다.  eval_tile=64 라 하네스 평가는 언제나 64 (= ema_net 자신) 지만, 다른 크기로 불려도
        # L 이 조용히 빠지지 않게 한다.
        orig_make = m._make

        def _make(H):                           # __init__ 은 이미 끝났다 -- 이 뒤의 _make 는 평가 사본뿐이다
            return _pd_install_site(orig_make(H), ms_ch, box, frozen=True)
        m._make = _make
        return m

    def _pd_pad(model, L):
        return SV.module_at(model.net, _PD_PAD)(L)              # 자리 conv 앞의 ReplicationPad2d 를 그대로

    @torch.no_grad()
    def _pd_sample(mm, S, L, pan, fixed_noise):
        """lib/specs/pandit.py:192-227 _PanDiT.sample 그대로 -- 다른 것은 청크마다 `SV.extra(mm, E[i:i+cs])` 하나다.
        (sample 은 B*(H/8)^2 토큰 예산으로 배치를 청크로 쪼개 디노이저를 부르므로 E 도 같은 청크로 잘라야 한다.)"""
        H = int(S.shape[-1]); dev = S.device
        assert H % P._PATCH == 0, f"H={H} must be divisible by patch_size {P._PATCH}"
        view, diff = mm._view(H, dev)
        view.eval()
        cs = max(1, P._TOK_BUDGET // max(1, (H // P._PATCH) ** 2))
        devs = list(range(torch.cuda.device_count())) if dev.type == "cuda" else []
        E = _pd_pad(mm, L)
        out = []
        for i in range(0, S.shape[0], cs):
            l, p = S[i:i + cs], pan[i:i + cs]
            diff.set_new_noise_schedule(None, dev, betas=mm._betas())
            with SV.extra(mm, E[i:i + cs]):
                if fixed_noise and not P._tiled():
                    with torch.random.fork_rng(devices=devs):
                        torch.manual_seed(P._EVAL_SEED + i)
                        res = diff(P._cond(l, p), mode="ddim_sample", section_counts=P._DDIM)
                else:
                    res = diff(P._cond(l, p), mode="ddim_sample", section_counts=P._DDIM)
            out.append((res + l).clip(0, 1))
        return torch.cat(out, 0)

    def _pd_forward(mm, ms_up, pan, lrms, ctx):
        S, L = SV.split(ms_up)
        # 본래 forward(pandit.py:306-307)와 같은 잡음 규칙: 64x64 = valid 분할 -> 고정 seed.
        return _pd_sample(mm, S, L, pan, fixed_noise=int(S.shape[-1]) == P._TRAIN_SIZE)

    def _pd_train_step(model, batch, ctx, lossfn):
        # pandit.py:179-189 train_step: GaussianDiffusion.p_losses 가 lms 와 같은 배치로 net 을 한두 번(self-cond) 부른다.
        S, L = SV.split(batch["lms"])
        with SV.extra(model, _pd_pad(model, L)):
            return model.train_step(dict(batch, lms=S), ctx)

    pd = SV.make_in(SPECS, "PanDiT", "PanDiT-in", ["net." + _PD_SITE, "ema_net." + _PD_SITE], "L",
                    forward=_pd_forward, train_step=_pd_train_step,
                    note="자리: dit.layers.0.cond_inj.ms_conv 의 첫 Conv2d (PanVIT_modules.py:410-417/:471, 앞에 "
                         "ReplicationPad2d -- E 도 같은 패딩).  디노이저의 prev_conv(PanVIT.py:77)는 잡음 잔차만 읽는다.  "
                         "L 은 [0,1] (cond[:, :C] 의 lms 와 같은 스케일; 0-255 웨이블릿 채널은 S 로 만든 그대로).  "
                         "net 과 ema_net(평가) 이 같은 자리에 같은 구조의 skv_w 를 가진다 (ema 쪽은 requires_grad=False, "
                         "EMA zip 이 함께 평균한다).  sample 은 청크마다 E 를 건다.")
    pd["build"] = _pd_build

    # ---------------------------------------------------------------- TMDiff
    _tm_base = SPECS["TMDiff"]
    _TMcls = next(c.cell_contents for c in _tm_base["build"].__closure__
                  if isinstance(c.cell_contents, type) and c.cell_contents.__name__ == "TMDiff")

    def _tm_build(ms_ch=4, ratio=4):
        fp = _fp(_TMcls.sample)
        if fp != _TM_SAMPLE_FP:
            raise RuntimeError(f"TMDiff-in: tmdiff.py 의 TMDiff.sample 이 바뀌었다 ({fp} != {_TM_SAMPLE_FP}) -- "
                               "zz_skv_yes.py 의 _tm_forward 를 다시 확인해야 한다")
        return _tm_install(_tm_base["build"](ms_ch, ratio))

    def _tm_forward(mm, ms_up, pan, lrms, ctx):
        """tmdiff.py:146-202 sample 을 **청크마다** 부른다 (청크 크기 k = eval_pix // (H*W), :175).  청크 하나 크기의 입력은
        sample 안에서 청크 하나라 E 를 그 청크로 잘라 건다.  학습 중 검증(CRN, :180-183)의 fork_rng + manual_seed(eval_seed)
        는 sample 이 호출마다 다시 뿌리므로 여기서 한 번 걸고, 안쪽 호출은 _val_crn=False 로 같은 난수열에서 차례로 뽑게 한다
        (안쪽 sampler 는 'ddpm' -- val_sampler 와 같다는 것을 확인한다).  결과는 한 번에 부른 본래 sample 과 같다."""
        S, L = SV.split(ms_up)
        E = _tm_E(pan, L)
        H, W = S.shape[-2:]
        k = max(1, mm.eval_pix // (H * W))
        crn = bool(mm._val_crn)
        if crn and mm.val_sampler != "ddpm":
            raise RuntimeError(f"TMDiff-in: val_sampler={mm.val_sampler} 는 이 청크 감싸개가 재현하지 못한다")
        devs = [S.device] if S.is_cuda else []
        outs = []
        with (torch.random.fork_rng(devices=devs) if crn else contextlib.nullcontext()):
            if crn:
                torch.manual_seed(mm.eval_seed)
            mm._val_crn = False
            try:
                for s0 in range(0, S.shape[0], k):
                    with SV.extra(mm, E[s0:s0 + k]):
                        outs.append(mm.sample(S[s0:s0 + k], pan[s0:s0 + k]))
            finally:
                mm._val_crn = crn
        return torch.cat(outs, 0)

    def _tm_train_step(model, b, ctx, lossfn):
        # tmdiff.py:254-292: denoise(x_noisy, t, pan, lms) 를 전체 배치로 한 번 부른다.
        S, L = SV.split(b["lms"])
        with SV.extra(model, _tm_E(b["pan"], L)):
            return _tm_base["train_step"](model, dict(b, lms=S), ctx, lossfn)

    tm = SV.make_in(SPECS, "TMDiff", "TMDiff-in", [_TM_SITE], "L",
                    forward=_tm_forward, train_step=_tm_train_step,
                    note="자리: Hyper_unet_general.py:533/:161-162/:169/:611 conv1.conv20 = Conv3d(1, 32, 1) 이 "
                         "to3D(PAN - MS) 를 읽는다 (nn.Conv2d 없음).  추가 입력은 to3D(PAN - L) 을 두 번째 채널로 "
                         "(skv_w (32,1,1,1,1), 0 초기화) -- 조건과 같은 꼴 · 스케일.  sample 은 청크마다 E 를 건다.  CUDA 전용.")
    tm["build"] = _tm_build
