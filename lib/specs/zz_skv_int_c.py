"""Global-skip 변형 연구 (lib/skipvar.py) -- INT 방법 넷: FusionMamba · PSDNet · TS-Pan · UAPN.

넷 모두 저자 코드가 저해상 MS 를 **모델 안에서** 업샘플해 잔차 기준(global skip)으로 더한다 (하네스 `ms_up` 은 skip 이 아니다):
  FusionMamba  u2net.py:165,196-197,210   skip = PixelShuffle(ms) -- **학습되는** 업샘플러 (conv3x3 C->16C + PixelShuffle(4), :24-33)
  PSDNet       dbpn.py:795,802,822-823    bicubic(ms) 가 세 단계의 잔차 기준이다: img1(패치별 :845-846) · img2(:905) · img3(:918, 최종)
  TS-Pan       tspan.py:108-117,261,276   skip = Upsample(ms) -- **학습되는** 업샘플러 (conv3x3 C->16C + PixelShuffle(4))
  UAPN         UAPN_arch.py:35,41,45,47   ms_0 = bicubic(ms, align_corners=True);  out = conv_out(F2) + ms_0,  UE 의 MC 평균도 ms_0 위 잔차(:87)

sk 팔 `<M>-sk`: 본체 입력(내부 업샘플을 읽는 encoder 들)은 그대로 두고 잔차 기준(skip) 항만 하네스 `ms_up`(= --skip 의 S)으로 바꾼다.
  저장소 클래스 · 모듈은 건드리지 않는다 -- forward 를 저장소 줄 그대로 스펙 안에서 다시 쓰거나 (FusionMamba · TS-Pan · UAPN),
  이 스펙이 지은 모델 인스턴스의 `up` 모듈에만 hook 을 건다 (PSDNet).  base 스펙 · 진행 중 런은 영향이 없다.
  `skv_ref_up`: 결정적 업샘플(PSDNet · UAPN)은 저자와 같은 호출이다.  학습되는 업샘플러(FusionMamba · TS-Pan)는 lrms 만으로 정할 수
  없으므로 **이 프로세스에서 마지막으로 지은 `<M>-sk` 모델의 업샘플러**를 부른다 (검사 전용 -- 학습은 부르지 않는다).  skv_check 는
  base 와 sk 를 같은 seed 로 지은 직후 부르므로 그 모델의 업샘플러 = base 모델의 업샘플러다 -> sk(ms_up = u) == base 를 증명한다.
in 팔 `<M>-in`: `<M>-sk` 위에서 PAN 을 처음 읽는 PAN 해상도 conv 가 S 를 추가 채널로 더 받는다 (skv_w 0 초기화).
  넷 모두 입력은 [0,1] 그대로 모델에 들어가므로(정규화 없음) S 도 [0,1] 그대로다.
  평가 forward 가 배치를 쪼개는 TS-Pan · UAPN 은 청크마다 같은 조각의 S 를 건다.  PSDNet 의 자리는 단계마다 PAN 을 패치로 잘라
  읽으므로 forward pre-hook 이 그 패치와 같은 자리의 S 를 잘라 건다.
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import contextlib
import sys
import weakref

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, (_GB + "/lib"))
import skipvar as SV                                                          # noqa: E402

_LAST = {}          # sk 스펙 이름 -> 마지막으로 지은 모델 (weakref).  학습되는 업샘플러의 skv_ref_up 전용.


def _remember(key, build):
    def build_rec(ms_ch=4, ratio=4):
        m = build(ms_ch, ratio)
        _LAST[key] = weakref.ref(m)
        return m
    return build_rec


def _bound_ref(key, get_up):
    """학습되는 업샘플러의 skv_ref_up: 이 프로세스에서 마지막으로 지은 `key` 모델의 업샘플러로 lrms 를 올린다 (검사 전용)."""
    def ref_up(lrms):
        r = _LAST.get(key)
        m = r() if r is not None else None
        if m is None:
            raise RuntimeError(f"{key}: skip 이 학습되는 업샘플러라 skv_ref_up 은 이 프로세스에서 지은 {key} 모델이 있어야 한다")
        with torch.no_grad():
            return get_up(m)(lrms)
    return ref_up


class _CallAs:
    """base train_step 의 `model(...)` 호출 자리에 sk 함수를 끼운다 -- 손실 코드는 base 의 것을 그대로 쓴다.
    base train_step(TS-Pan `_step`, UAPN `train_step`)은 model 을 부르기만 하고 다른 속성을 읽지 않는다 (읽으면 여기서 멈춘다)."""
    __slots__ = ("_fn",)

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *a, **k):
        return self._fn(*a, **k)


def _maybe_extra(m, E):
    return SV.extra(m, E) if E is not None else contextlib.nullcontext()


# ============================================================================ PSDNet 보조
class _PsdSkip:
    """PSDNet-sk 의 `up` hook 이 읽는 상자: 모델에 들어간 lrms 객체와 S, 그리고 D = S - bicubic(lrms)."""
    __slots__ = ("ms", "S", "D")

    def __init__(self):
        self.ms = self.S = self.D = None


def _row_offset(x, ref):
    """x 가 ref 의 (H 축 · W 축) 슬라이스 뷰일 때 그 시작 (행, 열).  뷰가 아니면 멈춘다."""
    if (x.untyped_storage().data_ptr() != ref.untyped_storage().data_ptr() or x.stride() != ref.stride()
            or tuple(x.shape[:2]) != tuple(ref.shape[:2]) or ref.stride(-1) != 1):
        raise RuntimeError(f"PSDNet skipvar: 입력 {tuple(x.shape)} 가 기준 텐서 {tuple(ref.shape)} 의 H/W 슬라이스 뷰가 아니다")
    off = (x.data_ptr() - ref.data_ptr()) // x.element_size()
    r0, c0 = divmod(off, ref.stride(-2))
    if not (0 <= r0 and r0 + x.shape[-2] <= ref.shape[-2] and 0 <= c0 and c0 + x.shape[-1] <= ref.shape[-1]):
        raise RuntimeError(f"PSDNet skipvar: 뷰 위치 ({r0},{c0}) 크기 {tuple(x.shape[-2:])} 가 {tuple(ref.shape[-2:])} 밖이다")
    return r0, c0


def _psd_up_hook(mod, inp, out):
    """dbpn.py 의 `self.up` (nn.Upsample(scale_factor=4, mode='bicubic'), Net :795 / Net_GT :604) 은 forward 에서 세 번 불린다:
         :802/:611  bicubic_x   = up(ms)          -> img2(:905/:713) · img3(:918/:726) 의 잔차 기준       => S 로 바꾼다
         :822/:631  bicubic_xtop = up(ms[:, :, 0:H/2+4])   -> img1top(:845/:653) 의 잔차 기준  => up(patch) + D[해당 PAN 행]
         :823/:632  bicubic_xbot = up(ms[:, :, H/2-4:H])   -> img1bot(:846/:654) 의 잔차 기준  => up(patch) + D[해당 PAN 행]
       D = S - up(ms) 다.  패치 기준은 저자의 패치별 bicubic(패치 경계에서 복제 패딩) 에 같은 보정장 D 를 그 행만큼 더한 것 --
       = crop(S) + (저자의 패치 경계 효과).  S = up(ms) 이면 D = 0 이라 세 기준 모두 저자 것과 비트 단위로 같다."""
    b = mod._skv_psd
    if b.S is None:
        raise RuntimeError("PSDNet-sk: `up` 이 skip(S) 설정 없이 불렸다 -- _psd_skip(model, lrms, S) 안에서만 부른다")
    x, ms = inp[0], b.ms
    if x is ms or (x.data_ptr() == ms.data_ptr() and x.shape == ms.shape and x.stride() == ms.stride()):
        return b.S                                                           # :802/:611 bicubic_x -> S
    r0, c0 = _row_offset(x, ms)
    if c0 != 0 or x.shape[-1] != ms.shape[-1]:
        raise RuntimeError(f"PSDNet-sk: `up` 입력이 lrms 의 행 띠가 아니다 (열 {c0}, 폭 {x.shape[-1]})")
    if b.D is None:
        b.D = b.S - F.interpolate(ms, mod.size, mod.scale_factor, mod.mode, mod.align_corners,
                                  recompute_scale_factor=mod.recompute_scale_factor)   # = nn.Upsample.forward(ms)
    s = out.shape[-2] // x.shape[-2]
    return out + b.D[:, :, s * r0: s * (r0 + x.shape[-2])]                 # :822-823/:631-632 패치 기준 + D[행]


@contextlib.contextmanager
def _psd_skip(model, ms, S):
    boxes = {id(m._skv_psd): m._skv_psd for m in model.modules() if hasattr(m, "_skv_psd")}
    if not boxes:
        raise RuntimeError("PSDNet-sk: 이 모델에는 skip hook 이 없다 (PSDNet-sk build 를 거치지 않았다)")
    B, C, h, w = ms.shape
    if tuple(S.shape) != (B, C, 4 * h, 4 * w):
        raise RuntimeError(f"PSDNet-sk: S {tuple(S.shape)} 가 lrms {tuple(ms.shape)} 의 x4 가 아니다")
    prev = [(b, b.ms, b.S, b.D) for b in boxes.values()]
    for b in boxes.values():
        b.ms, b.S, b.D = ms, S, None
    try:
        yield
    finally:
        for b, a, s, d in prev:
            b.ms, b.S, b.D = a, s, d


class _PsdPan:
    """PSDNet-in 의 pre-hook 이 읽는 상자: 모델에 들어간 pan 객체와 S, 그리고 그 자리 conv 들."""
    __slots__ = ("pan", "S", "convs")

    def __init__(self):
        self.pan = self.S = None
        self.convs = []


def _psd_in_pre(mod, inp):
    """자리 conv 가 받는 PAN 패치(pan 의 뷰)와 같은 위치의 S 를 잘라 추가 입력으로 건다 (SV 의 hook 이 크기를 다시 검사한다)."""
    b = mod._skv_pan
    if b.pan is None:
        raise RuntimeError("PSDNet-in: 추가 입력 자리가 S 설정 없이 불렸다 -- _psd_pan_extra(model, pan, S) 안에서만 부른다")
    x = inp[0]
    r0, c0 = (0, 0) if x is b.pan else _row_offset(x, b.pan)
    mod._skv_box.E = b.S[:, :, r0: r0 + x.shape[-2], c0: c0 + x.shape[-1]]


@contextlib.contextmanager
def _psd_pan_extra(model, pan, S):
    boxes = {id(m._skv_pan): m._skv_pan for m in model.modules() if hasattr(m, "_skv_pan")}
    if not boxes:
        raise RuntimeError("PSDNet-in: 이 모델에는 추가 입력 pre-hook 이 없다 (PSDNet-in build 를 거치지 않았다)")
    if S.shape[0] != pan.shape[0] or tuple(S.shape[-2:]) != tuple(pan.shape[-2:]):
        raise RuntimeError(f"PSDNet-in: S {tuple(S.shape)} 와 pan {tuple(pan.shape)} 의 배치/크기가 다르다")
    for b in boxes.values():
        b.pan, b.S = pan, S
    try:
        yield
    finally:
        for b in boxes.values():
            b.pan = b.S = None
            for c in b.convs:
                c._skv_box.E = None


def register(SPECS, REPOS, _load, _pkg):

    # ======================================================================== FusionMamba
    # 본래: specs/fusionmamba.py forward = m(lrms, pan) -> Wrap.forward (H/W 비례 재설정) -> U2Net.forward (u2net.py:194-211)
    #   :165  self.upsample = PixelShuffle(ms_dim, 4)    <- conv3x3(C->16C, bias 없음) + nn.PixelShuffle(4) (:24-33) -- **학습된다**
    #   :196  ms = self.upsample(ms);  :197 skip = ms
    #   :198  pan = self.raise_pan_dim(pan)               <- in 자리
    #   :199  ms = self.raise_ms_dim(ms)                  <- 본체는 학습 업샘플을 읽는다 (sk 에서도 그대로)
    #   :202-206 stage0..stage4,  :208 spe_attn(lrms)
    #   :210  output = self.to_hrms(output) * spe_attn + skip   <- 이 `+ skip` 만 `+ ms_up` 으로 바꾼다
    # skip 이 lrms 의 결정적 함수가 아니다 (학습되는 PixelShuffle 업샘플러).  가장 충실한 sk: 맨 끝 덧셈의 skip 만 ms_up 으로,
    # 업샘플러는 본체 입력으로 남는다 (그 기울기는 본체 경로로만 온다).
    def _fm_resize(m, pan):
        """specs/fusionmamba.py Wrap.forward 와 같다: input_h/w 를 PAN 크기에 비례해 다시 쓴다 (BASE = 64)."""
        f = pan.shape[-2] / 64
        for mod in m.net.modules():
            if hasattr(mod, "_base_h"):
                mod.input_h = max(1, int(round(mod._base_h * f)))
                mod.input_w = max(1, int(round(mod._base_w * f)))

    def fm_forward_sk(m, ms_up, pan, lrms, ctx):
        _fm_resize(m, pan)
        n = m.net
        ms = n.upsample(lrms)                                                # u2net.py:196 (본체 입력, 그대로)
        p = n.raise_pan_dim(pan)                                             # :198
        q = n.raise_ms_dim(ms)                                               # :199
        p, q, ps0, qs0 = n.stage0(p, q)                                      # :202
        p, q, ps1, qs1 = n.stage1(p, q)                                      # :203
        p, q = n.stage2(p, q, ps1, qs1)                                      # :204
        p, q = n.stage3(p, q, ps0, qs0)                                      # :205
        o = n.stage4(p, q)                                                   # :206
        a = n.spe_attn(lrms)                                                 # :208
        return n.to_hrms(o) * a + ms_up                                      # :210, skip -> ms_up

    d = SV.make_sk(SPECS, "FusionMamba", "FusionMamba-sk", fm_forward_sk,
                   ref_up=_bound_ref("FusionMamba-sk", lambda m: m.net.upsample),
                   note="skip: u2net.py:210 `to_hrms(..) * spe_attn + skip` 의 skip(= :196-197 학습되는 PixelShuffle 업샘플, :24-33) "
                        "-> ms_up.  본체 raise_ms_dim(:199)은 여전히 학습 업샘플을 읽는다.  skv_ref_up 은 마지막으로 지은 sk 모델의 "
                        "업샘플러다 (검사 전용).  Mamba CUDA 커널이라 CPU 학습 불가.")
    d["build"] = _remember("FusionMamba-sk", SPECS["FusionMamba"]["build"])
    # in 자리: U2Net.raise_pan_dim[0] = Conv2d(1, 32, 3, 1, 1) (u2net.py:166-169), :198 `pan = self.raise_pan_dim(pan)` --
    # 본체에서 PAN 을 처음 읽는 PAN 해상도 conv.  forward 는 배치를 쪼개지 않으므로 기본 make_in 감싸개가 맞다 (train_step 없음).
    SV.make_in(SPECS, "FusionMamba-sk", "FusionMamba-in", ["net.raise_pan_dim.0"], "S",
               note="자리: u2net.py:166-169/198 raise_pan_dim[0](pan) -> raise_pan_dim[0](cat[pan, S]).  입력 [0,1] 그대로.")

    # ======================================================================== PSDNet
    # 본래: specs/psdnet.py forward = mm(lrms, pan) -> PSDNet.student(lrms, pan)[2];  train_step = 교사(Net_GT, 단계 A) /
    #       학생(Net)+증류(단계 B).  Networks/dbpn.py:
    #   Net    :795 self.up = nn.Upsample(scale_factor=4, mode='bicubic')
    #          :802 bicubic_x = up(ms)  -> :905 img2 = x3 + bicubic_x,  :918 img3 = x3 + bicubic_x (최종 예측, global skip)
    #          :822-823 bicubic_xtop/xbot = up(ms 위/아래 띠)  -> :845-846 img1top/bot = x + bicubic_xtop/xbot (단계 1 패치)
    #          단계 2 는 img1top/bot 을(:861-862), 단계 3 은 img2 를(:908) 본체 입력으로 읽는다 (점진 구조).
    #   Net_GT :604 self.up,  :611 bicubic_x -> :713 img2 · :726 img3,  :631-632 -> :653-654 img1top/bot  (같은 구조, 단계 2 는 GT)
    # sk: 이 세 잔차 기준을 모두 S 쪽으로 옮긴다 (`_psd_up_hook`: 전체 = S, 패치 = 저자 패치 bicubic + (S - bicubic(lrms)) 의 해당 행).
    #     교사(Net_GT)도 같다 -- 학생의 쌍둥이(같은 잔차 설계)이고 단계 3 특징(GT_feat2, 증류 대상)이 img2 기준을 거친다.
    #     본체 입력(body1_1 의 MS 패치, body*_2 의 PAN)은 그대로다.  hook 은 이 스펙이 지은 인스턴스의 `up` 모듈에만 걸린다.
    _psd_fwd, _psd_ts = SPECS["PSDNet"]["forward"], SPECS["PSDNet"]["train_step"]

    def psd_build_sk(ms_ch=4, ratio=4):
        m = SPECS["PSDNet"]["build"](ms_ch, ratio)
        box = _PsdSkip()
        for net in (m.student, m.teacher):
            net.up._skv_psd = box
            net.up.register_forward_hook(_psd_up_hook)
        return m

    def psd_forward_sk(mm, ms_up, pan, lrms, ctx):
        with _psd_skip(mm, lrms, ms_up):
            return _psd_fwd(mm, ms_up, pan, lrms, ctx)

    def psd_train_step_sk(model, b, ctx, lossfn):
        # base train_step 그대로 (단계 A 교사 L1 3항 / 단계 B 학생 L1 3항 + 1e-6 증류).  교사 · 학생 모두 lrms = b["ms"] 를 받는다.
        with _psd_skip(model, b["ms"], b["lms"]):
            return _psd_ts(model, b, ctx, lossfn)

    def psd_ref_up(lrms):
        return F.interpolate(lrms, None, 4.0, "bicubic", None, recompute_scale_factor=None)   # = dbpn.py:795 nn.Upsample.forward

    d = SV.make_sk(SPECS, "PSDNet", "PSDNet-sk", psd_forward_sk, psd_train_step_sk, ref_up=psd_ref_up,
                   note="skip: dbpn.py:802->:905/:918 bicubic_x -> S, :822-823->:845-846 패치 기준 -> 저자 패치 bicubic + (S - bicubic)[행] "
                        "(교사 Net_GT :611/:631-632 도 같다).  `up` 인스턴스 hook.  S = bicubic 이면 저자와 비트 단위로 같다.")
    d["build"] = psd_build_sk

    # in 자리: Net.body1_2[0] = conv(1, 28, 3) (dbpn.py:746-751, base_networks.py:429-432 nn.Conv2d(pad 1)), :829-832 에서 단계 1
    # PAN 패치를 처음 읽는다.  body1_2 · body2_2 · body3_2 는 **같은 모듈 목록**으로 지어져(:751-753) 이 conv 를 공유한다 -> S 는
    # 단계 1(네 패치 :829-832) · 단계 2(두 띠 :863-864) · 단계 3(전체 :909) 의 PAN 갈래 모두에 그 패치와 같은 자리로 들어간다.
    # 자리 conv 가 PAN 의 **뷰**(패치)를 받으므로 기본 감싸개(E = 전체 S)는 크기가 맞지 않는다 -> pre-hook 이 뷰의 위치로 S 를 자른다.
    # 교사(Net_GT)에는 넣지 않는다: 추론 네트워크는 학생이고, 교사는 sk 와 같게 두어 단계 A(교사만 학습, 에폭 1-9)가 sk 와 같은
    # 계산이 된다 (skv_w 는 난수를 쓰지 않는다).  증류 계수는 1e-6 이다 (optimize1.py:164).
    PSD_IN_SITES = ["student.body1_2.0"]

    def psd_forward_in(mm, ms_up, pan, lrms, ctx):
        S, _ = SV.split(ms_up)
        pan = pan.contiguous()
        with _psd_pan_extra(mm, pan, S):
            return psd_forward_sk(mm, S, pan, lrms, ctx)

    def psd_train_step_in(model, b, ctx, lossfn):
        S, _ = SV.split(b["lms"])
        pan = b["pan"].contiguous()
        with _psd_pan_extra(model, pan, S):
            return psd_train_step_sk(model, dict(b, lms=S, pan=pan), ctx, lossfn)

    d = SV.make_in(SPECS, "PSDNet-sk", "PSDNet-in", PSD_IN_SITES, "S", forward=psd_forward_in, train_step=psd_train_step_in,
                   note="자리: dbpn.py:747/751 body1_2[0](pan 패치) -- body2_2 · body3_2 와 공유(:751-753)라 세 단계의 PAN 갈래 "
                        "모두가 같은 자리의 S 패치를 더 받는다 (pre-hook 이 자른다).  학생만; 교사는 sk 와 같다.  입력 [0,1] 그대로.")
    _psd_in_build = d["build"]

    def psd_build_in(ms_ch=4, ratio=4):
        m = _psd_in_build(ms_ch, ratio)
        box = _PsdPan()
        for s in PSD_IN_SITES:
            c = SV.module_at(m, s)
            c._skv_pan = box
            box.convs.append(c)
            c.register_forward_pre_hook(_psd_in_pre)
        return m

    d["build"] = psd_build_in
    # 보고용 학생 파라미터 수 (specs/psdnet.py REPORT_PARAMS) + 추가 채널 skv_w (28 x C x 3 x 3)
    d["report_params"] = {c: n + 28 * c * 9 for c, n in SPECS["PSDNet"]["report_params"].items()}

    # ======================================================================== TS-Pan
    # 본래: specs/ts_pan.py forward `_forward` (픽셀 예산 청크, 청크마다 [0].clamp(0,1)), train_step `_step`
    #       -> Net.forward(ms, pan) (tspan.py:258-276)
    #   :235  self.up_sample = Upsample(ms_dim)      <- conv3x3(C->16C, bias 없음) + PixelShuffle(4) (:108-117) -- **학습된다**
    #   :261  ms_up = self.up_sample(ms)
    #   :263  pan_pre_exp = self.pan_pre_exp(pan)    <- in 자리
    #   :264  ms_pre_exp = self.ms_pre_exp(ms_up)    <- 본체는 학습 업샘플을 읽는다 (sk 에서도 그대로)
    #   :266-267 spa/spe 전문가,  :270 process(spa, spe, pan, ms_up)  <- 본체 (HFG_S2EN.py:126-129 고주파 안내)
    #   :276  return output_list[-1] + ms_up, spa_cofk, spe_cofk   <- 이 `+ ms_up` 만 하네스 ms_up(= S)으로 바꾼다
    # skip 이 학습되는 업샘플러다 (FusionMamba 와 같은 처리).  평가의 clamp(0,1)(ts_pan.py:163-165, 저장소 testsolver_1.py:66)는 skip 뒤에 걸린다.
    _tsp_ts = SPECS["TS-Pan"]["train_step"]
    _TSP_EVAL_PIX = 4 * 256 * 256                                            # ts_pan.py:142 `_EVAL_PIX`

    def _tsp_net_sk(n, ms, pan, skip):
        u = n.up_sample(ms)                                                  # tspan.py:261 (본체 입력, 그대로)
        p = n.pan_pre_exp(pan)                                               # :263
        q = n.ms_pre_exp(u)                                                  # :264
        spa, spa_cofk, _ = n.spa_exp(n.merge(torch.cat([p, q], dim=1)))      # :266
        spe, spe_cofk, _ = n.spe_exp(n.merge(torch.cat([p, q], dim=1)))      # :267
        process = n.process(spa, spe, pan, u)                                # :270
        output = n.post(process)                                             # :272
        return output + skip, spa_cofk, spe_cofk                             # :276, skip -> ms_up

    def tsp_forward_sk(mm, ms_up, pan, lrms, ctx, E=None):
        # ts_pan.py:160-165 와 같은 청크 (픽셀 예산) · 같은 clamp.  E 는 in 팔의 추가 입력 (청크마다 같은 조각).
        n, H, W = lrms.shape[0], pan.shape[-2], pan.shape[-1]
        c = max(1, _TSP_EVAL_PIX // max(1, H * W))
        if n <= c:
            with _maybe_extra(mm, E):
                return _tsp_net_sk(mm, lrms, pan, ms_up)[0].clamp(0, 1)
        outs = []
        for i in range(0, n, c):
            with _maybe_extra(mm, None if E is None else E[i:i + c]):
                outs.append(_tsp_net_sk(mm, lrms[i:i + c], pan[i:i + c], ms_up[i:i + c])[0].clamp(0, 1))
        return torch.cat(outs, dim=0)

    def tsp_train_step_sk(model, b, ctx, lossfn):
        # base `_step` 그대로 (L1 + 0.3*(CV(spa)+CV(spe)), solver.py:85-90) -- model(ms, pan) 자리만 sk forward.
        return _tsp_ts(_CallAs(lambda ms, pan: _tsp_net_sk(model, ms, pan, b["lms"])), b, ctx, lossfn)

    d = SV.make_sk(SPECS, "TS-Pan", "TS-Pan-sk", tsp_forward_sk, tsp_train_step_sk,
                   ref_up=_bound_ref("TS-Pan-sk", lambda m: m.up_sample),
                   note="skip: tspan.py:276 `output + ms_up` 의 ms_up(= :261 학습되는 PixelShuffle 업샘플, :108-117) -> 하네스 ms_up.  "
                        "본체(ms_pre_exp :264, process :270)는 여전히 학습 업샘플을 읽는다.  평가 clamp(0,1) 은 skip 뒤.  "
                        "skv_ref_up 은 마지막으로 지은 sk 모델의 업샘플러다 (검사 전용).  WKV CUDA 확장이라 CPU 학습 불가.")
    d["build"] = _remember("TS-Pan-sk", SPECS["TS-Pan"]["build"])

    # in 자리: Net.pan_pre_exp[0] = Conv2d(1, 32, 3, 1, 1, bias=False) (tspan.py:237-238), :263 -- 본체에서 PAN 을 처음 읽는 conv.
    # 평가 forward 가 배치를 쪼개므로 청크마다 S 조각을 건다.  학습(train_step)은 배치 통째 한 번이라 기본 감싸개가 맞다.
    def tsp_forward_in(mm, ms_up, pan, lrms, ctx):
        S, _ = SV.split(ms_up)
        return tsp_forward_sk(mm, S, pan, lrms, ctx, E=S)

    SV.make_in(SPECS, "TS-Pan-sk", "TS-Pan-in", ["pan_pre_exp.0"], "S", forward=tsp_forward_in,
               note="자리: tspan.py:237-238/263 pan_pre_exp[0](pan) -> pan_pre_exp[0](cat[pan, S]).  평가 청크마다 S 조각.  입력 [0,1] 그대로.")

    # ======================================================================== UAPN
    # 본래: specs/uapn.py forward (np 시드 고정 · 메모리 청크, 4번째 반환값), train_step 7항 -> UAPN.forward(pan, lms=저해상) (UAPN_arch.py:34-48)
    #   :35  ms_0 = F.interpolate(lms, scale_factor=4, mode='bicubic', align_corners=True)
    #   :36  Fin = self.conv_ms(ms_0)                 <- 본체 MS 입력 (sk 에서도 저자 bicubic 그대로)
    #   :37  panf = self.conv_pan(pan)                <- in 자리
    #   :41  UAFM1(F1, ms_0, panf) -> UE(F_1, ms_0) (:62) -> x_cur = out(mask(x)) + ms_0 (:87): MC 표본/평균(hrms1[0], 감독됨)의 잔차 기준
    #   :45  UAFM2(F2, ms_0, panf)                    (같다, hrms1[1])
    #   :47  out = self.conv_out(F2) + ms_0           <- 최종 global skip
    # sk: 잔차 기준 셋(:41 · :45 의 UE 기준, :47)을 ms_up 으로.  EU = var(표본) 는 기준 이동에 불변이라(:92) UAConv 변조는 수학적으로 같다.
    _uapn_ts = SPECS["UAPN"]["train_step"]
    _UAPN_SEED = sys.modules["panbase_spec_uapn"]._EVAL_SEED                 # uapn.py:13

    def _uapn_net_sk(n, pan, lms, skip):
        ms_0 = F.interpolate(lms, scale_factor=4, mode="bicubic", align_corners=True)   # UAPN_arch.py:35 (본체 입력)
        Fin = n.conv_ms(ms_0)                                                # :36
        panf = n.conv_pan(pan)                                               # :37
        F1 = n.FEM1(Fin)                                                     # :39
        F1 = n.skip1(torch.cat([Fin, F1], 1), F1)                            # :40
        AU1, EU1, mean1, F1 = n.UAFM1(F1, skip, panf)                        # :41, UE 잔차 기준 ms_0 -> skip
        F2 = n.FEM2(F1)                                                      # :43
        F2 = n.skip2(torch.cat([F1, F2], 1), F2)                             # :44
        AU2, EU2, mean2, F2 = n.UAFM2(F2, skip, panf)                        # :45, UE 잔차 기준 ms_0 -> skip
        out = n.conv_out(F2) + skip                                          # :47, skip -> ms_up
        return [AU1, AU2], [EU1, EU2], [mean1, mean2], out

    def uapn_forward_sk(m, ms_up, pan, lrms, ctx, E=None):
        # uapn.py:101-111 와 같은 청크 · 청크마다 같은 NumPy 시드 · 전역 상태 복원.  E 는 in 팔의 추가 입력 (청크마다 같은 조각).
        B, _, H, W = pan.shape
        per = max(1, min(B, int(2.0e9 // (H * W * 4320))))
        state = np.random.get_state()
        try:
            outs = []
            for i in range(0, B, per):
                np.random.seed(_UAPN_SEED)
                with _maybe_extra(m, None if E is None else E[i:i + per]):
                    outs.append(_uapn_net_sk(m, pan[i:i + per], lrms[i:i + per], ms_up[i:i + per])[3])
        finally:
            np.random.set_state(state)
        return torch.cat(outs, 0)

    def uapn_train_step_sk(model, b, ctx, lossfn):
        # base train_step 그대로 (process.py:102-114 의 7항) -- model(pan=, lms=) 자리만 sk forward.
        return _uapn_ts(_CallAs(lambda pan, lms: _uapn_net_sk(model, pan, lms, b["lms"])), b, ctx, lossfn)

    def uapn_ref_up(lrms):
        return F.interpolate(lrms, scale_factor=4, mode="bicubic", align_corners=True)  # UAPN_arch.py:35

    SV.make_sk(SPECS, "UAPN", "UAPN-sk", uapn_forward_sk, uapn_train_step_sk, ref_up=uapn_ref_up,
               note="skip: UAPN_arch.py:47 `conv_out(F2) + ms_0` 와 :41/:45 UAFM->UE(:87) 의 MC 잔차 기준 ms_0(= :35 bicubic "
                    "align_corners=True) -> ms_up.  본체 conv_ms(:36)는 여전히 ms_0 를 읽는다.  MC 드롭아웃 난수열은 base 와 같다.")
    # in 자리: UAPN.conv_pan = Conv2d(1, 12, 1, 1, 0) (UAPN_arch.py:19), :37 `panf = self.conv_pan(pan)` -- 본체에서 PAN 을 처음 읽는
    # PAN 해상도 conv (먼저 불리는 conv_ms(:36)는 MS 만 읽는다).  panf 는 두 UAFM 의 UAConv 입력 cat[F_1, panf](:63)으로 간다.
    # 평가 forward 가 배치를 쪼개므로 청크마다 S 조각을 건다.  학습은 배치 통째 한 번이라 기본 감싸개가 맞다.
    def uapn_forward_in(m, ms_up, pan, lrms, ctx):
        S, _ = SV.split(ms_up)
        return uapn_forward_sk(m, S, pan, lrms, ctx, E=S)

    SV.make_in(SPECS, "UAPN-sk", "UAPN-in", ["conv_pan"], "S", forward=uapn_forward_in,
               note="자리: UAPN_arch.py:19/37 conv_pan(pan) -> conv_pan(cat[pan, S]) (1x1).  평가 청크마다 S 조각.  입력 [0,1] 그대로.")
