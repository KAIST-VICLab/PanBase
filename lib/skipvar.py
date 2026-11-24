"""Global-skip 변형 연구 (2026-10-09~) 의 공통 장치 -- skip 만 바꾸는 변형(sk)과 입력에도 S 를 주는 변형(in).

팔(arm) 정의 -- 모든 방법에 같은 규칙이다:
  LRMS (본 칸)    본체 입력 {MS, PAN}, global skip = 저자 코드의 업샘플 MS.
  sk  (skip 만)   global skip = Stage-1 base S.
                  YES 방법(하네스 `lms` 를 skip 으로 더한다)은 데이터만 바뀐다: `train_pancoll --skip` 이면 모델이 받는
                  `lms` 가 S 다 (본체가 lms 를 읽는 자리도 S 를 본다).
                  INT 방법(내부 업샘플을 skip 으로 더한다)은 그 skip 을 하네스 `lms`(= S)로 바꾼 `<M>-sk` 스펙이다
                  (`make_sk`; 본체 입력은 그대로).
  in  (입력에도)  sk 위에, 본체의 첫 PAN 해상도 입력 conv 가 sk 에서 보지 못하는 것을 추가 채널로 더 받는다 (`make_in`):
                  YES 는 LRMS (sk 에서 `lms` 자리를 S 가 차지했다), INT 는 S (본체는 원래 MS · PAN 만 본다).
                  그래서 in 의 본체는 언제나 {LRMS, PAN, S} 를 보고 global skip 은 S 다.
                  추가 채널 가중치(`skv_w`)는 0 에서 시작한다 -- 같은 seed 에서 in 의 초기 함수 = sk 의 초기 함수 (짝 비교).
                  `skv_w` 는 난수를 쓰지 않으므로 본래 가중치의 초기값도 sk 와 같다.
데이터: in 런은 `lms` = cat[S, LRMS] (2C 채널; `pancoll.LMS_STACK`, 스펙의 `lms_stack=True` 를 보고 트레이너가 켠다).  한 텐서로
        싣는 까닭은 트레이너의 배치 인덱싱 · D4 증강 · 타일 평가가 두 텐서에 **같은** 변환을 걸게 하려는 것이다.  스펙의
        forward/train_step 이 `split` 으로 [S | LRMS] 를 쪼갠다.  FR 지표는 언제나 원래 lms (`lms_orig`) 로 잰다.

추가 입력을 넣는 방법: conv(cat[x, E]) = conv_x(x) + conv_E(E) (선형) 이므로 원래 conv 는 그대로 두고 (state_dict 키 · 초기 난수 ·
저장소 코드 불변) 같은 conv 에 `skv_w` 를 등록하고 forward hook 으로 conv_E(E) 를 더한다.  E 는 `extra(model, E)` 안에서만
설정된다 -- 밖에서 그 conv 가 불리면 조용히 넘어가지 않고 멈춘다 (설정 누락 · 배치/크기 불일치도 멈춘다).
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Box:
    """추가 입력 E 를 담는 상자.  nn.Module 이 아니라 등록되지 않는다 (deepcopy 는 따라간다)."""
    __slots__ = ("E",)

    def __init__(self):
        self.E = None


def module_at(model, path):
    m = model
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


def _hook(name):
    def h(mod, inp, out):
        E = mod._skv_box.E
        if E is None:
            raise RuntimeError(f"skipvar: '{name}' 의 추가 입력이 비어 있다 -- extra(model, E) 안에서만 부른다")
        x = inp[0]
        if E.shape[0] != x.shape[0] or tuple(E.shape[-2:]) != tuple(x.shape[-2:]):
            raise RuntimeError(f"skipvar: '{name}' 입력 {tuple(x.shape)} 와 추가 입력 {tuple(E.shape)} 의 배치/크기가 다르다")
        E = E.to(dtype=x.dtype)
        if mod.padding_mode != "zeros":
            y = F.conv2d(F.pad(E, mod._reversed_padding_repeated_twice, mode=mod.padding_mode), mod.skv_w, None,
                         mod.stride, 0, mod.dilation)
        else:
            y = F.conv2d(E, mod.skv_w, None, mod.stride, mod.padding, mod.dilation)
        return out + y
    return h


def add_extra_input(model, sites, extra_ch, box=None):
    """`sites` (점 경로들) 의 Conv2d 가 입력 뒤에 추가 입력 E (extra_ch 채널) 를 이어 붙인 것처럼 동작하게 한다.

    같은 모델의 여러 자리(예: EMA 사본)가 한 상자를 나누려면 `box` 를 넘긴다."""
    box = box or _Box()
    for s in sites:
        c = module_at(model, s)
        assert isinstance(c, nn.Conv2d) and c.groups == 1, (s, type(c).__name__, getattr(c, "groups", None))
        assert not hasattr(c, "skv_w"), f"{s} 에 이미 추가 입력이 있다"
        c.register_parameter("skv_w", nn.Parameter(torch.zeros(c.out_channels, extra_ch, *c.kernel_size,
                                                               dtype=c.weight.dtype, device=c.weight.device)))
        c._skv_box = box
        c.register_forward_hook(_hook(s))
    model._skv_sites = list(getattr(model, "_skv_sites", [])) + list(sites)
    return model


@contextlib.contextmanager
def extra(model, E):
    """이 블록 안에서 model 의 모든 추가 입력 자리가 E 를 본다."""
    boxes = {id(m._skv_box): m._skv_box for m in model.modules() if hasattr(m, "_skv_box")}
    if not boxes:
        raise RuntimeError("skipvar: 이 모델에는 추가 입력 자리가 없다 (add_extra_input 을 거치지 않았다)")
    prev = [(b, b.E) for b in boxes.values()]
    for b in boxes.values():
        b.E = E
    try:
        yield
    finally:
        for b, e in prev:
            b.E = e


def split(x):
    """stack 된 lms = cat[S, LRMS] -> (S, LRMS)."""
    C = x.shape[1] // 2
    assert x.shape[1] == 2 * C, ("lms_stack 런의 lms 는 2C 채널이어야 한다", tuple(x.shape))
    return x[:, :C], x[:, C:]


def make_sk(SPECS, base, name, forward, train_step=None, ref_up=None, note=""):
    """INT 방법의 sk 변형: 내부 업샘플 skip 을 하네스 `ms_up`(= S) 로 바꾼 forward (+ train_step).

    `ref_up(lrms) -> Tensor` 는 저자 코드가 skip 으로 더하던 업샘플이다 -- 검사(`skv_check.py`)가 ms_up 에 그것을 넣으면
    sk forward 가 본래 forward 와 같은지 본다.  학습에는 쓰이지 않는다."""
    b = SPECS[base]
    d = dict(b, tier="variant", forward=forward,
             skipvar=dict(kind="sk", base=base, note=note),
             diff=b["diff"] + f"  [{name}] global skip 변형 연구(2026-10-09): 내부 업샘플 skip 대신 하네스 ms_up "
                              f"(--skip 의 Stage-1 base S) 을 더한다.  나머지는 {base} 와 같다.  {note}")
    if train_step is not None:
        d["train_step"] = train_step
    elif "train_step" in b:
        raise ValueError(f"{base} 는 train_step 이 있다 -- sk 변형도 train_step 을 줘야 한다")
    if ref_up is not None:
        d["skv_ref_up"] = ref_up
    SPECS[name] = d
    return d


def make_in(SPECS, base, name, sites, extra_kind, forward=None, train_step=None, note=""):
    """in 변형: `base` (YES 는 본래 스펙, INT 는 `<M>-sk`) 의 `sites` conv 에 추가 입력 (extra_kind: "L" = LRMS, "S" = S).

    forward/train_step 을 주지 않으면 기본 감싸개를 쓴다: lms 를 [S | L] 로 쪼개 E 를 걸고 base 를 lms=S 로 부른다.
    base 의 train_step 이 lms 와 다른 배치/크기로 모델을 부르면 (증강 · 복제 · 크롭) 직접 train_step 을 줘야 한다 --
    hook 의 크기 검사가 그런 경우를 멈춰 세운다."""
    assert extra_kind in ("L", "S"), extra_kind
    b = SPECS[base]
    pick = (lambda S, L: L) if extra_kind == "L" else (lambda S, L: S)
    sites_fn = sites if callable(sites) else (lambda m, _s=tuple(sites): _s)

    def build(ms_ch=4, ratio=4):
        m = b["build"](ms_ch, ratio)
        return add_extra_input(m, sites_fn(m), ms_ch)

    def forward_in(m, ms_up, pan, lrms, ctx):
        S, L = split(ms_up)
        with extra(m, pick(S, L)):
            return b["forward"](m, S, pan, lrms, ctx)

    def train_step_in(m, batch, ctx, lossfn):
        S, L = split(batch["lms"])
        with extra(m, pick(S, L)):
            return b["train_step"](m, dict(batch, lms=S), ctx, lossfn)

    d = dict(b, tier="variant", build=build, forward=forward or forward_in, lms_stack=True,
             skipvar=dict(kind="in", base=base, extra=extra_kind,
                          sites=None if callable(sites) else list(sites), note=note),
             diff=b["diff"] + f"  [{name}] global skip 변형 연구 in 팔: {base} 위에 본체 입력 conv "
                              f"({'동적' if callable(sites) else ', '.join(sites)}) 가 "
                              f"{'LRMS' if extra_kind == 'L' else 'S'} 를 추가 채널로 더 받는다 (0 초기화).  {note}")
    if train_step is not None:
        d["train_step"] = train_step
    elif "train_step" in b:
        d["train_step"] = train_step_in
    SPECS[name] = d
    return d
