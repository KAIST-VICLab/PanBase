"""Global-skip 변형 연구 (lib/skipvar.py) -- 먼저 학습할 세 방법: PINO-in · U-Know-DiffPAN-in (YES) · PAN-Crafter-in (INT).

YES 방법(PINO, U-Know)은 sk 팔이 데이터만 바꾼 본래 스펙이고(`--skip`, lms = S), in 팔은 본체가 lms 를 처음 읽는 PAN 해상도
conv 가 LRMS 를 추가 채널로 더 받는다.  INT 방법(PAN-Crafter)은 sk 팔 `PAN-Crafter-sk` 가 lib/specs/pancrafter.py 끝에 이미
있고, in 팔은 그 위에서 PAN 을 처음 읽는 PAN 해상도 conv 가 S 를 추가 채널로 더 받는다.

세 방법 모두 기본 `make_in` 감싸개를 그대로 쓸 수 없다 (그래서 forward/train_step 을 직접 준다):
  PINO       모델이 [-1,1] (`_norm`) 공간에서 돌고, 평가가 EVAL_BSUB 장씩 `gen_feat` 을 부른다.
  U-Know     lms 를 읽는 망이 다섯(prior · teacher · teacher EMA · student · student EMA)이고, stage 2 가 export 부분 배치
             (lms[js]) 와 항목별 H/V 뒤집기로 lms 를 바꾼 뒤 망을 부른다.
  PAN-Crafter 입력이 [-1,1] (`_n11`) 이고 train_step 이 저자 feeder 증강 + 배치 복제(repeat 2) 뒤에 모델을 부른다.
본래 스펙(PINO · U-Know-DiffPAN · PAN-Crafter · PAN-Crafter-sk)의 함수는 하나도 바꾸지 않는다 -- 필요한 것은 그 함수들의
클로저 · 모듈 전역에서 **같은 객체**를 꺼내 쓴다 (정규화를 옮겨 적다 어긋나는 일이 없게).
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import contextlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, (_GB + "/lib"))
import skipvar as SV                                                          # noqa: E402


def _cell(fn, name):
    """스펙 함수의 클로저 변수 하나를 꺼낸다 (torch.no_grad 등으로 감싼 함수는 __wrapped__ 를 따라간다)."""
    fn = getattr(fn, "__wrapped__", fn)
    co = fn.__code__
    if name not in co.co_freevars:
        raise KeyError(f"{fn.__qualname__} 의 클로저에 {name} 가 없다 ({co.co_freevars})")
    return fn.__closure__[co.co_freevars.index(name)].cell_contents


# ====================================================================================================== PINO
def _register_pino(SPECS):
    B = SPECS["PINO"]
    base_fwd, base_ts = B["forward"], B["train_step"]
    _predict = _cell(base_fwd, "_predict")
    # lib/specs/pino.py:134-137 `_norm(m, x) = (x * m.pino_in_scale - 0.5) / 0.5` -- 학습 · 평가가 쓰는 바로 그 함수.
    _norm = _cell(base_ts, "_norm")
    assert _norm is _cell(_predict, "_norm")
    # lib/specs/pino.py:50,286-289  평가는 EVAL_BSUB 장씩 gen_feat 을 부른다 -- 같은 조각으로 나눠 부르면 base 와 같은 계산이다.
    bsub = int(_cell(_predict, "EVAL_BSUB"))
    assert bsub >= 1

    def forward_in(m, ms_up, pan, lrms, ctx):
        """ms_up = cat[S, L].  base forward(lms = S) 를 EVAL_BSUB 조각마다 부르고, 그 조각의 _norm(L) 을 추가 입력으로 건다
        (_predict 는 조각마다 `gen_feat(pan_n[b0:b1], lms_n[b0:b1], ms_n[b0:b1])` -- pino.py:289)."""
        S, L = SV.split(ms_up)
        Ln = _norm(m, L)
        outs = []
        for b0 in range(0, S.shape[0], bsub):
            b1 = min(S.shape[0], b0 + bsub)
            with SV.extra(m, Ln[b0:b1]):
                outs.append(base_fwd(m, S[b0:b1], pan[b0:b1], lrms[b0:b1], ctx))
        return torch.cat(outs, 0)

    def train_step_in(m, batch, ctx, lossfn):
        """base train_step 은 모델을 배치 전체로 한 번 부른다 (pino.py:260 `model(pan_n, lms_n, ms_n, res, crd, bc)`).
        D4 증강은 학습기가 stack 된 lms 에 이미 걸었다 (spec aug='d4') -- S 와 L 은 같은 변환을 받았다."""
        S, L = SV.split(batch["lms"])
        with SV.extra(m, _norm(m, L)):
            return base_ts(m, dict(batch, lms=S), ctx, lossfn)

    # 자리: encoder.t.direction1.conv -- PAN 해상도에서 lms 를 처음 읽는 conv.
    #   liif.py:86            encoder(hr_msi=pan_n, up_lr_hsi=lms_n, lr_hsi=ms_n)
    #   bidirection.py:660    xt = cat(lms, pan)            (BidinetOri.forward, 'bdt-ori')
    #   bidirection.py:578    Merge: D1 = direction1(xt)    (direction2 는 MS 해상도 ms 만 읽는다)
    #   bidirection.py:471,491 Direction1.conv(xt)  Conv2d(C+1 -> 64, 3x3, groups 1)
    # lms 가 들어가는 나머지 두 곳은 sk 와 같이 S 다: 인코더 내부 skip `result = w_out + self.lms` (bidirection.py:670) 과
    # LIIF 잔차 `ret += res_value` (liif.py:210; res_value 는 _norm(S) 에서 모은다 -- pino.py:250-253, 282/293).
    SV.make_in(SPECS, "PINO", "PINO-in", ["encoder.t.direction1.conv"], "L",
               forward=forward_in, train_step=train_step_in,
               note="자리: bidirection.py:471/491 Direction1.conv(xt = cat[lms, pan], :660) -- liif.py:86 encoder 의 첫 PAN 해상도 "
                    "conv.  추가 LRMS 는 base 와 같은 _norm([0,1]->[-1,1], pino_in_scale 포함)을 거친다.  평가는 base 의 "
                    "EVAL_BSUB 조각마다 그 조각의 LRMS 를 건다.  인코더 내부 skip(bidirection.py:670)과 LIIF 잔차(liif.py:210)는 "
                    "sk 와 같이 S.  1000x1000 입력(bidirection.py:661-667 의 pad 분기)은 hook 크기 검사가 멈춘다 -- 하네스는 "
                    "64/256/512 만 쓴다.")


# ============================================================================================ U-Know-DiffPAN
#: lms 를 PAN 해상도에서 읽는 모든 망의 첫 conv.  EMA 사본은 student/teacher 와 **같은 자리**에 skv_w 를 받아
#: `UKnow._ema` (lib/specs/uknow_diffpan.py:188-197) 의 parameters() zip 이 한 칸도 어긋나지 않는다.
#:   prior.conv1                  pannet_variance.py:49,70-71   conv1(cat[lms, pan])           (stage 0 학습, stage 1 · export 조건)
#:   teacher_diff.model.downs.0   FSA_T.py:76,167-168,184       downs[0](cat[self_cond, lms, pan, x_t])
#:   teacher_ema.downs.0          (teacher 의 deepcopy, uknow_diffpan.py:156; export 샘플러 _samplers[0] 의 디노이저)
#:   student_diff.model.downs.0   FSA_S.py:61,151-152,163       downs[0](cat[self_cond, lms, pan, x_t])
#:   student_ema.downs.0          (student 의 deepcopy, uknow_diffpan.py:157; 평가 샘플러 _samplers[1] 의 디노이저)
_UK_SITES = ["prior.conv1", "teacher_diff.model.downs.0", "teacher_ema.downs.0",
             "student_diff.model.downs.0", "student_ema.downs.0"]
_UK_EMA_SITES = ["teacher_ema.downs.0", "student_ema.downs.0"]


@contextlib.contextmanager
def _uk_stage2(m, S, L, idx):
    """stage 2 (student_step, uknow_diffpan.py:383-422) 동안만 이 모델 인스턴스의 `_export` · `_flip` 을 감싼다.

    base student_step 은 lms 를 두 번 바꾼 뒤 망을 부른다:
      (a) `_teacher_targets` -> `_export(lms[js], pan[js], ids)` (uknow_diffpan.py:319): 처음 보는 표본만 골라 prior 와
          EMA teacher 를 부른다 -> 추가 입력은 L[js].  js 는 base 와 같은 규칙(그 표본 인덱스가 배치에 처음 나온 자리,
          :313-318)으로 ids 에서 되찾고, `lms == S[js]` 를 확인한다.
      (b) `_flip(hf, vf, gt, lms, pan, ...)` (:406-407): 항목별 H/V 뒤집기 -> 같은 hf/vf 로 L 도 뒤집어 student 호출
          (:412) 의 추가 입력으로 건다.
    그 밖의 망 호출이 생기면 추가 입력이 비어 있어 hook 이 멈춘다 (조용히 틀린 E 를 쓰지 않는다).
    클래스나 모듈은 건드리지 않는다 -- 인스턴스 __dict__ 에 잠시 얹었다가 끝나면 지운다."""
    orig_export, orig_flip = m._export, m._flip
    first = {}
    if idx is not None:
        for j, i in enumerate(idx.detach().to("cpu", torch.long).reshape(-1).tolist()):
            first.setdefault(int(i), j)
    stack = contextlib.ExitStack()

    def export(lms, pan, ids):
        js = torch.tensor([first[int(i)] for i in ids.tolist()], dtype=torch.long, device=L.device)
        if not torch.equal(lms, S[js]):
            raise RuntimeError("U-Know-DiffPAN-in: export 부분 배치가 S[js] 와 다르다 -- 추가 입력 L[js] 와 어긋난다")
        with SV.extra(m, L[js]):
            return orig_export(lms, pan, ids)

    def flip(hf, vf, *ts):
        *out, Lf = orig_flip(hf, vf, *ts, L)
        stack.enter_context(SV.extra(m, Lf))
        return out

    m.__dict__["_export"] = export
    m.__dict__["_flip"] = flip
    try:
        with stack:
            yield
    finally:
        m.__dict__.pop("_export", None)
        m.__dict__.pop("_flip", None)


def _register_uknow(SPECS):
    B = SPECS["U-Know-DiffPAN"]
    base_ts = B["train_step"]
    # 단계 경계는 base 와 **같은 값 · 같은 비교** (uknow_diffpan.py:12, 449-453 `f < _P1` 이면 stage 0/1).
    P1 = float(base_ts.__globals__["_P1"])

    def train_step_in(m, b, ctx, lossfn):
        S, L = SV.split(b["lms"])
        bS = dict(b, lms=S)
        if ctx["epoch_frac"] < P1:
            # stage 0 prior_step(:354-359) · stage 1 teacher_step(:361-381): prior / teacher 를 배치 그대로 부른다.
            with SV.extra(m, L):
                return base_ts(m, bS, ctx, lossfn)
        with _uk_stage2(m, S, L, ctx.get("sample_idx")):
            return base_ts(m, bS, ctx, lossfn)

    d = SV.make_in(SPECS, "U-Know-DiffPAN", "U-Know-DiffPAN-in", _UK_SITES, "L", train_step=train_step_in,
                   note="자리(망 다섯, 같은 규칙): prior.conv1 (pannet_variance.py:49,70) · teacher/student UNetSR3 downs[0] "
                        "(FSA_T.py:76,167-168 · FSA_S.py:61,151-152) · 그 EMA 사본 둘 (uknow_diffpan.py:156-157) -- EMA zip "
                        "(uknow_diffpan.py:188-197)이 맞도록 같은 자리에 skv_w, EMA 쪽 skv_w 는 requires_grad=False (다른 EMA "
                        "가중치와 같이).  추가 LRMS 는 [0,1] 그대로 (base 도 lms 를 정규화 없이 cond 로 넣는다).  stage 2 의 "
                        "export 부분 배치(L[js])와 H/V 뒤집기(같은 hf/vf)를 추가 입력에도 건다.  평가(student EMA ddim25, "
                        "eval_tile 64)는 기본 감싸개 -- 타일러가 stack 된 lms 를 S 와 함께 자른다.  웨이블릿 조건 cA(lms) "
                        "(uknow_diffpan.py:98-106)와 잔차 기준선 gt-lms / +lms 는 sk 와 같이 S.")
    base_build = d["build"]

    def build(ms_ch=4, ratio=4):
        m = base_build(ms_ch, ratio)
        for s in _UK_EMA_SITES:
            SV.module_at(m, s).skv_w.requires_grad_(False)
        return m
    d["build"] = build


# =============================================================================================== PAN-Crafter
def _register_pancrafter(SPECS):
    K = SPECS["PAN-Crafter-sk"]
    fwd_sk, ts_sk = K["forward"], K["train_step"]
    G = ts_sk.__globals__                         # lib/specs/pancrafter.py 모듈 전역 -- sk 가 쓰는 바로 그 함수들
    feeder_aug, _n11, _n01, _sw, _lpan = (G[k] for k in ("feeder_aug", "_n11", "_n01", "_sw", "_lpan"))

    def forward_in(m, ms_up, pan, lrms, ctx):
        """forward_sk(pancrafter.py:257-260) 는 모델을 배치 전체로 한 번(switch=1) 부르고 skip 으로 _n11(ms_up) 을 더한다.
        추가 입력은 같은 _n11(S)."""
        S, _ = SV.split(ms_up)
        with SV.extra(m, _n11(S)):
            return fwd_sk(m, S, pan, lrms, ctx)

    def train_step_in(m, b, ctx, lossfn):
        """train_step_sk (pancrafter.py:262-276) 와 한 줄씩 같고, 모델 호출에 추가 입력만 건다.

        S 는 sk 와 똑같이 feeder_aug 의 lms 자리로 들어가 gt · ms · lpan · pan 과 **같은 추첨**(크롭 · 고정 flip · rot90)을
        받고(stack 의 L 절반은 PAN-Crafter 가 쓰지 않으므로 증강하지 않는다 -- feeder_aug 는 sk 와 같은 C 채널 텐서를 본다),
        같은 _n11 을 거친다.  배치 복제(repeat 2)의 두 절반(switch 0 = PAN, 1 = HRMS) 모두에 S 를 건다 -- 본체의 입력
        집합이 switch 에 따라 달라지지 않게.  skip 으로 더하는 텐서와 추가 입력은 같은 텐서다."""
        S, _ = SV.split(b["lms"])
        gt, ms, pan = b["gt"], b["ms"], b["pan"]
        lpan = _lpan(ctx)
        seed = int(torch.randint(0, 2 ** 62, (1,)))
        gt, S, ms, lpan, pan = feeder_aug(seed, gt, S, ms, lpan, pan)
        gt, S, ms, lpan, pan = _n11(gt), _n11(S), _n11(ms), _n11(lpan), _n11(pan)
        B, nb = gt.shape[0], gt.shape[1]
        s = torch.cat([_sw(B, 0.0, gt), _sw(B, 1.0, gt)], 0)
        S2 = S.repeat(2, 1, 1, 1)
        with SV.extra(m, S2):
            out = m(pan.repeat(2, 1, 1, 1), lpan.repeat(2, 1, 1, 1), ms.repeat(2, 1, 1, 1), s)
        sv = s.view(-1, 1, 1, 1)
        res_pan = F.interpolate(lpan.repeat(2, 1, 1, 1), scale_factor=4, mode="bicubic")
        out = out + S2 * sv + res_pan.repeat(1, nb, 1, 1) * (1.0 - sv)
        loss_pan = (pan.repeat(1, nb, 1, 1) - out[:B]).abs().mean() * 1.0
        loss_ms = (gt - out[B:]).abs().mean()
        return _n01(out[B:]), loss_pan + loss_ms

    # 자리: input -- PAN 해상도에서 pan 을 처음 읽는 conv.
    #   model/pancrafter.py:224  self.input = Conv2d(3*1 + C -> 128, 3x3, groups 1)
    #   model/pancrafter.py:323  x = input(cat[pan, bicubic(ms), bicubic(lpan), pan - bicubic(lpan)])
    SV.make_in(SPECS, "PAN-Crafter-sk", "PAN-Crafter-in", ["input"], "S",
               forward=forward_in, train_step=train_step_in,
               note="자리: model/pancrafter.py:224,323 input(cat[pan, bicubic(ms), bicubic(lpan), pan-bicubic(lpan)]).  추가 S 는 "
                    "pan · ms 와 같은 _n11([-1,1]).  학습은 S 를 feeder_aug 의 lms 자리로 같은 추첨에 태우고(sk 와 동일), 배치 "
                    "복제의 두 절반(switch 0/1) 모두에 건다.  나머지는 PAN-Crafter-sk 와 같다.")


def register(SPECS, REPOS, _load, _pkg):
    _register_pino(SPECS)
    _register_uknow(SPECS)
    _register_pancrafter(SPECS)
