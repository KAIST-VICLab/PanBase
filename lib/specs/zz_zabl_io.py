"""2x2 ablation (2026-10-16) 의 io 팔 -- "같은 정보, 다른 base".  in 팔과 같은 본체 · 같은 추가 입력 자리 · 같은 0 초기화인데
global skip(base) 만 **원래** 업샘플 MS 로 둔다.  S 는 추가 입력으로만 들어간다.

    본 칸 (LRMS)   skip = 원래 업샘플,  본체 입력 {MS, PAN}
    sk             skip = S,            본체 입력 {MS, PAN}          (YES 방법은 본체 lms 자리도 S)
    io (이 파일)   skip = 원래 업샘플,  본체 입력 {MS, PAN} + S       <- 새 팔
    in (PanBase)   skip = S,            본체 입력 {MS, PAN} + 다른 하나 (YES: + LRMS, INT: + S)

io 와 in 은 본체가 보는 정보가 같다 (둘 다 LRMS · PAN · S).  둘의 차이는 무엇을 base(global skip)로 더하느냐 하나다.
그래서 in > io 이면 이득은 "S 라는 정보" 가 아니라 "S 를 base 로 둔 것" 에서 온다.

데이터는 in 팔과 같다: `lms` = cat[S, LRMS] (스펙의 lms_stack, `--skip S_PAN|S_MS`).
  INT (PanNet, HFIN): 저자 forward 가 skip 으로 더하는 저자 업샘플(`skv_ref_up(lrms)`)을 그대로 쓰고, in 팔과 같은 자리에
       in 팔과 같은 E(S) 를 건다 (PanNet: _hp(S) -- PanNet 본체 입력은 고주파 영역; HFIN: S 그대로).
       `<M>-sk` 의 forward 에 ms_up = 저자 업샘플을 넣으면 본래 forward 와 같다 (skv_check 가 증명한 성질).
  YES (LAGConv): in 팔 forward 에 반쪽을 바꾼 stack cat[LRMS, S] 를 넘긴다 -- in 의 감싸개는 앞 반쪽을 base(=본체 lms
       자리 · skip)로, 뒤 반쪽을 추가 입력으로 쓴다.  그래서 base = LRMS, 추가 입력 = S.
원래 스펙 · 저장소 코드 · 기존 sk/in 스펙은 건드리지 않는다 (새 이름만 등록).  파일 이름이 zz_z 인 까닭: specs 는 이름순으로
등록되고, 이 파일은 zz_skv_* 의 sk/in 스펙이 모두 등록된 뒤에 읽혀야 한다.
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys

import torch

sys.path.insert(0, (_GB + "/lib"))
import skipvar as SV                                                          # noqa: E402


def _io_spec(SPECS, in_name, io_name, forward, note):
    b = SPECS[in_name]
    assert b.get("lms_stack") and (b.get("skipvar") or {}).get("kind") == "in", in_name
    assert "train_step" not in b, f"{in_name} 은 train_step 이 있다 -- io 감싸개를 따로 써야 한다"
    d = dict(b, forward=forward,
             skipvar=dict(b["skipvar"], kind="io", extra="S", in_spec=in_name, note=note),
             diff=b["diff"] + f"  [{io_name}] 2x2 ablation io 팔 (2026-10-16): {in_name} 와 같은 본체 · 추가 입력 자리인데 global skip 은 "
                              f"원래 업샘플이고 S 는 추가 입력으로만 들어간다.  {note}")
    SPECS[io_name] = d
    return d


def register(SPECS, REPOS, _load, _pkg):
    # -------------------------------------------------------------------------------- PanNet (INT)
    # PanNet-in (zz_skv_int_a.py): E = _hp(S) at Conv1, skip = S.   io: 같은 E, skip = 저자 bicubic(lrms) (PanNet.py:48).
    _hp = SPECS["PanNet"]["forward"].__globals__["_hp"]
    pn_sk = SPECS["PanNet-sk"]["forward"]
    pn_up = SPECS["PanNet-sk"]["skv_ref_up"]

    def pannet_io(m, ms_up, pan, lrms, ctx):
        S, _L = SV.split(ms_up)
        with SV.extra(m, _hp(S)):
            return pn_sk(m, pn_up(lrms), pan, lrms, ctx)

    _io_spec(SPECS, "PanNet-in", "PanNet-io", pannet_io,
             "skip: PanNet.py:48 bicubic(lrms) 그대로.  Conv1 추가 입력 _hp(S) (PanNet-in 과 같다).")

    # ---------------------------------------------------------------------------------- HFIN (INT)
    # HFIN-in (zz_skv_int_b.py): 기본 make_in 감싸개 -- E = S at process.conv_p1, skip = S.   io: 같은 E, skip = 저자 mHR.
    hf_sk = SPECS["HFIN-sk"]["forward"]
    hf_up = SPECS["HFIN-sk"]["skv_ref_up"]

    def hfin_io(m, ms_up, pan, lrms, ctx):
        S, _L = SV.split(ms_up)
        with SV.extra(m, S):
            return hf_sk(m, hf_up(lrms), pan, lrms, ctx)

    _io_spec(SPECS, "HFIN-in", "HFIN-io", hfin_io,
             "skip: HFIN.py:480-482 mHR (bicubic, align_corners=True) 그대로.  conv_p1 추가 입력 S (HFIN-in 과 같다).")

    # ------------------------------------------------------------------------------- LAGConv (YES)
    # LAGConv-in (zz_skv_yes.py): forward 가 stack 의 앞 반쪽을 base(본체 lms · skip), 뒤 반쪽(LRMS)을 추가 입력으로 쓴다.
    # io: 반쪽을 바꿔 넘긴다 -> base = LRMS (= 본래 LAGConv), 추가 입력 = S.
    lag_in = SPECS["LAGConv-in"]["forward"]

    def lagconv_io(m, ms_up, pan, lrms, ctx):
        S, L = SV.split(ms_up)
        return lag_in(m, torch.cat([L, S], 1), pan, lrms, ctx)

    _io_spec(SPECS, "LAGConv-in", "LAGConv-io", lagconv_io,
             "LAGConv-in forward 에 cat[LRMS, S] -- base = 하네스 lms(본래 LAGConv 의 skip), 세 추가 입력 자리에 S.")

    # ---------------------------------------------------------------------------------- PINO (YES, 2026-10-18)
    # PINO-in (zz_skv_prio.py): forward_in / train_step_in 이 stack 의 앞 반쪽(S)을 base 로 -- 본래 PINO 의 lms 세 자리
    # (Direction1 입력 xt = cat[lms, pan] bidirection.py:660 · 인코더 내부 skip :670 · LIIF 잔차 liif.py:210) -- 쓰고, 뒤 반쪽(L)을
    # _norm 거쳐 encoder.t.direction1.conv 의 추가 입력으로 건다.  io: LAGConv-io 처럼 반쪽을 바꿔 넘긴다 -> base = LRMS
    # (= 본래 PINO), 추가 입력 = _norm(S).  PINO-in 은 train_step 이 있어 _io_spec 대신 둘 다 감싼다 (같은 단언).
    # 사용자 승인 2026-10-18 ("둘 다 (추천)": PINO 2x2 + PINO-in seed).
    pino_in = SPECS["PINO-in"]
    assert pino_in.get("lms_stack") and (pino_in.get("skipvar") or {}).get("kind") == "in"
    pino_fwd, pino_ts = pino_in["forward"], pino_in["train_step"]

    def pino_io(m, ms_up, pan, lrms, ctx):
        S, L = SV.split(ms_up)
        return pino_fwd(m, torch.cat([L, S], 1), pan, lrms, ctx)

    def pino_io_train_step(m, batch, ctx, lossfn):
        S, L = SV.split(batch["lms"])
        return pino_ts(m, dict(batch, lms=torch.cat([L, S], 1)), ctx, lossfn)

    _pnote = ("PINO-in forward · train_step 에 cat[LRMS, S] -- base = 하네스 lms (본래 PINO 의 lms 세 자리), "
              "Direction1.conv 추가 입력에 _norm(S).")
    SPECS["PINO-io"] = dict(pino_in, forward=pino_io, train_step=pino_io_train_step,
                            skipvar=dict(pino_in["skipvar"], kind="io", extra="S", in_spec="PINO-in", note=_pnote),
                            diff=pino_in["diff"] + "  [PINO-io] 2x2 ablation io 팔 (2026-10-18): PINO-in 와 같은 본체 · 추가 입력 자리인데 "
                                                   f"global skip 은 원래 업샘플이고 S 는 추가 입력으로만 들어간다.  {_pnote}")
