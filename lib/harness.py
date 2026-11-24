"""스펙을 **하네스가 해석하는** 두 자리를 한 곳에 둔다 -- 옵티마이저와 평가 타일.

둘 다 "스펙에 적힌 값이 실제로 쓰이는가" 의 문제다.  학습기(train_pancoll) · 코퍼스 재학습
(train_baseline) · zero-shot 재채점(redo_zeroshot) · 그림(render_visual) 이 각자 해석하면
한 곳만 고치게 된다 ([[one-gate-not-two]] 와 같은 이야기).

(1) 옵티마이저.  예전 학습기는 `opt["kind"]` 를 **읽지 않고** 언제나
    `torch.optim.Adam(..., weight_decay=o["weight_decay"])` 를 지었다.  kind 가 "adam" 이 아닌
    스펙은 PAN-Crafter 하나였는데, 저자는 AdamW(repos/PAN-Crafter/train.py:61) 이고 wd=0.01 을
    Adam 에 넣으면 **결합 L2** 가 된다 -- 가중치의 85~91% 가 0 쪽으로 무너졌다.  적힌 값이
    조용히 무시되는 것이 문제의 본질이라, 모르는 kind 는 이제 멈춘다.
(2) 평가 타일.  UTeRM 의 저랭크 시작(UTeRM_MRA.py:85-86, W 축 FFT 의 t-product)은
    네트워크를 **학습 폭**에 묶는다.  저자도 학습 크기(256)로 잘라 추론한다
    (test_full_res.py:64-87).  우리는 64x64 패치로 학습하고 256(RR · WV2) · 512(FR) 를 한 번에
    넣었다 -- wv3 RR 15.61 dB.  겹치지 않는 64 타일로 이어 붙이면 35.96 dB 다.
"""
from __future__ import annotations

import threading

import torch

#: 타일 추론 중인가 (`tiled_forward` 가 fwd 를 부르는 동안만 켜진다).  입력 크기로 검증(64x64 valid 분할)과 시험을
#: 가르는 스펙(PanDiT 의 초기 잡음 규칙)은 타일이 학습 크기와 같아도 이것으로 **시험**임을 안다.
_TILED = threading.local()


def in_tiled_eval() -> bool:
    """`tiled_forward` 안에서 불린 forward 인가."""
    return bool(getattr(_TILED, "on", False))

#: `opt["kind"]` -> 클래스.  여기 없는 kind 는 **멈춘다** (조용히 Adam 으로 떨어지지 않는다).
OPT_KINDS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def optimizer_class(spec):
    """하네스가 `make_opt` 없이 지을 클래스.  모르는 kind 면 SystemExit."""
    o = spec["opt"]
    cls = OPT_KINDS.get(o.get("kind"))
    if cls is None:
        raise SystemExit(f"[거부] 모르는 옵티마이저 kind {o.get('kind')!r} -- "
                         f"harness.OPT_KINDS({sorted(OPT_KINDS)}) 에 넣거나 스펙에 make_opt 를 준다")
    if cls is torch.optim.AdamW and "weight_decay" not in o:
        # torch 의 AdamW 기본 wd 는 0.01 이고 Adam 은 0 이다.  어느 쪽인지 스펙이 말해야 한다.
        raise SystemExit("[거부] kind='adamw' 는 weight_decay 를 명시해야 한다 (torch 기본값 0.01 에 "
                         "기대지 않는다)")
    return cls


def make_optimizer(spec, model):
    """스펙의 옵티마이저.  `make_opt` 가 있으면 그것이 이긴다 (파라미터 그룹 · 상태 참조가 필요한
    저장소들).  없으면 `opt["kind"]` 가 정한 클래스를 적힌 값 그대로 짓는다."""
    o = spec["opt"]
    if "make_opt" in spec:
        return spec["make_opt"](model, o)
    cls = optimizer_class(spec)
    return cls(model.parameters(), lr=o["lr"], betas=tuple(o.get("betas", (0.9, 0.999))),
               eps=o.get("eps", 1e-8), weight_decay=o.get("weight_decay", 0.0))


# ------------------------------------------------------------------ 평가 타일
def _cut(x, t, nh, nw):
    """(B,C,nh*t,nw*t) -> (B*nh*nw, C, t, t).  표본 순서 -> 행 -> 열."""
    B, C = x.shape[:2]
    return (x.reshape(B, C, nh, t, nw, t).permute(0, 2, 4, 1, 3, 5)
            .reshape(B * nh * nw, C, t, t))


def _stitch(y, B, nh, nw):
    """`_cut` 의 역."""
    n, C, t, u = y.shape
    assert n == B * nh * nw and t == u, (tuple(y.shape), B, nh, nw)
    return (y.reshape(B, nh, nw, C, t, t).permute(0, 3, 1, 4, 2, 5)
            .reshape(B, C, nh * t, nw * t))


def tiled_forward(fwd, model, lms, pan, ms, ctx, tile, ratio=4):
    """`fwd` 를 겹치지 않는 tile x tile 타일마다 돌려 이어 붙인다 (타일은 한 번에 배치로 넣는다).

    - H, W 가 tile 과 같으면 **그대로** 부른다 (학습 중 64x64 검증은 이 길이다).
    - H, W 는 tile 의 배수여야 한다 (아니면 AssertionError -- 겹침 · 덧댐은 규약이 아니다).
    - 저해상 입력(ms, 그리고 ctx 의 MS 해상도 텐서 예: lpan)은 tile/ratio 로 **같은 격자**를 자른다.
      ctx 의 전해상도 텐서도 tile 로 자른다.  다른 값(step 등)은 그대로 넘긴다.
    - 출력이 튜플이면 (B*n, C, tile, tile) 모양의 원소만 이어 붙이고 나머지는 그대로 둔다.
    """
    H, W = pan.shape[-2:]
    if not tile or (H == tile and W == tile):
        return fwd(model, lms, pan, ms, ctx)
    assert H % tile == 0 and W % tile == 0, f"평가 크기 {H}x{W} 가 타일 {tile} 의 배수가 아니다"
    assert tile % ratio == 0, f"타일 {tile} 이 배율 {ratio} 로 나뉘지 않는다"
    assert tuple(lms.shape[-2:]) == (H, W), ("lms 는 PAN 격자여야 한다", tuple(lms.shape), (H, W))
    assert tuple(ms.shape[-2:]) == (H // ratio, W // ratio), \
        ("ms 는 PAN/ratio 격자여야 한다", tuple(ms.shape), (H // ratio, W // ratio))
    B, nh, nw, ts = pan.shape[0], H // tile, W // tile, tile // ratio

    def cut_any(v):
        if torch.is_tensor(v) and v.dim() == 4 and v.shape[0] == B:
            if tuple(v.shape[-2:]) == (H, W):
                return _cut(v, tile, nh, nw)
            if tuple(v.shape[-2:]) == (H // ratio, W // ratio):
                return _cut(v, ts, nh, nw)
        return v

    tctx = {k: cut_any(v) for k, v in (ctx or {}).items()}
    prev = in_tiled_eval()
    _TILED.on = True
    try:
        out = fwd(model, _cut(lms, tile, nh, nw), _cut(pan, tile, nh, nw), _cut(ms, ts, nh, nw), tctx)
    finally:
        _TILED.on = prev

    def back(y):
        if torch.is_tensor(y) and y.dim() == 4 and y.shape[0] == B * nh * nw \
                and tuple(y.shape[-2:]) == (tile, tile):
            return _stitch(y, B, nh, nw)
        return y

    return tuple(back(y) for y in out) if isinstance(out, tuple) else back(out)


def eval_forward(spec, ratio=4):
    """평가에 쓸 forward.  스펙에 `eval_tile` 이 있으면 타일 추론으로 감싼 것, 없으면 원래 것.

    **학습 forward 는 건드리지 않는다** -- 학습기는 여전히 `spec["forward"]`(또는 train_step)로
    학습하고, 이 감싸개는 평가 경로(val · FR · zero-shot · 그림 · 코퍼스 zero-shot)만 쓴다."""
    fwd, tile = spec["forward"], spec.get("eval_tile")
    if not tile:
        return fwd

    def tiled(m, lms, pan, ms, ctx):
        return tiled_forward(fwd, m, lms, pan, ms, ctx, int(tile), ratio)
    return tiled


# ------------------------------------------------------------------ 공개 (원장 · 표 각주 · 인쇄되는 캡션)
#: 타일로 평가한 행의 표시.  논문 표(make_tables 의 네 표 · 코퍼스 표)의 행 이름 · 캡션 문장(`tile_caption`)과 원장 1절이
#: **이것 하나**를 읽는다.  다른 기호는 이미 쓰인다: $^\dagger$ = WV2 표의 PAN-Crafter (lpan 합성),
#: $^\ast$ = 코퍼스 FR 표의 clamp, $^\ddagger$ = 코퍼스 표의 합성 lpan.
TILE_MARK = r"$^\S$"
#: 같은 표시의 md 표기 (zeroshot_qa_fr.md · zeroshot_corpus.md 의 † · ‡ · ⚠ 옆).
TILE_MARK_MD = "\u00a7"   # section sign: the table mark of tiled evaluation
#: 타일 평가가 **저자 시험 스크립트와 다른** 방법 -> 그 스크립트 위치 (기계가 읽는 기록).  UTeRM · PanDiT 은
#: 저자도 학습 크기로 추론하므로 여기 없다.  `TILE_AUTHOR` 의 문장 · 인쇄되는 캡션(`tile_caption`) · 원장 1절이 모두
#: 이것을 읽는다 -- 저자 스크립트가 영상 통째인 타일 방법이 새로 생기면 여기 한 줄을 더한다.
TILE_AUTHOR_WHOLE = {"U-Know-DiffPAN": "S2_FSA_S_test.py:40"}
#: 통째 추론 값의 보충 표가 **이미 있는** 방법 -> 캡션이 그 표를 가리킬 문구 (예: "Table~S3 of the supplementary material").
#: 재측정(seed 0 덤프 경로; gf2 는 학습 뒤)이 끝나 보충 표가 생기면 여기 한 줄을 더한다 -- 그때
#: `tile_caption` 이 그 문장을 싣는다.  비어 있는 동안 캡션은 보충 자료를 가리키지 않고(없는 표를 약속하지 않는다),
#: `tile_release_gaps()` 가 공개 전 체크리스트 항목을 낸다.
TILE_WHOLE_SUPPLEMENT = {}

#: 타일 추론을 하는 이유.  원장과 표 각주가 이 문장을 그대로 싣는다 -- 새 `eval_tile` 스펙이 여기
#: 없어도 공개는 빠지지 않는다 (`eval_tile_notes` 가 일반 문장으로 대신한다).
TILE_WHY = {
    "UTeRM": "저랭크 시작(UTeRM_MRA.py:85-86)이 W 축 FFT 의 t-product 라 네트워크가 학습 폭에 묶인다. "
             "영상 전체를 한 번에 넣으면 wv3 RR 이 15.61 dB 로 무너지고, 겹치지 않는 64 타일은 35.96 dB 다 "
             "(진단 2026-09-24). 저자도 학습 크기(256)에서만 추론한다 -- FR 영상은 256 타일로 자른다 (test_full_res.py:64-87)",
    "PanDiT": "생성 시점에 공간 크기가 박히는 구조다 (LayerNorm[C,H,H] · 패치 격자 H/8 · 위치 인코딩 토큰 수). "
              "저자는 한 크기(128)의 타일로만 학습하고 추론한다 (data/PanDataset.py 의 train128 폴더, predict.py "
              "image_size). 64 로 학습한 가중치에 256 영상을 통째로 넣으면 RR 이 qb 32.68 · gf2 39.16 dB 이고, "
              "겹치지 않는 64 타일은 37.10 · 42.00 dB 다 (진단 2026-09-30, 같은 best.pt)",
    # 수치 출처 (손으로 옮겼다 -- 학습 프로세스가 import 하는 모듈이라 import 때 파일을 읽지 않는다):
    #   wv3 통째 · 64 타일  _logs/py_ab_uknow_wv3.log 마지막 JSON 줄 (seed_all(0), best.pt 847,841 = 결과 JSON best_step)
    #   qb 1M 통째          _logs/pancoll_U-Know-DiffPAN_qb.log 의 resume 런 [final] (step 1,000,176, 규약 rr_dim_cut30;
    #                      학습기 평가라 seed_all(0) 이 아니다 -- 709k 에서는 학습기 값 = seed 0 A/B 값 37.381)
    #   qb 1M 64 타일       같은 로그의 10-01 06:56 eval 런 [final] = results/pancoll_U-Know-DiffPAN_qb.json rr.psnr
    #   qb 709k            _logs/py_ab_uknow_qb.log (통째 · 64 타일 · 128 타일, seed_all(0)) · 같은 로그 첫 [final] 의 FR HQNR
    #   709k 의 student 반복 = 709,206 - 700,123 (단계 경계 epoch_frac 0.70, specs/uknow_diffpan.py:12)
    # 저자 스크립트 · 저자 기본 가중치 · 보충 자료는 이유가 아니라 저자 추론과의 관계라 `TILE_AUTHOR` 에 둔다 (원장 1절).
    "U-Know-DiffPAN": "student 디노이저(UNetSR3: 가장 깊은 H/8 해상도의 self-attention, 그룹 1 GroupNorm)를 64 패치로만 "
                      "학습한다. 256 영상을 통째로 넣으면 RR 이 wv3 31.31 · qb 31.75 dB 이고, 겹치지 않는 64 타일은 37.44 · "
                      "37.00 dB 다 (같은 best.pt; wv3 는 진단 2026-09-30, qb 는 통째가 학습 끝 평가 · 64 타일이 재평가 "
                      "2026-10-01). qb 709k 체크포인트(student 9,083 반복)에서는 RR 통째 37.38 · 64 타일 37.37 로 거의 같았다 "
                      "(RR 256 한정; 같은 체크포인트의 FR 512 통째 HQNR 0.8981). 그 709k 에서 128 타일(37.58)이 더 높았지만 "
                      "타일 크기는 시험값이 아니라 학습 크기로 정했다",
}

#: 저자는 어떤 크기로 추론하나 -- 원장 1절 '편향 방향' 칸이 싣는다.  예전에는 타일 평가 방법 모두에 '저자 추론 규약과
#: 같다' 한 문구를 붙였는데, U-Know-DiffPAN 은 틀렸다 (저자 시험 스크립트는 통째).  방법마다 사실만 적는다 (10-01).
#: 저자와 다르다는 판정은 이 문장이 아니라 `TILE_AUTHOR_WHOLE` 이 정한다 (원장이 그것을 읽어 굵게 적는다).
#: 여기 없는 `eval_tile` 스펙은 원장이 '확인하지 않았다' 로 적는다 -- 같은 줄이 빠지지는 않는다.
TILE_AUTHOR = {
    "UTeRM": "저자도 학습 크기로만 추론한다 (저자 학습 256: RR 256 은 그대로, FR 은 256 타일 -- test_full_res.py:64-87; "
             "우리 학습 64 -> 64 타일)",
    "PanDiT": "저자도 학습 크기 타일로 추론한다 (저자 128 -- predict.py image_size, 우리 64)",
    "U-Know-DiffPAN": f"저자 시험 스크립트는 영상 통째다 ({TILE_AUTHOR_WHOLE['U-Know-DiffPAN']}). 저자 기본 가중치는 EMA 가 "
                      "아닌 student 1만 반복이다 (config/FSA_S_config.py:60) -- student 반복이 그와 비슷한 우리 qb 709k "
                      "체크포인트에서는 RR 통째와 64 타일이 거의 같았다 (이유 칸). 구조가 학습 크기에 묶이고 저자도 학습 크기로 추론한다는 "
                      "UTeRM · PanDiT 의 기준을 넓힌 것이다. "
                      + (f"통째 값의 보충 표: {TILE_WHOLE_SUPPLEMENT['U-Know-DiffPAN']}" if "U-Know-DiffPAN" in TILE_WHOLE_SUPPLEMENT
                         else "통째 값의 보충 표는 아직 없다 -- seed 0 덤프 경로에서 다시 재어 낸다 (이유 칸의 통째 값은 진단 · "
                              "학습 끝 평가 값이다)"),
}


def eval_tile_notes(specs, names=None):
    """`eval_tile` 이 있는 스펙마다 (이름, 타일, 공개 문장).  원장(make_ledger)과 표 각주(make_tables)가
    **같은 함수**로 만든다 -- 손으로 적으면 스펙과 어긋난다 (검토 2026-09-24: 원장이 '평가는 모든 방법에
    동일' 이라고 적은 채 UTeRM 만 타일로 평가할 뻔했다).  `names` 를 주면 그 방법들의 문장만 낸다 -- 문장 안의
    '타일로 추론하는 방법' 목록은 `names` 와 무관하게 `specs` 전체에서 뽑는다 (일부만 물어도 목록이 거짓이 되지 않게)."""
    import pancoll as PC                       # 무거운 의존이 없는 쪽만 늦게 부른다
    all_tiled = [n for n in sorted(specs) if specs[n].get("eval_tile")]
    tiled = [n for n in all_tiled if names is None or n in names]
    # 방법마다 '다른 방법은 통째' 라고 쓰면 타일 방법이 둘 이상일 때 문장끼리 어긋난다 (원장 10-01) -- 타일 방법 목록을 같이 싣는다.
    who = " · ".join(all_tiled)
    out = []
    for n in tiled:
        t = int(specs[n]["eval_tile"])
        why = TILE_WHY.get(n, "스펙(lib/registry.py)의 주석 참고")
        out.append((n, t, f"{n} 은 평가 영상(RR · WV2 256, FR 512)을 학습 크기 {t}x{t} 의 겹치지 않는 타일로 "
                          f"나눠 한 배치로 추론하고 이어 붙인다 (`harness.eval_forward`, 규약 "
                          f"`{PC.eval_protocol(specs[n])}`). 타일로 추론하는 방법은 {who} 이고, 나머지는 영상 전체를 "
                          f"한 번에 넣는다. 지표는 이어 붙인 전체 영상에서 똑같이 잰다. 이유: {why}"))
    return out


def tile_caption(marked, rest_whole=None):
    r"""인쇄되는 캡션 한 문장 (영문) -- 그 표에서 `TILE_MARK` 를 단 행 `marked` ({방법: 타일 크기}).  비어 있으면 ''.

    make_tables 의 네 표와 코퍼스 표(make_zs_qa_fr_table · make_zs_corpus_table)가 **이 함수 하나**로 쓴다 -- 표마다
    문구가 갈리지 않게.
    - 타일 크기는 `marked` 의 값 (make_tables: 스펙 `eval_tile`, 코퍼스 표: 조각 · 결과에 기록된 `eval_tile`).
    - '통째로 넣으면 PSNR 이 크게 낮다' 는 `TILE_WHY` 의 진단 (UTeRM wv3 · PanDiT qb/gf2 · U-Know-DiffPAN wv3/qb, 같은
      best.pt) 이라 '진단에서' 로 적고 칸마다 쟀다고 쓰지 않는다.
    - 저자 시험 스크립트와 다른 방법(`TILE_AUTHOR_WHOLE`)은 그 행이 표에 있을 때만 밝힌다.  통째 값의 보충 표는 아직
      없다 (seed 0 덤프 경로에서 다시 잰다) -- 그래서 보충 자료를 가리키지 않는다.  보충 표가 생기면
      `TILE_WHOLE_SUPPLEMENT` 에 한 줄을 더하고, 그때 이 함수가 그 표를 가리키는 문장을 싣는다.
    - `rest_whole` 을 주면 (코퍼스 표: "test tile" -- 시험 단위가 영상이 아니라 코퍼스 타일이다)
      'Every other method takes each <rest_whole> whole.' 를 덧붙인다."""
    if not marked:
        return ""
    ts = sorted(set(int(t) for t in marked.values()))
    size = (r"$%d\times%d$" % (ts[0], ts[0]) if len(ts) == 1 else
            ", ".join(r"%s $%d\times%d$" % (n, int(t), int(t)) for n, t in sorted(marked.items())))
    s = (TILE_MARK + r"Inference runs on non-overlapping tiles of the size these models are trained on here ("
         + size + r"), and the stitched output is scored like every other method's; in our diagnostics on the PanCollection test "
         r"images, whole-image inference with the same weights gave a markedly lower reduced-resolution PSNR. ")
    dev = [n for n in sorted(marked) if n in TILE_AUTHOR_WHOLE]
    if dev:
        one = len(dev) == 1
        nm = dev[0] if one else ", ".join(dev[:-1]) + " and " + dev[-1]
        s += (r"For %s this departs from the authors' test script%s, which process%s the whole image. "
              % (nm, "" if one else "s", "es" if one else ""))
        for n in dev:
            if n in TILE_WHOLE_SUPPLEMENT:
                s += r"Whole-image scores for %s are given in %s. " % (n, TILE_WHOLE_SUPPLEMENT[n])
    if rest_whole:
        s += r"Every other method takes each %s whole. " % rest_whole
    return s


def tile_release_gaps():
    """공개 전 체크리스트: 저자 시험 스크립트와 달리 타일로 평가한 방법(`TILE_AUTHOR_WHOLE`) 중 통째 값의
    보충 표(`TILE_WHOLE_SUPPLEMENT`)가 아직 없는 것.  비어 있지 않으면 공개 전에 경고한다 (막지는 않는다)."""
    return sorted(n for n in TILE_AUTHOR_WHOLE if n not in TILE_WHOLE_SUPPLEMENT)
