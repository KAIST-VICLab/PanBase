"""PanCollection 공식 split 에서 베이스라인을 재학습한다.

바꾸지 않는 것: 모델·손실·옵티마이저·학습률·배치·epoch 수 -- 전부 공식 저장소 값.
바꾸는 것: 학습 루프 보일러플레이트와 평가 코드뿐.  데이터는 애초에 그들의 것이다.
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__)))
import sys, os, argparse, json, time, socket, datetime, random, warnings
from pathlib import Path
import numpy as np, torch
# `$PANBASE_ROOT` 도 넣는다.  FR 지표는 `pancoll.hqnr` -> `ps_metrics.fr_metrics` 로
# 내려가므로(게이트를 하나로 두려고 그렇게 위임했다) 그 경로가 없으면 **FR 만 조용히
# 빠진다**.  워커 환경에는 PYTHONPATH 가 있어서 여태 드러나지 않았는데, 손으로
# `--eval_only` 를 돌리자 MMMamba/qb 가 RR 만 있는 결과를 냈다 (2026-09-23).
# 경로를 호출자에게 맡기면 호출 방법마다 결과의 내용이 달라진다.
sys.path[:0] = [(_GB + "/lib"), _GB]
import pancoll as PC, registry as R
import harness as H        # 옵티마이저 · 평가 타일 해석 -- 다른 스크립트와 같은 함수

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dataset", required=True, choices=["wv3", "qb", "gf2"])
ap.add_argument("--epochs", type=int, default=0, help="0 이면 저장소 기본값")
ap.add_argument("--batch", type=int, default=0)
ap.add_argument("--max_hours", type=float, default=48.0,
                help="안전 천장.  정지 규칙은 수렴이고, 이것은 큐가 한 칸에 "
                     "영원히 묶이지 않게 하는 장치다")
ap.add_argument("--patience", type=int, default=0,
                help="valid 최고 기록이 이 step 수 동안 안 갱신되면 중단. "
                     "0 이면 일정 길이에서 자동으로 정한다")
ap.add_argument("--eval_every", type=int, default=2000)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seed", type=int, default=20260919)
ap.add_argument("--tag", default="")
ap.add_argument("--eval_only", action="store_true",
                help="학습을 건너뛰고 저장된 best.pt 로 평가만 다시 낸다. "
                     "평가 **규약**만 바뀌었고 레시피와 가중치는 그대로인 칸을 "
                     "되살리는 용도다 -- 그런 칸을 버리면 이미 쓴 GPU-시간을 버리는 것이고, "
                     "다시 학습해도 같은 가중치가 나온다. 평가 코드는 학습 경로와 "
                     "**같은 함수**를 쓴다(따로 만들면 또 어긋난다).")
ap.add_argument("--vram_frac", type=float, default=0.0,
                help="이 프로세스가 쓸 수 있는 GPU 메모리 상한(전체 대비 비율). "
                     "0 이면 제한 없음 -- 큐가 돌리는 모든 학습의 기본값이라 기존 동작은 "
                     "그대로다. 한 장을 다른 작업과 나눠 쓸 때 **우리 프로세스가 먼저 "
                     "죽도록** 만드는 용도다: 상한이 없으면 메모리가 모자랄 때 누가 "
                     "떨어질지는 다음에 할당하는 쪽이 정하고, 그 쪽이 남의 학습일 수 있다.")
ap.add_argument("--out", default=(_GB + "/runs_pancoll"),
                help="런 폴더 뿌리 (<out>/<model>_<dataset><tag>).  --run_root 를 함께 주면 이것은 "
                     "요약 JSON 경로가 된다 (아래)")
ap.add_argument("--run_root", default="",
                help="런 폴더 뿌리.  주면 --out 은 **요약 JSON** 경로(.json)로 읽고, 결과 JSON 을 다 쓴 "
                     "뒤(FR 까지 있을 때만) 요약을 쓴다 -- 디스패처 `py` 줄로 넣으려는 것이다 (`py` 줄은 "
                     "--out 이 없으면 버리고, 있으면 끝난 줄로 본다).  없으면 예전 그대로다")
ap.add_argument("--grad_clip_override", type=float, default=None,
                help="기울기 클립 상한을 스펙 값 대신 이 값으로 (0 = 클립 없음).  **민감도 런 전용** "
                     "(-4): 비어 있지 않은 --tag 가 있어야 받고(seed 복제 이름 `_s<seed>` 는 안 된다), "
                     "스펙 값과 같으면 거부한다.  결과 JSON 에 grad_clip_override 를 남기고 레시피 "
                     "스냅샷의 grad_clip 도 이 값이 된다 -- 게이트가 본 칸 레시피로 읽을 수 없다")
ap.add_argument("--force", action="store_true",
                help="이미 표에 실릴 수 있는 칸이라도 다시 학습한다. 기본은 건너뛴다.")
ap.add_argument("--resume", action="store_true",
                help="같은 런 폴더에서 이어 돈다 (사용자 결정 2026-09-24). "
                     "state.pt 가 있고 레시피·seed·tag 가 같으면 exact -- 옵티마이저·스케줄러·난수까지 "
                     "되살려 끊김 없는 학습과 같은 궤적을 잇는다. 아니면 best.pt 에서 warm -- 가중치만 "
                     "되살리고 옵티마이저는 새로 만든다(기록된다). 둘 다 없으면 거부한다.")
ap.add_argument("--resume_reason", default="",
                help="이어 돌리는 이유 (결과 JSON 의 resumes[].reason 에 그대로 남는다)")
ap.add_argument("--results_dir", default=(_GB + "/results"),
                help="결과 JSON 을 읽고(이어 돌리기 · 재평가의 앞 구간 기록) 쓰는 곳")
ap.add_argument("--state_every_min", type=float, default=30.0,
                help="학습 상태(state.pt)를 이 간격(분)마다 평가 경계에서 저장한다. "
                     "천장·시험 정지 때는 간격과 무관하게 최종 평가 **전에** 저장한다")
ap.add_argument("--stop_after_steps", type=int, default=0,
                help="**시험 전용.** 이 step 에서 천장에 걸린 것처럼 멈춘다 (stop_reason=test_stop). "
                     "--tag 가 있고 --results_dir 가 기본값이 아닐 때만 받는다 -- 표의 칸을 "
                     "만들 수 없게 하려는 것이다.")
ap.add_argument("--test_limit", type=int, default=0,
                help="**시험 전용.** train · valid 를 앞의 N 표본으로 줄인다 (CPU 에서 이어 돌리기 "
                     "시험을 몇 분 안에 돌리려는 것 -- PanNet 도 CPU 한 에폭이 5분 걸린다). "
                     "--stop_after_steps 와 같은 조건에서만 받는다.")
ap.add_argument("--train_frac", type=float, default=1.0,
                help="데이터 효율 ablation (2026-10-16): 학습 표본을 고정 부분집합(이 비율, 부분집합 난수 12345 -- 런 seed 와 무관)으로 "
                     "줄인다.  일정(spe · steps)은 전체 표본 수로 정해 **같은 iteration 예산**을 쓴다.  valid · test 는 그대로.  "
                     "1 이 아니면 --tag 와 기본값이 아닌 --results_dir 가 있어야 한다 (표의 칸이 되지 않게).  레시피에 남는다")
ap.add_argument("--skip", default="lrms", choices=list(PC.SKIPS),
                help="global skip 변형 연구 (2026-10-09): 모델이 받는 `lms` 를 Stage-1 base 로 바꾼다 (S_PAN = C2u, "
                     "S_MS = G1u; lib/pancoll.SKIP).  lrms 가 아니면 --tag 가 `_skip` 으로 시작하고 --results_dir 가 기본값이 "
                     "아니어야 한다 -- 표의 칸이 되지 않게.  FR 지표는 원래 lms 로 잰다.  레시피 스냅샷에 skip 이 남는다")
ap.add_argument("--gt_geo", default="ms", choices=["ms", "pan_detail"],
                help="학습 · 검증 GT 의 기하 (2026-10-13 파일럿, stage1/make_gt_geo.py; lib/pancoll.GT_GEO).  pan_detail = 디테일만 PAN "
                     "기하로 옮긴 GT.  test 의 RR · FR 은 원래 데이터로 잰다.  ms 가 아니면 --tag 가 `_geo` 로 끝나고 --results_dir 가 "
                     "기본값이 아니어야 한다 -- 표의 칸이 되지 않게")
a = ap.parse_args()
if a.gt_geo != "ms":
    if not a.tag.endswith("_geo") or os.path.realpath(a.results_dir) == os.path.realpath((_GB + "/results")):
        raise SystemExit("[거부] --gt_geo 변형은 --tag ..._geo 와 기본값이 아닌 --results_dir 가 있어야 한다 (표의 칸이 되지 않게)")
if a.train_frac != 1.0:
    if not (0.0 < a.train_frac < 1.0) or not a.tag or os.path.realpath(a.results_dir) == os.path.realpath((_GB + "/results")):
        raise SystemExit("[거부] --train_frac 은 0 과 1 사이이고, --tag 와 기본값이 아닌 --results_dir 가 있어야 한다 (표의 칸이 되지 않게)")
if a.skip != "lrms":
    if not a.tag.startswith("_skip") or os.path.realpath(a.results_dir) == os.path.realpath((_GB + "/results")):
        raise SystemExit("[거부] --skip 변형은 --tag _skip... 와 기본값이 아닌 --results_dir 가 있어야 한다 (표의 칸이 되지 않게)")
    # 변형 런은 **step 으로** 본 칸에 맞춘다 (2026-10-09).  본 칸이 시간 상한에 잘린 두 칸은 --stop_after_steps 로 그 step 에서
    # 멈추고, 나머지 본 칸은 일정 · 수렴(patience)으로 끝났다 -- 변형이 다른(느린, 공유) 카드에서 96h 총 상한에 잘리면 본 칸보다
    # 적은 step 을 학습한 채 비교된다.  그래서 변형 런에만 총 상한을 걷어낸다.  48h 구간 천장은 그대로이고, 끊긴 런은
    # `_logs/skipvar_watch.py` 가 exact 이어 돌리기로 잇는다 (옵티마이저 · 스케줄러 · 난수 · patience 상태 복원 = 끊김 없는 런).
    import resume_state as _RS_SKIP
    _RS_SKIP.TOTAL_CAP_H = 1.0e5

# `--run_root` 가 있으면 `--out` 은 런 폴더 뿌리가 아니라 **요약 파일**이다.  디스패처의 `py` 줄은 `--out`
# 이 없으면 버리고 있으면 끝난 것으로 보는데(`_logs/dispatch.py` verdict), 이 스크립트의 `--out` 은 원래
# 런 폴더 뿌리라 그 뜻을 바꿀 수 없다 -- 기존 호출(큐 · 시험)이 전부 그 뜻으로 쓴다.  그래서 이 조합에서만
# 켜고, 아래는 전부 예전처럼 `a.out` 을 런 폴더 뿌리로 읽는다.
SUMMARY = None
if a.run_root:
    if not a.out.endswith(".json") or os.path.isdir(a.out):
        raise SystemExit("[거부] --run_root 를 주면 --out 은 요약 JSON 경로(.json)여야 한다")
    if os.path.exists(a.out):
        raise SystemExit(f"[건너뜀] 요약 {a.out} 이 이미 있다 -- 끝난 런이다 (다시 돌리려면 지워라)")
    SUMMARY, a.out = a.out, a.run_root
elif a.out.endswith(".json"):
    # 런 폴더 뿌리가 `x.json` 이면 그 이름의 디렉터리가 생기고, 디스패처는 그것을 '이미 있는 출력'으로 본다.
    raise SystemExit("[거부] --out 이 .json 이다 -- 요약으로 쓰려면 --run_root 로 런 폴더 뿌리를 함께 줘라")
# 기울기 클립 덮어쓰기 (-4 민감도 런).  **표의 칸이 될 수 없게** 막는다: tag 필수 · seed 복제 이름 금지
# (산포 측정에 섞인다) · 스펙 값과 같은 값 금지 (덮어쓸 것이 없고, 스냅샷이 본 칸과 같아진다).  기록되는
# 레시피 스냅샷의 grad_clip 이 덮어쓴 값이 되므로 게이트(`make_tables.recipe_diff`)는 파일 이름을 바꿔도
# 이 결과를 현재 레시피로 읽지 않는다.  재평가는 이전 결과의 기록을 그대로 가져오므로 받지 않는다.
if a.grad_clip_override is not None:
    import math as _math, re as _re
    _gco = a.grad_clip_override
    if not a.tag:
        raise SystemExit("[거부] --grad_clip_override 는 --tag 가 있을 때만 받는다 (표의 칸이 되지 않게)")
    if _re.fullmatch(r"_s\d+", a.tag):
        raise SystemExit(f"[거부] tag {a.tag!r} 는 seed 복제 이름이다 -- 클립을 바꾼 런이 seed 산포에 섞인다")
    if a.eval_only:
        raise SystemExit("[거부] --grad_clip_override 는 학습에만 쓴다 -- 재평가(--eval_only)는 이전 결과의 "
                         "grad_clip · 레시피 기록을 그대로 가져온다")
    if not (_math.isfinite(_gco) and _gco >= 0):
        raise SystemExit(f"[거부] --grad_clip_override {_gco} -- 0(클립 없음) 이상의 유한한 값만 받는다")
    _sp0 = R.SPECS.get(a.model) or {}
    _gc0 = float({**_sp0, **((_sp0.get("per_dataset") or {}).get(a.dataset) or {})}.get("grad_clip", 10.0))
    if _gco == _gc0:
        raise SystemExit(f"[거부] --grad_clip_override {_gco:g} 가 스펙 값과 같다 -- 덮어쓸 것이 없다")

RESULTS_DEFAULT = (_GB + "/results")
#: 진짜 런 폴더들.  시험 전용 인자는 여기에 쓰지 못한다 (아래).
REAL_OUTS = ((_GB + "/runs_pancoll"), (_GB + "/runs_sweep"))
if a.resume and a.eval_only:
    raise SystemExit("[거부] --resume 과 --eval_only 는 함께 쓸 수 없다 -- 재평가는 학습을 하지 않는다")
# 결과 JSON 만 막아서는 부족하다 (검토, 2026-09-24).  --out 이 기본값(runs_pancoll)이면 tag 가 진짜
# seed 복제와 같을 때(예: _s20260922) 시험 런이 **진짜 런 폴더**에서 돈다 -- state.pt 를 치우고, 앞
# N 표본으로 학습한 가중치로 best.pt 를 덮고, test_stop 상태를 남긴다.  그러면 진짜 결과 JSON 이 자기가
# 내지 않은 가중치를 가리키고, 그 가중치가 체크포인트 · 그림으로 나간다.
if (a.stop_after_steps or a.test_limit) and (
        not a.tag or os.path.realpath(a.results_dir) == os.path.realpath(RESULTS_DEFAULT)
        or os.path.realpath(a.out) in {os.path.realpath(x) for x in REAL_OUTS}):
    raise SystemExit("[거부] --stop_after_steps · --test_limit 은 시험 전용이다: --tag 를 주고 "
                     "--results_dir 와 --out 을 기본값(진짜 결과 · 런 폴더)이 아닌 곳으로 돌려야 한다 "
                     "(표의 칸이나 진짜 체크포인트가 되지 않게)")
if a.resume:
    # 앞선 이어 돌리기가 **거부된 채로** 있으면 카드를 잡고 데이터를 올리기 전에 멈춘다.  거부는
    # 결정적이라 다시 해도 같은 자리에서 거부된다 (`resume_state.block_reason`).  원인을 고쳤으면
    # 런 폴더의 resume_refused.json 을 지운다.  시도 횟수는 여기서 보지 않는다 -- 그것은 자동 재큐
    # (reconcile · 디스패처)의 멈춤 장치이고, 사람이 조사한 뒤 손으로 다시 돌리는 것까지 막지 않는다.
    import resume_state as _RS0
    _br = _RS0.block_reason(str(Path(a.out) / f"{a.model}_{a.dataset}{a.tag}"),
                            f"{a.results_dir}/pancoll_{a.model}_{a.dataset}{a.tag}.json",
                            attempts=False)
    if _br:
        raise SystemExit(f"[이어 돌리기 거부] {a.model}/{a.dataset}{a.tag}: {_br}")

# 계단식 lr 감쇠 전 patience 정지의 연장 (사전 등록 2026-09-25, `lib/milestone_policy.py`).
# **스위치(`milestone_policy.SWITCH`)가 있을 때만** 켜진다 -- 학습하는 런 전부 (처음부터 · 이어 돌리기, 본 칸 · seed 복제;
# 규칙은 스펙 단위다, 검토 2026-09-25).  재평가(--eval_only)는 학습하지 않으므로 아니다.  꺼져 있으면 아래 어느 줄도
# 예전과 다르게 돌지 않는다.
#   MS_ON    스위치가 켜진 학습 런
#   MS_PEND  이어 돌리기이고 지금 결과 JSON 이 이 규칙의 연장 대상이다 (`pending`) -- 'converged' 런의 이어 돌리기 거부를
#            풀고, 시작 가드의 '이미 끝난 칸' 을 통과시키고, 구간 기록에 reason "milestone" 을 적고, 싣고 있는 best.pt 를
#            구간이 끝날 때까지 건드리지 않는다 (새 best 는 `_MP.STAGE`)
#   MS_LAST  이 런의 patience 판정에 쓰는 마지막 감쇠 step (아래 학습 루프 앞에서 정한다; None 이면 예전 규칙)
#   MS_REC   처음부터 도는 런의 규칙 기록 (결과 JSON 최상위 `milestone`; 이어 돌리기는 resumes[].milestone)
MS_ON, MS_PEND, MS_LAST, MS_REC, _MSJ = False, False, None, None, None
if not a.eval_only:
    try:
        import milestone_policy as _MP
    except ImportError:                     # 모듈이 없으면 규칙이 없다 (fail-closed)
        _MP = None
    if _MP is not None and _MP.enabled():
        MS_ON = True
        if a.resume:
            try:
                _MSJ = json.load(open(f"{a.results_dir}/pancoll_{a.model}_{a.dataset}{a.tag}.json"))
            except (OSError, ValueError):
                _MSJ = None
            MS_PEND = _MP.pending(_MSJ)

# **저장소 코드가 하나라도 import 되기 전에** CUDA 를 초기화해 장치를 고정한다.  그 뒤에
# 저장소가 CUDA_VISIBLE_DEVICES 를 바꿔도 이 프로세스의 장치는 움직이지 않는다.
# 이 줄이 없어서 PanTCR 이 원격 GPU 0(다른 사용자 카드)에 올라갔다.
# 아래 시작 가드보다 **앞**이어야 한다 -- 가드의 reconcile.classify() 가 레시피 대조
# 때문에 모델을 한 번 짓는데(make_tables._recipe_now), 그때 PanTCR-GF2/net.py:11 이
# 값을 '0' 으로 바꾼다.  2026-09-23 에 pin 을 가드 뒤에 두었다가 이 구멍을 찾았다.
import gpu_pin
gpu_pin.pin(lambda m: print(m, flush=True))   # `log` 는 런 폴더가 생긴 뒤에야 생긴다

# 돌기 전에 두 가지를 묻는다.  큐는 줄을 꺼내 **무조건** 실행하므로, 판정은 여기서
# 해야 한다.  워커 스크립트에 넣을 수는 없다 -- bash 는 실행 중인 파일을 바이트
# 오프셋으로 계속 읽어서, 돌고 있는 워커를 고치면 엉뚱한 줄을 실행하게 된다.
#
# (1) 표에서 제외된 방법인가.  PGCU 가 그렇다: `DownSamplingBlock` 이 블록당 1/4 로
#     줄이고 `NumberBlocks=3` 이라, 학습 크기 64x64 에서 어텐션 key 격자가 **1x1** 이
#     된다 -- softmax 가 상수라 논문의 기여인 전역 교차어텐션이 학습 내내 무력화됐다가
#     평가(256x256, 4x4)에서만 살아난다.  그 결과가 gf2 23.910 dB 다.  이 수를 싣는
#     것은 PGCU 에 불공정하고, 다시 돌려도 같은 수가 나온다.
# (2) 이미 표에 실릴 수 있는 칸인가.  판정은 reconcile.classify() 하나를 쓴다 --
#     같은 규칙을 복사해 두면 한 곳만 고치게 된다(이 저장소에서 세 번 겪었다).
if not (a.eval_only or a.force or a.tag):
    _tier = R.SPECS.get(a.model, {}).get("tier")
    if _tier == "excluded":
        raise SystemExit(f"[건너뜀] {a.model} 은 표에서 제외된 방법이다 (tier=excluded). "
                         f"다시 돌리려면 --force.")
    # The research queue's completion gate (reconcile / make_tables) is not part of the public release.

# 실행 중인 큐 워커를 멈추지 않고 정책을 바꾸기 위한 장치.  워커는 작업마다
# 파이썬을 새로 띄우므로, 이 파일을 쓰면 **다음 작업부터** 새 천장이 적용된다.
# 워커 스크립트를 제자리에서 고치는 것은 위험하다 -- bash 는 실행 중인 파일을
# 바이트 오프셋으로 계속 읽으므로, 돌고 있는 워커가 엉뚱한 줄을 실행하게 된다.
_ovr = Path((_GB + "/_logs/TIME_CEILING_HOURS"))
if _ovr.exists():
    a.max_hours = float(_ovr.read_text().strip())

# 같은 칸(= 같은 런 폴더)이 두 번 동시에 돌면 best.pt · log.txt · 결과 JSON 을 서로
# 덮는다.  큐는 경쟁 상태(재큐와 꺼내기 사이, ssh 실패)로 같은 줄을 두 번 줄 수 있으므로
# 마지막 방어선을 여기 둔다.  잠금은 프로세스가 살아 있는 동안만 유지되고, 죽으면
# 커널이 풀어 준다 -- 남은 잠금 파일이 다음 런을 막는 일은 없다.
import fcntl as _fcntl
_rundir = Path(a.out) / f"{a.model}_{a.dataset}{a.tag}"
_rundir.mkdir(parents=True, exist_ok=True)
_RUNLOCK = open(_rundir / ".running.lock", "w")
try:
    _fcntl.flock(_RUNLOCK, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(f"[건너뜀] {_rundir.name} 은 다른 프로세스가 이미 돌리고 있다 (중복 실행 방지).")

dev = torch.device(a.device); torch.manual_seed(a.seed)
# Python · numpy 난수도 처음부터 도는 런에서 씨드한다.  자기조건화 동전을 `random.random()` 으로 던지는
# 방법(PanDiT · U-Know-DiffPAN)은 이것 없이 같은 씨드의 두 런이 달랐다.  이어 돌리기는 아래에서 다시 정한다
# (exact: state.pt 의 상태를 되살림, warm: derived_seed).
random.seed(a.seed); np.random.seed(a.seed % (2 ** 32))
if a.vram_frac:
    # 할당자 상한은 (프로세스, 장치) 단위다.  모델을 올리기 전에 걸어야 의미가 있다.
    torch.cuda.set_per_process_memory_fraction(a.vram_frac, device=dev.index or 0)
    _tot = torch.cuda.get_device_properties(dev.index or 0).total_memory / 2 ** 30
    print(f"[vram] 이 프로세스 상한 {a.vram_frac * _tot:.1f} GiB / {_tot:.1f} GiB", flush=True)
nb = PC.SPEC[a.dataset][2]
model, spec = R.build(a.model, ms_ch=nb, ratio=4); model = model.to(dev)

# 저장소가 데이터셋마다 다른 학습률/스케줄/배치를 쓰면 그것도 레시피의 일부다.
# (예: CANConv 는 WV3 에서 lr 1e-3 / step 250, GF2·QB 에서 5e-4 / step 500.)
# 한 값으로 뭉뚱그리면 두 데이터셋 중 하나는 저자가 쓰지 않은 설정으로 학습된다.
_pd = spec.get("per_dataset", {}).get(a.dataset)
if _pd:
    spec = dict(spec)
    for _k, _v in _pd.items():
        spec[_k] = {**spec[_k], **_v} if isinstance(_v, dict) and isinstance(spec.get(_k), dict) else _v

batch = a.batch or spec["batch"]; epochs = a.epochs or spec.get("epochs", 500)
fwd = spec["forward"]

PC.SKIP = a.skip                     # lrms 이면 아무것도 바뀌지 않는다
PC.GT_GEO = a.gt_geo                 # ms 이면 아무것도 바뀌지 않는다 (train · valid 의 GT 만 바뀐다)
# in 팔 (lib/skipvar.py): 스펙이 lms = cat[S, 원래 lms] 를 요구한다.  --skip 없이는 S 가 없으니 멈춘다.
if spec.get("lms_stack"):
    if a.skip == "lrms":
        raise SystemExit(f"[거부] {a.model} 는 lms_stack 스펙이다 -- --skip S_PAN|S_MS 가 있어야 한다")
    PC.LMS_STACK = True
tr = PC.load(a.dataset, "train"); va = PC.load(a.dataset, "valid")
if a.test_limit:                     # 시험 전용 (위에서 --tag · --results_dir 를 확인했다)
    tr = {k: v[:a.test_limit] for k, v in tr.items()}
    va = {k: v[:a.test_limit] for k, v in va.items()}
# 데이터 효율 ablation (--train_frac): 학습 표본만 고정 부분집합으로 줄이고, 일정은 전체 표본 수로 정한다 (같은 iteration 예산).
# 1.0 이면 N_FULL == N 이라 예전과 같다.
N_FULL = tr["gt"].shape[0]
if a.train_frac < 1.0:
    _nsub = max(batch, int(round(a.train_frac * N_FULL)))
    _sub = torch.randperm(N_FULL, generator=torch.Generator().manual_seed(12345))[:_nsub].sort().values
    tr = {k: v[_sub] for k, v in tr.items()}
N = tr["gt"].shape[0]
spe = int(np.ceil(N_FULL / batch)); steps = epochs * spe
run = Path(a.out) / f"{a.model}_{a.dataset}{a.tag}"; run.mkdir(parents=True, exist_ok=True)
def log(m): print(m, flush=True); (run / "log.txt").open("a").write(m + "\n")

o = spec["opt"]
# `opt["kind"]` 가 **구속력 있다**.  예전에는 kind 를 읽지 않고 언제나 Adam 을 지어서,
# kind="adamw", weight_decay=0.01 인 PAN-Crafter 가 결합 L2 로 학습됐다 (가중치 85~91% 붕괴).
# make_opt 가 있으면 그것이 이기고, 모르는 kind 는 멈춘다 (`lib/harness.make_optimizer`).
opt = H.make_optimizer(spec, model)

# 저장소가 학습률 스케줄을 쓰면 그대로 재현한다.  에폭 단위로 step() 한다.
sched, _sc = None, spec.get("sched")
if _sc and _sc["kind"] == "step":
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=_sc["step_size"], gamma=_sc["gamma"])
elif _sc and _sc["kind"] == "multistep":
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(_sc["milestones"]),
                                                 gamma=_sc["gamma"])
elif _sc and _sc["kind"] == "cosine":
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_sc["T_max"],
                                                       eta_min=_sc.get("eta_min", 0.0))
if spec["loss"] == "mse":
    lossfn = torch.nn.MSELoss()
elif spec["loss"] == "l1_mean_over_2b":
    # MMMamba 의 손실.  utils/utils.py:51 이 reduction='mean' 이다 -- MDCUN utils.py:38 /
    # MSDDN utils.py:48 의 'sum' 과 다르다.  sum 으로 두면 기울기 노름이 0.32 대신
    # 5,250 이 되어 우리 기본 클립 10.0 에 매 스텝 걸린다(= 실질 학습률을 우리가 정하게 된다).
    _m = torch.nn.L1Loss(reduction="mean")
    lossfn = lambda a, b: _m(a, b) / (a.shape[0] * 2)
elif spec["loss"] == "l1_sum_over_2b":
    # MDCUN 의 손실.  mean 으로 바꾸면 유효 학습률이 C*H*W/2 배 작아진다.
    _s = torch.nn.L1Loss(reduction="sum")
    lossfn = lambda a, b: _s(a, b) / (a.shape[0] * 2)
else:
    lossfn = torch.nn.L1Loss()

# 머리줄은 런의 **종류**를 적는다 -- log.txt 는 런마다 이어 쓰므로, 나중에 한 파일에서
# "지금 가중치를 낸 런이 어디서 시작했는가"를 가려야 한다 (`lib/resume_state.py`).
# 처음부터 도는 학습은 예전 형식 그대로다 (예전 로그와 같은 규칙으로 읽힌다).
_hmark = ("[이어 돌리기 구간] " if a.resume else "[재평가] " if a.eval_only else "")
log(f"[{a.model}] {_hmark}{spec.get('venue','')} · {a.dataset} · params {R.n_params(model):,} "
    f"· batch {batch} · epochs {epochs} (= {steps:,} step, {spe}/epoch) · loss {spec['loss']} · opt {o}")
log(f"[data] train {N:,}" + (f" (--train_frac {a.train_frac:g} of {N_FULL:,}, 고정 부분집합)" if N != N_FULL else "") + f" · valid {va['gt'].shape[0]:,} · {nb}밴드 · 나눗수 {PC.SPEC[a.dataset][3]:g} "
    f"(PanCollection 공식 split, 우리 사전학습 코퍼스에 없음)")
log(f"[diff] {spec.get('diff','없음')}")
log(f"[opt] {type(opt).__name__}" + (" (스펙의 make_opt)" if "make_opt" in spec else f" (kind={o.get('kind')})"))
if spec.get("eval_tile"):
    log(f"[평가] {spec['eval_tile']}x{spec['eval_tile']} 타일 추론 -- 학습 크기보다 큰 평가 영상은 겹치지 "
        f"않는 타일로 나눠 넣고 이어 붙인다 (규약 {PC.eval_protocol(spec)})")

CLIP = float(spec.get("grad_clip", 10.0))   # 저장소가 값을 지정하면 그것을 쓴다
# `clip_grad_norm_` 에 실제로 넘기는 상한.  덮어쓰기가 없으면 CLIP 그대로다 (기존 런과 같다).
# 스펙 값 0 은 "걸지 않는다" (아래 덮어쓰기와 같은 뜻) -- 저장소가 클립하지 않는 방법 (MMNet).
# clip_grad_norm_(.., 0) 은 기울기를 0 으로 만들므로 상한을 inf 로 넘겨 노름만 잰다.  CLIP > 0 인 스펙은 예전과 같다.
CLIP_MAX = CLIP if CLIP > 0 else float("inf")
if CLIP == 0 and a.grad_clip_override is None:
    log("[클립] 스펙 grad_clip 0 -- 클립 없음 (저장소와 같다).  기울기 노름만 잰다")
if a.grad_clip_override is not None:
    # 0 은 "걸지 않는다" (저장소 solver 의 `if gclip > 0` 과 같은 뜻).  clip_grad_norm_(.., 0) 은 기울기를
    # 0 으로 만든다 -- 상한을 inf 로 넘겨 노름만 잰다 (계수가 1.0 으로 잘려 기울기는 그대로다).
    # 기록(레시피 스냅샷 · 결과 JSON 의 grad_clip)은 덮어쓴 값 그대로다 (0 = 클립 없음).
    CLIP = float(a.grad_clip_override)
    CLIP_MAX = CLIP if CLIP > 0 else float("inf")
    log(f"[클립] --grad_clip_override {CLIP:g}{' (클립 없음)' if CLIP == 0 else ''} -- 스펙 값 "
        f"{float(spec.get('grad_clip', 10.0)):g} 대신.  tag {a.tag} 민감도 런이다 (표의 칸이 아니다-4)")

# 이 결과가 **어떤 레시피로** 나왔는지 결과 파일이 스스로 들고 있게 한다.
# 레시피를 고치면(예: MDCUN 의 T, PanNet 의 고주파 정의, ARConv 의 스케줄러) 그
# 이전 결과는 조용히 낡는다.  사람이 기억해서 격리하는 방식은 실제로 실패했다 --
# 원격 15칸이 수정 전 설정으로 돌아간 것을 나중에야 발견했다.  make_tables.py 가
# 이 스냅샷을 현재 레지스트리와 대조해 낡은 행을 자동으로 뺀다.
#
# 학습 **전에** 만든다 (예전에는 끝에서 만들었다).  이어 돌리기가 state.pt · best.pt ·
# 결과 JSON 의 스냅샷을 **이것과** 대조하고, best.pt · state.pt 에도 같은 스냅샷을 넣는다.
# 값은 학습 중에 바뀌지 않는다 (params 는 모델 구조에서만 나온다).
_recipe = {k: spec.get(k) for k in
           ("batch", "epochs", "loss", "opt", "sched", "per_dataset",
            "aux_weights", "out_index", "aug", "val_cap", "eval_bs", "eval_after_frac")}
# 실제 적용된 값으로 기록한다.  spec.get("grad_clip") 은 지정 안 한 방법에서 None 이라,
# 기본값 10.0 을 채우는 비교 쪽과 어긋나 멀쩡한 칸이 전부 거부됐다.
_recipe["grad_clip"] = CLIP
_recipe["params"] = R.n_params(model)
# 평가 규약도 레시피의 일부다.  2026-09-20 에 RR 테두리 컷(dim_cut=30)을 도입했고,
# 그 이전 수치는 같은 가중치라도 다른 수를 낸다.  여기 박아 두면 make_tables 가
# 자동으로 걸러낸다 -- 사람이 기억하는 방식은 이미 두 번 실패했다.
# 문자열은 `pancoll.eval_protocol` 하나가 정한다 -- 게이트(make_tables._recipe_now)가 같은 함수로 대조한다.
# `eval_tile` 이 있는 스펙은 "+tile<N>" 이 붙는다 (UTeRM).
_recipe["eval_protocol"] = PC.eval_protocol(spec)
# 스펙의 **함수**들(build/forward/make_opt/train_step/on_*)에 대한 바이트코드 지문.
# 위 키들은 값만 비교하므로, 학습을 실제로 수행하는 함수를 고치면 이전 결과가 조용히
# 유효한 채로 남는다 -- DCPNet 의 `forward` 에서 `[0]` 을 뗀 수정이 딱 그랬다.
_recipe["code_digest"] = R.spec_code_digest(spec)
# global skip 변형 (2026-10-09).  lrms 런에는 키가 없다 -- 기존 칸의 스냅샷은 그대로다.
if a.skip != "lrms":
    _recipe["skip"] = a.skip
if a.gt_geo != "ms":
    _recipe["gt_geo"] = a.gt_geo
if a.train_frac != 1.0:
    _recipe["train_frac"] = a.train_frac
if spec.get("skipvar"):
    _recipe["skipvar"] = {k: v for k, v in spec["skipvar"].items() if k != "note"}

import resume_state as RS
# best.pt · state.pt 에 넣는 사본.  원시 자료형만 남긴다 -- `torch.load` 기본값
# (weights_only=True)으로 best.pt 를 읽는 도구(render_visual, redo_zeroshot)가 있다.
_RECIPE_N = RS.norm(_recipe)
HOST = socket.gethostname()
# 시험 런(--test_limit)이 낸 가중치 · 상태에 붙이는 표시.  데이터를 줄인 것은 레시피 키가 아니라서
# recipe_diff 로는 안 걸린다 (양쪽에 다 있는 키만 비교한다).  이어 돌리기는 이 값이 지금 런과 다르면
# 거부한다 -- 시험 가중치가 진짜 런으로 이어지지 않게, 진짜 가중치가 시험 런으로 섞이지 않게.
# 진짜 런에는 키 자체가 없다 (예전 best.pt · state.pt 와 같은 모양).
_TEST_MARK = {"test_limit": a.test_limit} if a.test_limit else {}


# 평가 배치 크기.  메모리를 많이 먹는 모델은 전수 평가에서 터진다 -- MDCUN 의
# blockNL 은 F.unfold(fs=15) 를 표본마다 만들어 8밴드 H=256 에서 1.4 GB/표본을 쓴다.
# 기본 64 로는 reduced_examples 20장이 한 배치로 들어가 28 GB 를 요구한다.
# 수치는 배치 크기와 무관하므로(표본별 지표를 평균할 뿐) 나누는 것은 공짜다.
EVAL_BS = int(spec.get("eval_bs", 64))


# 평가에서 실제로 쓰인 최소 배치.  자동 분할이 걸렸는지 결과 JSON 에 남긴다.
EVAL_BS_USED_MIN = EVAL_BS

# 평가 forward.  `eval_tile` 이 있는 스펙(UTeRM)은 학습 크기 타일로 나눠 추론하고 이어 붙인다.
# 없으면 `fwd` 그대로다.  학습 루프는 여전히 `fwd`(또는 train_step)를 쓴다 -- 학습은 바뀌지 않는다.
# 학습 중 검증(64x64 패치)은 타일과 크기가 같아 그대로 통과한다.
fwd_eval = H.eval_forward(spec, ratio=4)


@torch.no_grad()
def val(d, step=None, bs=None, full=False):
    bs = bs or EVAL_BS
    model.eval(); acc = {}
    # 느린 모델(표본별 루프 + 언롤링)은 valid 전수 평가가 예산을 잡아먹는다.  체크포인트
    # 선택에만 쓰는 값이므로 부분집합으로 충분하다.  표에 실리는 최종 수치(full=True,
    # reduced/full examples)는 언제나 전수다.
    n = d["gt"].shape[0]
    if step is not None and spec.get("val_cap"):
        n = min(n, int(spec["val_cap"]))

    def chunk(i, j):
        """[i, j) 행의 지표를 누적한다.  안 들어가면 반으로 쪼갠다.

        지표는 **표본별로** 계산해 평균하므로 배치 크기가 수치를 바꾸지 않는다
        (원장 1절 '평가' 항목).  따라서 쪼개는 것은 공짜이고, 공정성에 영향이 없다.

        왜 자동인가: `eval_bs` 를 손으로 정하던 때, 학습 패치(64x64)에서는 돌지만
        전해상도(256x256, FR 은 512x512)에서 터지는 방법이 있었다.  터지면 런이
        통째로 죽고 그 칸은 조용히 비어 있게 된다 -- WFANet 이 세 데이터셋에서
        그렇게 죽었다(전역 어텐션 점수 행렬 80 GiB).  거기서는 저장소 수식을
        SDPA 로 바꾸는 것이 옳은 답이었지만(patches/WFANet.patch, float64 에서
        2e-16 일치), 손으로 고른 상수에 기대는 구조 자체가 같은 사고를 부른다.
        """
        global EVAL_BS_USED_MIN
        oom = False
        try:
            lms, pan = d["lms"][i:j].to(dev), d["pan"][i:j].to(dev)
            gt, ms = d["gt"][i:j].to(dev), d["ms"][i:j].to(dev)
            pr = fwd_eval(model, lms, pan, ms,
                     {"step": step if step is not None else steps, "total": steps,
                      # `lpan`(MS 해상도 저역 PAN)은 PAN-Crafter 만 쓴다.  ctx 로 싣는
                      # 이유는 `fwd` 시그니처를 바꾸지 않기 위해서다 -- 다른 스펙들은
                      # 모르는 키를 그냥 무시한다.  데이터에 없으면 None 이다.
                      "lpan": d["lpan"][i:j].to(dev) if "lpan" in d else None})
            pr = (pr[0] if isinstance(pr, tuple) else pr).clamp(0, 1)
            if full:
                # PanCollection 의 RR 규약: 지표 전에 테두리 30픽셀을 잘라낸다
                # (256x256 -> 197x197).  공개 수치가 전부 이 규약이라, 안 자르면
                # 우리 표의 절대값이 문헌과 비교 불가능해진다.  pancoll.rr_cut 주석 참고.
                # `full=False` 인 64x64 검증 패치에는 적용하지 않는다.
                pr, gt = PC.rr_cut(pr), PC.rr_cut(gt)
            for k, v in PC.all_metrics(
                    pr, gt, divisor=(PC.SPEC[a.dataset][3] if full else None)).items():
                acc.setdefault(k, []).append(v.cpu())
        except torch.OutOfMemoryError:
            if j - i <= 1:
                raise                      # 한 장도 안 들어가면 진짜로 못 도는 것이다
            oom = True
        if oom:
            # 쪼갠 재시도는 **except 블록 밖에서** 한다 (2026-10-13).  블록 안에서 부르면 처리 중인 예외의
            # traceback 이 실패한 forward 의 프레임(중간 활성값)을 붙잡고 있어 empty_cache 가 아무것도 못 돌려준다
            # -- ARConv/qb S_PAN 이 일정을 다 돌고 RR 평가에서 쪼갤수록 점유가 늘어(22.7 -> 23.5 GiB) 한 장에서 죽었다.
            lms = pan = gt = ms = pr = None
            torch.cuda.empty_cache()
            mid = i + (j - i) // 2
            # **쪼갠 경우에만** 기록한다.  성공한 덩어리 크기를 그대로 적으면 집합
            # 끝의 자투리 배치(full_examples 는 20장이다)가 '자동 분할'로 둔갑한다.
            EVAL_BS_USED_MIN = min(EVAL_BS_USED_MIN, mid - i, j - mid)
            log(f"  [평가] {j-i}장이 안 들어감 -- {mid-i}/{j-mid} 로 쪼갬")
            chunk(i, mid); chunk(mid, j)

    for i in range(0, n, bs):
        chunk(i, min(i + bs, n))
    model.train()
    return {k: float(torch.cat(v).mean()) for k, v in acc.items()}




def _d4(g, *ts):
    """표본별 D4 (이면체군) 증강: 90도 회전 0~3회 + 축별 독립 뒤집기.

    저장소가 증강을 쓰면 그것도 레시피의 일부다.  우리가 빼면 그 베이스라인만
    데이터를 덜 보게 되어, 유리한 쪽이 우리가 된다 -- 그건 비교가 아니다.
    gt/lms/ms/pan 에 **같은** 변환을 걸어야 기하 정합이 깨지지 않는다.
    ms 는 1/4 크기지만 정사각이라 같은 연산이 성립한다."""
    outs = [t.clone() for t in ts]
    B = ts[0].shape[0]
    for i in range(B):
        k  = int(torch.randint(0, 4, (1,), generator=g))
        vf = int(torch.randint(0, 2, (1,), generator=g))
        hf = int(torch.randint(0, 2, (1,), generator=g))
        for o in outs:
            v = o[i]
            if k:  v = torch.rot90(v, k, dims=(-2, -1))
            if vf: v = torch.flip(v, dims=(-1,))
            if hf: v = torch.flip(v, dims=(-2,))
            o[i] = v
    return outs


def _plateau(h):
    """평탄화 증명서.  전역 최고 valid PSNR 이 마지막 25% 평가 구간 안에 있으면
    학습을 일찍 끊지 않았다는 뜻이다.  `plateau_gap_db` 가 0 보다 크면 그만큼
    더 나은 지점이 앞쪽에 있었다는 것이고, 그건 오히려 과적합 신호다.
    `still_improving` 이 True 면 상한 때문에 잘린 것이므로 그 칸은 재실행해야 한다."""
    if len(h) < 8:
        return {"plateau_n_evals": len(h), "still_improving": None}
    ps = [p for _, p in h]
    k = max(2, len(ps) // 4)
    tail = ps[-k:]
    gap = max(ps) - max(tail)
    # 마지막 구간 전반부 평균 대비 후반부 평균이 0.05 dB 이상 오르면 아직 오르는 중이다
    half = len(tail) // 2
    rising = (sum(tail[half:]) / len(tail[half:])) - (sum(tail[:half]) / max(1, len(tail[:half])))
    return {"plateau_n_evals": len(h), "plateau_gap_db": round(gap, 4),
            "plateau_tail_rise_db": round(rising, 4),
            "still_improving": bool(rising > 0.05)}


# 정지 규칙: 벽시계가 아니라 수렴.
#
# 5시간 상한은 저장소마다 공식 일정 길이가 100배 넘게 차이 나기 때문에 방법을
# 가려 버린다 -- LAGConv 은 73% 를 돌고 MDCUN 은 10% 를 돌았다.  그건 방법의
# 차이가 아니라 시계의 차이다.
#
# 대신 valid 최고 기록이 오래 갱신되지 않으면 멈춘다.  **우리는 valid 기준 best
# 체크포인트를 보고하므로, 이 규칙은 보고 수치를 낮출 수 없다** -- 이미 나온 최고를
# 그대로 들고 있고, 새 최고가 안 나오는 구간의 계산만 아낀다.  인내 구간은 일정
# 길이에 비례시키되 위로 잘라, 1,900만 step 짜리 일정(S2DBPN)이 인내만으로 며칠을
# 먹지 않게 한다.
PATIENCE = a.patience or max(20 * a.eval_every, min(int(0.10 * steps), 200_000))
log(f"[정지] 수렴 정지: valid 최고가 {PATIENCE:,} step 동안 안 갱신되면 중단 "
    f"(안전 천장 {a.max_hours}h)")

g = torch.Generator().manual_seed(a.seed)
gnorm_max, gnorm_hits = 0.0, 0
best, best_at, t0 = -1e9, 0, time.time()
STOP = {"why": "schedule_complete"}
vhist = []                      # (step, psnr) -- 평탄화 판정용
# 저장소가 표본 인덱스에 묶인 상태(예: CANConv 의 k-means 군집 캐시)를 들고 다니면
# 그 상태의 수명 주기도 레시피다.  캐시를 끄면 CANConv 는 11 배 느려져 같은 5시간
# 상한에서 일정의 5% 만 돌게 된다 -- 그건 우리가 그 방법을 불리하게 만든 것이다.
if "on_train_start" in spec:
    spec["on_train_start"](model, N, dev)

STATE = run / "state.pt"
BEST = run / "best.pt"
#: 결과 JSON 과 짝인 가중치 (표 · 체크포인트 · 그림 · zero-shot 이 읽는다).  보통은 새 best 도 여기(BEST)에 쓴다.
BEST_PUB = BEST
if MS_PEND:
    # 연장 구간: 지금 결과는 게이트에서 유효(표에 실림)이고 연장이 끝날 때까지 그대로 싣는다 -- 그 결과와 짝인 best.pt 를
    # 이 구간이 결과 JSON 을 쓸 때까지 건드리지 않는다.  새 best 는 옆 파일에 쓰고, 결과 JSON 을 쓴 **직후** best.pt 로
    # 옮긴다 (검토 2026-09-25: best.pt 를 결과 JSON 과 짝으로 읽는 곳 -- make_checkpoints · render_visual · zero-shot QA 의
    # sha -- 이 연장 중에도 맞게).
    BEST = run / _MP.STAGE
    if BEST.exists():
        # 앞 연장 구간이 결과를 못 쓰고 죽었다 -- 그 궤적은 버린다 (exact 는 state.pt 의 best 사본으로 아래에서 되살린다).
        os.replace(BEST, run / (_MP.STAGE + ".stale"))
        log(f"[이어 돌리기] 앞 연장 구간의 {_MP.STAGE} 는 결과 없이 끝난 궤적이다 -- {_MP.STAGE}.stale 로 치웠다")


def _save_best(step, m):
    """best.pt 를 쓴다.  레시피 스냅샷 · seed · tag 를 같이 넣는다 -- 나중에 이 가중치만
    남았을 때(warm 이어 돌리기) 어떤 레시피의 것인지 파일 스스로 답하게 한다.
    임시 파일 + 이름 바꾸기로 쓴다: 쓰다 죽으면 반쯤 쓴 best.pt 가 남는다."""
    RS.atomic_save({"model": model.state_dict(), "step": step, "valid": m,
                    "recipe": _RECIPE_N, "seed": a.seed, "tag": a.tag, **_TEST_MARK}, BEST)


# 이어 돌리기의 앞 구간 기록.  처음부터 도는 런이면 전부 비어 있다.
START = 0                 # 이 구간이 출발하는 step (가중치가 이 step 의 것)
PRIOR_CUM_H = 0.0         # 앞 구간들의 벽시계 합
PRIOR_RESUMES = []        # 앞 구간들의 이어 돌리기 기록
CUR_RESUME = None         # 이 구간의 기록 (segment_hours 는 끝날 때 채운다)
GRAD_NOTE = None
# 칸당 총 학습 상한 (사용자 결정 2026-09-24 14:10).  이 구간의 천장은
# min(--max_hours, TOTAL_CAP_H - PRIOR_CUM_H) 다 (`resume_state.segment_budget`, 아래 학습 루프 앞에서 정한다).
SEG_CEIL_H = a.max_hours  # 이 구간의 실제 천장 (시간)
CAP_BINDS = False         # 총 상한이 구간 천장보다 먼저 걸린다 -- 천장에서 멈추면 total_cap_reached
CAP_EXACT = True          # 앞 구간 누적 시간을 안다 (모르면 0 으로 두고 결과에 total_cap_exact=false)
# NaN 가드 (스펙 `nan_guard` 선택 B · `lib/nan_guard.py`).  켠 스펙만 학습 루프 앞에서 만든다 --
# 끈 스펙(나머지 전부)은 아래 어느 경로에서도 가드 코드를 거치지 않는다.
NG = None
_NG_LOAD = None           # exact 이어 돌리기: state.pt 의 가드 상태 (셈 · 창을 그대로 잇는다)
_NG_PRIOR = None          # warm 이어 돌리기: 앞 구간 결과 JSON 의 가드 기록 (그대로 싣는다)


def _save_state(step, stop=None):
    """학습 상태 전부를 state.pt 에 원자적으로 쓴다.  실패해도 학습은 계속한다.

    `best_ckpt` 로 best.pt 의 사본도 넣는다.  주기 저장 뒤에 best.pt 가 더 갱신되고
    프로세스가 죽으면, 이어 돌리기는 state 의 step 부터 다시 가므로 그 뒤의 best.pt 는
    **버려질 궤적**의 것이다.  exact 모드는 그 사본으로 best.pt 를 되돌려 곡선(vhist)과
    가중치를 맞춘다."""
    seg_h = (time.time() - t0) / 3600
    try:
        _bs = BEST if BEST.exists() else BEST_PUB      # 연장 구간에 아직 새 best 가 없으면 싣고 있는 것 (보통은 같은 파일)
        bk = torch.load(_bs, map_location="cpu", weights_only=False) if _bs.exists() else None
        st = {"model": model.state_dict(), "optimizer": opt.state_dict(),
              "scheduler": sched.state_dict() if sched is not None else None,
              "step": int(step),
              "rng": {"sample": g.get_state(), "cpu": torch.get_rng_state(),
                      "cuda": (torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None),
                      "py": random.getstate(), "np": np.random.get_state()},
              "best": float(best), "best_at": int(best_at), "vhist": [list(x) for x in vhist],
              "gnorm_max": float(gnorm_max), "gnorm_hits": int(gnorm_hits),
              "wall_hours": PRIOR_CUM_H + seg_h, "segment_hours": seg_h,
              "resumes": PRIOR_RESUMES + ([dict(CUR_RESUME, segment_hours=seg_h)]
                                          if CUR_RESUME else []),
              "grad_stats_note": GRAD_NOTE,
              # 변형 런의 본 칸 step 맞춤 (2026-10-09).  이어 돌리기 줄에 인자가 빠져도 exact 모드가 되살린다.
              "stop_after_steps": int(a.stop_after_steps or 0),
              # 총 상한.  재평가 복구(결과 JSON 이 없는 칸)가 이 값을 되살린다.
              "total_cap_h": RS.TOTAL_CAP_H,
              "total_cap_reached": bool(CAP_BINDS and stop == "time_ceiling"),
              "total_cap_exact": CAP_EXACT,
              "recipe": _RECIPE_N, "seed": a.seed, "tag": a.tag, "steps": steps, "spe": spe,
              "PATIENCE": PATIENCE, "eval_every": a.eval_every, **_TEST_MARK,
              # 레시피가 멈춘 런(converged / schedule_complete)의 마지막 상태는 **이어 돌리지
              # 않는다**는 표시다.  천장 · 시험 정지 · 주기 저장(None)만 이어 돈다.
              "stop_reason": stop, "best_ckpt": bk, "host": HOST,
              # 가드를 켠 런만 (끈 런의 state.pt 는 예전 모양 그대로다).
              **({"nan_guard": NG.state()} if NG is not None else {}),
              **({"milestone": MS_REC} if MS_REC else {}),       # 적용한 처음부터 도는 런만
              "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        RS.atomic_save(st, STATE)
        return True
    except Exception as e:                     # 상태 저장 실패로 학습을 죽이지 않는다
        log(f"[state] 저장 실패 (step {step:,}): {type(e).__name__}: {str(e)[:160]}")
        return False


if not (a.resume or a.eval_only) and STATE.exists():
    # 처음부터 도는 런이다.  남아 있는 state.pt 는 **다른 런**의 것이고, 그대로 두면 나중의
    # `--resume` 이 그 상태를 이 런의 것으로 알고 이어 붙인다.  지우지 않고 옆으로 치운다.
    os.replace(STATE, run / "state.pt.stale")
    log("[state] 처음부터 도는 런이다 -- 예전 state.pt 를 state.pt.stale 로 치웠다 "
        "(이어 돌리기가 다른 런의 상태를 집지 않게)")

PRIOR = {}
if a.eval_only:
    if not (run / "best.pt").exists():
        raise SystemExit(f"[eval_only] {run}/best.pt 가 없다 -- 되살릴 학습이 없다")
    _pj = f"{a.results_dir}/pancoll_{a.model}_{a.dataset}{a.tag}.json"
    if os.path.exists(_pj):
        PRIOR = json.load(open(_pj))
    if "plateau_n_evals" not in PRIOR and (run / "log.txt").exists():
        # 결과 JSON 이 없다 -- 학습은 끝났는데 **그 뒤 평가 단계에서 죽은** 런이다.
        # (FusionMamba/qb 가 그랬다: 4.1 시간을 돌려 RR/FR 수치까지 찍고, 마지막
        # 64x64 valid 에서 크기 상태 버그로 죽어 JSON 이 안 써졌다.)
        #
        # 그냥 `--eval_only` 로 되살리면 평탄성 근거(`plateau_*`, `still_improving`)와
        # 학습 시간이 사라진다.  그 둘은 공정성 원장이 "조기 종료 없음"을 주장하는
        # 근거다.  없애고 싶지 않고, 4 시간을 다시 쓰고 싶지도 않다.
        #
        # 그런데 그 증거는 우리가 쓴 로그에 그대로 있다.  형식은 우리가 정한 것이고
        # (`  [valid] step {step:7d}  PSNR {psnr:.3f}`), 평탄 판정은 아래에서 원래와
        # **같은 `_plateau()`** 에 같은 점들을 넣어 구한다 -- 다시 학습해도 나올 수는
        # 같다.  복구했다는 사실은 `metadata_recovered_from_log` 로 표시한다.
        #
        # **파일 전체가 아니라 best.pt 를 낸 사슬만 읽는다** (2026-09-24).  log.txt 는 런마다
        # 이어 쓰므로 예전에는 서로 무관한 런들의 valid 점이 한 곡선으로 섞였고, 정지 줄도
        # 아무 런의 것이나 잡혔다.  사슬 해석은 이어 돌리기와 같은 함수다 (`resume_state`).
        _txt = (run / "log.txt").read_text(errors="replace")
        try:
            _ck0 = torch.load(BEST, map_location="cpu", weights_only=False)
        except Exception:
            _ck0 = {}
        _ch = (RS.find_chain(_txt, step=_ck0.get("step"),
                             psnr=(_ck0.get("valid") or {}).get("psnr"))
               or RS.find_chain(_txt))
        _vh = _ch["vhist"] if _ch else []
        if _vh:
            # 정지 사유는 로그의 `[정지]` 줄이 말해 준다.  그 줄이 **없는데** 일정도
            # 안 끝났으면 밖에서 끊긴 것이다(우리가 중단했거나 죽었거나).  그것을
            # "일정 완주"로 적으면 원장이 거짓말을 하게 된다 -- 그런 런은 천장에
            # 잘린 것과 같은 처지이므로 같은 판정(평탄하면 싣고 아니면 재실행)을 받는다.
            # 로그 맨 위의 규칙 설명("[정지] 수렴 정지: valid 최고가 ... 안 갱신되면
            # 중단")과 **실제로 멈췄다는 줄**("[정지] 수렴 -- step N 이후 ...")을
            # 구별해야 한다.  전자를 잡으면 일정을 완주한 런이 '수렴 정지'로 기록된다
            # -- FusionMamba/gf2 가 ep 500/500 을 다 돌고도 그렇게 적혔다.
            # (그 구별은 이제 `resume_state.segments` 가 한다.)
            _why = _ch["stop_reason"] or "schedule_complete"
            PRIOR = dict(PRIOR)
            PRIOR.update(_plateau(_vh))
            PRIOR.setdefault("stop_reason", _why)
            PRIOR.setdefault("hit_time_ceiling", _why in ("time_ceiling", "interrupted"))
            if _ch["last_step"]:
                PRIOR.setdefault("last_step", int(_ch["last_step"]))
            # 구간 기록 (검토, 2026-09-24).  예전에는 여기서 resumes · host 가 사라지고
            # segment_wall_hours 가 재평가 시간이 됐다 -- 표 주석과 원장의 'Adam 상태 초기화' 공개가
            # 빠진다.  state.pt 가 이 사슬의 끝 상태면(학습 루프가 끝날 때 최종 평가 **전에** 쓴다)
            # 그것이 가장 자세하다: 구간 기록 · 학습 기계 · 기울기 통계 · 이 구간 시간이 그대로 있다.
            # 없으면 로그 사슬에서 되살린다 (이어 돌리기 구간의 모드 · 출발 step 뿐이다).
            try:
                _s0 = (torch.load(STATE, map_location="cpu", weights_only=False)
                       if STATE.exists() else None)
            except Exception:
                _s0 = None
            if (_s0 and _s0.get("seed") == a.seed and (_s0.get("tag") or "") == a.tag
                    and int(_s0.get("step") or -1) >= int(_ch["last_step"] or 0)):
                for _k, _sk in (("resumes", "resumes"), ("host", "host"),
                                ("segment_wall_hours", "segment_hours"),
                                ("grad_norm_max", "gnorm_max"), ("grad_clip_hits", "gnorm_hits"),
                                ("grad_stats_note", "grad_stats_note"), ("wall_hours", "wall_hours"),
                                ("total_cap_h", "total_cap_h"), ("total_cap_reached", "total_cap_reached"),
                                ("total_cap_exact", "total_cap_exact"), ("nan_guard", "nan_guard"),
                                ("milestone", "milestone")):
                    if _s0.get(_sk) is not None:
                        PRIOR.setdefault(_k, _s0[_sk])
                if PRIOR.get("resumes") == []:
                    PRIOR.pop("resumes")
                if PRIOR.get("total_cap_exact") is True:
                    PRIOR.pop("total_cap_exact")        # 결과 JSON 은 모를 때만 적는다 (아래 _RES)
            _rs = [{"mode": _sg["resume_mode"], "from_step": _sg["start"], "source": "log"}
                   for _sg in _ch["segments"] if _sg["kind"] == "resume"]
            if _rs:
                PRIOR.setdefault("resumes", _rs)
            _ls = _ch["segments"][-1]
            if _ls["last_step"] is not None and _ls["sps"] is not None:
                PRIOR.setdefault("segment_wall_hours",
                                 (_ls["last_step"] - _ls["start"]) * _ls["sps"] / 3600)
            if _ch["wall_hours"] is not None:
                # 로그의 `s/step` 은 그 구간 시작부터의 평균이므로 (마지막 step - 출발 step)
                # x 마지막 속도가 그 구간의 벽시계다.  유도값이라는 것을 표시와 함께 남긴다.
                PRIOR.setdefault("wall_hours", _ch["wall_hours"])
            if _ch.get("total_cap") and _why == "time_ceiling":
                # 마지막 구간의 `[정지]` 줄에 총 상한 표식이 있다 -- state.pt 가 없어도 되살린다.
                PRIOR.setdefault("total_cap_reached", True)
                PRIOR.setdefault("total_cap_h", RS.TOTAL_CAP_H)
            PRIOR["metadata_recovered_from_log"] = True
            log(f"[eval_only] 결과 JSON 이 없어 {run}/log.txt 에서 학습 기록을 복구했다: "
                f"평가점 {len(_vh)}개, 정지 '{_why}', "
                f"마지막 구간 {PRIOR.get('plateau_tail_rise_db', 0):+.3f} dB, "
                f"학습 {PRIOR.get('wall_hours', 0):.2f}h")
    if ("segment_wall_hours" not in PRIOR and not PRIOR.get("resumes")
            and PRIOR.get("wall_hours") is not None):
        PRIOR["segment_wall_hours"] = PRIOR["wall_hours"]      # 한 구간짜리 학습 -- 둘이 같다
    log(f"[eval_only] 학습 건너뜀.  {run}/best.pt 로 평가만 다시 낸다 "
        f"(원래 학습 {PRIOR.get('wall_hours', 0):.2f}h, {PRIOR.get('best_step', '?')} step)")

if a.resume:
    # ------------------------------------------------------------------ 이어 돌리기
    # 사용자 결정 (2026-09-24): 48h 천장에 잘렸는데 아직 오르는
    # 칸은 **처음부터가 아니라 이어서** 계속 돌린다.  처음부터 다시 돌면 첫 평가에서
    # best.pt 를 더 나쁜 가중치로 덮고, 일정이 천장보다 긴 방법은 같은 자리에서 또 잘린다.
    #
    # 두 모드가 있다.
    #   exact  state.pt 가 이 런의 것이다 (레시피 · seed · tag · 일정 · 평가 간격이 같다).
    #          가중치 · Adam 모멘트 · 스케줄러 · 표본 추출 난수 · 전역 난수 · best/patience ·
    #          곡선을 전부 되살린다.  CPU 에서는 끊김 없는 학습과 **비트 단위로** 같은
    #          가중치가 나온다 (tests/test_resume.py).  못 되살리는 것:
    #            - 천장 정지는 그 step 에서 평가를 한 번 더 한다(원래 규칙).  끊김 없는 런에는
    #              그 점이 없다 -- 곡선에 점 하나가 더 있고, 평가 중 난수를 쓰는 방법
    #              (PanFlowNet 의 표본 잡음)은 그만큼 난수 흐름이 달라진다.
    #            - CANConv 의 k-means 군집 캐시(GPU 852 MB)는 넣지 않는다.  캐시를 비우고
    #              일정(on_epoch)만 다시 밟는다 -- 처음 보는 표본을 만나면 그 스텝에서 다시
    #              군집한다 (`lib/specs/canconv.py`).  예전 배정 대신 지금 가중치의 배정을 쓴다.
    #            - GPU 커널의 비결정성은 원래도 같은 씨드 두 런을 비트 단위로 같게 만들지 않는다.
    #   warm   state.pt 가 없거나 이 런의 것이 아니다.  best.pt 의 가중치에서 출발하고
    #          (그 step 보다 뒤는 버린다), 옵티마이저는 **새로 만든다** -- Adam 모멘트가
    #          0 에서 다시 쌓인다.  표본 순서는 (seed, 출발 step) 에서 결정적으로 다시 뽑는다.
    #          둘 다 결과 JSON 의 resumes[] 에 남고 표 주석에 적힌다.
    # 어느 쪽이든 레시피를 확인할 수 있는 만큼 확인하고, 다른 것이 보이면 **거부한다.**
    import recipe_check as _MT            # 레시피 대조 규칙 (lib/recipe_check.py)
    _AT = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    _REASON = a.resume_reason or "(사유 미기재)"

    _PJ = f"{a.results_dir}/pancoll_{a.model}_{a.dataset}{a.tag}.json"

    def _refuse(msg):
        # 거부를 런 폴더에 **기록**한다.  결과 JSON 은 그대로라 칸은 여전히 천장-상승이고, 기록이
        # 없으면 reconcile 이 15분마다 같은 줄을 다시 넣는다 (검토, 2026-09-24).  reconcile · 디스패처 ·
        # 현황판이 이 파일을 읽고 [이상] 으로 사람을 부른다 (`resume_state.block_reason`).
        RS.record_refusal(str(run), _PJ, msg, source="train_pancoll")
        log(f"[이어 돌리기 거부] {msg}")
        raise SystemExit(f"[이어 돌리기 거부] {a.model}/{a.dataset}{a.tag}: {msg}")

    def _check_test_mark(ck, what):
        if int(ck.get("test_limit") or 0) != int(a.test_limit or 0):
            _refuse(f"{what} 는 --test_limit {ck.get('test_limit') or 0} 로 만든 것인데 지금은 "
                    f"{a.test_limit or 0} 이다 -- 시험 런과 진짜 런을 잇지 않는다")

    def _ms_snapshot(pre):
        """사슬의 첫 연장 전 결과 JSON · best.pt 를 `pre` 에 남기고 기록(dict)을 돌려준다.  이미 있으면 그대로 읽는다
        (앞 시도가 남긴 것 -- 결과 JSON 이 안 바뀌었으니 같은 것이다).  실패해도 학습은 막지 않는다 (기록에 error)."""
        import hashlib, shutil
        man = pre / "manifest.json"
        if man.exists():
            try:
                return dict(json.load(open(man)), reused=True)
            except ValueError:
                pass

        def _sha(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            return h.hexdigest()
        try:
            pre.mkdir(exist_ok=True)
            rec = {"dir": str(pre), "best_step": _MSJ.get("best_step"), "last_step": _MSJ.get("last_step"),
                   "stop_reason": _MSJ.get("stop_reason"), "rr_psnr": (_MSJ.get("rr") or {}).get("psnr"),
                   "hqnr": (_MSJ.get("fr") or {}).get("hqnr"),
                   "best_pt_step": torch.load(BEST_PUB, map_location="cpu", weights_only=False).get("step")}
            for src, key in ((Path(_PJ), "result"), (BEST_PUB, "best")):
                dst = pre / src.name
                shutil.copy2(src, f"{dst}.tmp")
                os.replace(f"{dst}.tmp", dst)
                rec[f"{key}_file"], rec[f"{key}_sha256"] = src.name, _sha(dst)
            with open(f"{man}.tmp", "w") as f:
                json.dump(rec, f, indent=1)
            os.replace(f"{man}.tmp", man)
            log(f"[이어 돌리기] 연장 전 결과를 남겼다 -> {pre} (best step {rec['best_step']}, best.pt sha256 "
                f"{rec['best_sha256'][:12]})")
            return rec
        except Exception as _e:                            # noqa: BLE001
            log(f"[이어 돌리기] 연장 전 결과를 남기지 못했다 ({type(_e).__name__}: {str(_e)[:120]}) -- 그대로 간다")
            return {"error": f"{type(_e).__name__}: {str(_e)[:200]}"}

    def _replay_epochs(upto):
        """on_epoch 훅을 1..(이미 시작한 에폭) 까지 다시 밟는다 (CANConv 의 캐시 일정)."""
        if "on_epoch" in spec:
            for _ep in range(1, (upto + spe - 1) // spe + 1):
                spec["on_epoch"](model, _ep)

    _log_txt = (run / "log.txt").read_text(errors="replace") if (run / "log.txt").exists() else ""
    _st, _why_not = None, None
    if STATE.exists():
        try:
            _st = torch.load(STATE, map_location="cpu", weights_only=False)
        except Exception as _e:
            _why_not = f"state.pt 를 읽지 못했다 ({type(_e).__name__}: {str(_e)[:100]})"
        if _st is not None:
            # 평가 전용 키(`eval_protocol`)만 다르면 학습 상태는 그대로 잇는다 -- 이 구간의 최종 평가가 지금
            # 규약으로 나온다 (make_tables.EVAL_ONLY_KEYS 10-01).
            _rd = _MT.recipe_diff_train(_st["recipe"], _recipe) if _st.get("recipe") else ["recipe 없음"]
            _rd_eval = sorted(set(_MT.recipe_diff(_st["recipe"], _recipe)) - set(_rd)) if _st.get("recipe") else []
            if _rd_eval and not _rd:
                log(f"[이어 돌리기] state.pt 와 평가 전용 키만 다르다 {_rd_eval} -- 학습 상태는 그대로 잇고 "
                    f"최종 평가는 지금 규약({_recipe.get('eval_protocol')})으로 낸다")
            if _rd:
                _why_not = f"레시피 불일치 {_rd}"
            elif _st.get("seed") != a.seed or (_st.get("tag") or "") != a.tag:
                _why_not = f"seed/tag 불일치 ({_st.get('seed')}{_st.get('tag')} != {a.seed}{a.tag})"
            elif _st.get("steps") != steps or _st.get("spe") != spe:
                _why_not = f"일정 불일치 ({_st.get('steps')}/{_st.get('spe')} != {steps}/{spe})"
            elif _st.get("eval_every") != a.eval_every or _st.get("PATIENCE") != PATIENCE:
                _why_not = (f"평가 간격 · patience 불일치 ({_st.get('eval_every')}/"
                            f"{_st.get('PATIENCE')} != {a.eval_every}/{PATIENCE})")
            if _why_not:
                _st = None
        if _why_not:
            log(f"[이어 돌리기] state.pt 를 쓰지 않는다: {_why_not} -- best.pt 에서 warm 으로 간다")

    if _st is not None:
        # ---------------------------------------------------------------- exact
        if _st.get("stop_reason") in ("converged", "schedule_complete") and not (
                #: 마지막 lr 감쇠 전의 patience 정지는 잇는다 -- 결과 JSON 이 **이 상태의** 정지를 적고 있을 때만
                MS_PEND and _st.get("stop_reason") == "converged" and _MSJ.get("stop_reason") == "converged"
                and _MSJ.get("last_step") == _st.get("step")):
            _refuse(f"state.pt 는 '{_st['stop_reason']}' 로 끝난 런의 마지막 상태다 -- 레시피가 "
                    f"스스로 멈춘 런은 이어 돌리지 않는다")
        if _st.get("stop_reason") == "nan_guard_exhausted":
            _refuse("state.pt 는 NaN 가드가 소진돼 멈춘 런의 마지막 상태다 -- 사전 등록 대체안 C: "
                    "best.pt 를 싣고 이어 돌리지 않는다 (같은 궤적을 다시 밟는다)")
        _check_test_mark(_st, "state.pt")
        _NG_LOAD = _st.get("nan_guard")
        START = int(_st["step"])
        if not a.stop_after_steps and int(_st.get("stop_after_steps") or 0):
            a.stop_after_steps = int(_st["stop_after_steps"])
            log(f"[이어 돌리기] state.pt 의 --stop_after_steps {a.stop_after_steps:,} 를 잇는다 (변형 런의 본 칸 step 맞춤)")
        if START >= steps:
            _refuse(f"state.pt 가 이미 일정 끝(step {START:,}/{steps:,})이다")
        model.load_state_dict(_st["model"])
        opt.load_state_dict(_st["optimizer"])
        if (sched is None) != (_st.get("scheduler") is None):
            _refuse("스케줄러 유무가 state.pt 와 다르다")
        if sched is not None:
            sched.load_state_dict(_st["scheduler"])
        _r = _st["rng"]
        g.set_state(_r["sample"])
        torch.set_rng_state(_r["cpu"])
        if dev.type == "cuda" and _r.get("cuda") is not None:
            torch.cuda.set_rng_state(_r["cuda"], dev)
        random.setstate(_r["py"])
        np.random.set_state(_r["np"])
        best, best_at = float(_st["best"]), int(_st["best_at"])
        vhist = [tuple(x) for x in _st["vhist"]]
        gnorm_max, gnorm_hits = float(_st["gnorm_max"]), int(_st["gnorm_hits"])
        PRIOR_CUM_H = float(_st["wall_hours"])
        PRIOR_RESUMES = list(_st.get("resumes") or [])
        GRAD_NOTE = _st.get("grad_stats_note")
        # 주기 저장 뒤에 best.pt 가 더 갱신됐다면 그것은 버리는 궤적의 가중치다.
        _bk = _st.get("best_ckpt")
        if _bk is not None:
            _bs = BEST if BEST.exists() else BEST_PUB  # 연장 구간: 되돌리는 곳은 옆 파일이다 (싣고 있는 best.pt 는 그대로)
            _cur = torch.load(_bs, map_location="cpu", weights_only=False) if _bs.exists() else {}
            if _cur.get("step") != _bk.get("step"):
                RS.atomic_save(_bk, BEST)
                log(f"[이어 돌리기] best.pt(step {_cur.get('step')}) 는 state 이후의 버리는 궤적이다 "
                    f"-- state 에 든 사본(step {_bk.get('step')})으로 되돌렸다")
        # 누적 벽시계.  state 는 최종 평가 **전에** 저장되므로 그 구간의 최종 평가 시간이 빠져
        # 있다.  같은 정지를 기록한 결과 JSON(끝난 step · 정지 사유 · seed/tag 가 같다)이 있으면
        # 그 값을 쓴다 -- 처음부터 돈 런의 wall_hours 도 최종 평가까지 센 값이다.
        _wsrc = "state"
        _pj = Path(a.results_dir) / f"pancoll_{a.model}_{a.dataset}{a.tag}.json"
        try:
            _J = json.load(open(_pj)) if _pj.exists() else None
        except ValueError:
            _J = None
        if (_J and _J.get("last_step") == START and _J.get("stop_reason") == _st.get("stop_reason")
                and _J.get("seed") in (None, a.seed) and _J.get("tag") in (None, a.tag)
                and _J.get("wall_hours") is not None):
            PRIOR_CUM_H = float(_J["wall_hours"])
            _wsrc = "json"
        # 앞 구간이 누적 시간을 몰랐으면(warm, 출처 unknown) 그 뒤의 누적도 정확하지 않다.
        if (_J if _wsrc == "json" else _st).get("total_cap_exact") is False:
            CAP_EXACT = False
        _ch = RS.find_chain(_log_txt) if _log_txt else None
        _prior_last = max(START, int((_ch or {}).get("last_step") or 0))
        _replay_epochs(START)
        CUR_RESUME = {"mode": "exact", "from_step": START, "prior_last_step": _prior_last,
                      "optimizer": "restored", "rng": "restored", "recipe_verified": "state",
                      "prior_wall_hours_source": _wsrc,
                      "at": _AT, "host": HOST, "segment_hours": None, "reason": _REASON}
    elif BEST_PUB.exists():
        # ---------------------------------------------------------------- warm
        _ck0 = torch.load(BEST_PUB, map_location="cpu", weights_only=False)
        _check_test_mark(_ck0, "best.pt")
        START = int(_ck0["step"])
        _bpsnr = float((_ck0.get("valid") or {}).get("psnr", float("nan")))
        if START >= steps:
            _refuse(f"best.pt 가 이미 일정 끝(step {START:,}/{steps:,})이다")
        _verified = []
        # (1) best.pt 스스로의 스냅샷 (2026-09-24 이후의 best.pt 에만 있다)
        if _ck0.get("recipe") is not None:
            _rd = _MT.recipe_diff_train(_ck0["recipe"], _recipe)      # 평가 전용 키는 빼고 (위 state.pt 와 같은 규칙)
            if _rd:
                _refuse(f"best.pt 의 레시피가 지금과 다르다: {_rd}")
            _verified.append("best.pt")
        if _ck0.get("seed") is not None and _ck0["seed"] != a.seed:
            _refuse(f"best.pt 의 seed {_ck0['seed']} != {a.seed}")
        if _ck0.get("tag") is not None and _ck0["tag"] != a.tag:
            _refuse(f"best.pt 의 tag {_ck0['tag']!r} != {a.tag!r}")
        # (2) 결과 JSON -- **이 런의 것일 때만** 앞 구간 기록으로 쓴다.
        #     이 런의 것 = seed/tag 가 맞고(예전 결과는 seed 필드가 없다 -- 모름은 불일치가
        #     아니다) best_step 이 best.pt 의 step 과 같다.  그러면 레시피도 지금 것이어야 한다.
        _pj = Path(a.results_dir) / f"pancoll_{a.model}_{a.dataset}{a.tag}.json"
        _J = json.load(open(_pj)) if _pj.exists() else None
        _same = False
        if _J is not None:
            _sok = _J.get("seed") in (None, a.seed) and _J.get("tag") in (None, a.tag)
            _jcur = isinstance(_J.get("recipe"), dict) and not _MT.recipe_diff_train(_J["recipe"], _recipe)
            if _sok and _J.get("best_step") == START:
                if not _jcur:
                    _refuse(f"결과 JSON 이 이 best.pt(step {START:,}) 를 **다른 레시피**의 결과로 "
                            f"기록하고 있다 -- 옛 레시피 가중치에서 이어 돌리지 않는다")
                _same = True
                _verified.append("json")
                if _J.get("stop_reason") in ("converged", "schedule_complete") and not (
                        MS_PEND and _J.get("stop_reason") == "converged"):    #: lr 감쇠 전 patience 정지는 잇는다
                    _refuse(f"결과 JSON 이 '{_J['stop_reason']}' 로 끝난 런이다 -- 레시피가 스스로 "
                            f"멈춘 런은 이어 돌리지 않는다")
                if _J.get("stop_reason") == "nan_guard_exhausted":
                    _refuse("결과 JSON 이 NaN 가드 소진으로 끝난 런이다 -- 사전 등록 대체안 C: "
                            "best.pt 를 싣고 이어 돌리지 않는다")
                # 앞 구간의 가드 기록.  warm 은 START 뒤를 버리므로 셈을 잇지 않고 기록째 싣는다.
                _NG_PRIOR = _J.get("nan_guard")
            elif _sok and _jcur:
                # 지금 레시피의 결과가 **다른 step** 을 가리킨다.  이 best.pt 는 그 결과를 낸
                # 런의 것이 아니다 (다른 런이 덮었거나, 다른 호스트의 사본이다).  결과 JSON 을
                # 보고 이어 돌리기로 정한 것이므로(reconcile) 이어 붙일 대상이 어긋났다.
                _refuse(f"best.pt step {START:,} != 결과 JSON best_step {_J.get('best_step')} -- "
                        f"이 best.pt 는 그 결과를 낸 런의 것이 아니다")
            else:
                log(f"[이어 돌리기] 결과 JSON 은 다른 런의 것이다 (best_step {_J.get('best_step')}, "
                    f"레시피 {'현재' if _jcur else '옛것'}) -- 앞 구간 기록으로 쓰지 않는다")
        # (3) log.txt 의 사슬 -- best.pt 의 step 에 *best* 가 찍힌 런.
        _ch = (RS.find_chain(_log_txt, step=START, psnr=_bpsnr) if _log_txt else None)
        if _log_txt and _ch is None and not _verified:
            _refuse(f"best.pt(step {START:,}, PSNR {_bpsnr:.3f}) 를 낸 런을 log.txt 에서 찾지 "
                    f"못했다 -- 어느 레시피의 가중치인지 확인할 길이 없다")
        if _ch is not None:
            for _sg in _ch["segments"]:
                _bad = RS.header_mismatch(_sg["header"], batch, epochs, spec["loss"], o,
                                          R.n_params(model))
                if _bad:
                    _refuse(f"best.pt 를 낸 런의 로그 머리줄이 지금 레시피와 다르다: {_bad}")
            _verified.append("log_header")
        elif _log_txt:
            log(f"[이어 돌리기] log.txt 에서 best.pt(step {START:,}) 의 런을 찾지 못했다 -- "
                f"곡선(vhist)은 이 구간부터 새로 쌓는다")
        if not _verified:
            _refuse("레시피를 확인할 근거가 없다 (best.pt 에 스냅샷 없음 · 이 런의 결과 JSON 없음 · "
                    "log.txt 없음)")
        _r = model.load_state_dict(_ck0["model"], strict=False)   # 최종 평가와 같은 적재
        if _r.missing_keys or _r.unexpected_keys:
            log(f"[이어 돌리기] best.pt 적재: 빠진 키 {len(_r.missing_keys)} · "
                f"남는 키 {len(_r.unexpected_keys)}")
        best, best_at = _bpsnr, START
        vhist = [(s_, p_) for s_, p_ in (_ch["vhist"] if _ch else []) if s_ <= START]
        if _same:
            gnorm_max = float(_J.get("grad_norm_max") or 0.0)
            gnorm_hits = int(_J.get("grad_clip_hits") or 0)
            PRIOR_CUM_H = float(_J.get("wall_hours") or 0.0)
            PRIOR_RESUMES = list(_J.get("resumes") or [])
            _prior_last = int(_J.get("last_step") or START)
            _wsrc = "json"
            if _J.get("total_cap_exact") is False:
                CAP_EXACT = False                  # 앞 구간이 누적 시간을 몰랐다
            GRAD_NOTE = _J.get("grad_stats_note")
            if _prior_last > START:
                GRAD_NOTE = ((GRAD_NOTE + " ") if GRAD_NOTE else "") + (
                    f"앞 구간 기울기 통계에는 버린 step {START + 1:,}~{_prior_last:,} 도 들어 있다.")
        else:
            GRAD_NOTE = (f"warm 이어 돌리기(step {START:,}): 앞 구간의 기울기 통계는 기록되지 "
                         f"않았다 -- grad_norm_max · grad_clip_hits 는 이 구간부터의 값이다.")
            _cw = _ch["wall_hours"] if _ch else None
            PRIOR_CUM_H = float(_cw or 0.0)
            _wsrc = "log" if _cw is not None else "unknown"
            # 사슬 안의 앞선 이어 돌리기 구간은 로그 줄에서만 안다.
            PRIOR_RESUMES = [{"mode": _sg["resume_mode"], "from_step": _sg["start"],
                              "source": "log"}
                             for _sg in (_ch["segments"] if _ch else []) if _sg["kind"] == "resume"]
            _prior_last = int((_ch or {}).get("last_step") or START)
        # 옵티마이저는 위에서 새로 만든 그대로다 (Adam 모멘트 0).  스케줄러는 에폭 수만큼
        # 앞으로 감는다 -- 학습 루프가 `step % spe == 0` 마다 한 번 부르는 것과 같은 횟수다.
        if sched is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")      # "optimizer.step() 전에 scheduler.step()"
                for _ in range(START // spe):
                    sched.step()
        _dseed = RS.derived_seed(a.seed, START)
        g.manual_seed(_dseed)
        torch.manual_seed(_dseed)
        random.seed(_dseed)
        np.random.seed(_dseed % (2 ** 32))
        _replay_epochs(START)
        CUR_RESUME = {"mode": "warm", "from_step": START, "prior_last_step": _prior_last,
                      "optimizer": "reset", "rng": "reseeded", "rng_seed": _dseed,
                      "recipe_verified": _verified[0], "recipe_checks": _verified,
                      "prior_wall_hours_source": _wsrc,
                      "at": _AT, "host": HOST, "segment_hours": None, "reason": _REASON}
    else:
        _refuse(f"{run} 에 state.pt 도 best.pt 도 없다 -- 이어 돌릴 것이 없다")
    # 모듈 트리 밖에 상태를 두는 스펙은 여기서 그 상태를 끊긴 step 에 맞춘다 (선택 훅).
    # 감사(2026-09-24): train_step 을 가진 스펙 13개 중 이것이 필요한 것은 BNNPan 하나다 -- 단계
    # 캐시가 None 이면 첫 스텝에서 되살린 Adam 상태를 비운다.  나머지의 트리 밖 상태는 스텝에서
    # 다시 계산되거나(TMDiff 의 lr, PanDiT · PINO 의 크기별 캐시) state_dict 안에 있다(PanDiT ·
    # U-Know 의 EMA 는 등록된 하위 모듈이다).  CANConv 의 k-means 캐시는 위 on_epoch 재생이 맡는다.
    if "on_resume" in spec:
        spec["on_resume"](model, {"step": START, "total": steps,
                                  "epoch_frac": START / steps, "epoch": max(1, (START - 1) // spe + 1)})
    # ---------------------------------------------------------------- 칸당 총 학습 상한
    # 사용자 결정 (2026-09-24 14:10): "96시간 같은 상한으로".  남은 예산이 한 구간(MIN_SEGMENT_H)이 안 되면
    # 띄우지 않는다 -- 그런 구간은 데이터 적재 · 최종 평가가 대부분이다.  **시도 기록 전에** 끝낸다: 실패가
    # 아니므로 MAX_ATTEMPTS 셈에 들어가면 안 된다.
    #   - 결과 JSON 이 이미 상한 소진을 말하면(`total_capped`) 게이트가 그 칸을 유효로 싣고 계획자 · 디스패처도
    #     넣지 않는다.  여기 오는 것은 손으로 띄운 줄(--force, seed 복제)뿐이다 -- 기록 없이 건너뛴다.
    #   - 결과 JSON 은 아직 상한 전인데 state.pt · 로그의 누적이 소진을 말하면: 뒤 구간이 결과를 못 쓰고 죽었다.
    #     기록하지 않으면 계획자가 15분마다 같은 줄을 다시 넣는다.  거부로 기록하고([이상]) 사람이 정한다.
    if _wsrc == "unknown":
        CAP_EXACT = False                          # 앞 구간 시간을 모른다 -- 0 으로 두고 그 사실을 남긴다
    _bud = RS.segment_budget(a.max_hours, PRIOR_CUM_H)
    if _bud["exhausted"]:
        try:
            _Jc = json.load(open(_PJ)) if os.path.exists(_PJ) else None
        except ValueError:
            _Jc = None
        _cmsg = (f"칸당 총 학습 상한 {RS.TOTAL_CAP_H:g}h 를 다 썼다 -- 앞 구간 누적 {PRIOR_CUM_H:.2f}h "
                 f"(출처 {_wsrc}), 남은 {max(0.0, _bud['remaining_h']):.2f}h 는 한 구간"
                 f"({RS.MIN_SEGMENT_H:g}h) 이 안 된다.  더 이어 돌리지 않는다 ({RS.CAP_DECIDED})")
        if RS.total_capped(_Jc):
            log(f"[건너뜀] {_cmsg} -- 결과 JSON 이 이미 상한 소진을 기록했다 (실패가 아니다)")
            raise SystemExit(f"[건너뜀] {a.model}/{a.dataset}{a.tag}: {_cmsg}")
        _refuse(f"{_cmsg} -- 그런데 결과 JSON 은 아직 상한 전이다 (누적 "
                f"{(_Jc or {}).get('wall_hours')}h): 뒤 구간이 결과를 쓰지 못하고 끝났다.  best.pt 를 "
                f"평가해 실을지 사람이 정한다")
    if MS_ON:
        #: 이 런의 스케줄러가 실제로 lr 을 계단식으로 줄이는 지점 (에폭당 한 번 부르는 대로 센다).
        _ms = _MP.milestones(_sc, epochs, spe)
        if MS_PEND and _ms["last_step"] != (_MP.result_milestones(_MSJ) or {}).get("last_step"):
            # 결과 JSON 의 기록된 일정으로 연장을 정했는데 이 런의 일정이 다르다 -- 풀어 준 'converged' 거부를 되돌린다.
            _refuse(f"연장: 결과 JSON 의 마지막 lr 감쇠 step "
                    f"{(_MP.result_milestones(_MSJ) or {}).get('last_step')} 과 이 런의 {_ms['last_step']} 이 다르다 "
                    f"(일정 {epochs} epoch x {spe}) -- 이어 돌리지 않는다")
    # 시도를 기록한다.  이 구간이 결과 JSON 을 쓰기 전에 죽으면(OOM, 중단) 기록이 쌓이고,
    # MAX_ATTEMPTS 번이면 reconcile · 디스패처가 더 넣지 않고 [이상] 으로 알린다.
    _nat = RS.record_attempt(str(run), _PJ, mode=CUR_RESUME["mode"], from_step=START,
                             pid=os.getpid())
    log(f"[이어 돌리기] mode={CUR_RESUME['mode']} from step {START:,} "
        f"(앞 구간 마지막 {CUR_RESUME['prior_last_step']:,}, 누적 {PRIOR_CUM_H:.2f}h) "
        f"optimizer {CUR_RESUME['optimizer']} rng {CUR_RESUME['rng']} "
        f"레시피 확인 {CUR_RESUME['recipe_verified']} · best {best:.3f} @ {best_at:,} · "
        f"곡선 {len(vhist)}점 · 사유 {_REASON} · 이 결과에 대한 시도 {_nat}번째")
    if MS_ON:
        #: 이산 감쇠 지점이 일정 안에 있는 스케줄이면 이 구간은 마지막 지점 전에 patience 로 멈추지 않고, 그 뒤에는
        # patience 를 그 지점부터 다시 센다 (`milestone_policy.patience_due`).  지점은 이 학습기가 스케줄러를 부르는
        # 대로(에폭당 한 번, `step % spe == 0`) 센다.  천장 · 총 상한 · 일정 끝은 그대로 멈춘다.
        if _ms["last_step"] is not None:
            MS_LAST = int(_ms["last_step"])
            CUR_RESUME["milestone"] = _MP.record(_ms, PATIENCE)
            if MS_PEND:                            # 이 구간은 이 규칙 **때문에** 돈다 -- 사유를 기록한다
                CUR_RESUME.update(reason=_MP.REASON, reason_requested=_REASON)
                if not _MP.caused(_MSJ):
                    # 사슬의 첫 연장 -- 지금 싣고 있는 결과 JSON 과 best.pt 를 sha256 과 함께 남긴다 (덮지 않는다).
                    # 연장이 끝나면 둘 다 바뀐다: 기준(MMNet/gf2 converged @ 716,000) · 전후 공개가 이것을 읽는다.
                    CUR_RESUME["milestone"]["published_before"] = _ms_snapshot(run / _MP.PRE_DIR)
            log(f"[이어 돌리기] lr 감쇠 규칙 ({_MP.DECIDED}): {_ms['kind']} 감쇠 epoch {_ms['epochs']} -> 마지막 "
                f"step {MS_LAST:,} 전에는 patience 로 멈추지 않는다, 그 뒤 patience {PATIENCE:,} step 을 "
                f"max(best, {MS_LAST:,}) 부터 센다"
                + (f" -- 결과 JSON 이 마지막 감쇠 전 정지(step {_MP.reached(_MSJ):,}, "
                   f"{_MSJ.get('stop_reason')})라 이어 돈다 (사유 {_MP.REASON})" if MS_PEND else ""))

if MS_ON and not a.resume:
    # 스펙 단위 규칙이다 (검토 2026-09-25) -- 처음부터 도는 런(본 칸 · seed 복제)도 마지막 감쇠 지점 전에는
    # patience 로 멈추지 않고, 그 뒤에는 그 지점부터 다시 센다.  예전 규칙으로 멈추는 step 까지는 궤적이 같고, 더 갈 뿐이다.
    _ms = _MP.milestones(_sc, epochs, spe)
    if _ms["last_step"] is not None:
        MS_LAST = int(_ms["last_step"])
        MS_REC = _MP.record(_ms, PATIENCE)
        log(f"[정지] lr 감쇠 규칙 ({_MP.DECIDED}): {_ms['kind']} 감쇠 epoch {_ms['epochs']} -> 마지막 step {MS_LAST:,} "
            f"전에는 patience 로 멈추지 않는다, 그 뒤 patience {PATIENCE:,} step 을 max(best, {MS_LAST:,}) 부터 센다")

if not a.eval_only:
    # 이 구간의 천장.  처음부터 도는 런은 앞 구간이 0 이라 --max_hours 그대로다 (천장을 96h 넘게
    # 올린 경우만 총 상한이 먼저 걸린다).  이어 돌리는 구간은 남은 예산이 더 짧으면 그것이 천장이다.
    _bud = RS.segment_budget(a.max_hours, PRIOR_CUM_H)
    SEG_CEIL_H, CAP_BINDS = _bud["ceiling_h"], _bud["cap_binds"]
    if CUR_RESUME is not None:
        CUR_RESUME.update(segment_ceiling_h=round(SEG_CEIL_H, 4), total_cap_exact=CAP_EXACT)
    log(f"[정지] 칸당 총 학습 상한 {RS.TOTAL_CAP_H:g}h ({RS.CAP_DECIDED}): 앞 구간 누적 "
        f"{PRIOR_CUM_H:.2f}h -> 이 구간 천장 {SEG_CEIL_H:.2f}h"
        + (f" (총 상한이 구간 천장 {a.max_hours:g}h 보다 먼저 걸린다)" if CAP_BINDS else "")
        + ("" if CAP_EXACT else " -- 앞 구간 시간을 몰라 0 으로 두었다: 상한을 정확히 지키지 못한다"))


def _train_update(lms, pan, ms, gt, ctx):
    """한 step 의 갱신 전체: forward -> 손실 -> zero_grad -> backward -> clip -> opt.step.  손실을 돌려준다.

    학습 루프가 이것을 **그대로** 부르거나(가드 없음 -- 나머지 모든 스펙) NaN 가드 아래에서 부른다
    (`NG.run`).  같은 함수 하나라 두 길의 연산과 순서가 같다 (예전에는 이 줄들이 루프 안에 있었다)."""
    global gnorm_max, gnorm_hits
    if "train_step" in spec:
        # 공식 손실이 단순 (예측, GT) 함수가 아닌 경우 -- 보조 잠재 항, 흐름 NLL,
        # 에폭 램프 등.  그런 저장소는 한 스텝 전체를 spec 이 맡는다.
        pr, loss = spec["train_step"](model, dict(lms=lms, pan=pan, ms=ms, gt=gt), ctx, lossfn)
    else:
        pr = fwd(model, lms, pan, ms, ctx)
        if isinstance(pr, tuple):
            # 저장소가 보조 출력을 함께 돌려주는 경우가 있다.  PGCU 의 공식 학습 루프
            # (train_pannet.py:71-76) 는 loss_1 + loss_2 로, 중간 up_ms 에도 같은 손실을
            # 건다.  aux_loss 를 켠 스펙은 그 항을 살린다.  끄면 첫 출력만 쓴다.
            w = spec.get("aux_weights")
            if w is not None:
                # (출력, 가중치) 짝.  out_index 가 최종 예측의 위치다 (기본 0).
                oi = spec.get("out_index", 0)
                aux = [(pr[i], w[k]) for k, i in enumerate(j for j in range(len(pr)) if j != oi)]
                pr = pr[oi]
            else:
                aux = [(x, 1.0) for x in pr[1:]] if spec.get("aux_loss") else []
                pr = pr[0]
        else:
            aux = []
        loss = lossfn(pr, gt)
        for extra, wgt in aux:
            loss = loss + wgt * lossfn(extra, gt)
    # 두 경로(train_step / 기본)는 여기서부터 **같은 줄**이다.  예전에는 아래 전체가 두 벌
    # 있었다 -- 이어 돌리기 · 상태 저장을 넣으면서 한쪽만 고치는 일을 막으려고 합쳤다
    # (연산 순서는 그대로다: zero_grad -> backward -> clip -> step -> sched).
    opt.zero_grad(set_to_none=True); loss.backward()
    # 저장소 대부분(LAGConv/ADWM/WFANet/ARConv/MMNet/MutInf ...)은 클리핑을 하지 않는다.  MDCUN(0.04) ·
    # MSDDN(40)은 저장소 값이 있지만 스펙에 옮기지 않아 10.0 이다 (-4 -- 둘 다 저장소 값도 매 step 걸려
    # Adam 갱신이 같다고 봤다, registry 의 diff).  우리는 안정성을 위해 10.0 을 건다.  그 상한이 실제로 걸렸는지
    # 기록해 두어야 "저장소 레시피에서 벗어났다"는 지적에 수치로 답할 수 있다.
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX))
    gnorm_max = max(gnorm_max, gn); gnorm_hits += (gn > CLIP_MAX)
    opt.step()
    return loss


# NaN 가드 (사용자 선택 B, 사전 등록; 설명은 `lib/nan_guard.py`).  스펙이 `nan_guard=True` 로 켠 방법만
# (지금은 MutInf 하나).  step 마다 가중치 · 버퍼 · 옵티마이저 상태를 떠 두고, 그 step 이 비유한 값을 내면
# 정확히 되돌리고 그 batch 를 건너뛴다.  **step 번호는 그대로 나아간다** -- batch 는 소비됐다(표본 추출 · 모델
# 안의 잡음 난수는 되돌리지 않는다).  그래서 평가 간격 · 스케줄러의 에폭 경계 · 천장 · 이어 돌리기는 건너뛴
# step 에도 원래대로 걸린다.  최근 1만 step 안에 100번을 넘으면 `nan_guard_exhausted` 로 멈추고 아래 **보통의
# 끝 경로**(best.pt 로 RR · FR · zero-shot, 결과 JSON)를 밟는다 -- 사전 등록 대체안 C.
# 켜는 것은 레시피가 아니다: `nan_guard` 는 레시피 키도 호출 가능한 필드도 아니라 게이트 판정이 바뀌지 않는다.
_ng_stop = False
if spec.get("nan_guard") and not a.eval_only:
    import nan_guard as _NGM
    NG = _NGM.NanGuard(model, opt, sched)
    NG.load(_NG_LOAD)
    if _NG_PRIOR is not None:
        NG.prior = _NG_PRIOR
    log(f"{_NGM.SKIP_MARK} 켜짐 (스펙 nan_guard) -- step 마다 가중치 · 버퍼 · 옵티마이저 상태를 떠 두고, "
        f"비유한 값(forward 예외 · 손실 · 갱신 뒤 상태)이 나오면 되돌리고 그 batch 를 건너뛴다 (step 번호는 "
        f"나아간다).  최근 {NG.window:,} step 안에 {NG.cap}번을 넘으면 '{_NGM.STOP_REASON}' 로 멈추고 "
        f"best.pt 를 평가해 싣는다"
        + (f".  state.pt 에서 이었다: 누적 {NG.skipped}번, 창 안 {len(NG.recent)}번" if _NG_LOAD else "")
        + (f".  warm 이전 구간의 기록(누적 {_NG_PRIOR.get('skipped')}번)은 prior 로 싣는다"
           if isinstance(_NG_PRIOR, dict) else ""))


def _ng_stop_log(step):
    """NaN 가드 소진 정지 줄.  `resume_state.segments` 가 머리(`nan_guard.STOP_LINE`)로 정지 사유를 읽는다."""
    log(f"{_NGM.STOP_LINE} -- 최근 {NG.window:,} step 안에 {len(NG.recent)}번 건너뜀 (> {NG.cap}, 누적 "
        f"{NG.skipped}번) -- step {step:,}/{steps:,} 에서 중단.  사전 등록 대체안 C: best.pt(step "
        f"{best_at:,}) 를 평가해 싣고 이어 돌리지 않는다")


model.train()
# `--eval_only` 면 아래 루프가 한 번도 돌지 않아 `step` 이 정의되지 않는다.
# 그 경우 마지막 step 은 되살리는 원래 런의 것이다.
step = int(PRIOR.get("last_step", PRIOR.get("best_step", 0)))
_last_state = time.time()
for step in (range(1, 1) if a.eval_only else range(START + 1, steps + 1)):
    if "on_epoch" in spec and (step - 1) % spe == 0:
        spec["on_epoch"](model, (step - 1) // spe + 1)
    # 표본 추출: **복원추출**이다.  이것은 하네스 차원의 선택이고, 모든 방법에
    # 똑같이 적용된다.
    # 저장소를 그대로 따를 수가 없다: 대부분 DataLoader(shuffle=True) 지만
    # MSDDN 은 solver/basesolver.py:30 에서 학습 로더를 shuffle=False 로 만든다.
    # 맞출 단일 관행이 없으므로 하나를 골라 전원에게 똑같이 적용하는 편이 공정하다.
    # 대신 두 가지를 기억해야 한다.
    #   (1) 한 "에폭"(spe 스텝)이 인덱스의 63.2% 에만 닿는다.  기댓값은 같지만
    #       비복원 재섞기보다 분산이 크다.
    #   (2) 표본 인덱스에 묶인 상태를 쓰는 저장소는 이 차이에 민감하다.  CANConv 의
    #       k-means 캐시가 실제로 깨졌고(37% 의 표본이 0 배정을 읽었다),
    #       lib/specs/canconv.py 가 처음 보는 인덱스를 만나면 init 을 돌려 막는다.
    idx = torch.randint(0, N, (batch,), generator=g)
    lms, pan = tr["lms"][idx].to(dev), tr["pan"][idx].to(dev)
    gt, ms = tr["gt"][idx].to(dev), tr["ms"][idx].to(dev)
    if spec.get("aug") == "d4":
        gt, lms, ms, pan = _d4(g, gt, lms, ms, pan)
    # 검증 경로는 sample_idx 를 넣지 않는다 -- 검증 표본은 학습 캐시에 자리가 없고,
    # 인덱스가 겹치면 남의 군집을 읽는다.  없으면 저장소 코드가 매번 새로 군집한다.
    ctx = {"step": step, "total": steps, "epoch_frac": step / steps,
           "epoch": (step - 1) // spe + 1, "sample_idx": idx.to(dev),
           "lpan": tr["lpan"][idx].to(dev) if "lpan" in tr else None}
    if NG is None:
        loss = _train_update(lms, pan, ms, gt, ctx)
    else:
        # 가드 아래 한 step.  건너뛰면 그 step 의 기울기 통계도 되돌린다 -- 되돌린 갱신의 노름(흔히 nan · inf)이
        # grad_norm_max · grad_clip_hits 에 남으면 원장의 클립 기록이 틀린다.
        _gstat = (gnorm_max, gnorm_hits)
        loss, _ng_why = NG.run(step, lambda: _train_update(lms, pan, ms, gt, ctx))
        if _ng_why is not None:
            gnorm_max, gnorm_hits = _gstat
            log(f"{_NGM.SKIP_MARK} step {step} 건너뜀 ({_ng_why})")
            _ng_stop = NG.exhausted
    if sched is not None and step % spe == 0: sched.step()
    if step % 500 == 0:
        # 속도는 **이 구간에서** 돈 step 으로 나눈다 -- 이어 돌리기 구간이 0 부터 센
        # step 으로 나누면 느린 카드도 빨라 보이고, 로그에서 벽시계를 되살릴 때 틀린다.
        # (가드가 건너뛴 step 에는 손실이 없다 -- nan 으로 적는다.)
        log(f"step {step:7d}/{steps} (ep {step//spe:3d})  "
            f"loss {(float('nan') if loss is None else loss.item()):.6f}  "
            f"{(time.time()-t0)/(step-START):.4f}s/step")
    over = (time.time() - t0) > SEG_CEIL_H * 3600         # min(구간 천장, 총 상한 - 누적)
    # 시험 전용 정지 (`--stop_after_steps`).  천장처럼 멈추되 **평가를 끼워 넣지 않는다** --
    # 그래야 이어 돌린 런이 끊김 없는 런과 같은 평가점을 갖는다 (tests/test_resume.py).
    tstop = bool(a.stop_after_steps) and step >= a.stop_after_steps
    if step < int(steps * spec.get("eval_after_frac", 0.0)):
        if _ng_stop:
            STOP["why"] = _NGM.STOP_REASON; _ng_stop_log(step); break
        if over:
            STOP["why"] = "time_ceiling"
            log("[정지] 구조 확정 전 천장 도달" + (f" {RS.CAP_MARK} -- 칸당 총 학습 상한 "
                                                  f"({RS.CAP_DECIDED})" if CAP_BINDS else ""))
            break
        if tstop: STOP["why"] = "test_stop"; log(f"[정지] 시험 정지 -- step {step:,}"); break
        # 평가 시작점 전에도 주기 저장한다.  이 구간에는 평가가 없어 난수를 쓰지 않으므로 어느 step 에서 저장해도
        # exact 이어 돌리기가 끊김 없는 런과 같다.  없으면 BNNPan(2/3) · U-Know(0.7) · TMDiff(0.75) 는 일정의 대부분을
        # state.pt 없이 돈다 -- 그 사이에 죽으면 처음부터다.  평가 경계(eval_every)에서만 본다: `--state_every_min 0`
        # 인 시험 런이 매 step 저장하지 않게.
        if step % a.eval_every == 0 and time.time() - _last_state >= a.state_every_min * 60:
            _save_state(step)
            _last_state = time.time()
        continue
    evald = False
    # NaN 가드 소진(`_ng_stop`)도 천장처럼 멈추는 step 에서 한 번 평가한다 (마지막 평가 뒤의 진전을 버리지 않는다).
    if step % a.eval_every == 0 or step == steps or over or _ng_stop:
        evald = True
        m = val(va, step)
        vhist.append((step, m["psnr"]))
        log(f"  [valid] step {step:7d}  PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}  "
            f"SAM {m['sam']:.3f}  ERGAS {m['ergas']:.3f}" + ("  *best*" if m["psnr"] > best else ""))
        if m["psnr"] > best:
            best, best_at = m["psnr"], step
            _save_best(step, m)
        elif (step - best_at >= PATIENCE) if MS_LAST is None else _MP.patience_due(step, best_at, PATIENCE, MS_LAST):
            # MS_LAST 는 켜진 이어 돌리기에서만 있다 (마지막 lr 감쇠 전에는 멈추지 않고 그 뒤 다시 센다).
            STOP["why"] = "converged"
            log(f"[정지] 수렴 -- step {best_at:,} 이후 {step-best_at:,} step 동안 "
                f"갱신 없음 (step {step:,}/{steps:,})"
                + ("" if MS_LAST is None else
                   f" -- 마지막 lr 감쇠 step {MS_LAST:,} 뒤 {step - MS_LAST:,} step")); break
    if _ng_stop:
        # 사전 등록 대체안 C.  천장보다 앞선다: 이어 돌려도 같은 궤적에서 또 소진된다.
        STOP["why"] = _NGM.STOP_REASON; _ng_stop_log(step); break
    if over:
        STOP["why"] = "time_ceiling"
        if CAP_BINDS:
            # `[정지] 안전 천장` 으로 시작해야 `resume_state.segments` 가 천장 정지로 읽는다.  CAP_MARK 는
            # 그 정지가 구간 천장이 아니라 칸당 총 상한이었다는 표식이다 (재평가 복구가 읽는다).
            log(f"[정지] 안전 천장 {RS.CAP_MARK} -- 칸당 총 학습 상한 {RS.TOTAL_CAP_H:g}h 에 닿았다 "
                f"(앞 구간 누적 {PRIOR_CUM_H:.2f}h + 이 구간 {SEG_CEIL_H:.2f}h) -- step {step:,}/{steps:,} "
                f"에서 중단.  아직 오르는 중이어도 더 이어 돌리지 않는다 ({RS.CAP_DECIDED})"); break
        log(f"[정지] 안전 천장 {a.max_hours}h -- step {step:,}/{steps:,} 에서 중단. "
            f"수렴 전이다 -- 아직 오르는 중이면 이 칸은 이어 돌린다 (--resume)"); break
    if tstop:
        STOP["why"] = "test_stop"
        log(f"[정지] 시험 정지 -- step {step:,}/{steps:,} (--stop_after_steps, 시험 전용)"); break
    # 주기 저장.  평가 경계에서만 한다 -- 평가가 난수를 쓰는 방법이 있어, 평가 뒤의 상태를
    # 저장해야 이어 돌린 런이 같은 난수 흐름을 잇는다.
    if evald and time.time() - _last_state >= a.state_every_min * 60:
        _save_state(step)
        _last_state = time.time()

if not a.eval_only:
    # **최종 평가 전에** 상태를 남긴다 -- 아래에서 best.pt 가 모델에 적재되므로 그 뒤에
    # 저장하면 마지막 가중치가 아니라 best 가중치가 state 에 들어간다.
    # 레시피가 멈춘 런(converged / schedule_complete)도 남긴다: 그 표시가 있으면
    # `--resume` 이 이어 돌리기를 거부한다 (주기 저장이 남긴 중간 상태로 되돌아가지 않게).
    if _save_state(step, stop=STOP["why"]):
        log(f"[state] 학습 상태 저장 -- step {step:,} ({STOP['why']}) -> {STATE.name}")

_BFIN = BEST if BEST.exists() else BEST_PUB     # 연장 구간이면 이 구간의 best, 없으면 싣고 있는 것 (보통은 같은 파일)
if not _BFIN.exists():
    _save_best(steps, val(va))
    _BFIN = BEST
ck = torch.load(_BFIN, map_location=dev)
if a.eval_only and PRIOR.get("best_step") is not None and ck.get("step") != PRIOR["best_step"]:
    # 재평가는 학습 쪽 기록(기울기 통계 · 구간 · 평탄성)을 **이전 결과에서** 가져온다.  best.pt 가 그
    # 결과를 낸 가중치가 아니면(다른 런이 덮었다) 남의 기록에 이 가중치의 지표를 붙이게 된다.
    raise SystemExit(f"[eval_only 거부] best.pt step {ck.get('step')} != 결과 JSON best_step "
                     f"{PRIOR['best_step']} -- 이 best.pt 는 그 결과를 낸 가중치가 아니다")
model.load_state_dict(ck["model"], strict=False)

# --- 문헌이 보고하는 test 집합에서 최종 평가 -------------------------------
# `valid` 는 체크포인트 선택에만 썼다.  표에 실리는 수는 아래 두 집합이다.
rr = PC.load_examples(a.dataset, "reduced")          # RR: gt 있음
m = val(rr, full=True)
log(f"[RR/reduced_examples] {a.model} {a.dataset}  PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}  "
    f"SAM {m['sam']:.3f}  ERGAS {m['ergas']:.3f}  Q2n {m.get('q2n', float('nan')):.4f}")

fr_m = {}
FR_SKIPPED = None
try:                                                  # FR: gt 없음 -> HQNR/D_lambda/D_s
    fe = PC.load_examples(a.dataset, "full")
    model.eval()
    outs = []
    with torch.no_grad():
        fbs = min(4, EVAL_BS)          # FR 은 H=512 라 RR 보다 4배 무겁다
        # 안 들어가면 RR 의 chunk() 처럼 반으로 줄인다 (2026-10-13: ARConv/qb 가 24 GB 카드에서 4장을 못 넣어
        # FR 이 통째로 빠졌다).  출력은 표본별이라 배치 크기가 수치를 바꾸지 않는다.  재시도는 except 밖에서 한다.
        i, n_fe = 0, fe["lms"].shape[0]
        while i < n_fe:
            j = min(i + fbs, n_fe)
            oom = False
            try:
                pr = fwd_eval(model, fe["lms"][i:j].to(dev), fe["pan"][i:j].to(dev),
                         fe["ms"][i:j].to(dev),
                         {"step": steps, "total": steps,
                          "lpan": fe["lpan"][i:j].to(dev) if "lpan" in fe else None})
            except torch.OutOfMemoryError:
                if j - i <= 1:
                    raise
                oom = True
            if oom:
                pr = None
                torch.cuda.empty_cache()
                fbs = max(1, (j - i) // 2)
                EVAL_BS_USED_MIN = min(EVAL_BS_USED_MIN, fbs)
                log(f"  [평가] FR {j-i}장이 안 들어감 -- {fbs}장씩")
                continue
            outs.append((pr[0] if isinstance(pr, tuple) else pr).clamp(0, 1).cpu())
            i = j
    fused = torch.cat(outs)
    h, dl, ds = PC.hqnr(fused, fe["ms"], fe.get("lms_orig", fe["lms"]), fe["pan"], a.dataset)   # 지표는 원래 lms
    fr_m = {"hqnr": float(h.mean()), "d_lambda": float(dl.mean()), "d_s": float(ds.mean())}
    log(f"[FR/full_examples]   {a.model} {a.dataset}  HQNR {fr_m['hqnr']:.4f}  "
        f"D_lambda {fr_m['d_lambda']:.4f}  D_s {fr_m['d_s']:.4f}")
    # 보조 FR 공간 지표 (2026-10-13, lib/fr_extra.py = rho-PNN 의 D_rho · D_sR, 낮을수록 좋음).  HQNR 의 D_s 는 디테일을 덜
    # 넣을수록 점수가 오른다 (GT 로 확인, memory fr-hqnr-detail-and-geometry) -- 성격이 다른 두 지표를 같이 남긴다.
    # 실패해도 위 FR 결과와 FR_SKIPPED 는 그대로다 (이 블록만 건너뛴다).
    try:
        import fr_extra as FX
        fr_m["d_rho"] = float(FX.d_rho(fused, fe["pan"]).mean())
        fr_m["d_sr"] = float(FX.d_sr(fused, fe["pan"]).mean())
        log(f"[FR/extra]           {a.model} {a.dataset}  D_rho {fr_m['d_rho']:.4f}  D_sR {fr_m['d_sr']:.4f}")
    except Exception as e:
        log(f"[FR/extra] 건너뜀: {type(e).__name__}: {str(e)[:120]}")
except Exception as e:
    log(f"[FR] 건너뜀: {type(e).__name__}: {str(e)[:120]}")
    FR_SKIPPED = f"{type(e).__name__}: {str(e)[:200]}"
# --- zero-shot: 학습하지 않은 봉인 센서 -----------------------------------
# WV2 는 PanCollection 에 학습 split 이 없다.  WV3 에서 학습한 8밴드 모델을 그대로
# 태워 cross-sensor 일반화를 잰다 -- 적응도 미세조정도 없다.
zs = {}
for zd in PC.ZEROSHOT.get(a.dataset, []):
    _keep = a.dataset
    try:
        zr = PC.load_examples(zd, "reduced")
        a.dataset = zd                      # Q2n 의 나눗수가 데이터셋에서 온다
        zm = val(zr, full=True)
        zs[zd] = zm
        log(f"[zero-shot/{zd}] {a.model}  PSNR {zm['psnr']:.3f}  SSIM {zm['ssim']:.4f}  "
            f"SAM {zm['sam']:.3f}  ERGAS {zm['ergas']:.3f}  Q2n {zm.get('q2n', float('nan')):.4f}")
    except Exception as e:
        log(f"[zero-shot/{zd}] 건너뜀: {type(e).__name__}: {str(e)[:100]}")
    finally:
        # 예외가 나도 **반드시** 되돌린다.  안 되돌리면 아래 결과가 `pancoll_<M>_wv2.json`,
        # dataset "wv2" 로 적혀 wv3 칸이 사라진다 -- PAN-Crafter/wv3 가 14 시간을 돌고 그렇게
        # 됐다 (WV2 에 lpan 이 없어 평가가 RuntimeError, 2026-09-24).
        # SystemExit 을 RuntimeError 로 바꿔 여기서 잡히게 만든 순간 열린 구멍이다.
        a.dataset = _keep

va_m = val(va, full=False)                            # 선택에 쓴 집합도 기록해 둔다
log(f"[final] {a.model} {a.dataset}  선택 step {ck['step']}/{steps}  "
    f"(valid PSNR {va_m['psnr']:.3f})")
# 레시피 스냅샷(`_recipe`)은 학습 전에 만들었다 (위, 머리줄 아래).
_SEG_H = (time.time() - t0) / 3600       # 이 구간(이 프로세스)의 벽시계
_RES = {"model": a.model, "dataset": a.dataset, "params": R.n_params(model),
           # seed 복제(`--tag _sS`)를 파일 이름이 아니라 **기록된 필드**로 알아볼 수 있게.
           "seed": a.seed, "tag": a.tag,
           **({"skip": a.skip} if a.skip != "lrms" else {}),
           **({"gt_geo": a.gt_geo} if a.gt_geo != "ms" else {}),
           "recipe": _recipe,
           "batch": batch, "epochs": epochs, "steps_planned": steps, "best_step": ck["step"],
           # `stopped_early` 는 "상한에 잘렸다"가 아니라 "best 가 마지막 step 이
           # 아니다" 라는 뜻이었다.  이름이 오해를 불러 아래 두 필드를 따로 둔다.
           "best_is_final": bool(ck["step"] >= steps),
           "stop_reason": PRIOR.get("stop_reason", STOP["why"]),   # schedule_complete | converged | time_ceiling
           "hit_time_ceiling": PRIOR.get("hit_time_ceiling", STOP["why"] == "time_ceiling"),
           "patience_steps": PATIENCE,
           # 재평가로 되살린 칸임을 남긴다.  가중치는 이전 학습의 것이고 평가만
           # 새 규약으로 다시 냈다 -- 레시피가 바뀌지 않은 방법에만 쓴다.
           **({"reeval_only": True} if a.eval_only else {}),
           # 학습 기록을 로그에서 되살렸다면 그 사실도 남긴다 -- 원장이 읽는다.
           **({"metadata_recovered_from_log": True}
              if PRIOR.get("metadata_recovered_from_log") else {}),
           "stopped_early": bool(ck["step"] < steps),
           # **누적**이다 -- 이어 돌린 칸은 앞 구간들의 합 + 이 구간.  처음부터 돈 런은
           # 예전과 같은 값(이 구간)이다.  이 구간만의 값은 `segment_wall_hours`.
           "wall_hours": PRIOR.get("wall_hours", PRIOR_CUM_H + _SEG_H),
           # 재평가는 학습 구간의 시간을 모르면 **적지 않는다** -- 재평가 시간을 넣으면 거짓이다.
           **({"segment_wall_hours": PRIOR["segment_wall_hours"]} if "segment_wall_hours" in PRIOR
              else {} if a.eval_only else {"segment_wall_hours": _SEG_H}),
           **_TEST_MARK,
           # 학습한 기계.  reconcile 이 이어 돌리기를 **이 기계의** 큐에 넣는다 (state.pt 와
           # 이어 쓸 log.txt 가 거기 있다).  재평가는 학습 기계를 그대로 물려받는다(아래).
           **({} if a.eval_only else {"host": HOST}),
           **({"resumes": PRIOR_RESUMES + [dict(CUR_RESUME, segment_hours=_SEG_H)]}
              if CUR_RESUME else {}),
           **({"grad_stats_note": GRAD_NOTE} if GRAD_NOTE and not a.eval_only else {}),
           # 적용한 처음부터 도는 런 (스위치가 켜졌을 때만; 이어 돌리기는 resumes[].milestone).
           **({"milestone": MS_REC} if MS_REC else {}),
           # 칸당 총 학습 상한.  `total_cap_reached` 는 **총 상한이** 구간 천장보다 먼저 걸려 멈췄다는
           # 뜻이다 (48h 구간 천장에서 멈춘 것과 구별한다).  게이트는 `resume_state.total_capped` 로 읽는다.
           # 앞 구간 시간을 몰랐으면 `total_cap_exact: false` (상한을 정확히 지키지 못했다).
           **({} if a.eval_only else {
               "total_cap_h": RS.TOTAL_CAP_H,
               "total_cap_reached": bool(CAP_BINDS and STOP["why"] == "time_ceiling"),
               **({} if CAP_EXACT else {"total_cap_exact": False})}),
           "rr": m, "fr": fr_m, "zeroshot": zs, "valid": va_m, "venue": spec.get("venue"),
           # FR 이 빈 이유를 결과에 남긴다.  없으면 표의 `--` 가 "아직 안 돌렸다"인지
           # "돌렸는데 실패했다"인지 구별되지 않는다.
           **({"fr_skipped_reason": FR_SKIPPED} if FR_SKIPPED else {}),
           "rr_set": "reduced_examples", "fr_set": "full_examples",
           "grad_clip": CLIP, "grad_norm_max": gnorm_max, "grad_clip_hits": int(gnorm_hits),
           # 민감도 런(-4)만 있다.  grad_clip 은 덮어쓴 값(0 = 클립 없음)이고 레시피 스냅샷도 같다.
           **({"grad_clip_override": CLIP} if a.grad_clip_override is not None else {}),
           # NaN 가드를 켠 런만 (공개 항목): {enabled, skipped, steps(앞 200개), last_reason, ...}.
           # 가드를 끈 런의 결과 JSON 은 예전 모양 그대로다.
           **({"nan_guard": NG.record()} if NG is not None else {}),
           # 마지막으로 **도달한** step.  `best_step` 은 보고하는 체크포인트의 step 이라
           # 서로 다르다.  천장에 잘린 칸이 일정의 어디까지 갔는지는 이 값이 답한다.
           "last_step": int(PRIOR.get("last_step", step) if a.eval_only else step),
           # 평가에서 실제로 쓴 최소 배치.  `eval_bs` 보다 작으면 자동 분할이 걸린
           # 것이다.  지표는 표본별 평균이라 수치는 같지만, 걸렸다는 사실은 남긴다.
           "eval_bs": EVAL_BS, "eval_bs_used_min": EVAL_BS_USED_MIN,
           **({k: PRIOR[k] for k in PRIOR if k.startswith("plateau") or k == "still_improving"}
              if a.eval_only else _plateau(vhist))}
if not a.eval_only and STOP["why"] == "time_ceiling" and step < int(steps * spec.get("eval_after_frac", 0.0)):
    # 평가 시작점(eval_after_frac) 전에 천장에 잘렸다.  평가점이 없어 `_plateau` 는 still_improving=None 을
    # 주고, reconcile 은 그것을 "평탄 판정 불가 -- 재실행" 으로 읽어 **처음부터** 다시 넣는다 (48 GPU-시간짜리
    # 고리).  이 런은 아직 구조가 확정되지 않았을 뿐 학습 중이다 -- 오르는 중으로 적어 `ceiling_rising` 이
    # state.pt 에서 이어 돌게 한다.  위의 best.pt · 지표는 학습 전 가중치의 것이라 표에 싣지 않는다 (천장-상승
    # 칸은 classify 에서 유효가 아니다).
    _RES["still_improving"] = True
    _RES["ceiling_before_gate"] = True
if a.eval_only:
    # **학습 때 기록한 값은 학습의 것이다.**  재평가는 학습 루프를 돌지 않으므로 위에서 만든
    # 기울기 통계는 0 이고, 레시피는 **지금의** 레지스트리에서 새로 찍힌다.  그대로 쓰면
    # (1) 원장이 "우리 클립이 실제로 걸렸는지 확인하라"고 가리키는 grad_norm_max ·
    #     grad_clip_hits 가 0 으로 지워지고 -- 2026-09-23 에 다섯 칸이 실제로 지워졌다
    #     (PanFlowNet/wv3 는 128만 번 클립됐었다),
    # (2) 옛 레시피로 학습한 가중치에 현재 레시피 도장이 찍혀 is_current 를 통과할 수 있다.
    # 그래서 학습 쪽 필드는 이전 결과에서 가져오고, 레시피는 **평가 규약만** 새로 쓴다.
    # `resumes` · `host` 도 학습의 것이다 -- 이어 돌린 칸을 재평가해도 구간 기록과 학습
    # 기계가 사라지면 안 된다.  재평가한 기계는 따로 적는다.
    for _k in ("grad_clip", "grad_clip_override", "grad_norm_max", "grad_clip_hits", "patience_steps",
               "seed", "tag", "grad_stats_note", "resumes", "host",
               "total_cap_h", "total_cap_reached", "total_cap_exact", "nan_guard", "milestone",
               # 평가 시작점 전 천장 표시 -- 재평가가 지우면 총 상한에서 학습 전 가중치의 지표가 실린다
               "ceiling_before_gate"):
        if _k in PRIOR:
            _RES[_k] = PRIOR[_k]
    _RES["eval_host"] = HOST
    if isinstance(PRIOR.get("recipe"), dict):
        _RES["recipe"] = {**PRIOR["recipe"], "eval_protocol": _recipe["eval_protocol"]}
    else:
        _RES.pop("recipe", None)         # 레시피 기록이 없던 학습 -- 재평가로 현재가 되지 않는다
# 임시 파일 + 이름 바꾸기.  reconcile · 표 · 디스패처가 이 파일을 수시로 읽는다 --
# 쓰는 도중에 읽으면 JSON 이 깨져 보인다.
_out = f"{a.results_dir}/pancoll_{a.model}_{a.dataset}{a.tag}.json"
os.makedirs(a.results_dir, exist_ok=True)
with open(_out + f".tmp.{os.getpid()}", "w") as _f:
    json.dump(_RES, _f, indent=1)
os.replace(_out + f".tmp.{os.getpid()}", _out)
if BEST != BEST_PUB and BEST.exists():
    # 연장 구간의 새 best -- 결과 JSON 을 쓴 **직후** 옮긴다 (그 전까지 best.pt 는 싣고 있던 결과의 것이었다).
    os.replace(BEST, BEST_PUB)
    log(f"[이어 돌리기] 연장 구간의 best(step {ck['step']:,}) 를 결과 JSON 과 함께 {BEST_PUB.name} 로 옮겼다")
log("[done]")
if SUMMARY:
    # 디스패처 `py` 줄의 끝 표시 (`--run_root` 와 함께 준 `--out`).  **완전히 성공했을 때만** 쓴다 --
    # 디스패처는 이 파일이 있으면 끝난 줄로 본다.  FR 이 빠진 결과는 표에서 `--` 이고 민감도 판정(HQNR)도
    # 못 한다: 요약을 쓰지 않고 실패로 끝낸다.  결과 JSON 은 남았으니 같은 인자에 --eval_only 를 붙여
    # 평가만 다시 낸다 (클립 덮어쓰기 · 레시피는 이전 결과에서 그대로 가져온다).
    if FR_SKIPPED:
        log(f"[요약 안 씀] FR 이 빠졌다 ({FR_SKIPPED[:120]}) -- 결과 {_out} 는 남았다.  --eval_only 로 "
            f"평가만 다시 내라")
        raise SystemExit(f"[요약 안 씀] {a.model}/{a.dataset}{a.tag}: FR 이 빠졌다 -- --eval_only 로 다시 평가")
    _SUM = {"model": a.model, "dataset": a.dataset, "tag": a.tag, "seed": a.seed, "result": _out,
            "run_dir": str(run), "reeval_only": bool(a.eval_only),
            **{k: _RES.get(k) for k in ("grad_clip", "grad_clip_override", "grad_norm_max", "grad_clip_hits",
                                        "best_step", "last_step", "steps_planned", "stop_reason",
                                        "wall_hours")},
            "rr": _RES["rr"], "fr": _RES["fr"], "recipe": _RES.get("recipe"),
            "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(os.path.abspath(SUMMARY)), exist_ok=True)
    with open(SUMMARY + f".tmp.{os.getpid()}", "w") as _f:
        json.dump(_SUM, _f, indent=1)
    os.replace(SUMMARY + f".tmp.{os.getpid()}", SUMMARY)
    log(f"[요약] {SUMMARY}")
