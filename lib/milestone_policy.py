"""계단식 lr 감쇠 전에 patience 로 멈춘 칸의 연장 -- 사전 등록 규칙 (2026-09-25).

문제.  우리 patience 정지는 "평탄하면 더 돌려도 같다" 를 가정한다.  계단식(StepLR · MultiStepLR) 일정에서는
틀리다 -- 감쇠 한 번이 valid 를 올린다 (MSDDN/wv3 +0.22 dB, PanTCR/wv3 반감 +0.05~0.08 dB).
규칙.  이산 감쇠 지점이 있는 스펙은 계획 일정 안의 **마지막 감쇠 지점을 지나기 전에는** patience 로 끝내지 않는다.
그 전에 'converged' 로 멈춘 칸은 **이어 돈다** (state.pt 가 있으면 exact, 없으면 best.pt 에서 warm -- 원장 공개).
끝나는 조건은 마지막 감쇠 뒤의 patience(그 지점부터 다시 센다) · 일정 완주 · 칸당 96h 상한 중 먼저.
cosine · 상수 · 선형 일정은 대상이 아니다.  매 에폭 감쇠(StepLR step_size=1, WFANet)도 계단이 아니라 지수 감쇠라 뺀다.

**학습기가 스케줄러를 부르는 법** (train_pancoll.py): `if sched is not None and step % spe == 0: sched.step()` --
에폭당 한 번, 그 에폭 마지막 갱신 **뒤**.  그래서 에폭 e 의 감쇠(StepLR: e % step_size == 0 · MultiStepLR: e in
milestones)는 step e*spe 에서 걸리고, 줄어든 lr 로 도는 첫 갱신은 e*spe + 1 이다.  e == epochs 의 감쇠는 마지막 갱신
뒤라 학습에 닿지 않는다 -- '일정 안' 은 0 < e < epochs (CANConv qb/gf2 의 StepLR(500) · 500 epoch 은 감쇠가 없다).
PanTCR 저장소는 에폭당 두 번 부르지만 우리 레시피는 한 번이다 -- 여기는 **우리 학습기의** 지점을 센다
(StepLR(500) · 700 epoch -> epoch 500 하나).  warm 이어 돌리기는 스케줄러를 START // spe 번 앞으로 감으므로 같은
지점을 지난다.  학습기는 모르는 kind 에는 스케줄러를 만들지 않는다 (상수 lr) -- 여기서도 계단이 아니다.
스펙 `min_lr`(AWFLN)은 학습기의 StepLR 이 읽지 않는다 -- 여기서도 읽지 않는다.
train_step 안에서 lr 을 직접 바꾸는 스펙(BNNPan · TMDiff, 표에 없는 extra)은 `sched` 에 드러나지 않아 대상이 아니다.

레지스트리에서 쓰는 kind (2026-09-25): step (ADWM · ARConv · AWFLN · CANConv · FusionMamba · MMNet · MSDDN · PanTCR ·
WFANet[step_size=1] + extra 여럿) · multistep (RAMSF) · cosine (MMMamba · PAN-Crafter · FAME-Net) · 없음(상수).

이 파일이 판정의 **유일한** 정의다 -- reconcile(게이트 안내 · 계획자) · 디스패처 · 학습기 · 현황판 · 원장이 여기서
읽는다 ([[one-gate-not-two]]).  기록된 필드(결과 JSON 의 recipe.sched · epochs · steps_planned · stop_reason ·
last_step · best_step · patience_steps · resumes · milestone)만 본다.  **스위치(`SWITCH`)가 없으면 어느 쪽도 이 규칙을
쓰지 않는다** -- 게이트 · 계획자 · 학습기 동작이 예전과 바이트 단위로 같다 (교훈: 검토 전에 라이브 동작을 바꾸지 않는다).
torch 없이 읽는다 (디스패처 · 계획자가 부른다).

수정 단계 (검토 2026-09-25) -- 스위치가 켜졌을 때:
  - **스펙 단위 규칙이다.**  처음부터 도는 런(본 칸 · seed 복제)도 마지막 감쇠 전에는 patience 로 멈추지 않는다 (결과
    JSON 최상위 `milestone` 에 적는다).  이미 멈춘 seed 복제(`_sS`)도 본 칸처럼 이어 돈다 -- 본 칸만 늘리면 seed 산포가
    두 정지 규칙을 섞는다.
  - 대상 판정은 멈추는 판정과 **같은 함수**다: 'converged' 는 이 규칙 아래서도 멈췄을 정지가 아니면 대상 (`patience_due`).
  - 이 규칙을 적용한 사슬이 구간 천장에서 잘리면: 마지막 감쇠 전이면 평탄해도, 지났으면 아직 오르는 동안 잇는다 -- 그동안
    지금 결과는 **유효로 남는다** (천장-상승으로 넘기면 게이트에서 빠져 표에서 사라진다).
  - 연장 구간은 싣고 있는 best.pt 를 건드리지 않는다: 새 best 는 `STAGE` 에 쓰고 결과 JSON 을 쓴 직후 best.pt 로 옮긴다
    (게이트 유효 칸의 best.pt 를 결과 JSON 과 짝으로 읽는 곳 -- make_checkpoints · render_visual · zero-shot QA 의 sha --
    이 연장 중에도 맞는다).  사슬의 첫 연장은 그 전의 결과 JSON 과 best.pt 를 `PRE_DIR` 에 sha256 과 함께 남긴다
    (기준 · 전후 공개).
"""
from __future__ import annotations
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

import os

#: 스위치.  **없으면 아무 곳도 이 규칙을 쓰지 않는다** (fail-closed).  검토가 끝난 뒤 사람(주 세션)이 만든다 --
#: 양쪽 호스트에 코드를 맞추고 디스패처를 다시 띄운 **뒤에**.  지우면 새 연장이 멈춘다 (도는 구간은 그대로).
SWITCH = (_GB + "/_logs/milestone_policy.enabled")
#: 이 규칙이 일으킨 이어 돌리기 구간의 `resumes[].reason`.
REASON = "milestone"
#: 안내 · 현황판 · 원장의 머리말.
NOTE_HEAD = "연장 (lr 감쇠 전 정지)"
DECIDED = "사전 등록 2026-09-25"
#: 이산 감쇠 지점을 가질 수 있는 kind (학습기가 StepLR · MultiStepLR 을 짓는 두 kind).
DISCRETE = ("step", "multistep")
#: 연장 구간의 새 best (런 폴더 안).  결과 JSON 을 쓴 직후 best.pt 로 옮긴다 -- 그 전까지 best.pt 는 싣고 있는 결과의 것이다.
STAGE = "best_ms42.pt"
#: 사슬의 첫 연장이 그 전의 결과 JSON · best.pt 를 남기는 곳 (런 폴더 안).  덮지 않는다.
PRE_DIR = "pre42"


def enabled(path=None) -> bool:
    """스위치가 있는가.  `path` 는 시험용 (기본은 `SWITCH`)."""
    return os.path.exists(path or SWITCH)


def milestones(sched, epochs, spe):
    """스펙의 `sched` 와 계획 epochs · 에폭당 step(spe) -> 학습기가 실제로 lr 을 계단식으로 줄이는 지점.

    돌려주는 값 (dict):
      kind         step | multistep | cosine | constant | exponential(StepLR step_size=1) | nodecay(gamma=1) | unknown:<k>
      discrete     이산 감쇠 지점이 **일정 안에** 하나라도 있는가 (규칙의 대상인가)
      epochs       일정 안 감쇠 에폭 (0 < e < epochs, 오름차순)
      steps        그 step (= e * spe; 이 step 의 갱신 뒤에 감쇠가 걸린다)
      last_epoch · last_step   마지막 감쇠 지점 (없으면 None)
    """
    epochs, spe = int(epochs or 0), int(spe or 0)
    sc = sched if isinstance(sched, dict) else None
    kind = (sc or {}).get("kind")
    eps = []
    if sc is None or kind is None:
        kind = "constant"                           # 스케줄러 없음 -- lr 상수
    elif kind == "step":
        ss, g = int(sc["step_size"]), float(sc.get("gamma", 0.1))     # torch StepLR 기본 gamma 0.1
        if g == 1.0:
            kind = "nodecay"
        elif ss <= 1:
            kind = "exponential"                    # 매 에폭 감쇠 = 지수 감쇠 (계단이 아니다)
        else:
            eps = list(range(ss, epochs, ss))
    elif kind == "multistep":
        if float(sc.get("gamma", 0.1)) == 1.0:
            kind = "nodecay"
        else:
            eps = sorted({int(m) for m in sc.get("milestones") or [] if 0 < int(m) < epochs})
    elif kind != "cosine":
        kind = f"unknown:{kind}"                    # 학습기는 스케줄러를 짓지 않는다 -- 상수 lr
    steps = [e * spe for e in eps] if spe > 0 else []
    return {"kind": kind, "discrete": bool(steps), "epochs": eps if steps else [], "steps": steps,
            "last_epoch": eps[-1] if steps else None, "last_step": steps[-1] if steps else None}


def result_milestones(d):
    """결과 JSON 의 기록(recipe.sched · epochs · steps_planned)으로 `milestones`.  읽을 수 없으면 None."""
    if not isinstance(d, dict) or not isinstance(d.get("recipe"), dict):
        return None
    try:
        ep, sp = int(d.get("epochs") or 0), int(d.get("steps_planned") or 0)
    except (TypeError, ValueError):
        return None
    if ep <= 0 or sp <= 0 or sp % ep:
        return None                                  # steps = epochs * spe 여야 한다 (학습기)
    return milestones(d["recipe"].get("sched"), ep, sp // ep)


def reached(d):
    """마지막으로 도달한 step (기록된 `last_step`).  없는 예전 결과의 수렴 정지는 가장 이른 정지 step
    `best_step + patience_steps` (patience 는 best 뒤 그만큼 지나야 걸린다) -- 모르면 None."""
    try:
        if d.get("last_step") is not None:
            return int(d["last_step"])
        if d.get("stop_reason") == "converged" and d.get("patience_steps") is not None:
            return int(d.get("best_step") or 0) + int(d["patience_steps"])
    except (TypeError, ValueError):
        pass
    return None


def extended(d) -> bool:
    """이 사슬에 이 규칙(마지막 감쇠 전 patience 금지)을 적용한 구간이 있는가 (`resumes[].milestone`, 처음부터 돈
    구간은 결과 JSON 최상위 `milestone`)."""
    return isinstance(d.get("milestone"), dict) or any(
        isinstance(r, dict) and r.get("milestone") for r in (d.get("resumes") or []))


def caused(d) -> bool:
    """이 규칙 **때문에** 이어 돌린 구간이 있는가 (`resumes[].reason == REASON`) -- 표시 · 원장이 쓴다."""
    return any(isinstance(r, dict) and r.get("reason") == REASON for r in (d.get("resumes") or []))


def pending(d) -> bool:
    """이 결과의 칸을 **이 규칙으로 이어 돌려야 하는가** -- 기록된 필드만 본다 (레시피 현재 여부는 부르는 쪽이 본다:
    reconcile 은 `is_current`, 디스패처는 게이트 유효, 학습기는 스냅샷 대조).

      1. 칸당 총 학습 상한을 다 쓰지 않았다 (`resume_state.total_capped`).
      2. 기록된 스케줄에 일정 안의 이산 감쇠 지점이 있다.
      3. 정지가 patience 다 (`converged`) -- 그리고 **이 규칙 아래서는 멈추지 않았을** 정지다: 멈추는 판정과 같은 함수
         `patience_due(도달, best, patience, 마지막 지점)` 가 거짓 (도달 step 이 마지막 지점 이하면 언제나 대상 -- 그 지점의
         갱신 뒤에 감쇠가 걸리므로 같은 step 에서 멈췄으면 줄어든 lr 로는 한 번도 돌지 않았다).  patience 기록이 없는 예전
         결과는 도달 step <= 마지막 지점.
         또는 이 규칙을 적용한 사슬(`extended`)이 구간 천장에서 잘렸다 -- 구간 천장은 끝나는 조건이 아니다: 마지막 지점
         전이면 평탄해도, 지났으면 아직 오르는 동안(`still_improving`, 같은 '오르면 계속') 잇는다.  그동안 지금
         결과는 유효로 남는다 (천장-상승으로 넘기면 게이트에서 빠져 표에서 사라진다).
    일정 완주 · NaN 가드 소진 · 시험 정지는 해당하지 않는다."""
    if not isinstance(d, dict):
        return False
    import resume_state as RS                          # 총 상한의 유일한 정의
    if RS.total_capped(d):
        return False
    ms = result_milestones(d)
    if not ms or ms["last_step"] is None:
        return False
    r = reached(d)
    if r is None:
        return False
    last = ms["last_step"]
    stop = d.get("stop_reason")
    if stop == "converged":
        try:
            pat = int(d["patience_steps"])
        except (KeyError, TypeError, ValueError):
            return r <= last
        return not patience_due(r, int(d.get("best_step") or 0), pat, last)
    if stop == "time_ceiling" and bool(d.get("hit_time_ceiling")) and extended(d):
        return r <= last or bool(d.get("still_improving"))
    return False


def patience_due(step, best_at, patience, last_step) -> bool:
    """이 규칙 아래의 patience 정지 판정.  마지막 감쇠 지점까지는 멈추지 않고, 그 뒤에는 patience 를 그 지점부터
    다시 센다: step - max(best_at, 마지막 지점) >= patience.  `last_step` 이 None 이면 예전 규칙 그대로다."""
    if last_step is None:
        return step - best_at >= patience
    return step - max(best_at, last_step) >= patience


def record(ms, patience):
    """이어 돌리기 구간 기록(`resumes[].milestone`) -- 이 구간에 규칙을 적용했다는 표시와 근거."""
    return {"rule": "milestone-extension", "kind": ms["kind"], "milestone_epochs": list(ms["epochs"]),
            "milestone_steps": list(ms["steps"]), "last_milestone_step": ms["last_step"],
            "last_milestone_epoch": ms["last_epoch"], "patience_steps": int(patience),
            "patience_from": "max(best_step, last_milestone_step)"}


def describe(d):
    """안내 · 현황판 · 원장이 쓰는 사실 (pending 인 칸).  (도달 step, 마지막 지점 step, 그 에폭, 정지 말)."""
    ms = result_milestones(d) or {}
    stop = "patience 로" if d.get("stop_reason") == "converged" else "구간 천장에서"
    return reached(d), ms.get("last_step"), ms.get("last_epoch"), stop


def phase(d) -> str:
    """마지막 감쇠 지점에 대한 위치 (pending 인 칸) -- 안내 · 계획자 · 현황판 · 원장이 같은 말을 쓴다."""
    r, ls, _, _ = describe(d)
    if r is None or ls is None or r <= ls:
        return "전이다"
    if d.get("stop_reason") == "converged":
        return (f"뒤 {r - ls:,} step 에서 멈췄다 -- patience {int(d.get('patience_steps') or 0):,} step 이 그 지점부터 "
                f"안 찼다")
    return f"뒤다 -- 아직 +{float(d.get('plateau_tail_rise_db') or 0):.3f} dB 오르는 중이다"


def where(d) -> str:
    """계획자 알림 · 현황판의 괄호 안 사실 한 줄 (pending 인 칸)."""
    r, ls, le, stop = describe(d)
    return (f"step {r:,}/{int(d.get('steps_planned') or 0):,} 에서 {stop} 멈춤, 마지막 lr 감쇠 step {ls:,} = epoch {le}"
            + ("" if r <= ls else f" {phase(d)}"))


def note(key, d, hold=None, blocked=None) -> str:
    """게이트(`reconcile.classify`)의 안내 줄 -- 지금 결과는 유효로 싣는다.  `hold` (resume-ok 없는 보류 사유) ·
    `blocked` (이어 돌릴 수 없는 이유) 가 있으면 '이어 돈다' 고 적지 않는다 (reconcile.milestone_state 가 정한다)."""
    r, ls, le, stop = describe(d)
    sp = int(d.get("steps_planned") or 0)
    ep = int(d.get("epochs") or 0)
    spe = sp // ep if ep else 0
    at = f" (epoch {r / spe:.0f})" if spe and r is not None else ""
    bs = f"best step {int(d.get('best_step') or 0):,}"
    if hold:
        tail = (f"대상이지만 보류라 이어 돌지 않는다 (hold.txt: {hold}) ({DECIDED}).  지금 결과({bs})를 그대로 싣는다")
    elif blocked:
        tail = (f"대상이지만 이어 돌릴 수 없다 ({blocked}) -- 사람이 정한다 ({DECIDED}).  지금 결과({bs})를 그대로 싣는다")
    else:
        tail = f"이어 돈다 ({DECIDED}).  지금 결과({bs})는 유효로 싣고, 연장이 끝나면 바뀐다"
    return (f"{key[0]}/{key[1]} {NOTE_HEAD} -- step {r:,}{at}/{sp:,} 에서 {stop} 멈췄는데 마지막 lr 감쇠 "
            f"(step {ls:,} = epoch {le}) {phase(d)} -- {tail}")
