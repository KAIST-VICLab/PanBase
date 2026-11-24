"""이어 돌리기(resume)에 필요한 것 중 **학습 스크립트 밖에 둘 수 있는 것** (2026-09-24).

왜 따로 두나.  `train_pancoll.py` 는 스크립트형이라 import 하는 순간 인자를 읽고 학습을
시작한다.  로그 사슬 해석처럼 틀리기 쉬운 부분을 거기 넣으면 시험할 방법이 없다 -- 그래서
부작용 없는 함수만 여기 모으고, 학습기와 시험이 같은 함수를 쓴다.

**로그 사슬이 왜 필요한가.**  `runs_pancoll/<칸>/log.txt` 는 런마다 **이어 쓴다**.  한 파일
안에 서로 무관한 예전 런(옛 레시피, 5시간 천장 시절, 재평가)이 줄줄이 붙어 있다 -- MDCUN/qb
의 로그 앞 587줄은 T=4 시절(params 98,347)의 런이고, 지금 가중치는 588줄부터의 T=6 런이다.
파일 전체에서 valid 점을 모으면 두 런의 곡선이 한 곡선으로 섞인다.  `--eval_only` 의 복구
경로가 실제로 그렇게 모으고 있었다.

그래서 머리줄 종류를 구별한다:
  `[M] venue · ds · ...`                  처음부터 시작한 학습 (예전 로그도 전부 이 형식)
  `[M] [이어 돌리기 구간] venue · ...`     이어 돌리기 구간
  `[M] [재평가] venue · ...`               `--eval_only` (예전 로그는 표식 없이 뒤에
                                           `[eval_only]` 줄이 온다 -- 그것으로도 알아본다)
사슬 = '처음부터' 머리줄 하나 + 그 가중치를 이은 이어 돌리기 구간들 (`chains` 설명).

아래쪽에는 이어 돌리기 자동 재큐의 멈춤 장치(거부 · 시도 기록, `resume-ok` 보류)와 칸당 총 학습
상한(`TOTAL_CAP_H`, 사용자 결정 2026-09-24 14:10)이 있다.
"""
from __future__ import annotations

import ast
import json
import os
import re

#: 이어 돌리기 구간 머리줄 표식.  바꾸면 예전 로그를 못 읽는다 -- 바꾸지 말 것.
RESUME_MARK = "[이어 돌리기 구간]"
#: 재평가 머리줄 표식.
EVAL_MARK = "[재평가]"
#: 이어 돌리기 시작 줄.  `from step N` 이 구간의 출발 step 이다 (그보다 뒤의 점은 버려졌다).
RESUME_LINE = "[이어 돌리기]"

_HDR = re.compile(
    r"^\[(?P<model>[^\]]+)\] (?P<mark>\[이어 돌리기 구간\] |\[재평가\] )?(?P<venue>.*?) · "
    r"(?P<ds>wv3|qb|gf2) · params (?P<params>[\d,]+|\?) · batch (?P<batch>\d+) · "
    r"epochs (?P<epochs>\d+) \(= (?P<steps>[\d,]+) step, (?P<spe>\d+)/epoch\) · "
    r"loss (?P<loss>\S+) · opt (?P<opt>.*)$")
_VALID = re.compile(r"\[valid\] step\s+(\d+)\s+PSNR\s+(-?[\d.]+|nan|inf|-inf)(.*)$")
_STEP = re.compile(r"^step\s+(\d+)/(\d+).*?([\d.]+)s/step")
_FROM = re.compile(r"^\[이어 돌리기\] mode=(\w+) from step ([\d,]+)")
#: 총 상한 때문에 멈춘 `[정지]` 줄의 표식 (`CAP_MARK`).  숫자는 가리지 않는다 -- 상한이 바뀌어도 예전 로그를 읽는다.
_CAPRE = re.compile(r"\[총 [\d.]+h 상한\]")


def norm(x):
    """JSON 왕복.  튜플은 리스트로, 모르는 객체는 문자열로 -- 결과 JSON 과 같은 모양이 된다.

    best.pt · state.pt 에 레시피를 넣을 때도 이것을 거친다.  `torch.load` 의 기본값이
    `weights_only=True` 라, 원시 자료형 밖의 객체가 섞이면 **읽는 쪽**(render_visual,
    redo_zeroshot) 이 죽는다."""
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def parse_header(line: str):
    """학습 머리줄 -> dict.  아니면 None.  `kind` 는 fresh | resume | eval."""
    m = _HDR.match(line.rstrip("\n"))
    if not m:
        return None
    mark = (m.group("mark") or "").strip()
    kind = {"": "fresh", RESUME_MARK: "resume", EVAL_MARK: "eval"}[mark]
    try:
        opt = ast.literal_eval(m.group("opt"))
    except (ValueError, SyntaxError):
        opt = m.group("opt")
    p = m.group("params")
    return {"kind": kind, "model": m.group("model"), "dataset": m.group("ds"),
            "params": None if p == "?" else int(p.replace(",", "")),
            "batch": int(m.group("batch")), "epochs": int(m.group("epochs")),
            "steps": int(m.group("steps").replace(",", "")), "spe": int(m.group("spe")),
            "loss": m.group("loss"), "opt": opt}


def segments(text: str):
    """로그를 머리줄마다 끊어 구간 목록으로.  머리줄 앞의 줄은 버린다."""
    out, cur = [], None
    for ln in text.splitlines():
        h = parse_header(ln)
        if h is not None:
            cur = {"kind": h["kind"], "header": h, "start": 0 if h["kind"] == "fresh" else None,
                   "resume_mode": None, "vhist": [], "best_marks": [], "last_step": None,
                   "sps": None, "stop": None, "total_cap": False, "done": False}
            out.append(cur)
            continue
        if cur is None:
            continue
        s = ln.strip()
        if s.startswith("[eval_only]") and cur["kind"] == "fresh":
            cur["kind"] = "eval"             # 표식 이전의 재평가 머리줄
        m = _FROM.match(s)
        if m:
            cur["resume_mode"] = m.group(1)
            cur["start"] = int(m.group(2).replace(",", ""))
            continue
        m = _VALID.search(ln)
        if m:
            st, ps = int(m.group(1)), float(m.group(2))
            cur["vhist"].append((st, ps))
            if "*best*" in m.group(3):
                cur["best_marks"].append((st, ps))
            continue
        m = _STEP.match(s)
        if m:
            cur["last_step"], cur["sps"] = int(m.group(1)), float(m.group(3))
            continue
        # 규칙 설명 줄("[정지] 수렴 정지: ...")과 실제로 멈춘 줄("[정지] 수렴 -- ...")을 가른다.
        if s.startswith("[정지] 안전 천장") or s.startswith("[정지] 구조 확정 전 천장"):
            cur["stop"] = "time_ceiling"
            # 구간 천장이 아니라 칸당 총 상한이 멈춘 구간이다 -- 재평가 복구가 이 사실을 되살린다.
            cur["total_cap"] = bool(_CAPRE.search(s))
        elif s.startswith("[정지] 수렴 --"):
            cur["stop"] = "converged"
        elif s.startswith("[정지] 시험 정지"):
            cur["stop"] = "test_stop"
        elif s.startswith("[정지] NaN 가드 소진"):      # `nan_guard.STOP_LINE` (torch 없이 읽으려고 글자로 둔다)
            cur["stop"] = "nan_guard_exhausted"
        elif s == "[done]":
            cur["done"] = True
    for c in out:
        # 평가점이 마지막 `step N/..` 줄보다 뒤일 수 있다 (500 step 마다 찍으므로).
        pts = [st for st, _ in c["vhist"]] + ([c["last_step"]] if c["last_step"] else [])
        c["reached"] = max(pts) if pts else None
    return out


def _chain_summary(segs):
    """사슬(처음부터 구간 + 이어 돌리기 구간들)의 누적 곡선 · 벽시계 · 마지막 상태."""
    vh, wall, wall_ok = [], 0.0, True
    for s in segs:
        if s["kind"] == "resume":
            # 이 구간은 `start` 의 가중치에서 출발했다.  그보다 뒤의 점은 버린 가중치의 것이다.
            vh = [(st, p) for st, p in vh if st <= s["start"]]
        vh += s["vhist"]
        if s["last_step"] is not None and s["sps"] is not None:
            # 로그의 s/step 은 **이 구간 시작부터의** 평균이다 (예전 로그는 start=0).
            wall += (s["last_step"] - s["start"]) * s["sps"] / 3600
        elif s["vhist"]:
            wall_ok = False                  # 점은 있는데 속도 줄이 없다 -- 모른다
    last = segs[-1]
    stop = last["stop"]
    if stop is None:
        stop = "schedule_complete" if (last["reached"] and last["reached"] >= last["header"]["steps"]
                                       ) else ("interrupted" if last["reached"] else None)
    trains = [s for s in segs if s["reached"] is not None]
    return {"segments": segs, "vhist": vh, "wall_hours": wall if wall_ok else None,
            "last_step": (trains[-1]["reached"] if trains else None), "stop_reason": stop,
            "header": segs[-1]["header"], "fresh_header": segs[0]["header"],
            "n_segments": len(segs), "total_cap": bool(last.get("total_cap"))}


def chains(text: str):
    """모든 사슬 (**마지막으로 움직인 순서**, 오래된 것부터).  재평가 구간과 출발 step 이 없는
    (시작하다 거부된) 이어 돌리기 구간은 사슬에 넣지 않는다.

    이어 돌리기 구간은 **그 가중치를 낸 사슬**에 붙인다 -- 로그에서 바로 앞의 '처음부터'
    머리줄이 아니다.  warm 구간은 best.pt 에서 출발하므로 출발 step 에 `*best*` 가 찍힌 가장
    최근 사슬이 그 주인이다.  그 사이에 처음부터 돈 런이 첫 평가 전에 죽으면 best.pt 는
    그대로 남는다 (ARConv/qb 로그에 실제로 있다: 다시 돈 런이 eval_after_frac 구간 안의
    94,500 에서 끊겼다).  위치로 붙이면 이어 돌린 구간이 죽은 런에 붙어, 그 뒤의 모든
    읽기(다음 warm 의 곡선 · 벽시계, 재평가 복구의 평탄 판정)가 앞 구간 곡선을 잃는다
    (검토, 2026-09-24).
    exact 구간은 위치 그대로다 -- state.pt 는 처음부터 도는 런이 시작할 때 치우므로
    (train_pancoll), exact 로 이어 붙는 사슬은 언제나 로그의 마지막 사슬이다.
    맞는 사슬이 없으면 예전처럼 위치로 붙인다."""
    groups, cur = [], None
    for i, s in enumerate(segments(text)):
        s["_pos"] = i
        if s["kind"] == "fresh":
            cur = [s]
            groups.append(cur)
        elif s["kind"] == "resume" and cur is not None and s["start"] is not None:
            tgt = cur
            if s["resume_mode"] != "exact":
                for g in reversed(groups):
                    if any(st == s["start"] for sg in g for st, _ in sg["best_marks"]):
                        tgt = g
                        break
            tgt.append(s)
            cur = tgt                     # 다음 이어 돌리기 구간도 이 사슬을 잇는다
    groups.sort(key=lambda g: g[-1]["_pos"])
    return [_chain_summary(c) for c in groups]


def find_chain(text: str, step=None, psnr=None, tol=6e-4):
    """best.pt 를 낸 사슬.  `step` 을 주면 그 step 에 `*best*` 가 찍힌 **가장 최근** 사슬,
    `psnr` 까지 주면 값도 맞아야 한다 (로그는 소수 셋째 자리).  없으면 None.
    `step` 이 None 이면 마지막으로 움직인 사슬."""
    cs = chains(text)
    if step is None:
        return cs[-1] if cs else None
    for c in reversed(cs):
        for s in c["segments"]:
            for st, p in s["best_marks"]:
                if st == step and (psnr is None or abs(p - float(psnr)) <= tol):
                    return c
    return None


def header_mismatch(h, batch, epochs, loss, opt, params=None):
    """로그 머리줄과 지금 레시피가 다른 항목.  빈 목록이면 확인되지 않은 불일치는 없다."""
    bad = []
    if h["batch"] != batch:
        bad.append(f"batch {h['batch']} != {batch}")
    if h["epochs"] != epochs:
        bad.append(f"epochs {h['epochs']} != {epochs}")
    if h["loss"] != loss:
        bad.append(f"loss {h['loss']} != {loss}")
    if norm(h["opt"]) != norm(opt):
        bad.append(f"opt {h['opt']} != {opt}")
    if params is not None and h.get("params") is not None and h["params"] != params:
        bad.append(f"params {h['params']:,} != {params:,}")
    return bad


def derived_seed(seed: int, start: int) -> int:
    """warm 이어 돌리기의 표본 추출 씨드.  (seed, 출발 step) 에서 결정적으로 나온다.

    원래 씨드를 그대로 다시 쓰면 첫 구간과 **같은 표본 순서**를 처음부터 다시 밟는다 --
    그것은 이어 돌리기가 아니라 같은 데이터를 두 번 보는 것이다."""
    return (int(seed) * 1_000_003 + int(start) * 7_919 + 0x5EED) % (2 ** 63 - 1)


def atomic_save(obj, path):
    """임시 파일에 쓴 뒤 이름을 바꾼다.  쓰다 죽어도 반쯤 쓴 파일이 원래 이름에 남지 않는다
    ([[partial-artifacts-that-look-complete]])."""
    import torch
    path = str(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 자동 재큐의 멈춤 장치
# 이어 돌리기는 reconcile(15분마다)이 넣고 디스패처가 띄운다.  그런데 학습기의 거부는 **결정적**이다
# (best.pt step != 결과 JSON best_step, 머리줄 불일치, 확인 근거 없음, 일정 끝 ...) -- 거부해도 결과
# JSON 은 그대로라 칸은 여전히 천장-상승이고, 다음 주기가 같은 줄을 또 맨 앞에 넣는다.  매번 카드에
# 데이터를 올리고 모델을 짓고 log.txt 에 거부 두 줄을 남긴 채 rc=1 로 끝나는 고리다 (검토, 2026-09-24).
# 도중에 죽는 구간(OOM 등)도 같다.  그래서 거부와 시도를 **런 폴더의 기록된 파일**로 남기고,
# reconcile · 디스패처 · 현황판이 그 파일 하나를 읽는다 ([[one-gate-not-two]]).
#
# 기록의 효력은 **그 칸의 결과 JSON 이 바뀔 때까지**다 (내용 해시로 비교한다 -- 두 호스트의 results 는
# rsync 로 맞춰지므로 어느 기계에서 봐도 같은 값이다).  새 구간이 결과를 쓰면 예전 거부 · 시도는 저절로
# 무효가 된다.  사람이 원인을 고쳤으면 파일을 지운다 -- 원격에서 난 기록은 reconcile.pull() 이 로컬로도
# 가져오므로 **두 호스트 모두**에서 지워야 한다.
REFUSED = "resume_refused.json"
ATTEMPTS = "resume_attempts.json"
#: 결과 JSON 이 한 번도 안 바뀐 채 띄운 이어 돌리기가 이만큼 쌓이면 자동으로는 더 넣지 않는다.
MAX_ATTEMPTS = 3
#: hold.txt 사유에 이 낱말이 있으면 그 칸은 **이어 돌리기만** 허용한다 -- 처음부터 도는 줄(plain)과
#: reconcile 의 재큐는 여전히 보류로 본다.  보류를 풀어야 이어 돌릴 수 있게 두면, 이어 돌리기가 거부되거나
#: 죽는 순간 다음 주기가 처음부터 도는 줄을 넣어 best.pt 를 덮는다 (PanTCR, 검토 2026-09-24).
RESUME_OK = "resume-ok"


def resume_ok_hold(reason) -> bool:
    """hold.txt 의 사유가 이어 돌리기를 허용하는가 (`resume-ok` 낱말)."""
    return bool(reason) and RESUME_OK in re.findall(r"[\w-]+", str(reason))


def result_sha(path):
    """결과 JSON 의 내용 해시 (없으면 None)."""
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except OSError:
        return None


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path, obj):
    path = str(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _stamp():
    import datetime
    import socket
    return {"at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname()}


def record_refusal(run_dir, result_json, reason, source):
    """이어 돌리기 거부를 `run_dir/resume_refused.json` 에 남긴다.  실패해도 예외를 내지 않는다
    (거부 자체는 이미 결정됐다 -- 기록 실패가 그 결정을 가리면 안 된다)."""
    try:
        os.makedirs(run_dir, exist_ok=True)
        _write_json(os.path.join(run_dir, REFUSED),
                    dict(_stamp(), reason=str(reason), source=source,
                         result_sha=result_sha(result_json)))
    except OSError:
        pass


def record_attempt(run_dir, result_json, **info):
    """이어 돌리기 시도 하나를 `run_dir/resume_attempts.json` 에 더한다.  지금 결과 JSON 과 해시가
    다른(= 그 뒤에 결과가 나온) 예전 시도는 버린다.  돌려주는 값: 지금 결과에 대한 시도 수."""
    p = os.path.join(run_dir, ATTEMPTS)
    sha = result_sha(result_json)
    prev = [x for x in (_read_json(p) or []) if isinstance(x, dict) and x.get("result_sha") == sha]
    prev.append(dict(_stamp(), result_sha=sha, **info))
    try:
        _write_json(p, prev)
    except OSError:
        pass
    return len(prev)


def block_reason(run_dir, result_json, attempts=True):
    """이 칸의 이어 돌리기를 (자동으로) 다시 띄우지 말아야 할 이유.  없으면 None.

    `attempts=False` 면 거부 기록만 본다 (학습기가 쓴다 -- 사람이 조사한 뒤 손으로 다시 돌리는 것은
    시도 횟수로 막지 않는다).  reconcile · 디스패처는 둘 다 본다."""
    sha = result_sha(result_json)
    r = _read_json(os.path.join(run_dir, REFUSED))
    if isinstance(r, dict) and r.get("result_sha") == sha:
        return (f"이어 돌리기가 거부됐다 ({r.get('at')}, {r.get('host')}, {r.get('source')}): "
                f"{r.get('reason')} -- 결과 JSON 이 바뀌거나 사람이 {REFUSED} 를 지울 때까지 "
                f"다시 넣지 않는다")
    if attempts:
        att = [x for x in (_read_json(os.path.join(run_dir, ATTEMPTS)) or [])
               if isinstance(x, dict) and x.get("result_sha") == sha]
        if len(att) >= MAX_ATTEMPTS:
            return (f"이어 돌리기를 {len(att)}번 띄웠는데 결과 JSON 이 한 번도 안 나왔다 "
                    f"(마지막 {att[-1].get('at')}, {att[-1].get('host')}) -- 도중에 죽는다.  "
                    f"로그를 보고 원인을 고친 뒤 {ATTEMPTS} 를 지워라")
    return None


# ---------------------------------------------------------------- 칸당 총 학습 상한
# 사용자 결정 (2026-09-24 14:10): "96시간 같은 상한으로".  "오르면 계속"은
# 끝나는 조건을 레시피(평탄 · patience · 일정 완주)에만 두었다 -- 공식 일정을 다 돌리면 S2DBPN/gf2 는
# 291h, HFIN/gf2 164h, ARConv/gf2 127h 라 최악에는 카드 몇 장이 1~2주 묶인다.  그래서 **칸 하나가 받는
# 학습 시간의 합**(구간 합 = 결과 JSON 의 누적 `wall_hours`)에 상한을 둔다.  구간 천장(`--max_hours`,
# 48h)은 그대로다 -- 이어 도는 구간의 천장은 둘 중 먼저 오는 쪽이다 (`segment_budget`).
#
# 상한에서 잘린 칸은 아직 오르는 중이어도 **그대로 싣는다** (reconcile.classify 의 안내 줄 · 원장의
# 공개 표).  더 돌리면 오를 수 있었다는 뜻이라 베이스라인에 불리한 쪽이다 -- 숨기지 않는다.
#
# 이 파일이 상한의 **유일한** 정의다.  학습기 · reconcile · 디스패처 · 표 · 원장이 모두 여기서 읽는다
# ([[one-gate-not-two]]).
#: 칸당 총 학습 상한 (시간, 구간 합).
TOTAL_CAP_H = 96.0
#: 남은 예산이 이 이하이면 **소진**으로 본다.  그보다 짧은 구간은 데이터 적재 · 모델 짓기 · 최종 평가가
#: 대부분이라 학습이 거의 늘지 않는다 -- 그런 구간은 띄우지 않는다.
MIN_SEGMENT_H = 4.0
#: 총 상한 때문에 멈춘 구간의 `[정지] 안전 천장 ...` 줄에 넣는 표식 (`segments` 가 읽는다).
CAP_MARK = f"[총 {TOTAL_CAP_H:g}h 상한]"
CAP_DECIDED = "사용자 결정 2026-09-24"


def total_capped(d) -> bool:
    """결과 JSON `d` 의 칸이 총 학습 상한을 다 썼는가.  **기록된 필드만** 본다.

      - `total_cap_reached` -- 상한이 구간 천장보다 먼저 걸려 멈춘 구간을 학습기가 이렇게 적는다.
      - 누적 `wall_hours` >= TOTAL_CAP_H - MIN_SEGMENT_H -- 남은 예산이 한 구간이 안 된다.  학습기는
        이런 칸의 이어 돌리기를 거부한다(`segment_budget` 의 exhausted 와 같은 경계).  상한 이전 코드로
        돈 구간(48h 구간 천장만 아는 학습기)이 쓴 결과도 이것으로 잡힌다.

    `wall_hours` 는 새 코드의 결과면 구간 합이고, 예전 결과면 한 구간짜리다 -- 예전 구간 천장은 48h 를
    넘은 적이 없어 예전 결과가 여기 걸리지 않는다."""
    if not isinstance(d, dict):
        return False
    if d.get("total_cap_reached"):
        return True
    try:
        return float(d.get("wall_hours")) >= TOTAL_CAP_H - MIN_SEGMENT_H
    except (TypeError, ValueError):
        return False


def cap_words(d):
    """상한 칸(`total_capped`)을 부르는 말 `(머리, 덧붙임)`.  안내 줄 · 현황판 · 원장이 같이 쓴다.

    `total_capped` 는 누적 92h(= 96 - 한 구간 4h)부터 참이라, 상한 칸이 모두 96h 를 채운 것은 아니다 --
    예컨대 상한 이전 코드로 45.54h 에서 48h 구간을 이어 돈 MutInf/qb 는 약 93.5h 에서 멈춘다.  그런 칸을
    '96h 에서 잘림'이라 부르면 받은 학습을 부풀려 적는 셈이다.  그래서
      - 상한이 실제로 구간을 끊었거나(`total_cap_reached`) 누적이 96h 이상이면 머리는 '총 96h 상한에서 잘림',
      - 아니면 '총 96h 상한 소진'이고, 덧붙임에 남은 예산이 한 구간이 안 된다는 사실을 적는다.
    누적 시간은 부르는 쪽이 늘 함께 적는다."""
    try:
        wh = float(d.get("wall_hours") or 0.0)
    except (TypeError, ValueError):
        wh = 0.0
    if d.get("total_cap_reached") or wh >= TOTAL_CAP_H:
        return f"총 {TOTAL_CAP_H:g}h 상한에서 잘림", ""
    return (f"총 {TOTAL_CAP_H:g}h 상한 소진",
            f"남은 {max(TOTAL_CAP_H - wh, 0.0):.1f}h 는 한 구간({MIN_SEGMENT_H:g}h)이 안 된다")


def segment_budget(max_hours, prior_hours):
    """이 구간이 쓸 수 있는 시간.

    돌려주는 값 (dict):
      ceiling_h    이 구간의 천장 = min(구간 천장, 총 상한 - 앞 구간 누적)
      cap_binds    총 상한이 구간 천장보다 **먼저** 걸린다 -- 천장에서 멈추면 total_cap_reached
      remaining_h  총 상한까지 남은 시간 (음수일 수 있다)
      exhausted    남은 시간이 MIN_SEGMENT_H 이하 -- 이어 돌리지 않는다 (`total_capped` 와 같은 경계)
    """
    prior = float(prior_hours or 0.0)
    rem = TOTAL_CAP_H - prior
    return {"ceiling_h": min(float(max_hours), rem), "cap_binds": rem < float(max_hours),
            "remaining_h": rem, "exhausted": rem <= MIN_SEGMENT_H}
