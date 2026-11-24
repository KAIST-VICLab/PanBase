"""NaN 가드 -- 스펙 키 `nan_guard=True` 인 방법만 켠다 (선택 B 사전 등록).

왜 있나.  MutInf/gf2 가 step 570,500 에서 `Normal(loc=mu_rgb)` 에 NaN 이 들어가 죽었고 (GPPNN.py:410,
mu_rgb 는 tanh 출력이라 그 앞이 이미 NaN), best.pt(506,000)에서 가드 없이 이어 돌린 구간도 ~513,000 에서
같은 자리에서 죽었다.  이어 돌리기의 난수는 `resume_state.derived_seed(seed, START)` 라 그냥 다시 돌면
**같은 batch 순서**를 밟는다 -- 재시도로는 안 된다.  사용자가 고른 대체안이 이 가드다.

하는 일 (학습기 `train_pancoll.py` 가 step 마다 `NanGuard.run(step, fn)` 을 부른다):
  1. step 전에 떠 둔다: 모델의 **파라미터와 버퍼 전부**(named_parameters · named_buffers -- 영속 아닌 버퍼 포함),
     옵티마이저 상태(텐서는 값, 나머지는 사본)와 param_groups 의 하이퍼파라미터, 스케줄러 상태(있으면).
     AMP GradScaler 는 이 학습기에 없다 -- 있으면 여기에 넣어야 한다.
  2. step 을 돈다 (`fn` = 학습기의 한 step: forward -> 손실 -> zero_grad -> backward -> clip -> opt.step).
  3. 다음 중 하나면 떠 둔 상태로 **정확히** 되돌리고 기울기를 비우고 그 batch 를 건너뛴다:
       (a) step 이 비유한 값을 알리는 예외를 냈다 -- ValueError/RuntimeError 이고 메시지가 nan · inf ·
           non-finite · 'invalid values' 를 말한다 (torch.distributions 의 인자 검사, autograd anomaly 의
           'returned nan values' 가 여기 걸린다).  메모리 부족(OutOfMemoryError)은 아니다.
       (b) 손실이 유한하지 않다.
       (c) 갱신 뒤 파라미터 · 버퍼 · **옵티마이저 상태** 중 하나라도 유한하지 않다.  옵티마이저 상태를 같이
           보는 것은 사전 등록(파라미터)보다 한 걸음 넓다: Adam 모멘트만 inf 가 되고 가중치는 멀쩡한 step 을
           통과시키면, 다음 step 부터 떠 두는 상태 자체가 오염되어 모든 step 이 건너뛰어진다.
     그 밖의 예외는 **그대로** 올라간다 (같은 예외 객체, 가드는 아무것도 되돌리지 않는다).
  4. step 번호는 그대로 나아간다 -- 그 batch 는 **소비됐다** (표본 추출 난수 · 증강 난수 · 모델 안의 잡음
     난수는 되돌리지 않는다).  그래서 이어 돌리기 · 평가 간격 · 스케줄러의 에폭 경계는 step 번호를 그대로
     따른다.  건너뛴 step 은 이 모듈이 센다.
  5. 최근 WINDOW(10,000) step 안에 건너뛴 것이 CAP(100)번을 **넘으면** `exhausted` 가 참이 된다.  학습기는
     정지 사유 STOP_REASON 으로 멈추고 **보통의 끝 경로**(best.pt 로 RR · FR · zero-shot 평가, 결과 JSON)를
     밟는다 -- 사전 등록 대체안 C (가중치 자체가 나쁜 곳에 있다는 뜻 -> best.pt 를 싣는다).  게이트가 그 결과를
     싣는 것은 이미 학습된 체크포인트에서 이어 돈 런뿐이다 (`reconcile.guard_loadable`; 아니면 [이상], 사람이 정한다).

되돌리지 **않는** 것: 스펙이 모듈 트리 밖에 두는 상태.  MutInf 의 `Mutual_info_reg` 는 일부러 등록을 피한
모듈인데(옵티마이저에 없어 랜덤 초기화로 고정 -- registry 의 Wrap 설명), 값이 바뀌지 않으므로 되돌릴 것이
없다 (그 .grad 는 아무도 읽지 않는다).  트리 밖 캐시가 있는 다른 스펙(CANConv 의 k-means 캐시, BNNPan 의
단계 캐시)에 이 가드를 켜려면 그 상태도 여기서 다뤄야 한다 -- 그래서 opt-in 이다.

게이트와 무관하다: `nan_guard` 는 호출 가능한 필드가 아니고(`registry.spec_code_digest`) 레시피 키도 아니다
(`make_tables._RECIPE_KEYS`, 학습기의 `_recipe`).  레시피 스냅샷 · code_digest 가 바뀌지 않는다.
"""
from __future__ import annotations

import collections
import copy
import re

import torch

#: 최근 이 step 수 안에서 센다.
WINDOW = 10_000
#: 이보다 **많이** 건너뛰면 멈춘다 ("1만 step 안에 100번을 넘으면").
CAP = 100
#: 결과 JSON 에 남기는 건너뛴 step (앞에서부터).
KEEP = 200
#: 소진으로 멈춘 런의 정지 사유 (결과 JSON · state.pt 의 stop_reason).
STOP_REASON = "nan_guard_exhausted"
#: 소진 정지 줄의 머리.  `resume_state.segments` 가 이것으로 로그에서 정지 사유를 되살린다.
STOP_LINE = "[정지] NaN 가드 소진"
#: 건너뜀 줄의 머리 ('[NaN 가드] step N 건너뜀 (사유)').
SKIP_MARK = "[NaN 가드]"

# 단어 경계로 찾는다 -- 'inf' 는 'info' · 'inference' · 'Mutual_info_reg' 안에도 있다.
_MSG = re.compile(r"(?<![A-Za-z0-9_])(?:nan|[+-]?inf|infinity|infinite|non-?finite)(?![A-Za-z0-9_])"
                  r"|invalid values?|not finite", re.IGNORECASE)
_OOM = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                         getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)) if t)


def nonfinite_error(e: BaseException) -> bool:
    """이 예외가 비유한 값을 알리는가 (가드가 잡는 예외인가)."""
    if _OOM and isinstance(e, _OOM):
        return False
    return isinstance(e, (ValueError, RuntimeError)) and bool(_MSG.search(str(e)))


def _short(e: BaseException, n: int = 200) -> str:
    return f"{type(e).__name__}: {' '.join(str(e).split())[:n]}"


class NanGuard:
    """한 학습 런의 NaN 가드.  `run(step, fn)` 하나로 쓴다 (모듈 설명)."""

    def __init__(self, model, opt, sched=None, window: int = WINDOW, cap: int = CAP, keep: int = KEEP):
        self.model, self.opt, self.sched = model, opt, sched
        self.window, self.cap, self.keep = int(window), int(cap), int(keep)
        self.skipped = 0
        self.steps: list[int] = []
        self.last_reason: str | None = None
        self.recent: collections.deque[int] = collections.deque()
        #: warm 이어 돌리기: 앞 구간 결과 JSON 의 기록 (그대로 싣는다 -- 그 구간의 START 이후 건너뜀은 버린 궤적의 것)
        self.prior = None
        # (모듈, 이름) 자리를 한 번만 찾는다 -- 매 step 모듈 트리를 두 번 훑으면 MutInf 에서 ~2 ms 다.  값은
        # 매번 그 자리에서 **지금 객체**를 읽는다 (버퍼를 바꿔 끼우는 모듈도 따라간다).  공유 파라미터 · 버퍼는
        # named_parameters · named_buffers 처럼 한 번만 센다.
        self._pslots, self._bslots, seen_p, seen_b = [], [], set(), set()
        for mname, mod in model.named_modules():
            for leaf, t in mod._parameters.items():
                if t is not None and id(t) not in seen_p:
                    seen_p.add(id(t))
                    self._pslots.append((mod, leaf, f"{mname}.{leaf}" if mname else leaf))
            for leaf, t in mod._buffers.items():
                if t is not None and id(t) not in seen_b:
                    seen_b.add(id(t))
                    self._bslots.append((mod, leaf, f"{mname}.{leaf}" if mname else leaf))
        self._pname = {id(mod._parameters[leaf]): n for mod, leaf, n in self._pslots}
        self._live = None           # [(kind, key, tensor)] -- key: param/buffer 는 (모듈, 이름, 전체 이름), opt 는 (param, 키)
        self._ids = None
        self._buf = None            # 떠 둔 텐서 (_live 와 같은 순서)
        self._groups = None         # [(_live 안의 위치 목록, 버퍼 목록)] -- (장치, dtype) 별 (_foreach_copy_ 묶음)
        self._optvals = None        # {param: {key: 텐서 아닌 값}}
        self._optkeys = None        # {param: (key, ...)}  -- 구조
        self._hyper = None          # param_groups 의 'params' 밖 항목
        self._sched = None

    # ------------------------------------------------------------------ 기록
    @property
    def exhausted(self) -> bool:
        return len(self.recent) > self.cap

    def record(self) -> dict:
        """결과 JSON 의 `nan_guard`."""
        r = {"enabled": True, "skipped": int(self.skipped), "steps": list(self.steps),
             "last_reason": self.last_reason, "window_steps": self.window, "cap": self.cap,
             "steps_kept": self.keep, "exhausted": bool(self.exhausted)}
        if self.prior is not None:
            r["prior"] = self.prior
        return r

    def state(self) -> dict:
        """state.pt 의 `nan_guard` -- exact 이어 돌리기가 셈과 창을 그대로 잇는다."""
        return dict(self.record(), recent=list(self.recent))

    def load(self, st: dict | None):
        """state.pt 에 저장된 `state()` 를 되살린다 (exact 이어 돌리기)."""
        if not st:
            return
        self.skipped = int(st.get("skipped") or 0)
        self.steps = [int(x) for x in (st.get("steps") or [])]
        self.last_reason = st.get("last_reason")
        self.recent = collections.deque(int(x) for x in (st.get("recent") or []))
        if st.get("prior") is not None:
            self.prior = st["prior"]

    def note_skip(self, step: int, reason: str) -> bool:
        """건너뛴 step 하나를 센다.  소진이면 True."""
        step = int(step)
        self.skipped += 1
        if len(self.steps) < self.keep:
            self.steps.append(step)
        self.last_reason = reason
        self.recent.append(step)
        while self.recent and self.recent[0] <= step - self.window:
            self.recent.popleft()
        return self.exhausted

    # ------------------------------------------------------------------ 상태 수집
    def _collect(self):
        items = [("param", sl, sl[0]._parameters[sl[1]]) for sl in self._pslots]
        items += [("buffer", sl, b) for sl in self._bslots if (b := sl[0]._buffers.get(sl[1])) is not None]
        for p, st in self.opt.state.items():
            for k, v in st.items():
                if torch.is_tensor(v):
                    items.append(("opt", (p, k), v))
        return items

    def _label(self, kind, key):
        if kind == "opt":
            p, k = key
            return f"옵티마이저 상태 {self._pname.get(id(p), '?')}.{k}"
        return f"{'파라미터' if kind == 'param' else '버퍼'} {key[2]}"

    @torch.no_grad()
    def snapshot(self):
        """지금 상태를 떠 둔다.  구조가 그대로면 미리 잡은 버퍼에 `_foreach_copy_` 로 복사한다."""
        live = self._collect()
        ids = [id(t) for _, _, t in live]
        fast = ids == self._ids
        if fast:
            # 원본은 매번 지금 객체에서 읽는다 (떼어 둔 view 는 `p.data = ...` 뒤에 낡는다).
            try:
                for idx, dst in self._groups:
                    torch._foreach_copy_(dst, [live[i][2] for i in idx])
                self._live = live
            except RuntimeError:          # 모양이 바뀐 텐서 (제자리 resize) -- 새로 잡는다
                fast = False
        if not fast:
            self._live, self._ids = live, ids
            self._buf = [t.detach().clone() for _, _, t in live]
            g = collections.OrderedDict()
            for i, ((_, _, t), b) in enumerate(zip(live, self._buf)):
                d = g.setdefault((t.device, t.dtype), ([], []))
                d[0].append(i)
                d[1].append(b)
            self._groups = list(g.values())
        self._optkeys = {p: tuple(st.keys()) for p, st in self.opt.state.items()}
        self._optvals = {p: {k: copy.deepcopy(v) for k, v in st.items() if not torch.is_tensor(v)}
                         for p, st in self.opt.state.items()}
        self._hyper = [copy.deepcopy({k: v for k, v in grp.items() if k != "params"})
                       for grp in self.opt.param_groups]
        self._sched = copy.deepcopy(self.sched.state_dict()) if self.sched is not None else None

    @torch.no_grad()
    def restore(self):
        """떠 둔 상태로 정확히 되돌리고 기울기를 비운다.  파라미터는 제자리 복사다 (옵티마이저가 쥔 객체 그대로)."""
        optsnap = {}
        for (kind, key, _), b in zip(self._live, self._buf):
            if kind == "opt":
                optsnap.setdefault(key[0], {})[key[1]] = b
                continue
            mod, leaf, name = key
            cur = (mod._parameters if kind == "param" else mod._buffers).get(leaf)
            if cur is not None and cur.shape == b.shape and cur.dtype == b.dtype and cur.device == b.device:
                cur.copy_(b)
            elif kind == "buffer":        # 모듈이 버퍼 객체를 바꿔 끼웠다 -- 사본을 다시 꽂는다
                mod._buffers[leaf] = b.clone()
            else:
                raise RuntimeError(f"[NaN 가드] 파라미터 {name} 를 되돌릴 수 없다 (모양이 바뀌었다)")
        st = self.opt.state
        for p in list(st.keys()):
            if p not in self._optkeys:
                del st[p]                  # 실패한 step 이 새로 만든 상태 (첫 step)
        for p, keys in self._optkeys.items():
            d = st[p]
            for k in list(d.keys()):
                if k not in keys:
                    del d[k]
            for k in keys:
                if k in optsnap.get(p, {}):
                    b, t = optsnap[p][k], d.get(k)
                    if torch.is_tensor(t) and t.shape == b.shape and t.dtype == b.dtype and t.device == b.device:
                        t.copy_(b)
                    else:
                        d[k] = b.clone()
                else:
                    d[k] = copy.deepcopy(self._optvals[p][k])
        for grp, h in zip(self.opt.param_groups, self._hyper):
            for k in [k for k in grp if k != "params" and k not in h]:
                del grp[k]
            grp.update(copy.deepcopy(h))
        if self.sched is not None:
            self.sched.load_state_dict(copy.deepcopy(self._sched))
        self.opt.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def check(self, loss):
        """step 뒤 상태가 유한한가.  유한하면 None, 아니면 사유 문자열.

        빠른 판정은 (장치, dtype) 묶음마다 묶음 연산 둘이다: `x * 0` 은 유한한 x 에서 0, inf · NaN 에서 NaN
        이므로 `_foreach_norm(ts * 0)` 이 전부 0 이면 묶음 전체가 유한하다.  값 자체를 제곱하지 않으니 아주 큰
        유한값에서도 넘치지 않는다 -- 예전의 L2 노름 판정은 원소 하나가 약 1.8e19 를 넘으면 inf 가 되어 매 step
        텐서 단위 확인(GPU 에서 텐서마다 동기화)으로 떨어졌다.  MutInf 의 `CDC.HP_branch` 는 초기화하지 않은
        메모리(`torch.FloatTensor(1)`)라 실제 체크포인트에 -1.16e30 (wv3) 같은 값이 있다.  텐서 단위 확인
        (`_first_bad`)은 빠른 판정이 비유한을 찾았을 때 **어느 텐서인가**를 적는 데만 쓴다."""
        lt = loss.detach() if torch.is_tensor(loss) else torch.tensor(float(loss))
        if self._all_finite(lt, (live := self._collect())):
            return None
        if not bool(torch.isfinite(lt).all()):
            v = lt.float().flatten()
            return f"손실 비유한 ({v[~torch.isfinite(v)][0].item()})"
        return self._first_bad(live)

    @staticmethod
    def _all_finite(lt, live) -> bool:
        """손실과 `live` 의 부동소수 텐서가 전부 유한한가 -- 넘침 없는 빠른 판정 (`check` 설명)."""
        flags = [torch.isfinite(lt).all()]
        g = collections.OrderedDict()
        for _, _, t in live:
            if t.is_floating_point():
                g.setdefault((t.device, t.dtype, t.dim() == 0), []).append(t)
        for (_, _, scalar), ts in g.items():
            # 0 차원(Adam 의 step 셈 -- CPU 에 있다)은 한 번에 쌓아 본다.  나머지는 묶음마다 `ts * 0` 의 노름.
            v = torch.stack(ts) if scalar else torch.stack(torch._foreach_norm(torch._foreach_mul(ts, 0.0), 2))
            flags.append(v.isfinite().all())
        return all(bool(f) for f in flags)

    def _first_bad(self, live):
        """비유한 원소가 있는 첫 텐서의 사유 (없으면 None).  빠른 판정이 실패한 뒤에만 부른다."""
        for kind, key, t in live:
            if t.is_floating_point() and not bool(torch.isfinite(t).all()):
                n = int((~torch.isfinite(t)).sum())
                return f"갱신 뒤 {self._label(kind, key)} 비유한 ({n}/{t.numel()} 원소)"
        return None

    # ------------------------------------------------------------------ 한 step
    def run(self, step: int, fn):
        """`fn()` (한 step 전체, 손실을 돌려준다) 을 가드 아래에서 돈다.

        돌려주는 값: (손실, None) -- 정상.  (None, 사유) -- 되돌리고 건너뛰었다 (`note_skip` 까지 했다)."""
        self.snapshot()
        try:
            loss = fn()
        except (ValueError, RuntimeError) as e:
            if not nonfinite_error(e):
                raise
            reason = _short(e)
        else:
            reason = self.check(loss)
        if reason is None:
            return loss, None
        self.restore()                     # 예외의 traceback(그래프)은 except 를 벗어나며 풀렸다
        self.note_skip(step, reason)
        return None, reason

