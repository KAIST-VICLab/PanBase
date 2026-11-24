# SPDX-License-Identifier: Apache-2.0
"""Recipe comparison of `train_pancoll.py --resume`: the recipe snapshots in state.pt / best.pt / the results JSON
against the current recipe.  Taken unchanged from the research table tooling, which is not part of this release."""
import json


def recipe_diff(snap, cur):
    """레시피 스냅샷 `snap` 이 지금 레시피 `cur` 와 다른 키 목록.  빈 목록이면 같다.

    `is_current` 의 대조 규칙 **그 자체**다 -- 이어 돌리기(train_pancoll `--resume`)도
    state.pt · best.pt · 결과 JSON 의 스냅샷을 이 함수로 대조한다.  같은 질문에 규칙이
    둘이면 한쪽만 고치게 된다.
    """
    # JSON 왕복으로 양쪽을 맞춘다.  `betas` 가 spec 에서는 튜플이지만 결과
    # 파일에서는 리스트로 되살아나, 내용이 같아도 != 로 판정됐다.
    norm = lambda x: json.loads(json.dumps(x, sort_keys=True, default=str))
    a, b = norm(snap), norm(cur)
    # `per_dataset` 는 **기록하되 비교하지 않는다.**
    #
    # 학습기도 표도 시작할 때 `per_dataset[ds]` 를 스펙 위에 병합하므로, 그 덮어쓰기가
    # 실제로 바꾸는 값은 이미 병합된 최상위 키(`epochs`, `sched`, `opt`,
    # `eval_after_frac` -- 레지스트리에서 per_dataset 가 덮는 키는 이 넷뿐이고 모두
    # 비교 대상이다)로 대조된다.  원본 dict 까지 같이 비교하면 **다른 데이터셋의**
    # 항목이 바뀐 것만으로 이 칸이 낡았다고 판정된다.
    #
    # 실제로 그렇게 됐다: ARConv 의 `per_dataset["qb"]` 에 `eval_after_frac` 를 더했더니
    # 18.6 시간을 완주한 **wv3** 칸이 거부됐다.  wv3 학습은 그 항목을 읽지도 않는다.
    # (같은 수정이 진짜로 건드린 `qb` 칸은 병합된 `eval_after_frac` 로 여전히 걸린다 --
    # 판별력은 그대로다.)
    a.pop("per_dataset", None)
    b.pop("per_dataset", None)
    # `params` 는 레시피 손잡이가 아니라 **측정값**이다 (`_RECIPE_KEYS` 에 없다).
    # `_recipe_now` 는 모델을 실제로 만들어 세는데, 그 기계에 저장소 의존성이
    # 없으면 -- CANConv 의 libKMCUDA, AWFLN 의 torchvision -- 조용히 None 이 되고,
    # 그러면 **완주한 칸이 전부 '낡음'으로 뒤집힌다**.  reconcile 이 같은 게이트를
    # 쓰므로 그 순간 끝난 학습들이 큐로 되돌아간다.  한쪽이라도 셀 수 없으면
    # 이 항목은 판정에서 뺀다 -- 양쪽 다 있을 때만 대조한다.
    if a.get("params") is None or b.get("params") is None:
        a.pop("params", None)
        b.pop("params", None)
    # **양쪽에 다 있는 키만 비교한다.**  판정 항목을 새로 추가하면(예: `code_digest`,
    # `aux_loss`) 그 이전 결과에는 그 키가 아예 없다.  없는 것을 '불일치'로 읽으면
    # 완주한 칸이 한꺼번에 낡음으로 뒤집힌다 -- 이 저장소에서 이미 두 번 그랬다
    # (`per_dataset`, `params`).  새 항목은 결과가 그 키를 갖기 시작한 뒤부터
    # 효력을 갖는다.  그게 정확한 셈이다: 모르는 것은 모른다고 두는 것이다.
    common = a.keys() & b.keys()
    return sorted(k for k in common if a[k] != b[k])


EVAL_ONLY_KEYS = ("eval_protocol",)


def recipe_diff_train(snap, cur):
    """`recipe_diff` 에서 평가 전용 키(`EVAL_ONLY_KEYS`)를 뺀 것 -- 학습 상태(state.pt · best.pt)를 이어도
    되는가의 대조.  규칙은 `recipe_diff` 하나에서 나온다."""
    return [k for k in recipe_diff(snap, cur) if k not in EVAL_ONLY_KEYS]
