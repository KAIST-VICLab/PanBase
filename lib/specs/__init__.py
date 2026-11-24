"""추가 베이스라인 spec 들.

`registry.py` 가 이 디렉터리의 모든 모듈을 읽어 `register(SPECS, REPOS, _load, _pkg)`
를 호출한다.  한 방법 = 한 파일이라 서로 충돌 없이 추가/제거할 수 있다.

각 spec 은 최소한 다음을 담는다:
    build       (ms_ch, ratio) -> nn.Module
    venue       발표처
    loss        "l1" | "mse" | "l1_sum_over_2b"   (train_step 이 있으면 무시된다)
    batch/epochs/opt/sched       저장소 공식 값
    forward     (m, ms_up, pan, lrms, ctx) -> Tensor   평가용
    tier        "confirmed"(학습 진행) | "extra"(코드만 준비)
    diff        저장소 대비 우리가 바꾼 것과 그 이유
선택:
    train_step  (model, batch, ctx, lossfn) -> (pred, loss)   손실이 복합일 때
    aux_weights / out_index      튜플 출력에 가중 보조손실을 걸 때
    val_cap / aug / grad_clip / eval_after_frac


`per_dataset`  (선택)
    데이터셋별 덮어쓰기.  `{"gf2": {"opt": {"lr": 5e-4}, "sched": {"step_size": 500}}}`
    형태이고, 사전 값은 얕게 병합된다(`opt` 안의 나머지 키는 살아남는다).
    저장소가 벤치마크마다 다른 학습률을 쓰는 경우에만 쓴다 -- 우리가 고르는 값이
    아니라 저자가 쓴 값을 그대로 옮기는 자리다.

`on_train_start(model, n_train, device)` / `on_epoch(model, epoch)`  (선택)
    표본 인덱스에 묶인 상태를 저장소 방식대로 초기화/갱신하는 훅.  학습 ctx 에는
    `sample_idx` (배치 표본의 데이터셋 인덱스, device 위) 와 `epoch` (1부터) 이 실린다.
    검증 ctx 에는 `sample_idx` 가 **없다** -- 검증 표본은 학습 캐시에 자리가 없다.

`eval_tile`  (선택, int)
    평가(val(full=True) · FR · zero-shot · 그림 · 코퍼스 zero-shot)를 이 크기의 겹치지 않는 타일로
    나눠 추론하고 이어 붙인다 (`lib/harness.eval_forward`).  네트워크가 학습 크기에 묶인 방법에만
    쓴다 -- UTeRM 의 W 축 t-product 저랭크 시작이 그렇다.  학습 forward 는 바뀌지 않고,
    결과의 `eval_protocol` 에 "+tile<N>" 이 붙는다 (`pancoll.eval_protocol`).

`eval_bs`  (선택, 기본 64)
    최종 평가(reduced/full examples)의 배치 크기.  지표는 표본별로 계산해 평균하므로
    배치 크기가 수치를 바꾸지 않는다 -- 메모리만 바뀐다.  전수 평가에서 OOM 이 나는
    무거운 모델에만 쓴다.  FR(H=512)은 자동으로 min(4, eval_bs) 를 쓴다.
"""
