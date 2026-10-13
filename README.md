# VLA 실패 예측·교정 작업공간

π0.5 (VLA) 를 LIBERO 에서 돌리면서 **실패를 미리 알아볼 수 있는가**, 그리고 **알아본 것으로
무언가를 고칠 수 있는가**를 측정한 기록. π0.5 자체는 모든 실험에서 고정이고, 학습되는 것은
탐지기·헤드뿐이다.

저장소 뿌리는 `~/yuna` 지만 추적하는 것은 아래 다섯 폴더뿐이다 (`.gitignore` 참고).
원시 수집 데이터와 체크포인트(합계 10 GB 이상)는 추적하지 않고, 각 README 의 "재현" 절에
재생성 명령을 적어 둔다.

## 두 갈래

### A. Gemma 로 "대신 무엇을 했어야 하나"를 묻는 쪽

| 폴더 | 내용 |
|---|---|
| [`counterfactual_pilot/`](counterfactual_pilot/README.md) | 실패 직전 장면에서 Gemma 가 8개 행동 중 하나를 고를 수 있는가. 1–5단계 전체 서술 |
| [`failure_prediction/`](failure_prediction/README.md) | 위에서 갈라져 나온 3단계 — "실패 몇 스텝 전부터 VLM 이 알아보는가" → "어느 결함이냐가 거의 전부" |

### B. π0.5 자신의 잠재변수로 실패를 읽는 쪽 (C1 체인)

| 폴더 | 질문 | 결론 |
|---|---|---|
| [`safe_detector/`](safe_detector/README.md) | 잠재변수로 **에피소드 실패**를 맞히는가 | 익숙한 task 0.834, 처음 보는 task **0.719**. `full` 창의 0.942 는 길이 누수 |
| [`xattn_detector/`](xattn_detector/README.md) | 왜 미학습 task 로 안 넘어가는가 → 목표를 바꾸면? | 잠재변수의 99.9% 가 **task 식별**. 목표를 10스텝 안 **국소 조건 위반(C1)** 으로 바꾸면 미학습 task macro **0.891** |
| [`reward_guidance/`](reward_guidance/README.md) | 그 탐지기를 denoising 샘플러의 **보상**으로 쓸 수 있는가 | **감시는 되지만 조종은 되지 않는다.** 상태를 고정하면 탐지기가 읽는 것은 사실상 그리퍼 명령 하나 |

B 의 세 폴더는 import 로 묶여 있어 **형제로 함께 있어야 한다** — `reward_guidance` 가
`../safe_detector/train_detector.py` 와 `../xattn_detector/conditions.py` 를 `sys.path` 로 끌어다 쓴다.

## C1 체인을 한 줄로

에피소드 결과를 목표로 하면 처음 보는 task 로 넘어가지 않는다. 잠재변수가 실패가 아니라
**어느 task 인지**를 담고 있기 때문이다. 목표를 "지금부터 10 환경 스텝 안에 그리퍼가 빈 채로
닫히는가"로 바꾸면 미학습 task 에서 0.891 이 나온다. 그런데 그 탐지기를 보상으로 써서
샘플러를 밀어 보면, 점수는 3.49 logit 움직이는데 실제 C1 은 2.37 %p 밖에 줄지 않고,
같은 보폭으로 방향만 뒤섞은 대조군과 구분되지 않는다. 상태 내 판별로 직접 학습해도 같다
(최선 −0.023 [−0.100, +0.058], 그리퍼 스칼라 대비).

**감시자가 곧 보상은 아니다.**

## 환경

```bash
conda activate fasi          # 중범님이 만든 공유 환경 — 패키지를 설치하지 않는다
cd ~/yuna
```

`safe_detector/jb.py` 가 `~/바탕화면/jungbeom` 의 `fasi.*` 를 읽기 전용으로 끌어온다.
그 트리에는 아무것도 쓰지 않는다. π0.5 체크포인트는 그쪽에서 사라져 `~/yuna/ckpt/` 에 복구해
두었다 (`reward_guidance/vla_setup.py`, `reward_guidance/rebuild_tokenizer.py`).

추적하지 않는 폴더: `branch_rollout/`(시뮬레이터 오라클 분기 실험), `Dreamerv3_study/`(별도 저장소),
`ckpt/`, `data/`, `models/`, `results/`, `videos/`. 필요해지면 `.gitignore` 에 `!` 줄을 추가한다.
