# C1 실패 탐지기 — 에피소드 결과에서 국소 조건 위반으로

[`../safe_detector/`](../safe_detector/README.md) 가 남긴 숫자는 "처음 보는 task 에서 0.719" 였다.
쓸 만한지 아닌지 애매하고, 무엇을 보고 그 점수를 내는지도 불분명하다. 이 폴더는 그 둘을 파고들다가
**목표 자체를 바꾼 기록**이다. π0.5 는 여기서도 끝까지 고정이다.

결론 먼저: 에피소드 결과를 목표로 하면 미학습 task 로 넘어가지 않는다. 잠재변수의 거의 전부가
"어느 task 인가"이기 때문이다. 목표를 **10 환경 스텝 안의 국소 조건 위반**으로 바꾸면
미학습 task macro AUROC **0.891 ± 0.010** 이 나온다.

요약 그림은 `figures/c1_summary.png`.

---

## 1부 — 모델을 키워도 소용없다 (`train_xattn.py`, `models.py`)

호출 하나가 50개 action 토큰을 낳고, `prefix8` 창은 그런 호출 8개다. 두 축이 따로 있다 —
**호출 안(50 토큰)** 과 **호출 사이(8 호출)**. 네 모델이 서로 다른 축에 앉는다.

| | 호출 안 | 호출 사이 |
|---|---|---|
| LogReg | 평균, 가중 없음 | 없음 |
| LSTM | 평균, 가중 없음 | 8 호출에 걸쳐 recurrent |
| AttnPool | 토큰에 학습된 attention | 없음 |
| CrossAttn | 조건부 attention | 없음 |

784 롤아웃, `safe_detector` 와 **같은 분할**로 (`results/xattn_all784.json`):

| 모델 | 파라미터 | ms/호출 | episode AUROC (init) | episode AUROC (**task**) |
|---|---|---|---|---|
| LogReg | 1,025 | 0.02 | 0.818 ± 0.029 | **0.730 ± 0.064** |
| LSTM | 590,977 | 0.05 | 0.827 ± 0.027 | 0.709 ± 0.086 |
| AttnPool | 132,353 | 0.04 | 0.809 ± 0.039 | 0.721 ± 0.059 |
| CrossAttn | 181,889 | 0.11 | 0.819 ± 0.028 | **0.730 ± 0.065** |

미학습 task 에서 **177배 큰 cross-attention 이 선형 탐침과 소수점 셋째 자리까지 같다**.
이후 모든 실험이 LogReg 를 쓰는 이유다.

## 2부 — 왜 미학습 task 로 안 넘어가는가 (`score_timeline.py`, `transfer_probe.py`, `task_id_probe.py`)

`prefix8` 이 그냥 너무 이른 것인지, 아니면 잠재변수가 같은 task 안에서 실패를 구분하지 못하는 것인지
갈라야 한다. 롤아웃 전체 길이를 호출마다 채점했다 (`results/score_timeline_main.json`):

* 학습한 task — call 3 에서 task-macro **0.620** 까지 오른다
* 처음 보는 task — 끝까지 **0.46 – 0.52**, 즉 우연

시점 문제가 아니다. 분산 분해와 네 가지 변환으로 원인을 짚었다 (`results/transfer_probe_main.json`):

| 표현 | 학습한 task | 처음 보는 task |
|---|---|---|
| 원본 | 0.622 ± 0.081 | 0.471 ± 0.044 |
| task 중심화 | 0.554 ± 0.080 | 0.533 ± 0.054 |
| 에피소드 중심화 | 0.519 ± 0.049 | 0.508 ± 0.093 |

**task 평균을 빼면 학습한 task 의 우위가 같이 사라진다.** 즉 0.622 는 "실패를 본다"가 아니라
"이 task 의 평소 모습에서 얼마나 벗어났나"였고, 그 평소 모습은 처음 보는 task 에 없다.

못을 박은 것은 task 식별 탐침이다 (`results/task_id_probe.json`): 호출 하나의 `suf_mean` 에서
**40개 task 중 어느 것인지를 선형 모델이 99.9% 정확도로 읽는다** (우연 2.5%, top-3 100%).

온라인 루프로 데이터를 더 넣어도 움직이지 않았다 (`online_runner.py`,
`results/online_curve_main.json`): 오프라인 454 + 온라인 0/60/120/180/234 롤아웃에서
미학습 task macro 가 0.471 → 0.470. **데이터가 부족해서가 아니라 목표가 틀렸다.**

## 3부 — 목표를 바꾼다 (`conditions.py`, `collect_conditions.py`)

에피소드 결과 대신 **국소 조건 위반**을 묻는다. H = 10 환경 스텝은 호출 하나의 action chunk 가
실제로 구동하는 구간이고, guidance 신호가 답해야 할 질문과 정확히 같은 창이다.

`conditions.py` 는 두 가지를 일부러 분리한다. `StateLogger` 는 환경 스텝마다 **원시량만** 기록하고
(물체 자세, 접촉, 그리퍼 개폐, 관절 여유, goal predicate) 아무것도 판정하지 않는다. `evaluate` 가
나중에 문턱을 받아 조건 series 와 onset 을 만든다. 문턱을 수집에서 빼둔 덕분에 롤아웃을 한 번도
다시 돌리지 않고 측정된 분포에서 문턱을 고칠 수 있다 — 실제로 이전 단계에서 **명령된 행동**으로
그리퍼 문턱을 잡아 80 프레임 중 9개를 잘못 라벨링한 적이 있고(`results/label_check.json`),
원시 개폐값을 남겨 두었기 때문에 잡혔다.

> **조건 onset 은 "되돌릴 수 없게 된 순간"이 아니다.** 이름 붙은 조건이 처음 참이 된 스텝일 뿐이다.
> 40 에피소드에서 onset 과 최종 실패는 32/40 만 일치한다 — onset 이 있었는데 성공한 에피소드가 5개,
> onset 없이 실패한 에피소드가 3개다 (`results/temporal_main.json`).

네 조건 중 어느 것이 읽히는지 먼저 쟀다 (`violation_probe.py`, `results/violation_probe_main.json`, H=10):

| 조건 | `suf_mean` | 그리퍼 개폐만 | call index |
|---|---|---|---|
| **C1 빈 그리퍼 닫힘** (active) | **0.888** | 0.806 | 0.449 |
| C1 (onset) | **0.830** | 0.641 | 0.657 |
| C3 비표적 접촉 | 0.604 | 0.679 | 0.445 |
| C4 물체 넘어짐 | 0.719 | 0.590 | 0.762 |
| 합성 C1C3C4 | 0.854 | 0.735 | 0.728 |

C1 만 분명하게 읽힌다. C4 는 **시계보다 못하다**. 이후는 C1 에 집중한다.

## 4부 — C1 전향 예측 (`c1_depth.py`)

`collect_conditions.py` 로 200 에피소드를 모았다 — 환경 스텝마다 상태, VLA 호출마다 `suf_mean` 과
`call_step`(그 호출이 나간 환경 스텝), onset 주변 프레임. **잠재변수와 라벨은 같은 실행에서 나온다.**
이전 실행을 재생해도 비트 단위로 재현되지 않으므로(97% 는 동일, 나머지는 최대 1.8 cm 차이), 새 상태
로그를 옛 잠재변수에 붙이면 두 궤적이 조용히 섞인다.

표본은 **C1 이 아직 참이 아닌** 호출뿐이고, 양성은 에피소드 전역으로 정의된 onset 이 (t, t+10] 에
떨어지는 경우다. 지표는 **미학습 task macro AUROC** — task-group 분할로 홀드아웃한 task 들의
per-task AUROC 평균.

> pooled 를 옆에 적는 이유: 이 프로젝트의 이전 단계에서 pooled 0.73 이 macro 0.50 이었다.
> pooled 는 "어느 롤아웃이 잘못되고 있나"가 아니라 "어느 task 가 어려운가"를 아는 것도 보상한다.

**4,994 호출 / 353 양성 (7.1%), 20 task, 5-fold × 20 반복** (`results/c1_features_n200.json`):

| 표현 | 차원 | macro AUROC | macro AUPRC |
|---|---|---|---|
| **`suf_mean`** (action expert) | 1024 | **0.891 ± 0.010** | **0.525** |
| VLM 이미지 토큰 | 2048 | 0.874 ± 0.006 | 0.411 |
| VLM 전체 | 4096 | 0.858 ± 0.013 | 0.398 |
| 상태 + 이력 (t, t−5, t−10) | 45 | 0.796 ± 0.009 | 0.324 |
| VLM 언어 토큰 | 2048 | 0.772 ± 0.010 | 0.286 |
| 순간 상태 | 15 | 0.766 ± 0.007 | 0.338 |
| 그리퍼 개폐 하나 | 1 | 0.551 ± 0.057 | 0.188 |
| call index | 2 | 0.492 | 0.086 |
| 무작위 | 4 | 0.524 ± 0.010 | 0.132 |

AUPRC 를 같이 읽어야 한다. 양성률이 7.1% 라 AUPRC 의 바닥은 0.5 가 아니라 0.071 이다.

## 5부 — 언제, 얼마나 멀리, 마지막 몇 스텝이 전부인가

**지평 (`c1_horizon.py`)** — H 를 넓히면 "더 먼 미래"와 "양성이 흔해짐"이 동시에 바뀌므로 양성률을
같이 적는다 (`results/c1_horizon_n200.json`):

| H | 양성 | 양성률 | macro AUROC | macro AUPRC |
|---|---|---|---|---|
| 5 | 169 | 3.4% | 0.914 ± 0.006 | 0.477 |
| **10** | 353 | 7.1% | **0.891 ± 0.010** | 0.525 |
| 20 | 648 | 13.0% | 0.855 ± 0.008 | 0.567 |
| 30 | 881 | 17.6% | 0.813 ± 0.008 | 0.550 |

**가드 (`c1_guard.py`)** — onset 이 코앞인 호출을 **빼고**(음성으로 바꾸지 않고) 채점한다.
k 스텝 안에 터지는 호출은 "t+k 이후에 터지나"의 깨끗한 양성도 참 음성도 아니기 때문이다
(`results/c1_guard_n200.json`):

| 가드 k | 남은 표본 | 양성 | macro AUROC |
|---|---|---|---|
| 0 | 4,994 | 353 | 0.891 ± 0.010 |
| 3 | 4,885 | 244 | 0.888 ± 0.005 |
| 5 | 4,825 | 184 | 0.886 ± 0.006 |

**마지막 몇 스텝에 기대고 있는 신호가 아니다.** 3 스텝을 비워도 거의 그대로다.

**거리 (`c1_distance.py`)** — 지평 넓히기로는 답이 안 나온다. H=30 의 0.813 이 "30 스텝 앞을 본다"인지
"0–10 구간이 여전히 들고 있다"인지 구분이 안 되기 때문이다. 여기서는 구간마다 따로 채점한다 —
음성은 항상 같은 집합이고 양성만 멀어진다 (`results/c1_distance_n200.json`):

| onset 까지 거리 | 양성 | `suf_mean` | 상태 |
|---|---|---|---|
| 0–10 스텝 | 353 | 0.891 ± 0.010 | 0.772 |
| 10–20 | 295 | 0.840 ± 0.006 | 0.703 |
| 20–30 | 233 | 0.793 ± 0.007 | 0.685 |
| 30–40 | 197 | 0.765 ± 0.009 | 0.643 |

멀어질수록 옅어지지만 40 스텝 밖에서도 상태 특징보다 위에 있다.

**task 별 (`c1_pertask.py`, `results/c1_pertask_n200.json`)** — 20개 task, 전부 out-of-fold.
`libero_goal/t2` 0.977 (양성 22) 부터 `libero_goal/t0` 0.765 (양성 **1개**) 까지. 양성이 적은 task 는
탐지기와 무관한 이유로 숫자가 크게 움직이므로 양성 수를 옆에 적고 `--min-pos` 미만은 따로 센다.
`reward_guidance` 가 쓰는 네 task 는 각각 goal/t2 0.977, object/t6 0.937, object/t0 0.815,
spatial/t9 0.830.

## 6부 — 눈으로 확인 (`c1_cases.py`, `replay_cases.py`, `make_sync_video.py`)

문턱 τ 는 **음성 호출의 90 백분위수**다. TP/FP/FN/TN 으로 에피소드를 나누기 위해서만 있고,
**작동점이 아니며 어떤 성능 수치도 거기서 보고하지 않는다.** 사례는 보기 전에 정한 규칙으로 고른다 —
각 범주에서 에피소드 점수가 그 범주의 **중앙값에 가장 가까운** 둘. 가장 높은 것도, 가장 선명한 것도 아니다.

`replay_cases.py` 는 그 여덟을 결정적으로 재생해 저장된 실행이 담지 못한 것을 기록한다. 재생이
어긋나는 에피소드는 **교체하지 않고 "어긋남"으로 보고한다** — 교체하면 고정된 선택이 조용히
필터링된 선택으로 바뀐다.

## 알려진 함정

* **`prospective_c1.py` 는 폐기됐다 (SUPERSEDED).** 창 안에서 debounce 를 판정해서, 연속 3 스텝이
  창 경계를 걸친 onset 을 음성으로 셌다. 200 에피소드 H=10 에서 양성을 353 이 아니라 300 으로 세고
  **0.891 대신 0.914 를 보고했다.** 과거 숫자의 출처로만 남겨 둔다. 이후 모든 결과는
  `c1_depth.build()` 의 에피소드 전역 onset 정의를 쓴다.
* `data/` (9.2 GB) 와 `results/replay/frames/` 는 추적하지 않는다. 재생성 명령은 아래에.

## 재현

```bash
conda activate fasi
cd ~/yuna

# 토큰 수집과 모델 비교 (1부)
python xattn_detector/collect_tokens.py
python xattn_detector/train_xattn.py --out results/xattn_all784.json

# 전이 진단 (2부)
python xattn_detector/score_timeline.py
python xattn_detector/transfer_probe.py
python xattn_detector/task_id_probe.py

# 조건 수집 (3부) — GPU, 수 시간
python xattn_detector/collect_conditions.py \
    --suites libero_spatial libero_object libero_goal libero_10 \
    --tasks 0,2,4,6,9 --inits 30-39 --out data/cond200
python xattn_detector/violation_probe.py --data data/cond200

# C1 전향 예측 (4·5부) — CPU, 각 수 분
python xattn_detector/c1_features.py --data data/cond200 --repeats 20
python xattn_detector/c1_horizon.py  --data data/cond200 --repeats 20
python xattn_detector/c1_guard.py    --data data/cond200 --repeats 20
python xattn_detector/c1_distance.py --data data/cond200 --repeats 20
python xattn_detector/c1_pertask.py  --data data/cond200 --repeats 20

# 사례와 그림 (6부)
python xattn_detector/c1_cases.py --data data/cond200
python xattn_detector/replay_cases.py
python xattn_detector/figures/make_c1_summary.py
```

## 다음

이 탐지기를 π0.5 의 denoising 샘플러에 **보상**으로 연결할 수 있는지는
[`../reward_guidance/`](../reward_guidance/README.md) 에서 측정한다. 짧게: 감시는 되지만 조종은 되지 않는다.
