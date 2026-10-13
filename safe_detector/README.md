# SAFE — π0.5 의 잠재변수로 에피소드 실패를 예측할 수 있는가

π0.5 를 LIBERO 에서 돌리면서 **action expert 의 hidden state 를 VLA 호출마다 한 벡터씩** 빼내고,
그 벡터로 "이 롤아웃은 실패로 끝난다"를 맞힐 수 있는지 본다. π0.5 는 끝까지 **고정**이다 —
불러서 돌리기만 하고 한 번도 최적화하지 않는다. 학습되는 것은 탐지기뿐이다.

이 폴더는 C1 체인의 **1단계**다. 여기서 "에피소드 결과"라는 목표의 한계가 드러나고,
[`../xattn_detector/`](../xattn_detector/README.md) 에서 목표를 국소 조건 위반으로 바꾼다.

## 파이프라인

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| 1 | `collect_rollouts.py` + `latent_hook.py` | 롤아웃을 돌리며 호출당 잠재변수 + 오라클 상태 + BDDL goal predicate 기록 |
| 2 | `train_detector.py` | 잠재변수 → 실패 점수. LogReg / LSTM, 두 분할(`init` / `task`) |
| 3 | `eval_leadtime.py` | 점수를 lead time 으로. 문턱은 **오경보 예산**으로 정한다 |
| — | `multiseed.py` | 모든 칸을 8 seed 로 반복, 평균 ± 표준편차 |
| — | `calibration.py` | 순위가 아니라 **확률**이 맞는지 (Brier, R²) |

`latent_hook.py` 는 lerobot 도 중범님 저장소도 건드리지 않고 `PaliGemmaWithExpertModel.forward` 의
**바인딩된 메서드를 감싼다** — 두 호출 지점 모두 `nn.Module.__call__` 을 건너뛰고 `.forward(...)` 를
직접 부르기 때문에 `register_forward_hook` 으로는 잡히지 않는다.

`jb.py` 는 중범님 저장소에 대한 **읽기 전용 다리**다. `~/바탕화면/jungbeom` 아래에는 아무것도 쓰지 않는다.

## 결과 (800 롤아웃, 8 seed, `results/multiseed.json`)

`act_mean` = action expert hidden state 의 평균 풀링. `init` 은 학습한 task 의 새 롤아웃을,
`task` 는 **task 통째로** 홀드아웃한다.

| 탐지기 | 분할 | 창 | call AUROC | episode AUROC | strat AUROC |
|---|---|---|---|---|---|
| LogReg act_mean | init | full | 0.872 ± 0.025 | 0.942 ± 0.023 | 0.833 |
| LogReg act_mean | task | full | 0.798 ± 0.051 | 0.878 ± 0.061 | 0.709 |
| LogReg act_mean | init | prefix8 | 0.834 ± 0.022 | 0.847 ± 0.025 | 0.834 |
| LogReg act_mean | task | prefix8 | **0.719 ± 0.052** | 0.722 ± 0.068 | 0.719 |
| LSTM act_mean | init | full | 0.793 ± 0.031 | 0.846 ± 0.022 | 0.697 |
| LogReg vlm_mean | init | full | 0.862 ± 0.027 | 0.915 ± 0.023 | 0.807 |
| **시계 (call index 만)** | init | full | 0.609 ± 0.219 | **0.838 ± 0.210** | **0.500** |
| **무작위** | init | full | 0.500 | **0.705 ± 0.030** | 0.499 |
| 시계 | init | prefix8 | 0.500 | **0.500** | 0.500 |

### 이 표에서 가장 중요한 줄은 아래 세 줄이다

LIBERO 의 성공 에피소드는 과제를 푼 순간 멈추고 실패 에피소드는 타임아웃까지 간다. 그래서
`full` 창의 **episode AUROC 는 길이 누수로 오염돼 있다** — 시계 하나로 0.838, 완전 무작위 점수로도
0.705 가 나온다. 0.942 를 그 옆에 두지 않고 읽으면 안 된다.

깨끗한 숫자는 두 가지다. `prefix8`(모든 에피소드의 첫 8 호출, 시계 baseline 이 정확히 0.500)과
`strat`(같은 call index 끼리만 비교해 길이가 기여할 수 없게 만든 AUROC). 그 둘로 보면
**처음 보는 task 에서 0.719** 이고, 이것이 1단계의 정직한 결론이다.

LSTM 은 LogReg 를 이기지 못한다. VLM 토큰도 action expert 를 이기지 못한다.

## 그래서 다음

`task` 분할 0.719 는 "쓸 만하다"와 "안 된다" 사이 어디쯤이고, 무엇을 보고 그 점수를 내는지가
불분명하다. [`../xattn_detector/`](../xattn_detector/README.md) 가 그걸 파고들어
**잠재변수의 대부분이 "어느 task 인가"** 라는 것을 찾아내고, 목표를 에피소드 결과에서
**국소 조건 위반**으로 바꾼다.

## 재현

```bash
conda activate fasi
cd ~/yuna

bash safe_detector/run_collect.sh          # 800 롤아웃 수집 (GPU, 수 시간) -> data/
bash safe_detector/run_analysis.sh         # 모든 탐지기 칸
python safe_detector/multiseed.py          # 8 seed 반복 -> results/multiseed.json
python safe_detector/summarize.py          # 한 장짜리 표
python safe_detector/calibration.py        # Brier / R²
python safe_detector/figures/make_results_figure.py
```

`data/` (241 MB) 는 추적하지 않는다. `results/` 의 JSON·npz(out-of-fold 점수)는 위 수치의 근거라
추적한다. 진행 상황은 `python safe_detector/progress.py` 로 읽기 전용 확인.
