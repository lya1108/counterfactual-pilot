# 실패 예측 (3단계)

`~/yuna_ws/counterfactual_pilot`의 1·2단계에서 갈라져 나온 실험이다. **전체 서술과 결과 해석은
[../counterfactual_pilot/README.md](../counterfactual_pilot/README.md)의 "3단계" 절**에 있고, 이
파일은 여기서 무엇을 어떻게 돌리는지만 적는다. 기록을 두 군데로 나누면 갈라지므로 본문은 한 곳에
둔다.

질문은 **"실패하기 몇 스텝 전부터 VLM이 실패를 알아보는가"**로 시작했고, 답이 **"lead time이
아니라 어느 결함이냐가 거의 전부"**로 나와서 지금은 **"필요한 관측을 축마다 어떻게 줄 것인가"**를
보고 있다.

## 옆 폴더와의 관계

`counterfactual_pilot`이 스크립트된 컨트롤러·환경 kwargs·프롬프트 어휘를 갖고 있고, 여기서는
그것을 **import해서 쓴다**(복사하지 않는다 — 물리 상수의 출처를 하나로 두기 위해서다).
`collect_leadtime.py`와 `fact_probe.py` 위쪽의 경로 한 줄이 전부이고, 옆 폴더의 파일은 수정하지
않는다.

`gemma_action_choice.py`(지각 프로브: `--task height` / `--task gripper_state`)는 1단계 도구라
옆 폴더에 남아 있다. 여기 이미지로 돌릴 때는 경로를 넘긴다 — 아래 재현 명령 참고.

## 파일

```
collect_leadtime.py   프레임 세트 수집. t_fail = t_commit 에서 -20/-10/-5/-1, 성공은 진행률로 짝지음
                      --chunk K --stride S 로 각 지점의 앞쪽 프레임까지 보관
leadtime_probe.py     SUCCESS/FAILURE 이진 예측. direct / twostage_nothink / twostage_think,
                      --chunk 로 다중 이미지 전송, --faults/--leads 로 슬라이스
fact_probe.py         --probe text (이미지 없이 기하만 문장으로 = 물리 확인)
                      --probe fact (참인 상태 문장을 덧붙여 결과 예측)
adaptive_probe.py     --mode self    현재 프레임만 주고 "이 한 장으로 되는가" (자기 불확실성)
                      --mode oracle  네 후보 설정 전수 = 선택의 천장 + 후보별 정답 라벨
                      --mode adaptive/placebo/fixed  선택 조건들
```

이미지 세트:

```
images_leadtime/          agentview 80장 (실패 40 + 성공 40), lead별 10+10
images_leadtime_wrist/    같은 80개 상태를 손목 시점으로
images_leadtime_chunk/    같은 80개 상태 × 3프레임 청크 (stride 10)
images_leadtime_blind10/  blind 결함 2종 × 10 에피소드 (실패 80 + 성공 80)
images_leadtime_blind30/  같은 설계 × 30 에피소드 (실패 240 + 성공 240)
images_leadtime_adaptive/ 4결함 × 10 에피소드 × 5프레임 (stride 5) — 후보 t-5/-10/-20 용
```

## 환경

```bash
conda activate fasi && export MUJOCO_GL=egl && cd ~/yuna_ws/failure_prediction
```

Gemma는 연구실 서버의 원격 vLLM이다(SSH 터널, 포트 18007, served name `gemma4`). 접속 설정은
하드코딩하지 않았다: `GEMMA_BASE_URL` / `GEMMA_MODEL` / `GEMMA_API_KEY` 또는
`--base-url` / `--model` / `--api-key`.

**`conda run`은 자식 출력을 버퍼링한다.** 진행 상황을 실시간으로 보려면 env의 python을 직접 부른다:

```bash
setsid nohup ~/miniconda3/envs/fasi/bin/python -u <스크립트> ... > results/run.log 2>&1 &
tail -f results/run.log
```

그리고 `pkill -f <패턴>`은 그 패턴을 명령줄에 담은 자기 셸까지 죽인다 — PID로 다룰 것.

## 재현

```bash
# 프레임 세트
python collect_leadtime.py --dry-run
python collect_leadtime.py --verify                       # --verify 가 t_commit 을 옆 폴더와 교차검증
python collect_leadtime.py --camera robot0_eye_in_hand --images-dir images_leadtime_wrist
python collect_leadtime.py --verify --chunk 3 --stride 10 --images-dir images_leadtime_chunk

# 이진 예측 (--limit 8 로 서버 확인 후 본 실행)
python leadtime_probe.py --limit 8 --conditions direct
python -u leadtime_probe.py
python -u leadtime_probe.py --chunk --images-dir images_leadtime_chunk --conditions direct,twostage_think

# 분해: 물리 / 사실 주입
python fact_probe.py --probe text
python fact_probe.py --probe fact --baseline results/leadtime_20260918_135728.jsonl

# 지각 프로브는 옆 폴더 도구를 쓴다 (경로만 넘긴다)
cd ../counterfactual_pilot
python gemma_action_choice.py --task height        --images-dir ../failure_prediction/images_leadtime       --limit 80 --robustness
python gemma_action_choice.py --task gripper_state --images-dir ../failure_prediction/images_leadtime       --limit 80 --robustness
python gemma_action_choice.py --task gripper_state --images-dir ../failure_prediction/images_leadtime_wrist --limit 80 --robustness
cd ../failure_prediction

# blind 결함 재현 (10 -> 30 에피소드; 30에서 청크 효과가 유의해진다)
python collect_leadtime.py --verify --types close_too_high,gripper_closed_early \
    --seeds $(seq -s, 0 29) --max-failures 60 --n-success 60 \
    --chunk 3 --stride 10 --images-dir images_leadtime_blind30
python -u leadtime_probe.py         --images-dir images_leadtime_blind30 --conditions direct
python -u leadtime_probe.py --chunk --images-dir images_leadtime_blind30 --conditions direct

# 적응적 시점 선택 — 게이트부터
python collect_leadtime.py --verify --seeds 0,1,2,3,4,5,6,7,8,9 --max-failures 40 --n-success 40 \
    --chunk 5 --stride 5 --images-dir images_leadtime_adaptive
python -u adaptive_probe.py --mode self      # 게이트 2: 자기 불확실성이 있나 (640콜)
python -u adaptive_probe.py --mode oracle    # 게이트 1: 고를 게 있나 (1280콜)
```

결과는 run id(기본 timestamp)로 분리되어 `results/`에 떨어지고 덮어쓰지 않는다. 사고 전문은
jsonl에만 들어가고 csv에서는 빠진다.

## 읽을 때 주의

- **답 분포를 정확도보다 먼저 본다.** 세트가 균형이라 상수 답은 정확히 50%가 되고, 이 파일럿은
  그 붕괴에 여러 번 걸렸다(`paraphrase`가 80장 전부 한쪽으로).
- **`--limit` 을 쓴 표의 수치는 결과가 아니다.** 서버가 답하는지만 보는 것이다.
- **결함별 셀이 2~3 에피소드면 방향 이상으로 읽지 않는다.** n=3에서 두 번 뒤집힌 전례가 있다
  (본 README의 정정 18·19번).
- **이진 과제의 Oracle@N은 구조적으로 부풀려진다.** 후보 4개가 독립 동전이면 "하나라도 맞음"이
  93.75%다. 천장은 프레임별 오라클이 아니라 **결함별 최적 고정**으로 잡는다.
