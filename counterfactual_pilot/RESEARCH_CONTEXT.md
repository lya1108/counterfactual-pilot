# 공동연구 컨텍스트

## 연구 주제

이미 학습된 Diffusion Policy를 유지한 상태에서, 실행 중 실패 위험을 감지하고 필요한 시점에만 추가 계산을 사용해 행동을 교정하는 test-time failure recovery 연구

기본 흐름

Diffusion Policy 실행
→ 실패 위험 감지
→ Gemma/VLM이 실패 상황 해석
→ counterfactual correction 또는 복구 목표/제약 생성
→ Diffusion Policy의 후보 생성 또는 sampling guidance에 반영
→ 후보 평가
→ 행동 실행

## 기존 공동연구에서 확인된 점

- VLM이 실패 원인을 잘 설명한다고 실제 제어 성능이 자동으로 좋아지지는 않았음
- 좋은 후보를 평가하는 것만으로는 부족했고, 후보 집합 안에 복구 가능한 trajectory가 없는 경우가 있었음
- 불필요한 개입은 기존 성공 episode를 망칠 수 있음
- recovery와 success preservation을 함께 봐야 함

## 내 역할 / 관심 부분

내 원래 연구 아이디어는 실패 경험에서

"어떤 행동이나 조건이 달랐다면 성공했을까?"

를 counterfactual하게 분석하고, 성공 가능성이 높은 행동 수정이나 subgoal을 얻는 것

공동연구에서는 이 counterfactual 요소를 기존 failure-aware Diffusion Policy 연구에 추가하는 방향을 검토 중

현재 관심 질문

실패 상황에서 Gemma가 단순히 실패 원인을 설명하는 데서 끝나는 것이 아니라,
원래 행동 대신 어떤 행동을 했어야 더 나았을지 counterfactual correction을 제안할 수 있는가?

## 현재 파일럿 실험

목적은 본 실험이 아니라 연구 주제와 Gemma의 역할을 이해하는 것

입력
- 실패 직전 이미지

Gemma 출력
- MOVE_LEFT
- MOVE_RIGHT
- MOVE_FORWARD
- MOVE_BACKWARD
- MOVE_UP
- MOVE_DOWN
- OPEN_GRIPPER
- CLOSE_GRIPPER

현재 Gemma가 선택한 행동은 정답으로 간주하지 않음
"원래 행동 대신 이 행동을 했으면 더 나았을 것"이라는 counterfactual proposal로만 봄

이번 단계에서 확인할 것
- 특정 행동만 반복하는지
- 이미지에 따라 선택이 달라지는지
- 사람이 보기에도 대안 행동이 말이 되는지
- structured output이 정상적으로 동작하는지

## 현재 환경

- 작업 폴더: /home/nuri5/yuna_ws
- 파일럿 폴더 예정: /home/nuri5/yuna_ws/counterfactual_pilot
- 원격 Gemma 서버 사용
- endpoint: http://localhost:18007
- model id: gemma4
- vision input 지원 확인
- structured_outputs.choice 지원 확인
- guided_choice 구버전 방식은 사용하지 않음

## 이번 단계에서 하지 않을 것

- 기존 중범님 코드 수정
- simulator oracle 평가
- snapshot restore
- Gemma가 선택한 행동 실제 실행
- accuracy / coverage 정식 평가
- Diffusion guidance 연결
- memory 구현
- 본 실험 성능 비교

## 작업 원칙

- 새 코드와 파일은 /home/nuri5/yuna_ws 아래에만 생성
- 중범님 코드는 read-only 참고용
- 다른 사람 파일, 프로세스, 서버 설정 수정 금지
- 먼저 작은 파일럿으로 확인한 뒤 다음 단계 결정
