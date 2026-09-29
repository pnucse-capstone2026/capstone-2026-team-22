# Pokemon Red RL — Exploration

포켓몬 레드를 **맵 탐험(exploration)** 중심으로 학습시키는 RL 코드베이스.  
원본 [`pokemonred_puffer`](https://github.com/drubinstein/pokemonred_puffer)를 기반으로 정리.

## 의도

- PyBoy 위에서 Gymnasium 환경을 돌리고, PufferLib + CleanRL 스타일 PPO로 학습한다.
- 저장한 정책을 외부 플레이 환경에서도 같은 관측과 액션 조건으로 실행한다.
- 기본 목표는 주어진 맵 내부를 탐험하고 주변 맵에 방문하는 것이다.
- 하이퍼파라미터·보상·래퍼는 `config.yaml`에서 바꾸고, 학습은 `train` CLI로 실행한다.

학습과 추론은 같은 `observation.py`, `actions.py`, `action_exec.py`를 사용합니다.
같은 게임 상태와 LSTM 상태에서 같은 행동을 실행했을 때 다음 관측과 좌표가 같아야 합니다.

## 설치 및 실행

```sh
cd pokemonred_puffer-main
pip3 install -e .
# ROM: red.gb, 시작 세이브: pyboy_states/ 또는 pokemonred_puffer/state_files/

# 현재 적용될 설정 확인
python -m pokemonred_puffer.train inspect-config --profile local
python -m pokemonred_puffer.train inspect-config --profile gpu

# 화면을 띄운 단일 환경 점검
python -m pokemonred_puffer.train debug --steps 500

# CPU/RAM 절약형 로컬 학습 (기본 프로필)
python -m pokemonred_puffer.train train --profile local

# CUDA 정책 추론/PPO 학습 + Weights & Biases 기록
python -m pokemonred_puffer.train train --profile gpu --track

# 이전 run을 무시하고 새 실험 시작
python -m pokemonred_puffer.train train --profile local --no-resume

# 특정 체크포인트에서 이어 학습
python -m pokemonred_puffer.train train --profile gpu --checkpoint runs/<run-id>

# trainer_state.pt 없이 저장 모델만으로 rollout 평가
python -m pokemonred_puffer.train evaluate --checkpoint runs/<run-id>/model_000010.pt

# 저장 모델 가중치만으로 새 optimizer부터 학습
python -m pokemonred_puffer.train train --profile gpu --checkpoint model_000010.pt

# 환경 병렬 설정 측정
python -m pokemonred_puffer.train autotune
```

`config.yaml` / `red.gb` / 세이브 경로는 패키지 루트 기준으로 찾습니다. 상위 폴더에서 실행해도 됩니다.

에피소드는 `env.max_steps`에서 truncate되고 reset됩니다.  
`--resume`은 기본 활성화이며 `train.data_dir`의 가장 최근 `model_*.pt` 가중치만 자동으로 불러옵니다. 이때 optimizer·학습 step은 새로 시작하며 `trainer_state.pt`는 사용하지 않습니다. 정확한 이어학습(optimizer/RNG/step 복원)은 `train --checkpoint runs/<run-id>`처럼 run 디렉터리를 지정하세요. 자세한 CLI는 `python -m pokemonred_puffer.train --help`를 참고하세요.

### 실행 프로필

`train.profiles`는 공통 PPO 설정 위에 실행 환경별 값을 병합합니다.

| 프로필 | 선택 | 주요 설정 | 용도 |
|---|---|---|---|
| `local` | `--profile local` (기본) | CPU, 1 env / 1 worker, rollout 16,384 | 로컬 개발·메모리 절약 |
| `gpu` | `--profile gpu` | CUDA, 64 env / 32 workers, rollout 32,768 | CUDA 사용 학습 |

GPU는 정책 추론과 PPO 업데이트를 가속합니다. PyBoy 에뮬레이터와 multiprocessing worker는 계속 CPU/RAM을 사용하므로, GPU 프로필에서도 환경 수와 rollout 크기를 무작정 키우면 안 됩니다. CUDA 지원 PyTorch와 호환 드라이버가 필요합니다.

`train.profile_cuda_sync`는 기본 `False`입니다. 정확한 GPU 타이밍이 필요할 때만 켜세요.

## 학습 흐름

```
config.yaml → train.py → environment + reward + wrappers + policy → cleanrl_puffer (PPO)
                                                              ↘ wandb_stats / eval (로깅·맵 오버레이)
```

기본값: reward `baseline.ExplorationReward`, wrappers `exploration`(= `EpisodeStatsWrapper`), policy `MultiConvolutionalPolicy`.

### 학습 안정화 (요약)

- RNN: 에피소드 `done`/`truncated` 슬롯의 LSTM state를 다음 추론 전에 초기화
- 체크포인트: 모델·trainer state 원자 저장 + RNG 복원 (`training/checkpointing.py`)
- 환경: `close()`에서 PyBoy 종료, 심볼 주소 캐시, 정지 시 LocalMap 스캔 스킵
- 로깅: `pokemon_exploration_map`은 에피소드 종료 시에만 생성·전송 (`render_global_map`)
- `use_global_explore_map`: 선택적 lifetime float `explore_map` (기본 OFF). `save_video`와 독립
- 관측: `screen`, `tracking_mask`, `map_scope`, `direction`, `last_action`, `movement_result` (`map_id`/`x`/`y` 없음)
  - `tracking_mask`는 플레이어 중심 20×18 타일의 관측 지식(관측/막힘/물/워프/NPC/오브젝트) 6채널
  - `map_scope`는 Target / Internal / External 분류 (scope FiLM + one-hot)
- 보상 (Target / Internal / External):
  - Target = 에피소드 시작 맵. 에피소드 중 고정
  - Internal = Target에서 `warps.json` 실내 warp로 이어지는 건물·층
  - External = 나머지 (동굴 Target이면 4번도로·3번도로 등 출구 밖 야외/도로)
  - `target_tile_reward`, `internal_tile_scale`, `observed_tile_reward`, `internal_entry_reward`, `external_entry_reward`, `external_return_reward`, `blocked_penalty`, `no_progress_*` (`config.yaml` rewards 참고)
- wandb: `reward/tile|tile_outside|entry|return|blocked|stagnation|sum`, `reward_config/*`

---

## 추론 환경 연동

`pokemonred_puffer/play_bridge.py`가 체크포인트 로드, 관측 생성, LSTM 상태와 액션
실행을 담당합니다. 외부 play.py는 `Skills/Exploration/exploration.py`가
`PlayExplorationPolicy`를 연결합니다.

- 기본 행동 선택: 학습과 같은 확률 샘플링 (`greedy=False`)
- 기본 전투 처리: 학습과 같은 `full` 프로필
- 액션 실행: 학습과 같은 `execute_rl_action()`
- `explore` 시작 시: 같은 PyBoy 세션이면 방문 마스크·Target·LSTM을 유지.
  세션 첫 explore만 비우고 Target을 고정. 스텝 상한만 다시 센다.
- 정책 로드나 추론 실패: 다른 알고리즘으로 자동 전환하지 않고 오류로 표시

`ExploreRunner.tick()`은 내부에서 한 액션의 모든 PyBoy 프레임을 실행합니다.
외부 앱은 탐험이 활성화된 동안 별도의 `pyboy.tick()`을 추가하면 안 됩니다.
추가 tick은 다음 관측 시점을 바꿔 학습 환경과 다른 동작을 만듭니다.

현재 정책의 액션은 `DOWN, LEFT, RIGHT, UP` 네 개입니다. 예전 6-action·`visited_mask`·`map_role` FiLM-only·`observed_map`/`coverage_convs` 체크포인트는 호환되지 않으며 로드 단계에서 거부됩니다.
현재 코드로 새로 학습한 체크포인트(`tracking_mask` 계약)를 사용하세요.

ROM과 체크포인트가 준비된 경우 다음 명령으로 공용 액션·관측 경로를 점검합니다.

```sh
python play_integration/verify_parity.py --checkpoint runs/<run-id>/model_XXXX.pt
```

---

## 파일별 역할

### 루트

| 파일 | 역할 |
|------|------|
| `config.yaml` | env / train / profiles / reward / policy / wrapper / wandb 설정 (학습 SoT) |
| `pyproject.toml` | 패키지 의존성·설치 정의 |
| `red.gb` | 포켓몬 레드 ROM (직접 준비) |
| `pyboy_states/` | 시작용 PyBoy 세이브 스테이트 |
| `sweep-config.yaml` | Wandb sweep 파라미터·메트릭 |

### `pokemonred_puffer/` (핵심)

| 파일 | 역할 |
|------|------|
| `train.py` | 학습 CLI (train / evaluate / autotune / debug / inspect-config), `--profile` 병합 |
| `environment.py` | 포켓몬 레드 Gymnasium 환경 (`RedGymEnv`) |
| `actions.py` | 방향 액션 인덱스·델타 상수 |
| `observation.py` | 화면·tracking_mask·스칼라 관측 |
| `observation_knowledge.py` | 관측 타일 직접/간접 경험 상태 |
| `episode_state.py` | 에피소드 탐험 상태 초기화 |
| `video_recorder.py` | rollout 비디오 writer |
| `emulator_runtime.py` | PyBoy 생성과 WRAM 심볼 주소 캐시 |
| `exploration_cheats.py` | 야생·트레이너 전투 스킵 |
| `play_bridge.py` | 외부 추론용 정책·관측·LSTM 연결부 |
| `play_env_parity.py` | 외부 PyBoy에 학습 전투 설정 적용 |
| `cleanrl_puffer.py` | PPO 학습 루프·체크포인트·대시보드 |
| `training/checkpointing.py` | 원자적 모델/state 저장, RNG 캡처·복원 |
| `training/experience.py` | observation 변환과 PPO rollout buffer |
| `training/rollout.py` | 단일 환경 interactive rollout |
| `rewards/baseline.py` | Target/Connected/Outside 탐험 보상 |
| `policies/multi_convolutional.py` | 화면+스칼라 CNN 정책 및 RNN 래퍼 |
| `wrappers/episode_stats.py` | 에피소드 return/length·탐험 통계 |
| `wrappers/coords_writer.py` | 좌표/액션 CSV 녹화 (`record` 프리셋) |
| `wandb_stats.py` | env info → wandb/대시보드 메트릭 |
| `eval.py` | 탐험 맵을 관동 지도 위에 겹친 시각화 |
| `global_map.py` | 로컬 ↔ 글로벌 좌표 변환 유틸 |
| `Map.py` | **지도 구현 SoT** — GlobalMap + LocalMap + Cell (`env.map_state`) |
| `map/` | 오버레이 헬퍼 (`render_global_map`) |
| `data/map_scope.py` | Target / Internal / External 범위 계산 |
| `data/map_categories.py` | 맵 유형 / Target·게이트 판정 |
| `profile.py` | SPS·CPU/GPU 프로파일링 |
| `sweep.py` | Wandb 하이퍼파라미터 스윕 |
| `map_data.json` / `warps.json` | 맵·워프 데이터 |
| `kanto_map_dsv.png` | 탐험 오버레이용 관동 맵 |
| `pokered.sym` | 롬 심볼 정의 |

### `play_integration/`

| 파일 | 역할 |
|------|------|
| `verify_parity.py` | ROM 기반 학습·추론 환경 동등성 점검 |

### `pokemonred_puffer/visualizations/`

| 파일 | 역할 |
|------|------|
| `actions_replay.py` | 액션 CSV → 게임플레이 mp4 |
| `coordinates_replay.py` | 좌표 CSV → 관동 맵 이동 mp4 |

녹화: `python -m pokemonred_puffer.train train --profile local -w record`  
재생 예:
```sh
python -m pokemonred_puffer.visualizations.actions_replay red.gb recordings out_actions
python -m pokemonred_puffer.visualizations.coordinates_replay recordings out_coords
```

## 수정할 때

1. **설정만** — `config.yaml` (보상 가중치, `num_envs`, `max_steps` 등)
2. **보상/정책/래퍼** — 각 디렉터리에 클래스 추가 후 `config.yaml`에 `module.Class`로 등록
3. **환경 본체** — `environment.py`
4. **지도** — `Map.py` (구현 SoT). 오버레이는 `map/overlay.py`
5. **학습·추론 공용 동작** — `actions.py`, `observation.py`, `action_exec.py` 중 한 곳에서 정의

래퍼는 기본으로 `EpisodeStatsWrapper`만 사용합니다. `--wrappers-name none`이면 래퍼 없이 돌릴 수 있습니다.

테스트는 `python -m pytest tests -q`로 실행합니다.
