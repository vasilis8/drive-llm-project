# 🏎️ DriveLLM: Language-Conditioned High-Speed Autonomous Racing

A reinforcement learning agent that maps **natural language commands** and **LiDAR scans** to continuous racing controls using the CARLA simulator.

## Architecture

```
                    ┌──────────────────┐
  "Push hard!" ───→ │  MiniLM Encoder  │──→ (384)─┐
                    └──────────────────┘           │
                    ┌──────────────────┐           ├─→ Fusion MLP ──→ PPO Actor-Critic
  LiDAR (1080) ──→ │     1D-CNN       │──→ (256)─┤
                    └──────────────────┘           │
  Vehicle State ──→ ────────────────────── (5)────┘
```

The agent takes a language command (e.g., "Push hard", "Conserve tires") and fuses it with LiDAR perception to produce steering and throttle outputs via PPO.

## Setup

### Local (Mac M1 / CPU)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Cloud (GPU instance with Docker)

```bash
bash scripts/setup_cloud.sh
```

## Usage

### Run Tests
```bash
python -m pytest tests/ -v
```

### Train (Local — DummyEnv)
```bash
python -m src.training.train --config configs/default.yaml --dummy
```

### Train (Cloud — CARLA)
```bash
bash scripts/run_training.sh
```

### Evaluate
```bash
python -m src.evaluation.evaluate --model checkpoints/drive_llm_final --dummy --output results/evaluation.json
```

### Visualize Results
```bash
python -m src.evaluation.visualize --results results/evaluation.json --output results/plots/
```

### Record Agent Driving (on cloud with CARLA)
```bash
python scripts/record_replay.py --model checkpoints/drive_llm_final --episodes 1
```
This generates MP4 videos with a HUD overlay showing the active command, speed, and steering for each command category.

## Project Structure

```
├── configs/default.yaml          # Hyperparameters & CARLA config
├── src/
│   ├── envs/
│   │   ├── carla_env.py          # CARLA gymnasium wrapper (cloud)
│   │   ├── dummy_env.py          # Mock env for local testing
│   │   └── rewards.py            # Command-aware reward function
│   ├── models/
│   │   ├── instruction_encoder.py # Frozen MiniLM sentence embeddings
│   │   └── policy.py             # Custom 1D-CNN + MLP feature extractor
│   ├── training/
│   │   ├── train.py              # Main training script
│   │   └── callbacks.py          # Curriculum & metrics callbacks
│   └── evaluation/
│       ├── evaluate.py           # Per-category evaluation
│       └── visualize.py          # Charts & radar plots
├── scripts/
│   ├── setup_cloud.sh            # Cloud GPU bootstrap
│   └── run_training.sh           # CARLA Docker + training launcher
├── tests/                        # Unit tests (31 tests)
├── requirements.txt              # Local dependencies
└── requirements-cloud.txt        # Cloud/CARLA dependencies
```

## Command Categories

| Category | Behavior | Example Commands |
|----------|----------|-----------------|
| 🔴 Aggressive | High speed, late braking | "Push hard", "Full attack", "Overtake now" |
| 🟢 Conservative | Smooth inputs, energy saving | "Conserve tires", "Cruise pace" |
| 🔵 Defensive | Hold position, block | "Defend inside line", "Hold position" |
| 🟣 Neutral | Standard driving | "Follow racing line", "Steady pace" |

## Tech Stack

- **Simulator:** CARLA 0.9.15 (headless Docker)
- **RL:** Stable-Baselines3 PPO
- **NLP:** all-MiniLM-L6-v2 (sentence-transformers)
- **Framework:** PyTorch, Gymnasium