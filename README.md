# 🎮 Geometry Dash — Reinforcement Learning Agent

A physics-accurate Geometry Dash simulator with a full **PPO training pipeline** built on Stable-Baselines3. The agent learns to jump over obstacles from a compact observation vector, trained entirely inside a custom pygame engine.

---

## 📸 Demo

![Trained PPO agent playing](ScreenRecording2026-03-31at11_14_02AM-ezgif_com-video-to-gif-converter.gif)

---

## 🗂️ Repository Structure

```
geometry-dash-reinforcement-learning/
└── Game/
    ├── constants.py            # Physics, display, and reward constants
    ├── game.py                 # Core game engine — physics, rendering, collision
    ├── level_generator.py      # Procedural level generation with difficulty control
    ├── gym_env.py              # Gymnasium wrapper (reset / step / reward shaping)
    ├── train_ppo.py            # PPO training — vectorized envs, checkpointing, curriculum
    ├── evaluate_ppo.py         # Evaluation across seeds and difficulty levels
    ├── watch_ppo.py            # Watch a trained agent play in real time
    ├── plot_training.py        # CLI tool to generate training plots
    ├── training_plots.py       # Plotting logic (reward curves, loss curves)
    └── test_observation_stream.py
```

**Pipeline:**
```
LevelGenerator → Game Engine → Gym Wrapper → PPO Training → Evaluate / Watch
```

---

## ⚙️ Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/danielh-hong/geometry-dash-reinforcement-learning
cd geometry-dash-reinforcement-learning
```

**Mac / Linux**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Key dependencies: `stable-baselines3`, `gymnasium`, `pygame`, `numpy`, `matplotlib`

---

## 🕹️ Play Manually

```bash
python Game/game.py
```

| Key | Action |
|-----|--------|
| `SPACE` / `↑` | Jump |
| `R` | Restart |
| `H` | Toggle hitbox overlay — draws the exact collision rectangles used by the engine, useful for verifying near-misses |
| `T` | Toggle telemetry panel |
| `ESC` / `Q` | Quit |

---

## 🤖 PPO Training

### Quick start

```bash
python Game/train_ppo.py \
    --timesteps 500000 \
    --difficulty 3 \
    --num-envs 8 \
    --seed 42 \
    --log-dir Game/logs_ppo
```

### Advanced — reward shaping + curriculum

```bash
python Game/train_ppo.py \
    --lr 0.00006 \
    --ent-coef 0.0003 \
    --target-kl 0.01 \
    --jump-penalty 0.04 \
    --air-jump-penalty 0.05 \
    --unnecessary-jump-penalty 0.20 \
    --jump-danger-distance 80 \
    --load-model Game/logs_custom/checkpoints/ppo_policy_final.zip \
    --difficulty 5 \
    --level-length 9000 \
    --timesteps 5000000 \
    --num-envs 8 \
    --progressive \
    --log-dir Game/logs_progressive
```

### Key arguments

| Argument | Description |
|----------|-------------|
| `--lr` | Learning rate for policy and value networks |
| `--ent-coef` | Entropy bonus — higher values push the agent to explore more |
| `--target-kl` | Stops a PPO update early if the policy changes too much in one step |
| `--jump-penalty` | Small penalty per jump — discourages mindless button mashing |
| `--air-jump-penalty` | Penalty for pressing jump while already airborne |
| `--unnecessary-jump-penalty` | Penalty for jumping when no obstacle is within range |
| `--jump-danger-distance` | Pixel distance that defines "near enough to need a jump" |
| `--progressive` | Ramps up obstacle density across the level instead of fixed difficulty |
| `--load-model` | Resume training from an existing `.zip` checkpoint |
| `--num-envs` | Number of parallel game instances — more = faster data collection |

---

## 📊 Evaluate Generalization

Once trained, this checks whether the policy actually generalizes — or if it just memorized one level layout.

```bash
python Game/evaluate_ppo.py \
    --model Game/logs_ppo/checkpoints/ppo_policy_final.zip \
    --difficulties 1 2 3 \
    --num-seeds 30
```

It runs the same policy over 30 different randomly-seeded levels at each difficulty and reports:

| Metric | What it tells you |
|--------|-------------------|
| `mean_reward` | How well the agent performs on average across unseen levels |
| `best_reward` | The single best run — shows the ceiling of what the policy can do |
| `mean_steps` | How long the agent survives on average before dying |
| `completion_rate` | % of runs where the agent survived to the time limit without dying |

---

## 👁️ Watch the Agent Play

```bash
python Game/watch_ppo.py \
    --model Game/logs_ppo/checkpoints/ppo_policy_final.zip \
    --difficulty 3 \
    --seed 42
```

Press `T` to open the telemetry panel — shows the agent's live jump probability and confidence score each frame. Press `H` to overlay the collision hitboxes.

---

## 📈 Plot Training Curves

```bash
python Game/plot_training.py \
    --metrics-file Game/logs_ppo/training_metrics_ppo_<timestamp>.csv \
    --output-dir Game/training_figures
```

Reads the CSV logged during training and produces a reward curve (and loss curve if available). The raw per-episode reward is plotted alongside a 100-episode rolling average so you can see both the noise and the overall trend. Saved as a PNG to `training_figures/`.

---

## 🧠 Observation Space

The agent sees a **28-float vector** at every step. A few of the most important features:

- **`player_vy` (index 1)** — vertical velocity, normalized to `[-1, 1]`. Lets the agent know if it's rising, falling, or on the ground.
- **`on_ground` (index 2)** — binary flag. The agent can only jump when this is `1.0`, so this is the gating signal for any jump decision.
- **`rel_x` (per obstacle)** — how far ahead the next obstacle is, normalized to `[0, 1]`. Drives the timing of jumps.
- **`time_to_reach` (per obstacle)** — seconds until the obstacle reaches the player at current speed. More intuitive than raw distance for learning jump timing.

The remaining features cover obstacle type, position, dimensions, and vertical clearance — giving the agent a full picture of the next 3 upcoming obstacles.

---

## 📦 Expected Outputs

After a training run, your `--log-dir` will contain:

```
logs_ppo/
├── checkpoints/
│   ├── ppo_policy_10000_steps.zip    # Periodic snapshots saved during training
│   ├── ppo_policy_50000_steps.zip
│   └── ppo_policy_final.zip          # Final policy — use this for evaluation and watching
└── training_metrics_ppo_<timestamp>.csv   # Per-episode log of reward, steps, and loss
```

And after running `plot_training.py`:

```
training_figures/
└── training_metrics_ppo_<timestamp>_reward.png   # Reward curve with rolling average overlay
```

**Checkpoints** (`.zip`) are loadable by both `watch_ppo.py` and `evaluate_ppo.py` and can also be passed to `--load-model` to resume training from that point.

**The metrics CSV** records one row per episode: episode number, reward, episode length in steps, and a 100-episode rolling average. Feed it into `plot_training.py` to visualize progress.

**The reward plot** shows raw per-episode reward (faint) alongside a smoothed rolling average — the rolling average is the clearest signal of whether the agent is actually improving over time.
