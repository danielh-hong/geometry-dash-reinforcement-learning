# Geometry Dash RL — Game Engine

Simple, clean Geometry Dash clone built to be played by humans **and** driven programmatically by RL agents or YOLO pipelines.

## Setup

**1. Create and activate a virtual environment** (do this once)
```bash
cd Game
python -m venv venv

# Mac / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate # make sure to activate venv
```
You'll see `(venv)` in your terminal prompt — that means it's active.

**2. Install dependencies**
pip install -r requirements.txt


**3. Play**
python game.py

> **Every time you open a new terminal**, re-run the activate command before working on the project.

> **Never commit `venv/`** — add it to `.gitignore`.

> **When you add new packages**, update the requirements file with:
> ```bash
> pip freeze > requirements.txt
> ```

Controls: `SPACE` / `UP` = jump &nbsp;|&nbsp; `R` = restart &nbsp;|&nbsp; `Q` = quit

---

## PPO + Gymnasium (Recommended Training Path)

This project now includes a Gymnasium-compatible wrapper and a full PPO trainer.

### What is Gymnasium?
Gymnasium is a standard API for RL environments (`reset`, `step`, `action_space`, `observation_space`).
It does **not** replace your game; it wraps `game.py` so libraries like Stable-Baselines3 can train reliably.

### Why PPO over REINFORCE?
- REINFORCE: great for learning fundamentals, but high variance.
- PPO: clipped updates + GAE, usually more stable and better for real performance.

### Install PPO dependencies
```bash
pip install -r requirements.txt
```

### Train PPO
```bash
python train_ppo.py --timesteps 500000 --difficulty 1 --num-envs 8
```

Outputs:
- PPO checkpoints in `logs_ppo/checkpoints/`
- CSV metrics in `logs_ppo/`
- Reward/loss training figure in `training_figures/`

### Evaluate generalization
```bash
python evaluate_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulties 1 2 3 --num-seeds 30
```

---

## Programmatic API

```python
from game import Game

# Headless (for training)
g = Game(render=False, seed=42)
obs = g.reset()

while True:
    action = 1 if should_jump(obs) else 0   # your logic here
    obs, reward, done = g.step(action)
    if done:
        obs = g.reset()
```

### `obs` dict

| Key | Type | Description |
|-----|------|-------------|
| `player_y` | float | Player top-left y (px) |
| `player_vy` | float | Vertical velocity |
| `on_ground` | bool | Is cube on the ground? |
| `alive` | bool | False after collision |
| `scroll_x` | float | Total px scrolled (proxy for progress) |
| `obstacles` | list[dict] | Upcoming obstacles, sorted by distance |

Each obstacle dict: `{type, x, y, w, h}`

### `step(action, dt=1/60)`

| Param | Values |
|-------|--------|
| `action` | `0` = no-op, `1` = jump |
| `dt` | seconds per step (default `1/60`) |

Returns `(obs, reward, done)`

---

## File structure

```
gd_game/
  game.py        ← entire game + API (start here)
  constants.py   ← all tunable values
  requirements.txt
```