# PPO Command Playbook (Geometry Dash RL)

This file is the practical, copy-paste workflow for training and testing PPO in this repo.

It is based on the current project scripts:
- `Game/train_ppo.py`
- `Game/evaluate_ppo.py`
- `Game/watch_ppo.py`
- `Game/gym_env.py`
- `Game/training_plots.py`

---

## 0) One-time setup

From repository root:

```powershell
cd Game
python -m venv venv
venv\Scripts\activate
pip install -r ..\requirements.txt
```

If you already have a venv, just activate it:

```powershell
cd Game
venv\Scripts\activate
```


## 2) Fast sanity run (pipeline check only)

```powershell
python train_ppo.py --timesteps 10000 --difficulty 1 --num-envs 1 --n-steps 128 --batch-size 128
```

Expected output artifacts:
- model: `logs_ppo/checkpoints/ppo_policy_final.zip`
- metrics: `logs_ppo/training_metrics_ppo_YYYYMMDD_HHMMSS.csv`
- figure: `training_figures/training_metrics_ppo_..._reward.png`

---

## 3) Recommended real training commands

## A) Baseline (good starting point)

```powershell
python train_ppo.py --timesteps 500000 --difficulty 1 --num-envs 8 --seed 42
```

## B) Stronger run (recommended)

```powershell
python train_ppo.py --timesteps 2000000 --difficulty 1 --num-envs 8 --seed 42
```

## C) More stable variant if you see jump-spam

```powershell
python train_ppo.py --timesteps 2000000 --difficulty 1 --num-envs 8 --seed 42 --ent-coef 0.003 --lr 0.0002 --target-kl 0.015
```

## C2) Strong anti-jump-spam variant (recommended if still over-jumping)

```powershell
python train_ppo.py --timesteps 2000000 --difficulty 1 --num-envs 8 --seed 42 --ent-coef 0.003 --lr 0.0002 --target-kl 0.015 --jump-penalty 0.01 --air-jump-penalty 0.02 --zero-last-action
```

## C3) 
python [evaluate_ppo.py](http://_vscodecontentref_/7) --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulties 1 2 3 --seed-start 1000 --num-seeds 30

If jump-spam still persists after this run, try slightly stronger shaping:

```powershell
python train_ppo.py --timesteps 2000000 --difficulty 1 --num-envs 8 --seed 42 --ent-coef 0.002 --lr 0.00015 --target-kl 0.012 --jump-penalty 0.02 --air-jump-penalty 0.04 --zero-last-action
```

## D) Curriculum flavor (optional)

```powershell
python train_ppo.py --timesteps 2000000 --difficulty 2 --num-envs 8 --seed 42 --progressive
```

Notes:
- Keep `--num-envs 8` on decent CPU; reduce to `4` if your machine struggles.
- Do not judge policy quality from very short training.

---

## 4) Evaluate generalization (important)

Evaluate a trained PPO `.zip` checkpoint on unseen seeds:

```powershell
python evaluate_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulties 1 2 3 --seed-start 1000 --num-seeds 30
```

Quick eval:

```powershell
python evaluate_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulties 1 --seed-start 1000 --num-seeds 10
```

Interpretation:
- If mean reward stays near `-1.0`, policy is still weak.
- Increasing mean reward and steps indicates better control.

---

## 5) Watch PPO play in the game window

Use PPO watcher (not `game.py --agent`, which is for `.pth`):

```powershell
python watch_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulty 1 --seed 42 --action-repeat 4
```

Diagnostic watch (shows jump/no-jump confidence in console + telemetry):

```powershell
python watch_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulty 1 --seed 42 --action-repeat 4 --telemetry --show-probs --probs-interval 10
```

Tip: if behavior looks odd, test the latest numbered checkpoint too (sometimes more stable than final):

```powershell
python watch_ppo.py --model logs_ppo/checkpoints/ppo_policy_2000000_steps.zip --difficulty 1 --seed 42 --action-repeat 4
```

Try several seeds to verify behavior is not overfit:

```powershell
python watch_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulty 1 --seed 12 --action-repeat 4
python watch_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulty 1 --seed 99 --action-repeat 4
python watch_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulty 2 --seed 42 --action-repeat 4
```

---

## 6) Best training workflow (step-by-step)

1. **Train** a meaningful run:
   ```powershell
   python train_ppo.py --timesteps 2000000 --difficulty 1 --num-envs 8 --seed 42
   ```
2. **Evaluate** across unseen seeds:
   ```powershell
   python evaluate_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulties 1 2 3 --seed-start 1000 --num-seeds 30
   ```
3. **Watch** visually on fixed seeds:
   ```powershell
   python watch_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulty 1 --seed 42 --action-repeat 4
   ```
4. If weak/jumpy: retrain with lower entropy and smaller LR:
   ```powershell
   python train_ppo.py --timesteps 2000000 --difficulty 1 --num-envs 8 --seed 42 --ent-coef 0.003 --lr 0.0002 --target-kl 0.015
   ```

---

## 7) Command knobs you will actually change

Most useful flags in `train_ppo.py`:

- `--timesteps`: total training budget (bigger = better, slower)
- `--num-envs`: parallel environments (bigger = faster sampling)
- `--difficulty`: level difficulty
- `--seed`: reproducibility
- `--ent-coef`: exploration strength (lower if jump-spam persists)
- `--lr`: learning rate
- `--target-kl`: stabilizes updates
- `--fixed-level`: disables per-episode randomization (use only for ablations)
- `--progressive`: curriculum generation inside levels

---

## 8) Common mistakes

1. **Using wrong runner for model type**
   - `.pth` (REINFORCE/SimplePolicyNetwork): use `game.py --agent --weights ...`
   - `.zip` (SB3 PPO): use `watch_ppo.py --model ...`

2. **Using smoke-test checkpoint as final quality check**
   - Tiny timesteps are only for validating scripts.

3. **Evaluating only one seed**
   - Always test many seeds (`--num-seeds 30` or more).

---

## 9) Practical presets

### CPU-friendly preset

```powershell
python train_ppo.py --timesteps 500000 --difficulty 1 --num-envs 4 --n-steps 256 --batch-size 256 --seed 42
```

### Throughput preset (strong machine)

```powershell
python train_ppo.py --timesteps 2000000 --difficulty 1 --num-envs 8 --n-steps 256 --batch-size 256 --seed 42
```

### Anti-jump-spam preset

```powershell
python train_ppo.py --timesteps 2000000 --difficulty 1 --num-envs 8 --seed 42 --ent-coef 0.003 --lr 0.0002 --target-kl 0.015 --jump-penalty 0.01 --air-jump-penalty 0.02 --zero-last-action --n-epochs 10
```

---

## 10) Quick checklist before trusting a model

- [ ] Trained at least `500k` timesteps (prefer `2M+`)
- [ ] Evaluation on unseen seeds done (`num-seeds >= 30`)
- [ ] Watched on multiple seeds and difficulties
- [ ] Reward plot generated in `training_figures/`

If all four are true, you can trust the policy much more.
