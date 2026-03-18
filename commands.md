## HOW TO TRAIN: (level 1, 10000 games, resume the logs)
python train.py --episodes 10000 --difficulty 1 --seed 42 --resume logs/checkpoints/policy_ep00500.pth

## PPO TRAINING (Stable-Baselines3 + Gymnasium)
# Why: PPO is generally more stable and sample-efficient than REINFORCE.
# Prereqs (once): pip install -r requirements.txt

# Quick PPO run (recommended starting point)
python train_ppo.py --timesteps 500000 --difficulty 1 --num-envs 8
NOTE: timesteps is the total number of environment intercation steps PPO collects during training

# Stronger run with randomized levels (better generalization)
python train_ppo.py --timesteps 2000000 --difficulty 1 --num-envs 8 --seed 42

# Fixed-level ablation (for direct overfitting comparison)
python train_ppo.py --timesteps 500000 --difficulty 1 --num-envs 8 --fixed-level

# Evaluate PPO checkpoint on unseen seeds/difficulties
python evaluate_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulties 1 2 3 --num-seeds 30

# Watch PPO checkpoint in live game window (.zip model from SB3)
python watch_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulty 1 --seed 42

# Watch agent on difficulty 1 (easy, fixed level)
# NOTE: this command is for SimplePolicyNetwork .pth checkpoints, not PPO .zip
python game.py --agent --weights logs/checkpoints/policy_final.pth
- Another ex: python game.py --agent --weights logs/checkpoints/policy_ep10000.pth --difficulty 1 --seed 42

# This now automatically:
# - Uses difficulty 1 (easiest)
# - Seed 42 (same level every time)
# - Shows same obstacles repeatedly so you see if agent improves

## EASY LEVEL SELECTION
# Specific difficulty levels
python game.py --agent --difficulty 1  # Stereo Madness (easiest)
python game.py --agent --difficulty 2  # Normal
python game.py --agent --difficulty 3  # Hard
python game.py --agent --difficulty 4  # Harder
python game.py --agent --difficulty 5  # Insane/Demon

# Longer levels
python game.py --agent --difficulty 1 --length 12000  # 40 seconds

# Different seed (different level layout)
python game.py --agent --difficulty 1 --seed 99

# Human play with LevelGenerator
python game.py --difficulty 2 --seed 42

## HOW TO RUN WEIGHTS:
# Step 1: TRAIN (weights improve over time)
python train.py --episodes 500 --difficulty 1
# ↓ saves to logs/checkpoints/policy_ep50.pth, policy_ep100.pth, etc.

# Step 2: WATCH (just inference, no learning)
python game.py --agent --difficulty 1 --weights logs/checkpoints/policy_ep100.pth

How to Resume Training 🔄
Continue from Episode 500:
Important Notes:
Weights are loaded ✅ - Network starts with learned knowledge
Optimizer resets ⚠️ - Learning rate momentum starts fresh
Episode counter restarts - Logs will show ep 1-500 again
Use SAME difficulty/seed - For consistency, use same settings