# Watch agent on difficulty 1 (easy, fixed level)
python game.py --agent --weights logs/checkpoints/policy_final.pth

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