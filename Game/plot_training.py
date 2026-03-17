import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ── change this to your actual CSV filename ───────────────────────────────────
CSV_PATH = "logs/training_metrics_20260302_130457.csv"

df = pd.read_csv(CSV_PATH)

fig, ax = plt.subplots(figsize=(8, 4))

# Raw episode reward (faint)
ax.plot(df["episode"], df["episode_reward"],
        color="steelblue", alpha=0.25, linewidth=0.8, label="Episode reward")

# 100-episode rolling average (bold)
ax.plot(df["episode"], df["avg_reward_100"],
        color="steelblue", linewidth=2.0, label="100-ep rolling average")

ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("Training Progress — REINFORCE on Fixed Difficulty-1 Level")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/training_curve.png", dpi=150)
print("Saved to figures/training_curve.png")