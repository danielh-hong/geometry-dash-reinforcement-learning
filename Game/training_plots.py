from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple


def _moving_average(values: List[float], window: int) -> List[float]:
    """Compute a simple trailing moving average."""
    if not values:
        return []
    window = max(1, window)
    averaged = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        count = min(index + 1, window)
        averaged.append(running_sum / count)
    return averaged


def _safe_float(value: str | None) -> Optional[float]:
    """Convert CSV value to float when possible; return None otherwise."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_metrics(metrics_file: str) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Load metrics with schema flexibility.

    Supports:
    - REINFORCE CSV (episode_reward, loss, avg_reward_100)
    - PPO CSV (episode_reward, episode_steps, timesteps, ...)

    Returns
    -------
    episodes, rewards, losses, avg_rewards_100
        losses may be empty when not available in the source CSV.
    """
    episodes: List[int] = []
    rewards: List[float] = []
    losses: List[float] = []
    avg_rewards_100: List[float] = []

    with open(metrics_file, "r", newline="") as file:
        reader = csv.DictReader(file)
        has_loss = reader.fieldnames is not None and "loss" in reader.fieldnames
        has_avg100 = reader.fieldnames is not None and "avg_reward_100" in reader.fieldnames

        for row in reader:
            episode_raw = row.get("episode")
            if episode_raw is None or episode_raw == "":
                episodes.append(len(episodes) + 1)
            else:
                episodes.append(int(episode_raw))

            reward = _safe_float(row.get("episode_reward"))
            rewards.append(0.0 if reward is None else reward)

            if has_loss:
                loss = _safe_float(row.get("loss"))
                if loss is not None:
                    losses.append(loss)

            if has_avg100:
                avg = _safe_float(row.get("avg_reward_100"))
                if avg is not None:
                    avg_rewards_100.append(avg)

    # Backfill rolling average if source CSV does not provide avg_reward_100.
    if len(avg_rewards_100) != len(rewards):
        avg_rewards_100 = _moving_average(rewards, window=min(100, len(rewards)))

    return episodes, rewards, losses, avg_rewards_100


def generate_training_plots(metrics_file: str, output_dir: str = "training_figures") -> Optional[Path]:
    """
    Generate and save reward/loss training plots from a metrics CSV file.

    Returns saved figure path on success, else None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        return None

    episodes, rewards, losses, avg_rewards_100 = _load_metrics(str(metrics_path))
    if len(episodes) == 0:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reward_ma_50 = _moving_average(rewards, window=min(50, len(rewards)))

    has_loss = len(losses) == len(episodes) and len(losses) > 0

    if has_loss:
        loss_ma_50 = _moving_average(losses, window=min(50, len(losses)))
        figure, (ax_reward, ax_loss) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    else:
        figure, ax_reward = plt.subplots(1, 1, figsize=(12, 5))
        ax_loss = None

    ax_reward.plot(
        episodes,
        rewards,
        color="steelblue",
        alpha=0.25,
        linewidth=0.8,
        label="Episode reward"
    )
    ax_reward.plot(
        episodes,
        avg_rewards_100,
        color="steelblue",
        linewidth=2.0,
        label="Avg reward (100 ep)"
    )
    ax_reward.plot(
        episodes,
        reward_ma_50,
        color="navy",
        linewidth=1.2,
        linestyle="--",
        label="Reward MA (50 ep)"
    )
    ax_reward.set_ylabel("Reward")
    ax_reward.set_title("Training Reward by Episode")
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend()

    if ax_loss is not None:
        ax_loss.plot(
            episodes,
            losses,
            color="darkorange",
            alpha=0.35,
            linewidth=0.8,
            label="Episode loss"
        )
        ax_loss.plot(
            episodes,
            loss_ma_50,
            color="orangered",
            linewidth=1.6,
            label="Loss MA (50 ep)"
        )
        ax_loss.set_xlabel("Episode")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Loss by Episode")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend()
    else:
        ax_reward.set_xlabel("Episode")
        ax_reward.set_title("Training Reward by Episode (loss unavailable in CSV)")

    figure.tight_layout()

    output_suffix = "reward_loss" if has_loss else "reward"
    output_file = output_path / f"{metrics_path.stem}_{output_suffix}.png"
    figure.savefig(output_file, dpi=160)
    plt.close(figure)
    return output_file
