"""
=============================================================================
evaluate_ppo.py
=============================================================================
Evaluate a trained PPO model across multiple seeds/difficulties.

Purpose
-------
Training reward alone can be misleading. This script answers:

  "Does the policy generalize beyond the exact training layout?"

It loads an SB3 PPO .zip model and runs deterministic episodes over a test grid
of seeds and difficulties, then prints aggregate statistics.
=============================================================================
"""

from __future__ import annotations

import argparse
from statistics import mean

import numpy as np

from stable_baselines3 import PPO

from gym_env import GeometryDashGymEnv, GdEnvConfig


def _adapt_obs_for_model(model: PPO, obs: np.ndarray) -> np.ndarray:
    """Pad/truncate observation to match loaded PPO model observation dimension."""
    expected_shape = getattr(model.observation_space, "shape", None)
    expected_dim = int(expected_shape[0]) if expected_shape else obs.shape[0]
    current_dim = int(obs.shape[0])

    if current_dim == expected_dim:
        return obs
    if current_dim < expected_dim:
        pad = np.zeros((expected_dim - current_dim,), dtype=np.float32)
        return np.concatenate([obs.astype(np.float32, copy=False), pad], axis=0)
    return obs[:expected_dim].astype(np.float32, copy=False)


def evaluate(
    model_path: str,
    difficulties: list[int],
    seed_start: int,
    num_seeds: int,
    level_length: int,
    action_repeat: int,
    max_steps: int,
    staircase_only: bool,
    device: str,
) -> None:
    model = PPO.load(model_path, device=device)

    print("=" * 80)
    print("  PPO Evaluation")
    print("=" * 80)
    print(f"  Model            : {model_path}")
    print(f"  Difficulties     : {difficulties}")
    print(f"  Seed range       : [{seed_start}, {seed_start + num_seeds - 1}]")
    print("=" * 80)

    all_rewards = []
    all_completions = []

    for difficulty in difficulties:
        rewards = []
        lengths = []
        completions = []

        for seed in range(seed_start, seed_start + num_seeds):
            env = GeometryDashGymEnv(
                GdEnvConfig(
                    difficulty=difficulty,
                    level_length=level_length,
                    seed=seed,
                    randomize_level_each_episode=False,
                    progressive=False,
                    staircase_only=staircase_only,
                    action_repeat=action_repeat,
                    max_steps_per_episode=max_steps,
                    render=False,
                )
            )

            obs, _ = env.reset()
            obs = _adapt_obs_for_model(model, np.asarray(obs, dtype=np.float32))
            done = False
            episode_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                obs = _adapt_obs_for_model(model, np.asarray(obs, dtype=np.float32))
                episode_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)

            rewards.append(episode_reward)
            lengths.append(steps)
            completions.append(1 if (truncated and not terminated) else 0)
            env.close()

        diff_mean_reward = mean(rewards)
        diff_mean_steps = mean(lengths)
        diff_completion_rate = 100.0 * (sum(completions) / len(completions))
        all_rewards.extend(rewards)
        all_completions.extend(completions)

        print(
            f"Difficulty {difficulty}: "
            f"mean_reward={diff_mean_reward:8.3f} | "
            f"best_reward={max(rewards):8.3f} | "
            f"mean_steps={diff_mean_steps:8.2f} | "
            f"completion_rate={diff_completion_rate:6.2f}%"
        )

    print("-" * 80)
    print(f"Overall mean reward: {mean(all_rewards):.3f}")
    print(f"Overall completion rate: {100.0 * (sum(all_completions) / len(all_completions)):.2f}%")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint on seed/difficulty grid")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO .zip model")
    parser.add_argument("--difficulties", type=int, nargs="+", default=[1, 2, 3], help="Difficulty list")
    parser.add_argument("--seed-start", type=int, default=1000, help="First evaluation seed")
    parser.add_argument("--num-seeds", type=int, default=30, help="Number of seeds per difficulty")
    parser.add_argument("--level-length", type=int, default=9000, help="Level length in pixels")
    parser.add_argument("--action-repeat", type=int, default=1, help="Action repeat used in env")
    parser.add_argument("--max-steps", type=int, default=2500, help="Max env steps per episode")
    parser.add_argument("--staircase-only", action="store_true", help="Evaluate on staircase-only generated levels")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        model_path=args.model,
        difficulties=args.difficulties,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
        level_length=args.level_length,
        action_repeat=args.action_repeat,
        max_steps=args.max_steps,
        staircase_only=args.staircase_only,
        device=args.device,
    )


if __name__ == "__main__":
    main()
