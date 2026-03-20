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

from stable_baselines3 import PPO

from gym_env import GeometryDashGymEnv, GdEnvConfig


def evaluate(
    model_path: str,
    difficulties: list[int],
    seed_start: int,
    num_seeds: int,
    level_length: int,
    action_repeat: int,
    max_steps: int,
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

    for difficulty in difficulties:
        rewards = []
        lengths = []

        for seed in range(seed_start, seed_start + num_seeds):
            env = GeometryDashGymEnv(
                GdEnvConfig(
                    difficulty=difficulty,
                    level_length=level_length,
                    seed=seed,
                    randomize_level_each_episode=False,
                    progressive=False,
                    action_repeat=action_repeat,
                    max_steps_per_episode=max_steps,
                    render=False,
                )
            )

            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)

            rewards.append(episode_reward)
            lengths.append(steps)
            env.close()

        diff_mean_reward = mean(rewards)
        diff_mean_steps = mean(lengths)
        all_rewards.extend(rewards)

        print(
            f"Difficulty {difficulty}: "
            f"mean_reward={diff_mean_reward:8.3f} | "
            f"best_reward={max(rewards):8.3f} | "
            f"mean_steps={diff_mean_steps:8.2f}"
        )

    print("-" * 80)
    print(f"Overall mean reward: {mean(all_rewards):.3f}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint on seed/difficulty grid")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO .zip model")
    parser.add_argument("--difficulties", type=int, nargs="+", default=[1, 2, 3], help="Difficulty list")
    parser.add_argument("--seed-start", type=int, default=1000, help="First evaluation seed")
    parser.add_argument("--num-seeds", type=int, default=30, help="Number of seeds per difficulty")
    parser.add_argument("--level-length", type=int, default=6000, help="Level length in pixels")
    parser.add_argument("--action-repeat", type=int, default=4, help="Action repeat used in env")
    parser.add_argument("--max-steps", type=int, default=2500, help="Max env steps per episode")
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
        device=args.device,
    )


if __name__ == "__main__":
    main()
