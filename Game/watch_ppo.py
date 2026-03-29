"""
=============================================================================
watch_ppo.py
=============================================================================
Visual runner for PPO checkpoints saved by Stable-Baselines3.

Use this to WATCH a PPO model play in the normal game window.
This is separate from game.py --agent because game.py expects a .pth policy
from SimplePolicyNetwork, while PPO training saves .zip actor-critic models.

Example:
    python watch_ppo.py --model logs_ppo/checkpoints/ppo_policy_final.zip --difficulty 1 --seed 42
=============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame
import torch
from stable_baselines3 import PPO

from game import Game
from level_generator import LevelGenerator


def _adapt_obs_for_model(model: PPO, obs_norm: np.ndarray) -> np.ndarray:
    """Pad/truncate observation so it matches the loaded PPO model input size."""
    expected_shape = getattr(model.observation_space, "shape", None)
    expected_dim = int(expected_shape[0]) if expected_shape else obs_norm.shape[0]
    current_dim = int(obs_norm.shape[0])

    if current_dim == expected_dim:
        return obs_norm
    if current_dim < expected_dim:
        pad = np.zeros((expected_dim - current_dim,), dtype=np.float32)
        return np.concatenate([obs_norm.astype(np.float32, copy=False), pad], axis=0)
    return obs_norm[:expected_dim].astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch PPO model play Geometry Dash")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO .zip model")
    parser.add_argument("--difficulty", type=int, default=1, help="Level difficulty (1-5)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible level")
    parser.add_argument("--length", type=int, default=9000, help="Level length in pixels")
    parser.add_argument("--staircase-only", action="store_true", help="Use staircase-only generated test level")
    parser.add_argument("--triple-only", action="store_true", help="Use triple-spike-only generated test level")
    parser.add_argument("--progressive", action="store_true", help="Enable progressive curriculum (ramps up difficulty)")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Hold each predicted action for N frames (default: 1, matches PPO training env)",
    )
    parser.add_argument(
        "--show-probs",
        action="store_true",
        help="Print PPO jump/no-jump probabilities to console during play",
    )
    parser.add_argument(
        "--probs-interval",
        type=int,
        default=15,
        help="Console probability print interval in decision steps (default: 15)",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--telemetry", action="store_true", help="Enable in-game telemetry panel at start")
    return parser.parse_args()


def _get_policy_action_and_probs(model: PPO, obs_norm: np.ndarray) -> tuple[int, np.ndarray]:
    """Return deterministic action and class probabilities [no_jump, jump]."""
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs_norm)
        distribution = model.policy.get_distribution(obs_tensor)

        # Discrete action space -> categorical probabilities.
        probs_tensor = distribution.distribution.probs
        probs = probs_tensor.detach().cpu().numpy()[0]

    action = int(np.argmax(probs))
    return action, probs


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(str(model_path), device=args.device)

    level_gen = LevelGenerator(
        difficulty=args.difficulty,
        seed=args.seed,
        progressive=args.progressive,
    )
    if args.triple_only:
        level_obstacles = level_gen.generate_triple_only(length=args.length)
    elif args.staircase_only:
        level_obstacles = level_gen.generate_staircase_only(length=args.length)
    else:
        level_obstacles = level_gen.generate(length=args.length)

    game = Game(render=True, seed=args.seed, debug=False, agent_policy=None)
    game.load_level(level_obstacles)

    if args.telemetry:
        game.toggle_telemetry()

    attempts = 0
    best_px = 0
    frame_index = 0
    decision_index = 0
    current_action = 0
    current_probs = np.asarray([0.5, 0.5], dtype=np.float32)
    total_reward = 0.0

    print("=" * 70)
    print("  GEOMETRY DASH RL — PPO Watch Mode")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  Difficulty: {args.difficulty} | Seed: {args.seed} | Obstacles: {len(level_obstacles)}")
    print(f"  Action repeat: {args.action_repeat} frame(s)")
    print("  R = restart | H = debug hitboxes | T = telemetry | Q/ESC = quit")
    if args.show_probs:
        print(f"  Console probs: enabled (interval={max(1, args.probs_interval)} decisions)")
    print("=" * 70)

    # Reuse existing game telemetry panel fields for live AI confidence display.
    game._agent_enabled = True

    running = True
    while running:
        dt = game.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.load_level(level_obstacles)
                    attempts += 1
                    frame_index = 0
                    decision_index = 0
                    current_action = 0
                    current_probs = np.asarray([0.5, 0.5], dtype=np.float32)
                elif event.key == pygame.K_h:
                    game.toggle_debug()
                elif event.key == pygame.K_t:
                    game.toggle_telemetry()
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # Match training behavior: sample a new action every N frames and hold
        # that action between decisions.
        if frame_index % max(1, args.action_repeat) == 0:
            obs_norm = np.asarray(game.get_normalized_observation(), dtype=np.float32)
            obs_norm = _adapt_obs_for_model(model, obs_norm)
            current_action, current_probs = _get_policy_action_and_probs(model, obs_norm)
            decision_index += 1

            # Update in-game telemetry values shown in Game._draw_telemetry_panel()
            game._agent_action_probs = [float(current_probs[0]), float(current_probs[1])]
            game._agent_predicted_action = int(current_action)
            game._agent_confidence = float(current_probs[current_action])

            if args.show_probs and (decision_index % max(1, args.probs_interval) == 0):
                print(
                    f"Decision {decision_index:5d} | "
                    f"action={'JUMP' if current_action == 1 else 'WAIT':>4} | "
                    f"p(wait)={current_probs[0]:.3f} p(jump)={current_probs[1]:.3f}"
                )

        # Keep physics integration closer to training by using default fixed dt.
        obs, reward, done = game.step(current_action)
        total_reward += reward
        game.render()
        frame_index += 1

        best_px = max(best_px, int(game._scroll_x))

        if done:
            attempts += 1
            print(f"  Attempt {attempts:>3}  |  dist = {int(game._scroll_x):>6} px  |  best = {best_px:>6} px  |  total_reward = {total_reward:.2f}")
            pygame.time.wait(300)
            game.load_level(level_obstacles)
            total_reward = 0.0
            frame_index = 0
            decision_index = 0
            current_action = 0
            current_probs = np.asarray([0.5, 0.5], dtype=np.float32)

    game.close()
    print(f"\nSession over. {attempts} attempts. Best distance: {best_px} px.")


if __name__ == "__main__":
    main()
