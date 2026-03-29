"""
=============================================================================
train_ppo.py
=============================================================================
Production PPO training entrypoint for Geometry Dash RL.

This script uses:
  - Gymnasium wrapper (gym_env.py)
  - Stable-Baselines3 PPO
  - Vectorized environments (parallel rollout workers)
  - Periodic checkpointing
  - CSV metrics logging
  - Optional training figure generation

Why this script exists
----------------------
Your existing train.py (REINFORCE) is excellent for educational clarity but has
higher variance and slower convergence. PPO is the standard practical upgrade:

  - clipped policy updates -> more stable learning
  - GAE -> lower-variance advantages
  - vectorized rollouts -> better throughput

This file is intentionally verbose and heavily commented so it doubles as
reference documentation for your future experiments.
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from gym_env import GeometryDashGymEnv, GdEnvConfig
from training_plots import generate_training_plots

import threading
import time
class VisualEvalCallback(BaseCallback):
    """
    Periodically runs the current policy visually on the current training seed/level.
    """
    def __init__(self, eval_env_config: GdEnvConfig, eval_interval: int = 50000, max_steps: int = 1000):
        super().__init__()
        self.eval_env_config = eval_env_config
        self.eval_interval = eval_interval
        self.max_steps = max_steps
        self.last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_interval:
            self.last_eval = self.num_timesteps
            # threading.Thread(target=self.run_visual_eval, daemon=True).start()
            self.run_visual_eval()  # <-- Run directly on main thread
        return True

    def run_visual_eval(self):
        import pygame
        from game import Game
        from stable_baselines3 import PPO
        # Use the current seed and config
        env_cfg = self.eval_env_config
        # Use the current model weights
        model = self.model
        # Build the level
        from level_generator import LevelGenerator
        level_gen = LevelGenerator(
            difficulty=env_cfg.difficulty,
            seed=env_cfg.seed,
            progressive=env_cfg.progressive,
        )
        if env_cfg.triple_only:
            level_obstacles = level_gen.generate_triple_only(length=env_cfg.level_length)
            print(f"[DEBUG][VisualEval] triple_only={env_cfg.triple_only} | First 5 obstacles: {[o['type'] for o in level_obstacles[:5]]}")
        elif env_cfg.staircase_only:
            level_obstacles = level_gen.generate_staircase_only(length=env_cfg.level_length)
            print(f"[DEBUG][VisualEval] staircase_only={env_cfg.staircase_only} | First 5 obstacles: {[o['type'] for o in level_obstacles[:5]]}")
        else:
            level_obstacles = level_gen.generate(length=env_cfg.level_length)
            print(f"[DEBUG][VisualEval] default level | First 5 obstacles: {[o['type'] for o in level_obstacles[:5]]}")
        game = Game(render=True, seed=env_cfg.seed, debug=False, agent_policy=None)
        game.load_level(level_obstacles)
        obs = game.get_normalized_observation()
        done = False
        steps = 0
        while not done and steps < self.max_steps:
            # If obs is a dict, flatten to array (should not happen with get_normalized_observation, but safe)
            if isinstance(obs, dict):
                # Try to use get_normalized_observation if available
                if hasattr(game, 'get_normalized_observation'):
                    obs_input = np.asarray(game.get_normalized_observation(), dtype=np.float32)
                else:
                    # Fallback: flatten dict values (not expected)
                    obs_input = np.asarray(list(obs.values()), dtype=np.float32)
            else:
                obs_input = np.asarray(obs, dtype=np.float32)
            action, _ = model.predict(obs_input, deterministic=True)
            obs, reward, done = game.step(action)
            # After step, if obs is dict, get normalized again
            if isinstance(obs, dict) and hasattr(game, 'get_normalized_observation'):
                obs = game.get_normalized_observation()
            game.render()
            steps += 1
            time.sleep(1.0 / 60.0)
        game.close()
# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class PpoTrainConfig:
    # Training budget
    total_timesteps: int = 2_000_000

    # Environment generation
    difficulty: int = 3
    level_length: int = 9000
    seed: int = 42
    randomize_level_each_episode: bool = True
    progressive: bool = True
    staircase_only: bool = False
    triple_only: bool = False

    # Runtime controls
    action_repeat: int = 1
    max_steps_per_episode: int = 2500
    num_envs: int = 8

    # Optional reward shaping and feature control.
    alive_reward: float = 0.001  # Slight survival bonus per RL step
    jump_action_penalty: float = 0.002
    air_jump_penalty: float = 0.01
    unnecessary_jump_penalty: float = 0.03
    jump_danger_distance_px: float = 250.0
    # dangerous_jump_penalty removed

    # PPO hyperparameters (solid starting defaults)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 10
    target_kl: float = 0.02

    # Network architecture
    net_arch_pi: tuple[int, ...] = (256, 128)
    net_arch_vf: tuple[int, ...] = (256, 128)

    # Device and outputs
    device: str = "auto"
    log_dir: str = "logs_ppo"
    checkpoint_interval_steps: int = 100_000
    plot_after_training: bool = True
    figure_dir: str = "training_figures"


# -----------------------------------------------------------------------------
# Logging callback
# -----------------------------------------------------------------------------

class CsvEpisodeLoggerCallback(BaseCallback):
    """
    Write episode metrics to CSV while PPO trains.

    SB3 stores episode summaries in self.model.ep_info_buffer (rolling buffer
    from Monitor/VecMonitor), containing fields such as:
      - r: episode reward
      - l: episode length
      - t: wall-clock seconds

    This callback snapshots the latest episode summary whenever a new one appears.
    """

    def __init__(self, csv_file: Path):
        super().__init__()
        self.csv_file = csv_file
        self._episode_counter = 0

    def _on_training_start(self) -> None:
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "episode",
                "episode_reward",
                "episode_steps",
                "timesteps",
                "fps",
                "timestamp",
                "seed",
            ])

    def _on_step(self) -> bool:
        # VecMonitor injects an "episode" dict into info when an episode ends.
        # This is reliable even for long runs where ep_info_buffer length no
        # longer increases.
        infos = self.locals.get("infos", [])
        if infos is None:
            infos = []

        rows = []
        for info in infos:
            if not isinstance(info, dict):
                continue

            # Handle multiple possible info layouts used by VecEnv/Gymnasium.
            candidate_episode_infos = []
            episode_seed = None

            episode_info = info.get("episode")
            if isinstance(episode_info, dict):
                candidate_episode_infos.append(episode_info)
            # Try to get the seed from info (Gymnasium envs pass it in info)
            if "seed" in info:
                episode_seed = info["seed"]

            final_info = info.get("final_info")
            if isinstance(final_info, dict):
                nested_episode = final_info.get("episode")
                if isinstance(nested_episode, dict):
                    candidate_episode_infos.append(nested_episode)
                if "seed" in final_info:
                    episode_seed = final_info["seed"]
            elif isinstance(final_info, (list, tuple)):
                for nested in final_info:
                    if isinstance(nested, dict):
                        nested_episode = nested.get("episode")
                        if isinstance(nested_episode, dict):
                            candidate_episode_infos.append(nested_episode)
                        if "seed" in nested:
                            episode_seed = nested["seed"]

            for episode_info in candidate_episode_infos:
                self._episode_counter += 1
                # Try to get the seed from episode_info if not found yet
                seed_val = episode_seed
                if seed_val is None and "seed" in episode_info:
                    seed_val = episode_info["seed"]
                rows.append([
                    self._episode_counter,
                    f"{float(episode_info.get('r', np.nan)):.6f}",
                    int(episode_info.get("l", 0)),
                    self.num_timesteps,
                    int(self.model.logger.name_to_value.get("time/fps", 0)),
                    datetime.now().isoformat(),
                    seed_val if seed_val is not None else "",
                ])

        if rows:
            with open(self.csv_file, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(rows)

        return True


# -----------------------------------------------------------------------------
# Env factory helpers
# -----------------------------------------------------------------------------

def _make_env(config: PpoTrainConfig, rank: int):
    """Factory for creating one monitored environment instance."""

    def _init():
        env_cfg = GdEnvConfig(
            difficulty=config.difficulty,
            level_length=config.level_length,
            seed=config.seed + rank * 10_000,
            randomize_level_each_episode=config.randomize_level_each_episode,
            progressive=config.progressive,
            staircase_only=config.staircase_only,
            triple_only=config.triple_only,
            action_repeat=config.action_repeat,
            max_steps_per_episode=config.max_steps_per_episode,
            jump_action_penalty=config.jump_action_penalty,
            air_jump_penalty=config.air_jump_penalty,
            unnecessary_jump_penalty=config.unnecessary_jump_penalty,
            jump_danger_distance_px=config.jump_danger_distance_px,
            render=False,
        )
        print(f"[DEBUG][_make_env] rank={rank} triple_only={env_cfg.triple_only}")
        return GeometryDashGymEnv(config=env_cfg)

    return _init


def build_vec_env(config: PpoTrainConfig):
    """Create vectorized env (subprocess when num_envs>1, else dummy)."""
    env_fns = [_make_env(config, i) for i in range(config.num_envs)]
    if config.num_envs > 1:
        vec = SubprocVecEnv(env_fns)
    else:
        vec = DummyVecEnv(env_fns)
    return VecMonitor(vec)


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------

def train_ppo(config: PpoTrainConfig) -> tuple[Path, Path, Optional[Path]]:
    """
    Train PPO and return key output paths:
      (final_model_zip, metrics_csv, figure_path_or_none)
    """
    log_path = Path(config.log_dir)
    ckpt_dir = log_path / "checkpoints"
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv = log_path / f"training_metrics_ppo_{timestamp}.csv"

    vec_env = build_vec_env(config)

    policy_kwargs = {
        "net_arch": {
            "pi": list(config.net_arch_pi),
            "vf": list(config.net_arch_vf),
        }
    }

    if hasattr(config, 'load_model') and config.load_model:
        print(f"\n[INFO] Continuing training from existing model: {config.load_model}")
        model = PPO.load(
            config.load_model, 
            env=vec_env, 
            device=config.device,
            # We explicitly override the learning rate in case you want to lower it in later phases
            custom_objects={"learning_rate": config.learning_rate} 
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config.device,
        )

    csv_logger = CsvEpisodeLoggerCallback(metrics_csv)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, config.checkpoint_interval_steps // max(1, config.num_envs)),
        save_path=str(ckpt_dir),
        name_prefix="ppo_policy",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    # Visual evaluation callback: use the same config as training, but only one env

    print("=" * 80)
    print("  GEOMETRY DASH RL — PPO Training (Stable-Baselines3)")
    print("=" * 80)
    print(f"  Total timesteps   : {config.total_timesteps}")
    print(f"  Num envs          : {config.num_envs}")
    print(f"  Difficulty        : {config.difficulty}")
    print(f"  Randomized levels : {config.randomize_level_each_episode}")
    print(f"  Progressive       : {config.progressive}")
    print(f"  Alive reward      : {config.alive_reward}")
    print(f"  Jump penalty      : {config.jump_action_penalty}")
    print(f"  Air-jump penalty  : {config.air_jump_penalty}")
    print(f"  Unnec. jump pen.  : {config.unnecessary_jump_penalty}")
    print(f"  Danger dist (px)  : {config.jump_danger_distance_px}")
    print(f"  Device            : {config.device}")
    print(f"  Logs              : {log_path}")
    # dangerous jump penalty print removed
    print("=" * 80)

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[csv_logger, checkpoint_cb],
        progress_bar=False,
    )

    final_model = ckpt_dir / "ppo_policy_final.zip"
    model.save(str(final_model))

    vec_env.close()

    figure_path = None
    if config.plot_after_training:
        figure_path = generate_training_plots(
            metrics_file=str(metrics_csv),
            output_dir=config.figure_dir,
        )

    print("\n" + "=" * 80)
    print("  PPO Training Complete")
    print("=" * 80)
    print(f"  Final model       : {final_model}")
    print(f"  Metrics CSV       : {metrics_csv}")
    if figure_path is not None:
        print(f"  Training figure   : {figure_path}")
    else:
        print("  Training figure   : skipped (missing matplotlib or empty metrics)")
    print("=" * 80)

    return final_model, metrics_csv, figure_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Geometry Dash agent with PPO (Stable-Baselines3)",
    )

    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total PPO training timesteps")
    parser.add_argument("--difficulty", type=int, default=3, help="Level difficulty (1-5)")
    parser.add_argument("--level-length", type=int, default=9000, help="Level length in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")

    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--action-repeat", type=int, default=1, help="Action repeat per step")
    parser.add_argument("--max-steps", type=int, default=2500, help="Max environment steps per episode")
    parser.add_argument(
        "--alive-reward",
        type=float,
        default=0.001,
        help="Bonus per RL step for staying alive (survival encouragement)",
    )
    parser.add_argument(
        "--jump-penalty",
        type=float,
        default=0.002,
        help="Penalty applied when action=jump each RL step (anti-spam shaping)",
    )
    parser.add_argument(
        "--air-jump-penalty",
        type=float,
        default=0.01,
        help="Extra penalty when jump is selected while airborne",
    )
    parser.add_argument(
        "--unnecessary-jump-penalty",
        type=float,
        default=0.03,
        help="Extra penalty for on-ground jumps when nearest obstacle is not imminent",
    )
    parser.add_argument(
        "--jump-danger-distance",
        type=float,
        default=250.0,
        help="Obstacle distance threshold (px) within which a jump is considered necessary",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping")
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout steps per env before update")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Optimization epochs per update")
    parser.add_argument("--target-kl", type=float, default=0.02, help="Optional KL early stopping target")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    parser.add_argument("--log-dir", type=str, default="logs_ppo", help="Output directory")
    parser.add_argument("--checkpoint-steps", type=int, default=100_000, help="Save interval in timesteps")
    parser.add_argument("--figure-dir", type=str, default="training_figures", help="Directory for generated figures")

    parser.add_argument("--fixed-level", action="store_true", help="Disable per-episode level randomization")
    parser.add_argument("--progressive", action="store_true", default=True, help="Enable progressive LevelGenerator curriculum")
    parser.add_argument("--staircase-only", action="store_true", help="Train on staircase patterns only (no spikes/clusters)")
    parser.add_argument("--triple-only", action="store_true", help="Train on triple spikes only (no other obstacles)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation after training")
    parser.add_argument("--load-model", type=str, default=None, help="Path to an existing .zip model to continue training")

    # --dangerous-jump-penalty argument removed

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PpoTrainConfig(
        total_timesteps=args.timesteps,
        difficulty=args.difficulty,
        level_length=args.level_length,
        seed=args.seed,
        randomize_level_each_episode=not args.fixed_level,
        progressive=args.progressive,
        staircase_only=args.staircase_only,
        triple_only=args.triple_only,
        action_repeat=args.action_repeat,
        max_steps_per_episode=args.max_steps,
        num_envs=max(1, args.num_envs),
        alive_reward=max(0.0, args.alive_reward),
        jump_action_penalty=max(0.0, args.jump_penalty),
        air_jump_penalty=max(0.0, args.air_jump_penalty),
        unnecessary_jump_penalty=max(0.0, args.unnecessary_jump_penalty),
        jump_danger_distance_px=max(1.0, args.jump_danger_distance),
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        target_kl=args.target_kl,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_interval_steps=max(1, args.checkpoint_steps),
        plot_after_training=not args.no_plot,
        figure_dir=args.figure_dir,
    )
    cfg.load_model = args.load_model

    train_ppo(cfg)


if __name__ == "__main__":
    main()
