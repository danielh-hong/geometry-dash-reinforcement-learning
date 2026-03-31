"""
=============================================================================
gym_env.py
=============================================================================
Gymnasium wrapper for the Geometry Dash game engine.

Why this file exists
--------------------
Stable-Baselines3 (SB3) algorithms, including PPO, expect environments that
follow the Gymnasium API contract:

    - env.reset() -> (observation, info)
    - env.step(action) -> (observation, reward, terminated, truncated, info)

Your core simulator already exists in game.py and is great for gameplay and
custom loops, but PPO needs this exact interface so it can:

    - Collect rollouts consistently
    - Compute advantage estimates (GAE)
    - Handle episode termination vs truncation correctly
    - Vectorize environments (parallel workers)

What Gym/Gymnasium actually is (plain English)
----------------------------------------------
Gymnasium is a standard interface for reinforcement learning environments.
Think of it as a "USB standard" for RL: once a game follows this interface,
any compatible RL library (SB3, RLlib, CleanRL, etc.) can train on it.

It does NOT replace your game logic.
It only wraps your game so RL code can plug in cleanly.

Design choices in this wrapper
------------------------------
1) Observation space is a fixed float vector of size 29, matching
   Game.get_normalized_observation().
2) Action space is Discrete(2):
      0 = no-op, 1 = jump
3) Action repeat is configurable (default 4), matching your existing trainer.
4) Episode truncation is enforced via max_steps_per_episode to prevent
   unbounded rollouts.
5) Level generation can be:
      - fixed seed (reproducible)
      - randomized every episode (generalization)
      - optional progressive curriculum via LevelGenerator

This wrapper is intentionally thin and documented heavily so future tweaks
are safe and obvious.
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game import Game
from level_generator import LevelGenerator


@dataclass
class GdEnvConfig:
    """Configuration for GeometryDashGymEnv."""

    difficulty: int = 1
    level_length: int = 9000
    seed: int = 42

    # If True, each episode uses a new seed (seed + episode_index), improving
    # generalization and reducing overfitting to one exact obstacle sequence.
    randomize_level_each_episode: bool = True

    # LevelGenerator progressive mode (curriculum inside each generated level).
    progressive: bool = False

    # Special training modes
    staircase_only: bool = False  # Only staircases, no spikes/clusters (for staircase mastery)
    triple_only: bool = False     # Only triple spikes, for focused triple-spike training

    # Environment runtime limits.
    action_repeat: int = 1
    max_steps_per_episode: int = 2500

    # Optional reward shaping to discourage jump-spam and encourage survival.
    # Applied once per RL decision step (not per repeated frame).
    alive_reward: float = 0.0  # Bonus per step for staying alive
    jump_action_penalty: float = 0.0
    air_jump_penalty: float = 0.0
    unnecessary_jump_penalty: float = 0.0
    jump_danger_distance_px: float = 50.0
    # dangerous_jump_penalty removed

    # Rendering is generally False for training speed, True for debugging.
    render: bool = False


class GeometryDashGymEnv(gym.Env):
    def _is_on_top_of_stairs_no_further_up(self) -> bool:
        """Detect if player is on the highest stair step (no further steps above)."""
        player = self._game.player
        player_x = player.x
        player_w = getattr(player, 'hitbox', None).width if hasattr(player, 'hitbox') else 112  # Default to 1 block
        player_y = player.y
        player_bottom = player_y + player_w  # y increases downward
        print(f"[DEBUG][stairs] Checking: player_x={player_x}, player_y={player_y}, player_w={player_w}, player_bottom={player_bottom}")

        # Find all blocks that horizontally overlap with the player
        blocks_under = [
            obs for obs in self._game.obstacles
            if getattr(obs, 'kind', None) == 'block'
            and (player_x + player_w > obs.x)
            and (player_x < obs.x + obs.w)
        ]
        if not blocks_under:
            print(f"[DEBUG][stairs] No blocks under player.")
            return False

        # Find the block directly under the player's feet (closest to player_bottom, but not above it)
        blocks_below = [b for b in blocks_under if b._y >= player_bottom - 8]
        if not blocks_below:
            print(f"[DEBUG][stairs] No blocks directly under player's feet.")
            return False
        top_block = min(blocks_below, key=lambda b: b._y)
        y_diff = abs(player_bottom - top_block._y)
        print(f"[DEBUG][stairs] Top block y={top_block._y}, y_diff={y_diff}")
        if y_diff > 8:
            print(f"[DEBUG][stairs] Player not close enough to top block (y_diff={y_diff}).")
            return False

        # Check if there is any block above this one (same x overlap, smaller y)
        blocks_above = [
            obs for obs in self._game.obstacles
            if getattr(obs, 'kind', None) == 'block'
            and (player_x + player_w > obs.x)
            and (player_x < obs.x + obs.w)
            and obs._y < top_block._y - 2
        ]
        if blocks_above:
            print(f"[DEBUG][stairs] There are further stairs above (blocks_above count: {len(blocks_above)}).")
            return False
        print(f"[DEBUG][stairs] Player is on top of stairs with no further up.")
        return True

    # _is_on_top_of_stairs_and_spike_ahead removed
    """
    Gymnasium-compatible environment for Geometry Dash.

    Observation
    -----------
    np.ndarray shape=(28,), dtype=np.float32

    Action
    ------
    Discrete(2)
      0 = no-op
      1 = jump

    Reward
    ------
    Delegated to Game.step(): alive/clear/death rewards from constants.py

    Episode end semantics
    ---------------------
    terminated=True  -> player died (game over)
    truncated=True   -> step budget exhausted (time limit)
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config: Optional[GdEnvConfig] = None):
        super().__init__()
        self.config = config or GdEnvConfig()

        self.action_space = spaces.Discrete(2)
        # Normalized features are expected in roughly [-1, 1], but we use a
        # wider safe range to avoid accidental clipping by wrappers.
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(28,),
            dtype=np.float32,
        )

        self._episode_index = 0
        self._step_count = 0
        self._prev_action_int = 0

        self._game: Optional[Game] = None
        self._current_level = None

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _episode_seed(self) -> int:
        """Return deterministic seed for this episode."""
        if self.config.randomize_level_each_episode:
            return int(self.config.seed + self._episode_index)
        return int(self.config.seed)

    def _build_level_for_episode(self) -> list[dict[str, Any]]:
        """Generate a level layout for the current episode."""
        seed_for_episode = self._episode_seed()
        level_gen = LevelGenerator(
            difficulty=self.config.difficulty,
            seed=seed_for_episode,
            progressive=self.config.progressive,
        )
        if self.config.triple_only:
            obstacles = level_gen.generate_triple_only(length=self.config.level_length)
            print(f"[DEBUG] triple_only={self.config.triple_only} | First 5 obstacles: {[o['type'] for o in obstacles[:5]]}")
            return obstacles
        elif self.config.staircase_only:
            obstacles = level_gen.generate_staircase_only(length=self.config.level_length)
            print(f"[DEBUG] staircase_only={self.config.staircase_only} | First 5 obstacles: {[o['type'] for o in obstacles[:5]]}")
            return obstacles
        else:
            obstacles = level_gen.generate(length=self.config.level_length)
            print(f"[DEBUG] default level | First 5 obstacles: {[o['type'] for o in obstacles[:5]]}")
            return obstacles

    def _ensure_game_initialized(self) -> None:
        """Create Game instance lazily so reset() can be called repeatedly."""
        if self._game is None:
            self._game = Game(render=self.config.render)

    def _current_observation(self) -> np.ndarray:
        """Get normalized observation as np.float32 vector."""
        assert self._game is not None
        return np.asarray(self._game.get_normalized_observation(), dtype=np.float32)

    def _nearest_obstacle_distance_px(self) -> Optional[float]:
        """Return distance in px from player front to nearest upcoming obstacle."""
        assert self._game is not None

        player_front_x = float(self._game.player.x + self._game.player.hitbox.width)
        nearest_distance: Optional[float] = None

        for obstacle in self._game.obstacles:
            obstacle_front_x = float(obstacle.x)

            # Ignore obstacles if the player's front edge has already passed 
            # the obstacle's front edge (e.g., player is currently on top of a wide block)
            if obstacle_front_x <= player_front_x:
                continue

            distance = obstacle_front_x - player_front_x
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance

        return nearest_distance

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Start a new episode and return first observation.

        Gymnasium convention:
            reset returns (obs, info)
        """
        super().reset(seed=seed)

        # Optional runtime override from SB3/user.
        if seed is not None:
            self.config.seed = int(seed)

        self._ensure_game_initialized()
        self._step_count = 0
        self._prev_action_int = 0

        self._current_level = self._build_level_for_episode()
        self._game.load_level(self._current_level)

        obs = self._current_observation()
        info = {
            "episode_index": self._episode_index,
            "seed": self._episode_seed(),
            "difficulty": self.config.difficulty,
        }

        self._episode_index += 1
        return obs, info

    def step(self, action: int):
        """
        Apply action, advance simulation, and return Gymnasium 5-tuple.
        """
        print(f"[DEBUG][step] Called with action={action}")
        assert self._game is not None, "Environment must be reset() before step()."

        self._step_count += 1

        action_int = int(action)
        jump_pressed = (action_int == 1) and (self._prev_action_int != 1)
        was_on_ground = bool(self._game.player.on_ground)
        jump_executed = (action_int == 1) and was_on_ground
        nearest_obstacle_distance = self._nearest_obstacle_distance_px()

        # Action repeat to match your existing trainer behavior.
        accumulated_reward = 0.0
        terminated = False

        for _ in range(self.config.action_repeat):
            _, reward, done = self._game.step(action_int)
            accumulated_reward += float(reward)
            if done:
                terminated = True
                break

        # Optional reward shaping: survival bonus + anti-spam penalties.
        accumulated_reward += float(self.config.alive_reward)
        # Penalize each actual jump event (on-ground jump), even if action=1 is held.
        on_top_of_stairs = self._is_on_top_of_stairs_no_further_up()
        # on_top_of_stairs_and_spike logic removed
        if on_top_of_stairs:
            print(f"[DEBUG][step] Player is on top of stairs (no further up). jump_executed={jump_executed}, jump_pressed={jump_pressed}")
        if jump_executed:
            print(f"[DEBUG][step] Jump executed. Applying jump penalty: {self.config.jump_action_penalty}")
            accumulated_reward -= float(self.config.jump_action_penalty)
            # dangerous jump penalty logic removed
            if (
                nearest_obstacle_distance is None
                or nearest_obstacle_distance > float(self.config.jump_danger_distance_px)
            ):
                print(f"[DEBUG][step] Unnecessary jump penalty applied: {self.config.unnecessary_jump_penalty}")
                accumulated_reward -= float(self.config.unnecessary_jump_penalty)
        else:
            if jump_pressed:
                print(f"[DEBUG][step] Air jump penalty applied: {self.config.air_jump_penalty}")
                accumulated_reward -= float(self.config.air_jump_penalty)

        self._prev_action_int = action_int

        truncated = self._step_count >= self.config.max_steps_per_episode
        obs = self._current_observation()

        info = {
            "step_count": self._step_count,
            "seed": self._episode_seed(),
            "difficulty": self.config.difficulty,
        }

        return obs, accumulated_reward, terminated, truncated, info

    def render(self):
        """Render current frame when render=True is enabled in config."""
        if self._game is not None:
            self._game.render()

    def close(self):
        """Release pygame resources."""
        if self._game is not None:
            self._game.close()
            self._game = None
