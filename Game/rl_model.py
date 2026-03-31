# =============================================================================
# rl_model.py
# =============================================================================
# Neural network policy and observation normalization for Geometry Dash RL agent.
#
# This module provides:
#   - Neural network architecture (simple feedforward MLP)
#   - Observation space specification and constants
#   - Policy wrapper for action prediction
#
# ── OBSERVATION SPACE ─────────────────────────────────────────────────────
#
#   The normalized observation vector has 28 floats:
#
#   [0:3]                Player state (3 values)
#       0: player_y         — vertical position normalized to [0, 1]
#       1: player_vy        — vertical velocity normalized to [-1, 1]
#       2: on_ground        — binary: 0.0 or 1.0
#
#   [3:27]               Upcoming obstacles (3 obstacles × 8 features each)
#       Per obstacle i:
#         0: type           — 0.0 (spike) or 1.0 (block)
#         1: rel_x          — horizontal distance to player, [0, 1]
#         2: rel_y          — vertical offset, [-1, 1]
#         3: width          — normalized obstacle width
#         4: height         — normalized obstacle height
#         5: time_to_reach  — seconds until obstacle reaches player, [0, 1]
#         6: gap_top        — vertical clearance above, [0, 1]
#         7: gap_bottom     — vertical clearance below, [0, 1]
#
#   [27]                 Derived feature (1 value)
#       27: is_jump_possible  — duplicate of on_ground (helps agent learn)
#
# ── NETWORK ARCHITECTURE ──────────────────────────────────────────────────
#
#   Input layer:  28 floats (normalized observation)
#   Hidden layer: 128 neurons (ReLU activation)
#   Hidden layer: 64 neurons (ReLU activation)
#   Output layer: 2 neurons (jump/no-jump logits, softmax for probability)
#
#   Total parameters: ~13k (small, trains fast)
#
# ── USAGE ─────────────────────────────────────────────────────────────────
#
#   from game import Game
#   from rl_model import SimplePolicyNetwork
#   import torch
#
#   game = Game(render=False)
#   policy = SimplePolicyNetwork(device="cpu")
#   obs = game.reset()
#
#   # Get normalized observation from game
#   obs_norm = game.get_normalized_observation()
#   obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
#
#   # Predict action
#   with torch.no_grad():
#       action_logits = policy(obs_tensor)
#       action = torch.argmax(action_logits, dim=1).item()
#
#   obs, reward, done = game.step(action)
#
# =============================================================================

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import Game.constants as C


# =============================================================================
# Observation Specification
# =============================================================================

class ObservationSpec:
    """
    Constants and utilities for normalized observation vectors.
    
    Defines min/max ranges for normalization and provides helper methods
    to convert between raw and normalized values.
    """

    # Observation vector size
    OBSERVATION_SIZE = 28
    
    # Player state normalization ranges
    PLAYER_Y_MAX = float(C.SCREEN_H)           # Screen height in pixels
    PLAYER_VY_MAX = float(C.MAX_FALL_SPEED)    # Max vertical velocity amplitude (px/s)
    
    # Obstacle state ranges (game-specific)
    SCREEN_W_MAX = float(C.SCREEN_W)           # Screen width for relative distance normalization
    GROUND_Y_MAX = float(C.SCREEN_H)           # Screen height for gap calculations
    BLOCK_SIZE = float(C.BLOCK_SIZE)           # One "block" in game units (pixels)
    GAME_SPEED = float(C.GAME_SPEED)           # Game scroll speed (px/s)
    
    # Time-to-reach calculation (how far ahead to look)
    TIME_HORIZONT_MAX = 6.0        # seconds — assume max obstacle distance
    
    # Obstacle dimensions
    MAX_OBSTACLE_WIDTH_BLOCKS = 5.0
    MAX_OBSTACLE_HEIGHT_BLOCKS = 5.0

    @staticmethod
    def normalize(raw_value: float, min_val: float, max_val: float) -> float:
        """Linearly normalize a raw value to [-1, 1] or [0, 1]."""
        if max_val == min_val:
            return 0.0
        normalized = (raw_value - min_val) / (max_val - min_val)
        # Clamp to valid range
        return max(-1.0, min(1.0, normalized))

    @staticmethod
    def denormalize(norm_value: float, min_val: float, max_val: float) -> float:
        """Reverse normalization from [-1, 1]/[0, 1] back to original range."""
        return norm_value * (max_val - min_val) + min_val


# =============================================================================
# Neural Network Policy
# =============================================================================

class SimplePolicyNetwork(nn.Module):
    r"""
    Simple feedforward policy network for binary action prediction (jump / no-jump).
    
    Architecture:
        Input(28) → Dense(128, ReLU) → Dense(64, ReLU) → Output(2, Softmax)
    
    This is a minimal network designed for:
    - Fast training on standard RL algorithms (PPO, DQN)
    - Interpretability (small enough to understand decision patterns)
    - Real-time inference (sub-millisecond prediction latency)
    
    The network learns to map normalized observations to jump probabilities.
    
    Parameters
    ----------
    input_size : int
        Size of the normalized observation vector (default: 28).
    hidden_size : int
        Number of neurons in hidden layers (default: 128).
    device : str
        PyTorch device ("cpu" or "cuda").
    
    Examples
    --------
    >>> import torch
    >>> policy = SimplePolicyNetwork(device="cpu")
    >>> obs = torch.randn(1, 28)
    >>> logits = policy(obs)  # Shape: (1, 2)
    >>> action_probs = torch.softmax(logits, dim=1)
    >>> action = torch.argmax(action_probs, dim=1).item()
    """

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 2)  # 2 action classes: jump / no-jump

        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Normalized observation batch, shape (batch_size, input_size).
        
        Returns
        -------
        torch.Tensor
            Action logits, shape (batch_size, 2).
            Higher logit at index 1 means "jump" is more likely.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def predict(self, obs: list[float] | torch.Tensor) -> int:
        """
        Predict action from a single observation.
        
        Parameters
        ----------
        obs : list[float] or torch.Tensor
            Normalized observation vector of size 28.
        
        Returns
        -------
        int
            Action: 0 (no-jump) or 1 (jump).
        """
        if isinstance(obs, list):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        elif not obs.is_cuda and self.device == "cuda":
            obs = obs.to(self.device)

        # Ensure batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(obs)
            action = torch.argmax(logits, dim=1).item()

        return action

    def predict_with_confidence(self, obs: list[float] | torch.Tensor) -> tuple[int, float]:
        """
        Predict action and return confidence score.
        
        Parameters
        ----------
        obs : list[float] or torch.Tensor
            Normalized observation vector.
        
        Returns
        -------
        tuple[int, float]
            (action, confidence_probability) where confidence is the max softmax value.
        """
        if isinstance(obs, list):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        elif not obs.is_cuda and self.device == "cuda":
            obs = obs.to(self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(obs)
            probs = torch.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()
            confidence = probs[0, action].item()

        return action, confidence

    def save(self, path: str) -> None:
        """Save network weights to file."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load network weights from file."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def to_device(self, device: str) -> None:
        """Move network to a different device."""
        self.device = device
        self.to(device)

    @property
    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test: verify network shapes and parameter count
    policy = SimplePolicyNetwork(device="cpu")
    print(f"Network architecture:\n{policy}")
    print(f"Total parameters: {policy.parameter_count}")

    # Test forward pass
    dummy_obs = torch.randn(8, 45)  # Batch of 8 observations
    output = policy(dummy_obs)
    print(f"\nInput shape:  {dummy_obs.shape}")
    print(f"Output shape: {output.shape}")

    # Test prediction on single observation
    action, conf = policy.predict_with_confidence(dummy_obs[0].tolist())
    print(f"\nSingle prediction: action={action}, confidence={conf:.3f}")
