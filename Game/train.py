# =============================================================================
# train.py
# =============================================================================
# Training Script for Geometry Dash RL Agent
#
# Algorithm: REINFORCE (Policy Gradient)
# Loss function: -E[log π(a|s) * G_t]
# where G_t = discounted return (sum of future rewards)
#
# This is the simplest policy gradient algorithm:
#   1. Collect full episodes
#   2. Compute returns backwards (G_t = r_t + γ*G_{t+1})
#   3. Normalize returns (reduces variance)
#   4. Compute loss = -mean(log_prob(action) * return)
#   5. Gradient step (Adam optimizer)
#
# Experience collection: Full trajectories (on-policy)
# Output: Logs and checkpoints saved to logs/ directory
#
# Usage:
#   python train.py                          # defaults (1000 episodes, diff 1)
#   python train.py --episodes 500 --difficulty 2 --lr 2e-3
#   python train.py --render                 # watch training progress
#   python train.py --device cuda            # use GPU if available
#
# =============================================================================

import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from typing import Tuple, List
import random

import constants as C
from game import Game
from level_generator import LevelGenerator
from rl_model import SimplePolicyNetwork


# =============================================================================
# Configuration
# =============================================================================

class TrainingConfig:
    """Centralized training hyperparameters."""
    
    num_episodes: int = 1000
    learning_rate: float = 1e-3
    discount_factor: float = 0.99  # γ (gamma) — discount future rewards
    device: str = "cpu"
    checkpoint_interval: int = 50  # Save weights every N episodes
    seed: int = 42
    difficulty: int = 1
    level_length: int = 6000  # pixels per level (~20 seconds at normal speed)
    render: bool = False  # Render game window during training
    log_dir: str = "logs"


# =============================================================================
# Utility Functions
# =============================================================================

def setup_logging(log_dir: str = "logs") -> Tuple[str, Path]:
    """
    Initialize logging directory structure and CSV metrics file.
    
    Creates:
        logs/
        ├── training_metrics_YYYYMMDD_HHMMSS.csv  (metrics log)
        └── checkpoints/                           (saved model weights)
    
    Parameters
    ----------
    log_dir : str
        Root directory for all logs and checkpoints
    
    Returns
    -------
    metrics_file : str
        Path to CSV file for recording training metrics
    checkpoint_dir : Path
        Path to directory for model checkpoints
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    checkpoint_dir = log_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create timestamped metrics file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = log_path / f"training_metrics_{timestamp}.csv"
    
    # Write CSV header
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",           # Episode number
            "episode_reward",    # Total reward in episode
            "loss",              # Policy gradient loss
            "avg_reward_100",    # Running average (last 100 episodes)
            "max_reward",        # Best episode reward so far
            "episode_steps",     # Number of steps in episode
            "timestamp"          # Wall clock time
        ])
    
    return str(metrics_file), checkpoint_dir


def compute_returns(
    rewards: List[float], 
    gamma: float = 0.99
) -> List[float]:
    """
    Compute discounted cumulative returns backwards through trajectory.
    
    This computes G_t for each timestep:
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    
    Working backwards is efficient: G_T = r_T, G_t = r_t + γ*G_{t+1}
    
    Parameters
    ----------
    rewards : list[float]
        Rewards collected during episode
    gamma : float
        Discount factor (0 < γ ≤ 1). Controls how much we care about 
        distant rewards. γ=0.99 means distant rewards are worth ~37% 
        after ~100 steps (0.99^100 ≈ 0.37).
    
    Returns
    -------
    returns : list[float]
        Discounted cumulative returns for each timestep
    """
    returns = []
    cumulative = 0.0
    
    # Work backwards through episode
    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        returns.insert(0, cumulative)
    
    return returns


def normalize_returns(returns: List[float]) -> List[float]:
    """
    Normalize returns to zero mean and unit variance.
    
    Normalization reduces variance in gradients and stabilizes training.
    Without this, large returns cause huge gradient updates.
    
    Normalized return = (return - mean) / std
    
    Parameters
    ----------
    returns : list[float]
        Raw returns from trajectory
    
    Returns
    -------
    normalized : list[float]
        Zero-mean, unit-variance returns
    """
    if len(returns) == 0:
        return []
    
    mean = sum(returns) / len(returns)
    
    if len(returns) > 1:
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = max(1e-8, variance ** 0.5)  # Avoid division by zero
    else:
        std = 1.0
    
    return [(r - mean) / std for r in returns]


# =============================================================================
# Training Loop
# =============================================================================

class PolicyGradientTrainer:
    """
    REINFORCE trainer — simplest policy gradient algorithm.
    
    The training loop is:
        1. Collect trajectory (observations, actions, rewards)
        2. Compute returns (discounted future rewards)
        3. Compute loss = -mean(log π(a|s) * G_t)
        4. Backprop and optimizer.step()
    
    This is on-policy (uses only current policy's trajectories) and
    unbiased but high-variance. Good for starting; upgrade to PPO/A3C later.
    """
    
    def __init__(self, config: TrainingConfig, resume_from: str = None):
        self.config = config
        
        # Policy network
        self.policy = SimplePolicyNetwork(device=config.device)
        
        # Optimizer: Adam with specified learning rate
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Track episode rewards for running average
        self.episode_rewards = []
        
        # Resume from checkpoint if specified
        if resume_from is not None:
            print(f"\nResuming training from checkpoint: {resume_from}")
            self.policy.load(resume_from)
            print(f"  ✓ Loaded policy weights ({self.policy.parameter_count:,} parameters)")
            print("  Note: Optimizer state is reset (learning starts fresh from loaded weights)\n")
    
    def collect_trajectory(self, game: Game) -> Tuple[List, List, List]:
        """
        Run one full episode and collect the trajectory.
        
        A trajectory is the sequence of (observation, action, reward) tuples
        collected during one episode.
        
        Parameters
        ----------
        game : Game
            The environment instance
        
        Returns
        -------
        observations : list[list[float]]
            Normalized observations from each step
        actions : list[int]
            Actions taken (0 = no-op, 1 = jump)
        rewards : list[float]
            Rewards received at each step
        """
        observations = []
        actions = []
        rewards = []
        
        obs = game.reset()
        done = False
        step = 0
        max_steps = 2500  # Increased step cap to allow full 20-second level completion
        
        while not done and step < max_steps:
            # Get normalized observation
            obs_norm = game.get_normalized_observation()
            observations.append(obs_norm)
            
            # Policy selects action (deterministic for now, could be stochastic)
            action = self.policy.predict(obs_norm)
            actions.append(action)
            
            # Action Repeat: Apply the same action for 4 consecutive frames
            accumulated_reward = 0.0
            for _ in range(4):
                obs, step_reward, done = game.step(action)
                accumulated_reward += step_reward
                if done:
                    break
            
            rewards.append(accumulated_reward)
            
            # We stepped up to 4 times
            step += 4
        
        return observations, actions, rewards
    
    def train_on_trajectory(
        self,
        observations: List[List[float]],
        actions: List[int],
        rewards: List[float]
    ) -> float:
        """
        Single gradient update from one trajectory using REINFORCE.
        
        REINFORCE loss:
            Loss = -E[log π(a|s) * G_t]
        
        This is derived from policy gradient theorem. The negative is there
        because we want to maximize expected return, but optimizers minimize loss.
        
        Interpretation:
        - log π(a|s) is higher when the action is more likely under current policy
        - G_t is higher when the trajectory is better (higher future rewards)
        - We increase probability of good actions and decrease for bad ones
        
        Parameters
        ----------
        observations : list
            Observations from trajectory
        actions : list[int]
            Actions taken
        rewards : list[float]
            Rewards received
        
        Returns
        -------
        loss : float
            Scalar loss value (for logging and monitoring)
        """
        if len(rewards) == 0:
            return 0.0
        
        # Step 1: Compute discounted returns
        returns = compute_returns(rewards, gamma=self.config.discount_factor)
        
        # Step 2: Normalize returns (reduces variance without biasing gradient)
        returns_normalized = normalize_returns(returns)
        
        # Step 3: Convert to PyTorch tensors
        obs_tensor = torch.tensor(
            observations,
            dtype=torch.float32,
            device=self.config.device
        )
        returns_tensor = torch.tensor(
            returns_normalized,
            dtype=torch.float32,
            device=self.config.device
        )
        actions_tensor = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.config.device
        )
        
        # Step 4: Forward pass — get action probabilities
        logits = self.policy.forward(obs_tensor)  # Shape: (T, 2)
        log_probs = F.log_softmax(logits, dim=1)  # Shape: (T, 2)
        
        # Step 5: Get log probability of taken actions
        # actions_tensor is (T,), we need to gather the correct logits
        action_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Step 6: Compute REINFORCE loss
        # Loss = -mean(log_prob * return)
        # This encourages actions with positive returns and discourages negative ones
        loss = -(action_log_probs * returns_tensor).mean()
        
        # Step 7: Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, metrics_file: str, checkpoint_dir: Path) -> None:
        """
        Main training loop. Runs for num_episodes, collecting trajectories
        and performing gradient updates.
        
        Outputs:
        - Progress to console every 10 episodes
        - Metrics to CSV file after each episode
        - Model checkpoints every checkpoint_interval episodes
        
        Parameters
        ----------
        metrics_file : str
            Path to CSV file for metrics
        checkpoint_dir : Path
            Directory to save model checkpoints
        """
        # Print header
        print("=" * 80)
        print("  GEOMETRY DASH RL — REINFORCE Training")
        print("=" * 80)
        print(f"  Episodes          : {self.config.num_episodes}")
        print(f"  Learning rate     : {self.config.learning_rate}")
        print(f"  Discount factor γ : {self.config.discount_factor}")
        print(f"  Device            : {self.config.device}")
        print(f"  Difficulty        : {self.config.difficulty}")
        print(f"  Seed              : {self.config.seed}")
        print(f"  Metrics log       : {metrics_file}")
        print(f"  Checkpoints saved : {checkpoint_dir}")
        print("=" * 80)
        print()
        
        # Generate fixed level for all episodes (reproducible training)
        level_gen = LevelGenerator(
            difficulty=self.config.difficulty,
            seed=self.config.seed,
            progressive=False  # Use fixed difficulty, no ramping
        )
        level = level_gen.generate(length=self.config.level_length)
        
        # Initialize game
        game = Game(render=self.config.render)
        game.load_level(level)
        
        # Training loop
        for episode in range(self.config.num_episodes):
            # Collect trajectory from one episode
            observations, actions, rewards = self.collect_trajectory(game)
            episode_reward = sum(rewards)
            
            # Train on trajectory (single gradient update)
            loss = self.train_on_trajectory(observations, actions, rewards)
            
            # Track statistics
            self.episode_rewards.append(episode_reward)
            avg_reward_100 = (
                sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:])
            )
            max_reward = max(self.episode_rewards)
            
            # Log to CSV
            with open(metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode + 1,                    # Episode (1-indexed)
                    f"{episode_reward:.3f}",        # Total reward
                    f"{loss:.6f}",                  # Loss value
                    f"{avg_reward_100:.3f}",        # Running avg
                    f"{max_reward:.3f}",            # Best so far
                    len(rewards),                   # Episode length
                    datetime.now().isoformat()      # Timestamp
                ])
            
            # Console output (every 10 episodes)
            if (episode + 1) % 10 == 0:
                print(
                    f"Ep {episode+1:5d}  |  "
                    f"Reward: {episode_reward:8.2f}  |  "
                    f"Avg(100): {avg_reward_100:8.2f}  |  "
                    f"Loss: {loss:.6f}  |  "
                    f"Steps: {len(rewards):4d}"
                )
            
            # Save checkpoint
            if (episode + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"policy_ep{episode+1:05d}.pth"
                self.policy.save(str(checkpoint_path))
                print(f"        → Checkpoint saved to {checkpoint_path.name}")
        
        # Save final model
        final_path = checkpoint_dir / "policy_final.pth"
        self.policy.save(str(final_path))
        
        game.close()
        
        # Print summary
        print()
        print("=" * 80)
        print("  Training Complete!")
        print("=" * 80)
        print(f"  Final avg reward (100 ep): {avg_reward_100:.3f}")
        print(f"  Best episode reward:       {max_reward:.3f}")
        print(f"  Total episodes trained:    {self.config.num_episodes}")
        print()
        print(f"  Metrics log: {metrics_file}")
        print(f"  Final model: {final_path}")
        print(f"  All saved to: {self.config.log_dir}/")
        print("=" * 80)


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Parse command-line arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train Geometry Dash RL agent using REINFORCE policy gradient"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor 0<γ≤1 (default: 0.99)"
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=1,
        help="Level difficulty 1-5 (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render game window during training (slower, for debugging)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on (default: cpu)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs and checkpoints (default: logs)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint .pth file. "
             "Example: --resume logs/checkpoints/policy_ep00500.pth"
    )
    
    args = parser.parse_args()
    
    # Validate resume path if provided
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"ERROR: Resume checkpoint not found: {resume_path}")
            sys.exit(1)
    
    # Create configuration
    config = TrainingConfig()
    config.num_episodes = args.episodes
    config.learning_rate = args.lr
    config.discount_factor = args.gamma
    config.difficulty = args.difficulty
    config.seed = args.seed
    config.render = args.render
    config.device = args.device
    config.log_dir = args.log_dir
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Set up logging
    metrics_file, checkpoint_dir = setup_logging(args.log_dir)
    
    # Create trainer and run (with optional resume)
    trainer = PolicyGradientTrainer(config, resume_from=args.resume)
    trainer.train(metrics_file, checkpoint_dir)


if __name__ == "__main__":
    main()
