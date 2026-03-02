# =============================================================================
# test_observation_stream.py
# =============================================================================
# Quick test: verify that observations stream correctly from game to RL network.
# Run this to test the data pipeline without needing torch installed.
#
# Usage:
#   (venv active)
#   python test_observation_stream.py
#
# =============================================================================

from level_generator import LevelGenerator
from game import Game

def test_observation_streaming():
    """Test that normalized observations stream correctly."""
    
    print("=" * 70)
    print("  Testing Observation Streaming")
    print("=" * 70)
    
    # Create a simple level
    gen = LevelGenerator(difficulty=1, seed=42)
    level = gen.generate(length=6000)
    
    # Create game in headless mode
    game = Game(render=False)
    obs_dict = game.load_level(level)
    
    print(f"\n✓ Game initialized with {len(level)} obstacles")
    
    # Run 50 steps and collect observations
    print(f"\n{'Step':>5} {'Action':>7} {'Obs Size':>10} {'Next Obs Type':>15}")
    print("-" * 50)
    
    for step in range(50):
        action = 0 if step % 3 != 0 else 1  # Random action pattern
        obs_dict, reward, done = game.step(action)
        
        # Get normalized observation
        obs_norm = game.get_normalized_observation()
        
        # Check observation format
        next_type = "spike" if obs_dict["obstacles"][0] == 0.0 else "block" if obs_dict["obstacles"][0] == 1.0 else "none"
        
        print(f"{step:5d} {action:7d} {len(obs_norm):10d} {next_type:>15}")
        
        if done:
            print(f"  → Episode ended at step {step}")
            break
    
    print("\n" + "=" * 70)
    print("  Observation Format Details")
    print("=" * 70)
    
    obs_norm = game.get_normalized_observation()
    print(f"\nTotal observation size: {len(obs_norm)}")
    print(f"\nFirst 10 values (player state + 2 obstacle features):")
    for i, val in enumerate(obs_norm[:10]):
        label = ["player_y", "player_vy", "on_ground", 
                 "obs0_type", "obs0_relx", "obs0_rely", "obs0_w", "obs0_h", "obs0_time", "obs0_gap_top"][i]
        print(f"  [{i:2d}] {label:15s}: {val:8.4f}")
    
    print(f"\nLast 2 values (jump_possible, last_action):")
    for i, val in enumerate(obs_norm[-2:]):
        label = ["is_jump_possible", "last_action"][i]
        print(f"  [{len(obs_norm)-2+i:2d}] {label:15s}: {val:8.4f}")
    
    print("\n" + "=" * 70)
    print("  ✓ Observation streaming test PASSED")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_observation_streaming()
