"""
integrate_yolo_rl.py

Drop‑in replacement that:
- Captures ROI using YOLOObservationPipeline (which outputs raw pixel data)
- Normalizes the raw observation using the original YOLO pipeline's formulas
- Feeds it to a Stable‑Baselines3 PPO checkpoint
"""

from __future__ import annotations

import cv2
import mss
import numpy as np
import torch
from pathlib import Path

from stable_baselines3 import PPO

# Import YOLOObservationPipeline and screen capture region
from Cv.yolo import YOLOObservationPipeline, X0, Y0, X1, Y1

# -----------------------------------------------------------------------------
# Normalization constants (must match the original YOLO pipeline)
# -----------------------------------------------------------------------------
GROUND_Y = 864
BLOCK_SIZE = 112
GAME_SPEED = 1163.22
VISION_LIMIT_PX = 784.0
TIME_NORM_FACTOR = 6.0
PLAYER_X = 0
PLAYER_SIZE = BLOCK_SIZE

# Path to YOLO weights (update as needed)
YOLO_WEIGHTS = "runs/detect/train12/weights/best.pt"

# Path to PPO model zip file
RL_WEIGHTS = "Game/logs_custom/checkpoints/ppo_policy_final.zip"


def adapt_obs_for_model(model: PPO, obs: np.ndarray) -> np.ndarray:
    """Pad/truncate observation so it matches the PPO model observation size."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)

    expected_shape = getattr(model.observation_space, "shape", None)
    if not expected_shape or len(expected_shape) != 1:
        return obs

    expected_dim = int(expected_shape[0])
    current_dim = int(obs.shape[0])

    if current_dim == expected_dim:
        return obs
    if current_dim < expected_dim:
        pad = np.zeros((expected_dim - current_dim,), dtype=np.float32)
        return np.concatenate([obs, pad], axis=0)
    return obs[:expected_dim]


def load_ppo_model(model_path: str, device: str = "cpu") -> PPO:
    """Load a Stable‑Baselines3 PPO checkpoint from a .zip file."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"PPO model not found: {path}")

    model = PPO.load(str(path), device=device)
    return model


def normalize_observation(raw_obs: np.ndarray) -> np.ndarray:
    """
    Convert raw observation vector (from YOLOObservationPipeline) to the normalized
    28‑element vector expected by the PPO model.

    Raw vector format (28 floats):
        [0] player_y (raw top y)
        [1] player_vy (normalized by GAME_SPEED, i.e. raw_vy / GAME_SPEED)
        [2] on_ground (0/1)
        Then for up to 5 obstacles (5 values each):
            type, x, y, width, height (all raw pixels)
        Missing obstacles are zeros.

    Normalized output (28 floats):
        [0] player_y / GROUND_Y
        [1] player_vy / BLOCK_SIZE   (to match original normalization)
        [2] on_ground
        Then for 3 obstacles, each with 8 features:
            type,
            rel_x / VISION_LIMIT_PX,
            rel_y / GROUND_Y,
            width / BLOCK_SIZE / 5.0,
            height / BLOCK_SIZE / 5.0,
            time_to_reach / TIME_NORM_FACTOR,
            gap_top / GROUND_Y,
            gap_bottom / GROUND_Y
    """
    raw = np.asarray(raw_obs, dtype=np.float32).copy()
    if len(raw) != 28:
        print(f"Warning: raw observation has length {len(raw)}, expected 28. Returning as is.")
        return raw

    # Extract player state
    player_y_raw = raw[0]
    player_vy_norm = raw[1]       # already raw_vy / GAME_SPEED
    on_ground = raw[2]

    # Normalize player_y and player_vy
    player_y_norm = player_y_raw / GROUND_Y
    player_vy_norm_original = player_vy_norm / BLOCK_SIZE   # raw_vy / (GAME_SPEED * BLOCK_SIZE)

    player_y_norm = np.clip(player_y_norm, 0.0, 1.0)
    player_vy_norm_original = np.clip(player_vy_norm_original, -1.0, 1.0)

    norm_obs = [player_y_norm, player_vy_norm_original, on_ground]

    # Parse obstacles from raw vector (max 5 obstacles, each with 5 values)
    obstacles_raw = []
    for i in range(5):
        base = 3 + i * 5
        if base + 4 < len(raw):
            otype = raw[base]
            x = raw[base + 1]
            y = raw[base + 2]
            w = raw[base + 3]
            h = raw[base + 4]
            # A genuine obstacle has positive width/height (or type 0/1, but check size)
            if w > 0 and h > 0:
                obstacles_raw.append((otype, x, y, w, h))
        else:
            break

    # Take up to 3 nearest obstacles (they are already sorted by x in the pipeline)
    obstacles_raw = obstacles_raw[:3]

    # For each obstacle, compute the 8 normalized features
    for i in range(3):
        if i < len(obstacles_raw):
            otype, ox, oy, ow, oh = obstacles_raw[i]

            # rel_x = horizontal distance from player (PLAYER_X = 0)
            rel_x = (ox - PLAYER_X) / VISION_LIMIT_PX
            rel_x = np.clip(rel_x, 0.0, 1.0)

            # rel_y = vertical offset (obstacle top y - player top y)
            rel_y = (oy - player_y_raw) / GROUND_Y
            rel_y = np.clip(rel_y, -1.0, 1.0)

            # width / height normalized
            width_norm = ow / BLOCK_SIZE / 5.0
            height_norm = oh / BLOCK_SIZE / 5.0
            width_norm = np.clip(width_norm, 0.0, 1.0)
            height_norm = np.clip(height_norm, 0.0, 1.0)

            # time_to_reach = rel_x / GAME_SPEED, then normalized by TIME_NORM_FACTOR
            time_to_reach = rel_x / GAME_SPEED / TIME_NORM_FACTOR
            time_to_reach = np.clip(time_to_reach, 0.0, 1.0)

            # gap_top = vertical clearance above obstacle (top of obstacle - player's top)
            gap_top = max(0.0, oy - (player_y_raw + PLAYER_SIZE)) / GROUND_Y
            gap_top = np.clip(gap_top, 0.0, 1.0)

            # gap_bottom = vertical clearance below obstacle (only for blocks)
            if otype == 1.0:  # block
                gap_bottom = max(0.0, (player_y_raw - PLAYER_SIZE) - (oy + oh)) / GROUND_Y
            else:  # spike
                gap_bottom = 0.0
            gap_bottom = np.clip(gap_bottom, 0.0, 1.0)

            norm_obs.extend([otype, rel_x, rel_y, width_norm, height_norm,
                             time_to_reach, gap_top, gap_bottom])
        else:
            # No obstacle – fill 8 zeros
            norm_obs.extend([0.0] * 8)

    # Ensure exactly 28 elements
    if len(norm_obs) != 28:
        if len(norm_obs) < 28:
            norm_obs.extend([0.0] * (28 - len(norm_obs)))
        else:
            norm_obs = norm_obs[:28]

    return np.array(norm_obs, dtype=np.float32)


def predict_action_and_probs(model: PPO, obs: np.ndarray) -> tuple[int, np.ndarray]:
    """Return deterministic PPO action and class probabilities [no_jump, jump]."""
    obs = adapt_obs_for_model(model, obs)

    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(obs_tensor)
        probs_tensor = distribution.distribution.probs
        probs = probs_tensor.detach().cpu().numpy()[0]

    action, _ = model.predict(obs, deterministic=True)
    action = int(np.asarray(action).item())
    return action, probs.astype(np.float32, copy=False)


def main() -> None:
    # Instantiate YOLO observation pipeline
    yolo_pipe = YOLOObservationPipeline(YOLO_WEIGHTS)

    # Load PPO policy zip
    model = load_ppo_model(RL_WEIGHTS, device="cpu")
    print(f"Loaded PPO model from {RL_WEIGHTS}")
    print("Observation space:", model.observation_space)
    print("Low:", model.observation_space.low)
    print("High:", model.observation_space.high)

    # Set up screen capture
    sct = mss.mss()
    monitor = sct.monitors[1]

    # ROI dimensions (for debugging)
    roi_height = Y1 - Y0
    roi_width = X1 - X0
    print(f"ROI dimensions: {roi_width} x {roi_height}")

    # Create window and set always‑on‑top property
    window_name = "YOLO+PPO Live"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print("Press 'q' in the window to quit.")
    frame_idx = 0
    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        roi = frame[Y0:Y1, X0:X1]

        # Get raw observation vector from YOLO pipeline
        raw_obs = yolo_pipe.step(roi)
        raw_obs = np.asarray(raw_obs, dtype=np.float32).reshape(-1)

        if raw_obs.shape[0] == 0:
            print("Warning: empty observation from YOLO pipeline")
            continue

        # Warn if player not detected (first three values are zero)
        if raw_obs[0] == 0.0 and raw_obs[1] == 0.0 and raw_obs[2] == 0.0:
            print("Warning: No player detected in this frame")

        # Normalize the observation to match the model's expectations
        norm_obs = normalize_observation(raw_obs)

        # Optional: print first few raw and normalized values for debugging
        print(f"Frame {frame_idx}: raw (first 10) = {raw_obs[:10]}")
        print(f"Frame {frame_idx}: raw (full) = {raw_obs.tolist()}")
        print(f"Frame {frame_idx}: norm (first 10) = {norm_obs[:10]}")

        # Get PPO action and probabilities
        action, probs = predict_action_and_probs(model, norm_obs)

        print(f"Frame {frame_idx}: PPO action = {action} | p(wait)={probs[0]:.3f} p(jump)={probs[1]:.3f}")

        # Display the ROI with action annotation
        cv2.putText(
            roi,
            f"Action: {action}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow(window_name, cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()