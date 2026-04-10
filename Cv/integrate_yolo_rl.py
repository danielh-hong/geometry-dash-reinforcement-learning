"""
integrate_yolo_rl.py

Drop‑in replacement that:
- Captures ROI using YOLOObservationPipeline (which now outputs the exact
    normalized vector schema used by Game.get_normalized_observation)
- Captures ROI using YOLOObservationPipeline (which now outputs the exact
    normalized vector schema used by Game.get_normalized_observation)
- Feeds it to a Stable‑Baselines3 PPO checkpoint
"""

from __future__ import annotations

import cv2
import mss
import numpy as np
import torch
from pathlib import Path


from stable_baselines3 import PPO
# Import key press helper
from Cv.key_press import press_space

# Import YOLOObservationPipeline and screen capture region
from yolo import YOLOObservationPipeline, X0, Y0, X1, Y1

# Path to YOLO weights (update as needed)
YOLO_WEIGHTS = "../runs/detect/train12/weights/best.pt"

# Path to PPO model zip file
RL_WEIGHTS = "../Game/logs_custom/checkpoints/ppo_policy_final.zip"


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

        # Get player, obstacles, and obs vector from YOLO pipeline (step_with_debug)
        players, verified_obstacles, obs = yolo_pipe.step_with_debug(roi)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        if obs.shape[0] == 0:
            print("Warning: empty observation from YOLO pipeline")
            continue

        if obs.shape[0] != 28:
            print(f"Warning: observation has length {obs.shape[0]}, expected 28")

        # Warn if player not detected (first three values are zeros)
        if obs[0] == 0.0 and obs[1] == 0.0 and obs[2] == 0.0:
            if obs.shape[0] != 28:
                print(f"Warning: observation has length {obs.shape[0]}, expected 28")

        # Warn if player not detected (first three values are zeros)
        if obs[0] == 0.0 and obs[1] == 0.0 and obs[2] == 0.0:
            print("Warning: No player detected in this frame")

        # Optional debug print.
        print(f"Frame {frame_idx}: obs (first 10) = {obs[:10]}")
        # Optional debug print.
        print(f"Frame {frame_idx}: obs (first 10) = {obs[:10]}")

        # Get PPO action and probabilities
        action, probs = predict_action_and_probs(model, obs)
        action, probs = predict_action_and_probs(model, obs)

        print(f"Frame {frame_idx}: PPO action = {action} | p(wait)={probs[0]:.3f} p(jump)={probs[1]:.3f}")
        if action == 1:
            # press_space()
            pass

        # --- Overlay drawing (match yolo.py) ---
        annotated = roi.copy()

        # Draw player if detected
        if players:
            player = players[0]
            px1, py1, px2, py2 = int(player.x1), int(player.y1), int(player.x2), int(player.y2)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 4)
            cv2.putText(annotated, "PLAYER", (px1, max(0, py1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        else:
            cv2.putText(annotated, "PLAYER NOT DETECTED", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw obstacles
        for i, obj in enumerate(verified_obstacles[:7]):
            x1, y1, x2, y2 = int(obj.x1), int(obj.y1), int(obj.x2), int(obj.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.putText(
                annotated,
                f"#{i+1}: {obj.name}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3,
            )

        cv2.putText(
            annotated,
            f"obs_len={len(obs)} frame={frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # Show action in the corner as well
        cv2.putText(
            annotated,
            f"Action: {action}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        display_img = cv2.resize(annotated, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, display_img)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()