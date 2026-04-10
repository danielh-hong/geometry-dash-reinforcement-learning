"""
yolo.py

Outputs the same normalized 28-feature vector schema used by
Game.get_normalized_observation() during PPO training.
Outputs the same normalized 28-feature vector schema used by
Game.get_normalized_observation() during PPO training.
"""

from __future__ import annotations

import cv2
import mss
import numpy as np
import torch
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ultralytics import YOLO

from Game import constants as C

from Game import constants as C

# -----------------------------------------------------------------------------
# CUDA / device selection
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    print("CUDA is available. Using GPU for inference.")
    DEVICE = "cuda"
else:
    print("CUDA is not available. Using CPU for inference.")
    DEVICE = "cpu"

# -----------------------------------------------------------------------------
# Game / ROI constants
# -----------------------------------------------------------------------------
GROUND_Y = float(C.GROUND_Y)
BLOCK_SIZE = float(C.BLOCK_SIZE)
PLAYER_SIZE = float(C.PLAYER_SIZE)
GAME_SPEED = float(C.GAME_SPEED)
MAX_FALL_SPEED = float(C.MAX_FALL_SPEED)
GROUND_Y = float(C.GROUND_Y)
BLOCK_SIZE = float(C.BLOCK_SIZE)
PLAYER_SIZE = float(C.PLAYER_SIZE)
GAME_SPEED = float(C.GAME_SPEED)
MAX_FALL_SPEED = float(C.MAX_FALL_SPEED)

# PPO observation constants from Game.get_normalized_observation
MAX_OBSTACLES = 3
VISION_LIMIT_PX = 784.0
TIME_NORM_FACTOR = 6.0
# PPO observation constants from Game.get_normalized_observation
MAX_OBSTACLES = 3
VISION_LIMIT_PX = 784.0
TIME_NORM_FACTOR = 6.0

# Tracking constants
MAX_MISSED_FRAMES = 2
MIN_SEEN_FRAMES = 3
Y_TOLERANCE = 400
X_SPEED_TOLERANCE = 150
RIGHT_EDGE_TOLERANCE = 300

# Class name groups
SPIKE_NAMES = {"spike", "spike2", "spike3"}
BLOCK_NAMES = {"block", "block2", "block3"}
PLAYER_NAMES = {"player"}

# ROI within the full screenshot
X0, Y0, X1, Y1 = 758, 55, 2035, 1162

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class Detection:
    name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class TrackedObstacle:
    name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    seen: int = 1
    missed: int = 0

@dataclass
class ObservationState:
    prev_player_y: Optional[float] = None   # top y of player
    prev_time: Optional[float] = None

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def canonical_kind(name: str) -> str:
    if name in SPIKE_NAMES:
        return "spike"
    return "block"

def canonical_kind(name: str) -> str:
    if name in SPIKE_NAMES:
        return "spike"
    return "block"

def best_detection(detections: Sequence[Detection], name: str) -> Optional[Detection]:
    best = None
    for d in detections:
        if d.name != name:
            continue
        if best is None or d.conf > best.conf:
            best = d
    return best

def merge_adjacent_obstacles(obstacles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Optional merging – kept for reference but not used in observation."""
    obstacles = sorted(obstacles, key=lambda o: o["x"])
    merged: List[Dict[str, Any]] = []

    for o in obstacles:
        if not merged:
            merged.append(o.copy())
            continue

        last = merged[-1]

        same_kind = (o["kind"] in SPIKE_NAMES and last["kind"] in SPIKE_NAMES) or \
                    (o["kind"] in BLOCK_NAMES and last["kind"] in BLOCK_NAMES)

        adjacent = o["x"] <= last["x"] + last["w"] + 1.0

        if same_kind and adjacent:
            x1 = last["x"]
            y1 = min(last["y"], o["y"])
            x2 = max(last["x"] + last["w"], o["x"] + o["w"])
            y2 = max(last["y"] + last["h"], o["y"] + o["h"])
            last["x"] = x1
            last["y"] = y1
            last["w"] = x2 - x1
            last["h"] = y2 - y1
        else:
            merged.append(o.copy())

    return merged

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
class YOLOObservationPipeline:
    """Stateful pipeline that returns PPO-ready normalized observations."""
    """Stateful pipeline that returns PPO-ready normalized observations."""
    def __init__(
        self,
        model_path: str,
        conf: float = 0.4,
        iou: float = 0.3,
        imgsz: int = 640,
        device: str = DEVICE,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.state = ObservationState()
        self.tracked_obstacles: List[TrackedObstacle] = []

        if self.device == "cuda":
            self.model.to("cuda")

    def reset(self) -> None:
        self.state = ObservationState()
        self.tracked_obstacles = []

    def step(self, roi: np.ndarray, now_t: Optional[float] = None) -> List[float]:
        """Return the normalized 28-float PPO observation for one ROI frame."""
        """Return the normalized 28-float PPO observation for one ROI frame."""
        _, _, obs = self.step_with_debug(roi, now_t=now_t)
        return obs

    def step_with_debug(
        self, roi: np.ndarray, now_t: Optional[float] = None
    ) -> Tuple[List[Detection], List[TrackedObstacle], List[float]]:

        if now_t is None:
            now_t = perf_counter()

        results = self.model.predict(
            roi,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            iou=self.iou,
        )[0]

        detections = self._parse_detections(results)

        # Best‑player selection
        raw_player_detections = [d for d in detections if d.name in PLAYER_NAMES]
        best_player = max(raw_player_detections, key=lambda p: p.conf) if raw_player_detections else None

        filtered_detections: List[Detection] = []
        for d in detections:
            if d.name in PLAYER_NAMES:
                if best_player is not None and (d.x1, d.y1, d.x2, d.y2) != (best_player.x1, best_player.y1, best_player.x2, best_player.y2):
                    continue

            w = d.x2 - d.x1
            h = d.y2 - d.y1
            max_std_size = BLOCK_SIZE * 1.8
            is_valid_size = True

            if d.name == "block2":
                if best_player is not None:
                    px1, py1, px2, py2 = best_player.x1, best_player.y1, best_player.x2, best_player.y2
                    if not (d.x2 < px1 or d.x1 > px2 or d.y2 < py1 or d.y1 > py2):
                        is_valid_size = False
            elif d.name == "spike3":
                if h > max_std_size:
                    is_valid_size = False
            elif d.name == "block3":
                if w > max_std_size * 1.5 or h > max_std_size * 1.5:
                    is_valid_size = False
            else:
                if w > max_std_size or h > max_std_size:
                    is_valid_size = False

            if is_valid_size:
                filtered_detections.append(d)

        players = [d for d in filtered_detections if d.name in PLAYER_NAMES]
        current_obstacles = [d for d in filtered_detections if d.name in SPIKE_NAMES or d.name in BLOCK_NAMES]

        current_obstacles.sort(key=lambda d: d.x1)
        self.tracked_obstacles.sort(key=lambda t: t.x1)

        # Tracking logic – updates tracked obstacles with current detections (no expansion)
        matched_tracked_indices = set()
        new_tracked_obstacles: List[TrackedObstacle] = []

        for curr_obs in current_obstacles:
            best_match_idx = -1
            best_match_dist = float("inf")

            for i, tracked in enumerate(self.tracked_obstacles):
                if i in matched_tracked_indices:
                    continue

                is_same_type = (
                    (curr_obs.name in SPIKE_NAMES and tracked.name in SPIKE_NAMES)
                    or (curr_obs.name in BLOCK_NAMES and tracked.name in BLOCK_NAMES)
                )
                if not is_same_type:
                    continue

                if abs(curr_obs.y1 - tracked.y1) > Y_TOLERANCE or abs(curr_obs.y2 - tracked.y2) > Y_TOLERANCE:
                    continue

                dx = tracked.x1 - curr_obs.x1
                if -80 <= dx <= X_SPEED_TOLERANCE:
                    dist = abs(dx)
                    if dist < best_match_dist:
                        best_match_dist = dist
                        best_match_idx = i

            if best_match_idx != -1:
                t = self.tracked_obstacles[best_match_idx]
                # Replace tracked coordinates with current detection (no expansion)
                t.x1 = curr_obs.x1
                t.y1 = curr_obs.y1
                t.x2 = curr_obs.x2
                t.y2 = curr_obs.y2
                t.conf = curr_obs.conf
                t.name = curr_obs.name
                t.seen = max(t.seen + 1, MIN_SEEN_FRAMES if curr_obs.conf > 0.70 else 0)
                t.missed = 0
                matched_tracked_indices.add(best_match_idx)
            else:
                roi_width = X1 - X0
                dynamic_edge_tolerance = RIGHT_EDGE_TOLERANCE + (200 if curr_obs.conf > 0.70 else 0)

                if curr_obs.x1 >= (roi_width - dynamic_edge_tolerance):
                    initial_seen = MIN_SEEN_FRAMES if curr_obs.conf > 0.70 else 1
                    new_tracked_obstacles.append(
                        TrackedObstacle(
                            name=curr_obs.name,
                            conf=curr_obs.conf,
                            x1=curr_obs.x1,
                            y1=curr_obs.y1,
                            x2=curr_obs.x2,
                            y2=curr_obs.y2,
                            seen=initial_seen,
                            missed=0,
                        )
                    )

        for i, tracked in enumerate(self.tracked_obstacles):
            if i not in matched_tracked_indices:
                tracked.missed += 1

        self.tracked_obstacles = [t for t in self.tracked_obstacles if t.missed <= MAX_MISSED_FRAMES and t.x2 > -50]
        self.tracked_obstacles.extend(new_tracked_obstacles)

        verified_obstacles = [t for t in self.tracked_obstacles if t.seen >= MIN_SEEN_FRAMES]
        verified_obstacles.sort(key=lambda t: t.x1)

        # Build observation vector matching Game.get_normalized_observation.
        # Build observation vector matching Game.get_normalized_observation.
        obs = self._build_observation_vector(players, verified_obstacles, now_t)
        self.state.prev_time = now_t

        if players:
            self.state.prev_player_y = players[0].y1

        return players, verified_obstacles, obs

    def _parse_detections(self, results: Any) -> List[Detection]:
        detections: List[Detection] = []
        names = results.names

        for box in results.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)
            cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls[0].cpu().numpy())
            name = str(names[cls_id]).lower().strip()
            conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf[0].cpu().numpy())
            detections.append(Detection(name=name, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2))

        return detections

    def _infer_on_ground(
        self,
        player: Detection,
        verified_obstacles: Sequence[TrackedObstacle],
    ) -> float:
        player_left = float(player.x1)
        player_right = float(player.x2)
        player_bottom = float(player.y2)

        # Ground support.
        if abs(player_bottom - GROUND_Y) <= 6.0:
            return 1.0

        # Block-top support.
        for o in verified_obstacles:
            if o.name not in BLOCK_NAMES:
                continue
            horizontal_overlap = (player_right > o.x1) and (player_left < o.x2)
            standing_on_top = abs(player_bottom - o.y1) <= 8.0
            if horizontal_overlap and standing_on_top:
                return 1.0

        return 0.0

    def _infer_on_ground(
        self,
        player: Detection,
        verified_obstacles: Sequence[TrackedObstacle],
    ) -> float:
        player_left = float(player.x1)
        player_right = float(player.x2)
        player_bottom = float(player.y2)

        # Ground support.
        if abs(player_bottom - GROUND_Y) <= 6.0:
            return 1.0

        # Block-top support.
        for o in verified_obstacles:
            if o.name not in BLOCK_NAMES:
                continue
            horizontal_overlap = (player_right > o.x1) and (player_left < o.x2)
            standing_on_top = abs(player_bottom - o.y1) <= 8.0
            if horizontal_overlap and standing_on_top:
                return 1.0

        return 0.0

    def _build_observation_vector(
        self,
        players: Sequence[Detection],
        verified_obstacles: Sequence[TrackedObstacle],
        now_t: float,
    ) -> List[float]:
        """
        Build the normalized observation vector of length 28 used in PPO training:
          [player_y_norm, player_vy_norm, on_ground,
           obstacle_1(8), obstacle_2(8), obstacle_3(8),
           is_jump_possible_now]
        Build the normalized observation vector of length 28 used in PPO training:
          [player_y_norm, player_vy_norm, on_ground,
           obstacle_1(8), obstacle_2(8), obstacle_3(8),
           is_jump_possible_now]
        """
        # ----- Player state -----
        if players:
            player = max(players, key=lambda p: p.conf)
            player_y = float(player.y1)
            player_x = float(player.x1)
            player_y = float(player.y1)
            player_x = float(player.x1)

            if self.state.prev_player_y is not None and self.state.prev_time is not None and now_t > self.state.prev_time:
                dt = now_t - self.state.prev_time
                dy = player_y - self.state.prev_player_y
                raw_vy = dy / dt  # px/s
                vy_norm = clamp(raw_vy / MAX_FALL_SPEED, -1.0, 1.0)
                raw_vy = dy / dt  # px/s
                vy_norm = clamp(raw_vy / MAX_FALL_SPEED, -1.0, 1.0)
            else:
                vy_norm = 0.0

            on_ground = self._infer_on_ground(player, verified_obstacles)
            on_ground = self._infer_on_ground(player, verified_obstacles)
        else:
            player_y = 0.0
            player_x = 0.0
            player_x = 0.0
            vy_norm = 0.0
            on_ground = 0.0

        obs = [
            clamp(player_y / GROUND_Y, 0.0, 1.0),
            float(vy_norm),
            float(on_ground),
        ]
        obs = [
            clamp(player_y / GROUND_Y, 0.0, 1.0),
            float(vy_norm),
            float(on_ground),
        ]

        # ----- Obstacles (canonicalized + merged exactly like game.py) -----
        # ----- Obstacles (canonicalized + merged exactly like game.py) -----
        obstacle_dicts = []
        for o in verified_obstacles:
            if o.x2 < player_x:
                continue
            obstacle_dicts.append({
                "kind": canonical_kind(o.name),
                "x": float(o.x1),
                "y": float(o.y1),
                "w": float(o.x2 - o.x1),
                "h": float(o.y2 - o.y1),
            })
            if o.x2 < player_x:
                continue
            obstacle_dicts.append({
                "kind": canonical_kind(o.name),
                "x": float(o.x1),
                "y": float(o.y1),
                "w": float(o.x2 - o.x1),
                "h": float(o.y2 - o.y1),
            })

        obstacle_dicts.sort(key=lambda o: o["x"])

        merged: List[Dict[str, float | str]] = []
        for o in obstacle_dicts:
            if not merged:
                merged.append(o.copy())
                continue

            last = merged[-1]
            if (
                o["kind"] == last["kind"]
                and float(o["x"]) <= float(last["x"]) + float(last["w"]) + 1.0
            ):
                new_right = max(float(last["x"]) + float(last["w"]), float(o["x"]) + float(o["w"]))
                last["w"] = new_right - float(last["x"])
                last["y"] = min(float(last["y"]), float(o["y"]))
                last["h"] = max(float(last["h"]), float(o["h"]))
            else:
                merged.append(o.copy())

        merged: List[Dict[str, float | str]] = []
        for o in obstacle_dicts:
            if not merged:
                merged.append(o.copy())
                continue

            last = merged[-1]
            if (
                o["kind"] == last["kind"]
                and float(o["x"]) <= float(last["x"]) + float(last["w"]) + 1.0
            ):
                new_right = max(float(last["x"]) + float(last["w"]), float(o["x"]) + float(o["w"]))
                last["w"] = new_right - float(last["x"])
                last["y"] = min(float(last["y"]), float(o["y"]))
                last["h"] = max(float(last["h"]), float(o["h"]))
            else:
                merged.append(o.copy())

        for i in range(MAX_OBSTACLES):
            if i < len(merged):
                o = merged[i]
                rel_x = float(o["x"]) - player_x
                if rel_x > VISION_LIMIT_PX:
                    obs.extend([0.0] * 8)
                    continue

                rel_y = float(o["y"]) - player_y
                time_to_reach = rel_x / GAME_SPEED if GAME_SPEED > 0.0 else 0.0
                gap_top = max(0.0, float(o["y"]) - (player_y + PLAYER_SIZE))
                gap_bot = (
                    max(0.0, (player_y - PLAYER_SIZE) - (float(o["y"]) + float(o["h"])))
                    if o["kind"] == "block"
                    else 0.0
                )

                obs.extend([
                    0.0 if o["kind"] == "spike" else 1.0,
                    clamp(rel_x / VISION_LIMIT_PX, 0.0, 1.0),
                    clamp(rel_y / GROUND_Y, -1.0, 1.0),
                    clamp((float(o["w"]) / BLOCK_SIZE) / 5.0, 0.0, 1.0),
                    clamp((float(o["h"]) / BLOCK_SIZE) / 5.0, 0.0, 1.0),
                    clamp(time_to_reach / TIME_NORM_FACTOR, 0.0, 1.0),
                    clamp(gap_top / GROUND_Y, 0.0, 1.0),
                    clamp(gap_bot / GROUND_Y, 0.0, 1.0),
                ])
            if i < len(merged):
                o = merged[i]
                rel_x = float(o["x"]) - player_x
                if rel_x > VISION_LIMIT_PX:
                    obs.extend([0.0] * 8)
                    continue

                rel_y = float(o["y"]) - player_y
                time_to_reach = rel_x / GAME_SPEED if GAME_SPEED > 0.0 else 0.0
                gap_top = max(0.0, float(o["y"]) - (player_y + PLAYER_SIZE))
                gap_bot = (
                    max(0.0, (player_y - PLAYER_SIZE) - (float(o["y"]) + float(o["h"])))
                    if o["kind"] == "block"
                    else 0.0
                )

                obs.extend([
                    0.0 if o["kind"] == "spike" else 1.0,
                    clamp(rel_x / VISION_LIMIT_PX, 0.0, 1.0),
                    clamp(rel_y / GROUND_Y, -1.0, 1.0),
                    clamp((float(o["w"]) / BLOCK_SIZE) / 5.0, 0.0, 1.0),
                    clamp((float(o["h"]) / BLOCK_SIZE) / 5.0, 0.0, 1.0),
                    clamp(time_to_reach / TIME_NORM_FACTOR, 0.0, 1.0),
                    clamp(gap_top / GROUND_Y, 0.0, 1.0),
                    clamp(gap_bot / GROUND_Y, 0.0, 1.0),
                ])
            else:
                obs.extend([0.0] * 8)

        # is_jump_possible_now in game.py is exactly obs["on_ground"].
        obs.append(float(on_ground))
        obs.extend([0.0] * 8)

        # is_jump_possible_now in game.py is exactly obs["on_ground"].
        obs.append(float(on_ground))

        # Ensure exactly 28 elements
        if len(obs) != 28:
            if len(obs) < 28:
                obs.extend([0.0] * (28 - len(obs)))
            else:
                obs = obs[:28]
        obs[2] = 1
        obs[-1] = 1
        obs[4] -= 0.15
        return obs

# -----------------------------------------------------------------------------
# Demo / standalone capture loop
# -----------------------------------------------------------------------------
def run_demo() -> None:
    model_path = "../runs/detect/train12/weights/best.pt"
    pipe = YOLOObservationPipeline(model_path)

    sct = mss.mss()
    monitor = sct.monitors[1]

    cv2.namedWindow("Detection", cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty("Detection", cv2.WND_PROP_TOPMOST, 1)

    frame_idx = 0

    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        roi = frame[Y0:Y1, X0:X1]

        players, verified_obstacles, obs = pipe.step_with_debug(roi)

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

        display_img = cv2.resize(annotated, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("Detection", display_img)

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()