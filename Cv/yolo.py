import cv2
import numpy as np
import mss
from ultralytics import YOLO

# Check of cuda availability for faster inference
import torch
if torch.cuda.is_available():
    print("CUDA is available. Using GPU for inference.")
else:
    print("CUDA is not available. Using CPU for inference.")

# Constants based on your game.py / constants.py
SCREEN_W = 1920
GROUND_Y = 864
BLOCK_SIZE = 112
GAME_SPEED = 1163.22

# model = YOLO("yolov8m.pt")
model = YOLO("../runs/detect/train11/weights/best.pt")
if torch.cuda.is_available():
    model.to("cuda")

SPIKE_NAMES = {"spike", "spike2", "spike3"}
BLOCK_NAMES = {"block", "block2", "block3"}
PLAYER_NAMES = {"player"}

sct = mss.mss()
monitor = sct.monitors[1]

# Adjust to your game lane
X0, Y0, X1, Y1 = 758, 55, 2035, 1162

last_spike = None
last_seen_frame = -999
frame_idx = 0

cv2.namedWindow("Detection", cv2.WINDOW_AUTOSIZE)
cv2.setWindowProperty("Detection", cv2.WND_PROP_TOPMOST, 1)

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    roi = frame[Y0:Y1, X0:X1]

    results = model.predict(roi, conf=0.4, imgsz=640, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())
        name = model.names[cls_id].lower().strip()
        conf = float(box.conf[0].cpu().numpy())
        detections.append((name, conf, x1, y1, x2, y2))

    spikes = [d for d in detections if d[0] in SPIKE_NAMES]
    players = [d for d in detections if d[0] in PLAYER_NAMES]

    # Player Tracking
    player_box = max(players, key=lambda d: d[1]) if players else None

    # Spike Temporal Persistence
    if spikes:
        best_spike = max(spikes, key=lambda d: d[1])
        last_spike = best_spike
        last_seen_frame = frame_idx
    elif frame_idx - last_seen_frame <= 4:
        best_spike = last_spike
    else:
        best_spike = None

    # Calculate the 8 RL Values
    if player_box and best_spike:
        p_name, p_conf, px, py, px2, py2 = player_box
        s_name, s_conf, sx, sy, sx2, sy2 = best_spike

        ph = py2 - py

        # Check if spike is actually in front of the player
        if sx > px:
            rel_x_pixels = sx - px
            rel_y_pixels = sy - py
            time_to_reach_sec = rel_x_pixels / GAME_SPEED
            gap_top_px = max(0.0, sy - (py + ph))
            
            # Normalize for Neural Network
            eight_values = [
                0.0, # Spike is 0.0 (Block is 1.0)
                max(0.0, min(1.0, rel_x_pixels / SCREEN_W)),
                max(-1.0, min(1.0, rel_y_pixels / GROUND_Y)),
                (sx2 - sx) / (BLOCK_SIZE * 5.0), # Width
                (sy2 - sy) / (BLOCK_SIZE * 5.0), # Height
                max(0.0, min(1.0, time_to_reach_sec / 6.0)),
                max(0.0, min(1.0, gap_top_px / GROUND_Y)),
                0.0 # Spike has no floor gap
            ]
            print(f"Next Obstacle (Spike): {['%.3f' % v for v in eight_values]}")
    else:
        print("Waiting for player or obstacle...")

    frame_idx += 1

    # In newer ultralytics versions, plot() accepts 'conf' as boolean, but filters via the predict threshold
    annotated = results.plot(conf=True)
    
    # Resize display to half size
    display_img = cv2.resize(annotated, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Detection", display_img)
    

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()