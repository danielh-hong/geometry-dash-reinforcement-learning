import cv2
import numpy as np
import mss
from ultralytics import YOLO

'''
Noise reduction:
Force objects tracked from right edge of screen, see constants for configs and tolerance.
Force sizing check (Only block2 and spike3 can have dynamic sizes, others must be consistent)
Immediately pass very high confidence detections as real
'''


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
model = YOLO("../runs/detect/train12/weights/best.pt")
if torch.cuda.is_available():
    model.to("cuda")

SPIKE_NAMES = {"spike", "spike2", "spike3"}
BLOCK_NAMES = {"block", "block2", "block3"}
PLAYER_NAMES = {"player"}

sct = mss.mss()
monitor = sct.monitors[1]

# Adjust to your game lane
X0, Y0, X1, Y1 = 758, 55, 2035, 1162

# Tracker state
tracked_obstacles = []  # List of dicts representing tracked objects
MAX_MISSED_FRAMES = 2
MIN_SEEN_FRAMES = 3
Y_TOLERANCE = 400  # pixels - Increased significantly to allow for player jumping up/down screen levels
X_SPEED_TOLERANCE = 150  # Max pixels an object can move left in one frame
RIGHT_EDGE_TOLERANCE = 300 # How close to the right edge (X1-X0) an object must spawn to be considered real

frame_idx = 0

cv2.namedWindow("Detection", cv2.WINDOW_AUTOSIZE)
cv2.setWindowProperty("Detection", cv2.WND_PROP_TOPMOST, 1)

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    roi = frame[Y0:Y1, X0:X1]

    results = model.predict(roi, conf=0.4, imgsz=640, verbose=False, iou=0.3)[0]

    detections = []
    # Filter and extract ONLY the best, most confident player
    # There should only be one player!
    raw_player_detections = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())
        name = model.names[cls_id].lower().strip()
        conf = float(box.conf[0].cpu().numpy())
        
        if name in PLAYER_NAMES:
            raw_player_detections.append({'box': (x1, y1, x2, y2), 'conf': conf, 'name': name})
            
    # Find the single most confident player (if any exist)
    best_player = None
    if raw_player_detections:
        best_player = max(raw_player_detections, key=lambda p: p['conf'])
        
    detections = []
    
    # Process all boxes again and apply constraints
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())
        name = model.names[cls_id].lower().strip()
        conf = float(box.conf[0].cpu().numpy())
        
        # If this is a player but NOT the best player, skip it completely (noise)
        if name in PLAYER_NAMES:
            if best_player and (x1, y1, x2, y2) != best_player['box']:
                continue
        
        # --- NOISE FILTER: SIZE CONSTRAINTS ---
        w = x2 - x1
        h = y2 - y1
        MAX_STD_SIZE = BLOCK_SIZE * 1.8  # Allow up to ~200px to account for slight glow/buffer
        
        is_valid_size = True
        if name == "block2":
            # Dynamic size allowed, but if it covers/overlaps the verified player, it's noise
            if best_player:
                px1, py1, px2, py2 = best_player['box']
                # Check for bounding box intersection
                if not (x2 < px1 or x1 > px2 or y2 < py1 or y1 > py2):
                    is_valid_size = False
        elif name == "spike3":
            if h > MAX_STD_SIZE: is_valid_size = False # Dynamic width allowed, but height must be small
        elif name == "block3":
            if w > MAX_STD_SIZE * 1.5 or h > MAX_STD_SIZE * 1.5: is_valid_size = False # Allow block3 to be slightly bigger if they cluster 
        else:
            # Everything else (spike, spike2, block, player) must be relatively small
            if w > MAX_STD_SIZE or h > MAX_STD_SIZE: is_valid_size = False
            
        if is_valid_size:
            detections.append((name, conf, x1, y1, x2, y2))

    players = [d for d in detections if d[0] in PLAYER_NAMES]
    current_obstacles = [d for d in detections if d[0] in SPIKE_NAMES or d[0] in BLOCK_NAMES]
    
    # Sort current frame detections left-to-right (by x1) to prevent inner-loop merging bugs
    current_obstacles.sort(key=lambda d: d[2])
    # Also sort tracked objects left-to-right to ensure spatial matching order
    tracked_obstacles.sort(key=lambda t: t['x1'])

    # --- TRACKING LOGIC ---
    matched_tracked_indices = set()
    new_tracked_obstacles = []

    for curr_obs in current_obstacles:
        name, conf, x1, y1, x2, y2 = curr_obs
        best_match_idx = -1
        best_match_dist = float('inf')

        for i, tracked in enumerate(tracked_obstacles):
            if i in matched_tracked_indices: continue
            
            # Constraint 1: Check Type (Spike vs Block)
            # block3 usually functions as a block, ensure it's categorized effectively
            is_same_type = (name in SPIKE_NAMES and tracked['name'] in SPIKE_NAMES) or \
                           (name in BLOCK_NAMES and tracked['name'] in BLOCK_NAMES)
            if not is_same_type: continue
            
            # Constraint 2: Verify Y height/level roughly matches (Increased tolerance for jumping)
            if abs(y1 - tracked['y1']) > Y_TOLERANCE or abs(y2 - tracked['y2']) > Y_TOLERANCE:
                continue

            # Constraint 3: Verify X moves generally leftwards
            dx = tracked['x1'] - x1 
            
            # For block3 or fast objects, we sometimes need more generous jitter/forward movement logic 
            # especially when they are clustered tight together
            if -80 <= dx <= X_SPEED_TOLERANCE:
                dist = abs(dx)
                if dist < best_match_dist:
                    # Additional check for clustered objects (block3s): 
                    # Only map if it's the closest spatial matching to avoid consuming neighbours
                    best_match_dist = dist
                    best_match_idx = i

        if best_match_idx != -1:
            # Update the existing object properties
            t = tracked_obstacles[best_match_idx]
            t['x1'], t['y1'], t['x2'], t['y2'] = x1, y1, max(x2, t['x2']), y2 # Let x2 expand if offscreen originally
            t['conf'], t['name'] = conf, name
            # If highly confident, instantly fully verify it, otherwise increment normally
            t['seen'] = max(t['seen'] + 1, MIN_SEEN_FRAMES if conf > 0.70 else 0)
            t['missed'] = 0
            matched_tracked_indices.add(best_match_idx)
        else:
            # Found a completely new object. LEFT edge must originate near the right edge!
            roi_width = X1 - X0
            
            # High confidence objects get a larger relaxed spawn area just in case they enter fast
            dynamic_edge_tolerance = RIGHT_EDGE_TOLERANCE + (200 if conf > 0.70 else 0)
            
            if x1 >= (roi_width - dynamic_edge_tolerance):
                # If confidence is high, instantly trust it and bypass the 3-frame waiting probation
                initial_seen = MIN_SEEN_FRAMES if conf > 0.70 else 1
                new_tracked_obstacles.append({
                    'name': name, 'conf': conf, 
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'seen': initial_seen, 'missed': 0
                })
            # else: Ignore it completely as it spontaneously popped up in the middle of the screen

    # Age objects that dissapeared in this frame
    for i, tracked in enumerate(tracked_obstacles):
        if i not in matched_tracked_indices:
            tracked['missed'] += 1

    # Keep only those that haven't missed too many frames, then add new objects
    tracked_obstacles = [t for t in tracked_obstacles if t['missed'] <= MAX_MISSED_FRAMES and t['x2'] > -50]
    tracked_obstacles.extend(new_tracked_obstacles)

    # --- FILTER TO FINAL LEFTMOST 7 RETAINED OBJECTS ---
    # Only trust objects that have persisted for at least MIN_SEEN_FRAMES
    verified_obstacles = [t for t in tracked_obstacles if t['seen'] >= MIN_SEEN_FRAMES]
    
    # Sort them primarily by x location (left to right)
    verified_obstacles.sort(key=lambda t: t['x1'])
    leftmost_7 = verified_obstacles[:7]

    # Print out debug stats to the terminal
    # print(f"Frame {frame_idx} -> Top 7 objects: {[obj['name'] for obj in leftmost_7]}")
    frame_idx += 1

    annotated = results.plot(conf=True)
    
    # --- VISUALIZE THE 7 TRACKED OBJECTS ---
    # Draw thick bright yellow boxes and numbers so you can clearly see them in your pop-up window
    for i, obj in enumerate(leftmost_7):
        x1, y1, x2, y2 = int(obj['x1']), int(obj['y1']), int(obj['x2']), int(obj['y2'])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 4) # Yellow thick bounding box
        cv2.putText(annotated, f"#{i+1}: {obj['name']}", (x1, max(0, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

    # Resize display to half size
    display_img = cv2.resize(annotated, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Detection", display_img)
    

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()