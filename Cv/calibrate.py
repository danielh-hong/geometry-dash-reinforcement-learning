import cv2
import mss
import numpy as np

# Initial coordinates for the cropping rectangle
# (X0, Y0) is top-left, (X1, Y1) is bottom-right
rect_params = {
    "X0": 120,
    "Y0": 250,
    "X1": 1178,
    "Y1": 720
}

# The handle for adjusting the crop box
current_drag = None

def mouse_callback(event, x, y, flags, param):
    global current_drag

    threshold = 20  # Pixels within which you can grab an edge

    # Check edges
    near_left = abs(x - rect_params["X0"]) < threshold
    near_right = abs(x - rect_params["X1"]) < threshold
    near_top = abs(y - rect_params["Y0"]) < threshold
    near_bottom = abs(y - rect_params["Y1"]) < threshold

    if event == cv2.EVENT_LBUTTONDOWN:
        if near_left: current_drag = "X0"
        elif near_right: current_drag = "X1"
        elif near_top: current_drag = "Y0"
        elif near_bottom: current_drag = "Y1"
        # Clicked inside the box - move the whole box
        elif rect_params["X0"] < x < rect_params["X1"] and rect_params["Y0"] < y < rect_params["Y1"]:
            current_drag = ("move", x, y, rect_params["X0"], rect_params["Y0"], rect_params["X1"], rect_params["Y1"])
            
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_drag == "X0": rect_params["X0"] = x
        elif current_drag == "X1": rect_params["X1"] = x
        elif current_drag == "Y0": rect_params["Y0"] = y
        elif current_drag == "Y1": rect_params["Y1"] = y
        elif isinstance(current_drag, tuple) and current_drag[0] == "move":
            _, start_x, start_y, orig_X0, orig_Y0, orig_X1, orig_Y1 = current_drag
            dx = x - start_x
            dy = y - start_y
            rect_params["X0"] = orig_X0 + dx
            rect_params["X1"] = orig_X1 + dx
            rect_params["Y0"] = orig_Y0 + dy
            rect_params["Y1"] = orig_Y1 + dy
            
    elif event == cv2.EVENT_LBUTTONUP:
        # Prevent inverted rectangles
        if rect_params["X0"] > rect_params["X1"]:
            rect_params["X0"], rect_params["X1"] = rect_params["X1"], rect_params["X0"]
        if rect_params["Y0"] > rect_params["Y1"]:
            rect_params["Y0"], rect_params["Y1"] = rect_params["Y1"], rect_params["Y0"]
        current_drag = None


def main():
    print("==================================================")
    print("Game Lane Calibration Tool")
    print("==================================================")
    print("1. A window should open showing your screen.")
    print("2. Click and drag the GREEN edges of the box to resize it.")
    print("3. Click and drag inside the box to move the whole thing.")
    print("4. Press 'Enter' or 'q' to save and exit.")
    print("==================================================\n")

    sct = mss.mss()
    monitor = sct.monitors[1]  # Primary monitor

    cv2.namedWindow("Calibrate Setup", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibrate Setup", mouse_callback)
    
    # Make window stay on top mostly
    cv2.setWindowProperty("Calibrate Setup", cv2.WND_PROP_TOPMOST, 1)

    while True:
        # Grab screen
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Draw dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        
        # Punch out the "bright" game lane inside the overlay
        X0, Y0, X1, Y1 = rect_params["X0"], rect_params["Y0"], rect_params["X1"], rect_params["Y1"]
        # Ensure bounds are within screen
        X0 = max(0, X0); Y0 = max(0, Y0); X1 = min(frame.shape[1], X1); Y1 = min(frame.shape[0], Y1)
        
        overlay[Y0:Y1, X0:X1] = frame[Y0:Y1, X0:X1]
        
        # Apply alpha blend to outside areas
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw the target rectangle outline
        cv2.rectangle(frame, (X0, Y0), (X1, Y1), (0, 255, 0), 2)
        
        # Add text instructions on screen
        cv2.putText(frame, "Drag edges to resize. Drag center to move.", (X0 + 10, Y0 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"X0:{X0} Y0:{Y0}  X1:{X1} Y1:{Y1}", (X0 + 10, Y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'Enter' to confirm and close", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Calibrate Setup", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 13 or key == ord('q'):  # 13 is Enter
            break

    cv2.destroyAllWindows()

    print("\n✅ Final Calibration Coordinates:")
    print(f"X0, Y0, X1, Y1 = {X0}, {Y0}, {X1}, {Y1}")
    print("\nCopy & paste this line into your yolo.py:")
    print("-" * 50)
    print(f"X0, Y0, X1, Y1 = {X0}, {Y0}, {X1}, {Y1}")
    print("-" * 50)

if __name__ == "__main__":
    main()