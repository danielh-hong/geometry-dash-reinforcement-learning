# keypress.py
"""
Send a spacebar keypress using Python (Windows only).
"""
import ctypes
import time

# Constants for key events
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002
VK_SPACE = 0x20

def press_space():
    # Press spacebar
    ctypes.windll.user32.keybd_event(VK_SPACE, 0, KEYEVENTF_KEYDOWN, 0)
    time.sleep(0.01)  # Hold for 10ms
    ctypes.windll.user32.keybd_event(VK_SPACE, 0, KEYEVENTF_KEYUP, 0)

if __name__ == "__main__":
    press_space()
