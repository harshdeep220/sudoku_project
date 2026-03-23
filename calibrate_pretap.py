"""
calibrate_pretap.py — Manual coordinate calibration for the pre-tap button.

Opens a device screenshot and allows you to click on the button
(e.g., "New Game") you want the solver to pre-tap before beginning
the solve sequence. The coordinates are saved to pretap.json for app.py.

Usage:
    python calibrate_pretap.py
"""

import argparse
import json
import os
import cv2
import time
from device_connector import DeviceConnector

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "pretap.json")

def main():
    parser = argparse.ArgumentParser(description="Pre-tap button calibration")
    parser.add_argument("--serial", type=str, default=None)
    args = parser.parse_args()

    device = DeviceConnector(device_serial=args.serial)
    
    print("[Calibrate] Taking screenshot…")
    screenshot = device.capture_screen()
    h, w = screenshot.shape[:2]

    # Scale for display
    max_display_h = 900
    scale = min(1.0, max_display_h / h)
    display_w = int(w * scale)
    display_h = int(h * scale)
    display_img = cv2.resize(screenshot, (display_w, display_h))

    picked = []

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x = int(mx / scale)
            real_y = int(my / scale)
            picked.clear()
            picked.append((real_x, real_y))
            
            # Show a green dot on the UI to indicate the click
            frame = display_img.copy()
            cv2.circle(frame, (mx, my), 6, (0, 255, 0), -1)
            cv2.rectangle(frame, (0, 0), (display_w, 40), (30, 30, 30), -1)
            cv2.putText(frame, f"Picked ({real_x}, {real_y}). Press Esc to exit, or Enter to Save.",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(win_name, frame)

    win_name = "Click the pre-tap button, then press Enter to save"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    # Initial frame
    frame = display_img.copy()
    cv2.rectangle(frame, (0, 0), (display_w, 40), (30, 30, 30), -1)
    cv2.putText(frame, "Click the pre-tap button, then press Enter to save",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow(win_name, frame)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # Esc
            print("[Calibrate] Aborted without saving.")
            break
        elif key == 13 or key == 10:  # Enter
            if picked:
                x, y = picked[0]
                with open(OUTPUT_FILE, "w") as f:
                    json.dump({"x": x, "y": y}, f, indent=2)
                print(f"\n[Calibrate] Pre-tap saved to {OUTPUT_FILE} → ({x}, {y})")
            else:
                print("\n[Calibrate] Nothing was clicked. Aborting.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
