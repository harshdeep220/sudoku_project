"""
debug_taps.py — Quick tap-test script to verify taps land correctly.

This script:
  1. Connects to the device.
  2. Takes a screenshot.
  3. Saves a debug image with all planned tap locations drawn on it.
  4. Performs a few test taps (number buttons 1, 5, 9) so you can
     visually confirm they land on the right spots.

Usage:
    python debug_taps.py
"""

import time
import cv2
import numpy as np

from device_connector import DeviceConnector
from vision import detect_grid_and_coords


def main():
    print("\n=== TAP DEBUG TOOL ===\n")

    # 1. Connect
    device = DeviceConnector()
    screen_w, screen_h = device.get_screen_size()

    # 2. Screenshot
    screenshot = device.capture_screen()

    # 3. Detect grid + coords
    cells, cell_coords, button_coords, _ = detect_grid_and_coords(
        screenshot, screen_w, screen_h,
    )

    # 4. Draw ALL tap targets on the screenshot
    debug_img = screenshot.copy()

    # Draw cell centres (green dots)
    for r in range(9):
        for c in range(9):
            x, y = cell_coords[r][c]
            cv2.circle(debug_img, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(debug_img, f"{r},{c}", (x - 10, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Draw button centres (red dots with digit labels)
    for digit, (bx, by) in button_coords.items():
        cv2.circle(debug_img, (bx, by), 15, (0, 0, 255), -1)
        cv2.putText(debug_img, str(digit), (bx - 8, by + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    debug_path = "debug_taps.png"
    cv2.imwrite(debug_path, debug_img)
    print(f"[Debug] Tap overlay saved → {debug_path}")
    print(f"[Debug] Check the image to see if dots align with the grid and buttons.\n")

    # 5. Test taps — tap number buttons 1, 5, 9 with 2-second gaps
    #    so you can watch the screen and see where they land
    test_digits = [1, 5, 9]
    for digit in test_digits:
        bx, by = button_coords[digit]
        print(f"[Debug] Tapping number {digit} at ({bx}, {by}) in 2 seconds …")
        time.sleep(2)
        device.tap(bx, by)
        print(f"[Debug]   → Tapped!")

    # 6. Also tap a few cell positions
    print(f"\n[Debug] Now tapping cell (0,0) at {cell_coords[0][0]} …")
    time.sleep(2)
    device.tap(*cell_coords[0][0])
    print(f"[Debug]   → Tapped!")

    print(f"\n[Debug] Now tapping cell (4,4) at {cell_coords[4][4]} …")
    time.sleep(2)
    device.tap(*cell_coords[4][4])
    print(f"[Debug]   → Tapped!")

    print("\n=== DONE ===")
    print("Check your device screen — did you see the taps land?")
    print(f"Also open {debug_path} to verify coordinate alignment.\n")


if __name__ == "__main__":
    main()
