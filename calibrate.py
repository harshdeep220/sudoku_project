"""
calibrate.py — Manual coordinate calibration tool (full 81-cell mode).

Opens a device screenshot in an OpenCV window and guides you to click
the centre of every grid cell (row by row, left to right) followed by
the 9 number buttons.  90 clicks total.

Saves all coordinates to  coords.json  so main.py can use them.

Usage
-----
    python calibrate.py
    python calibrate.py --serial <device_serial>

Controls
--------
    Left-click : mark the current point
    Z          : undo the last click
    Esc        : abort without saving
"""

import argparse
import json
import os
import cv2
import numpy as np
from device_connector import DeviceConnector

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "coords.json")


def main():
    parser = argparse.ArgumentParser(description="Sudoku coordinate calibration")
    parser.add_argument("--serial", type=str, default=None)
    args = parser.parse_args()

    # ── Connect & screenshot ─────────────────────────────────────────
    device = DeviceConnector(device_serial=args.serial)
    screenshot = device.capture_screen()
    h, w = screenshot.shape[:2]

    # Scale for display
    max_display_h = 900
    scale = min(1.0, max_display_h / h)
    display_w = int(w * scale)
    display_h = int(h * scale)
    base_img = cv2.resize(screenshot, (display_w, display_h))

    # ── Build the list of 90 prompts ─────────────────────────────────
    prompts: list[str] = []
    # 81 cells: row 1–9, col 1–9
    for r in range(9):
        for c in range(9):
            prompts.append(f"Cell  Row {r + 1}, Col {c + 1}")
    # 9 buttons
    for d in range(1, 10):
        prompts.append(f"Number button  {d}")

    total = len(prompts)  # 90
    all_clicks: list[tuple[int, int]] = []   # device-resolution coords
    display_marks: list[tuple[int, int]] = []  # display-resolution coords

    # ── Mouse callback ───────────────────────────────────────────────
    new_click: list[tuple[int, int]] = []

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            new_click.append((mx, my))

    win_name = "Calibration (Left-click = mark | Z = undo | Esc = abort)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    # ── Main loop ────────────────────────────────────────────────────
    idx = 0
    while idx < total:
        prompt = prompts[idx]
        progress = f"[{idx + 1}/{total}]"

        # Draw frame
        frame = base_img.copy()

        # Draw all previous marks
        for i, (dx, dy) in enumerate(display_marks):
            color = (0, 255, 0) if i < 81 else (0, 0, 255)
            cv2.circle(frame, (dx, dy), 4, color, -1)

        # Highlight which row we're in (subtle yellow overlay on current row)
        if idx < 81:
            cur_row = idx // 9
            # Draw a faint yellow stripe hint (just text is enough)

        # Top bar with prompt
        cv2.rectangle(frame, (0, 0), (display_w, 40), (30, 30, 30), -1)
        cv2.putText(frame, f"{progress}  {prompt}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Bottom bar with controls hint
        cv2.rectangle(frame, (0, display_h - 25), (display_w, display_h), (30, 30, 30), -1)
        cv2.putText(frame, "Click = mark  |  Z = undo  |  Esc = quit",
                    (10, display_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.imshow(win_name, frame)
        new_click.clear()

        key = cv2.waitKey(30) & 0xFF

        # Esc → abort
        if key == 27:
            print("[Calibrate] Aborted.")
            cv2.destroyAllWindows()
            return

        # Z → undo last click
        if key in (ord('z'), ord('Z')) and idx > 0:
            idx -= 1
            all_clicks.pop()
            display_marks.pop()
            print(f"  ↩  Undid click {idx + 1}")
            continue

        # Process click
        if new_click:
            mx, my = new_click[0]
            real_x = int(mx / scale)
            real_y = int(my / scale)
            all_clicks.append((real_x, real_y))
            display_marks.append((mx, my))
            print(f"  {progress}  {prompt}  →  ({real_x}, {real_y})")
            idx += 1

    cv2.destroyAllWindows()

    # ── Build JSON ───────────────────────────────────────────────────
    cell_coords: list[list[list[int]]] = []
    for r in range(9):
        row: list[list[int]] = []
        for c in range(9):
            pt = all_clicks[r * 9 + c]
            row.append([pt[0], pt[1]])
        cell_coords.append(row)

    button_coords: dict[str, list[int]] = {}
    for d in range(9):
        pt = all_clicks[81 + d]
        button_coords[str(d + 1)] = [pt[0], pt[1]]

    data = {
        "cell_coords": cell_coords,
        "button_coords": button_coords,
        "screen_width": device.screen_width,
        "screen_height": device.screen_height,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n{'═' * 40}")
    print(f"  ✓  Calibration saved to {OUTPUT_FILE}")
    print(f"     81 cell coords + 9 button coords")
    print(f"     Cell(0,0) = {cell_coords[0][0]}")
    print(f"     Cell(8,8) = {cell_coords[8][8]}")
    print(f"     Button 1  = {button_coords['1']}")
    print(f"     Button 9  = {button_coords['9']}")
    print(f"{'═' * 40}")
    print(f"\nRun  python main.py  — it will auto-load coords.json.\n")


if __name__ == "__main__":
    main()
