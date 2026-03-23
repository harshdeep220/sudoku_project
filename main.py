"""
main.py — Orchestrator for the Android Sudoku Solver.

Connects to the device, captures the screen, detects and reads the
Sudoku grid, solves it, and taps the solution into the app.

Usage
-----
    python main.py
    python main.py --serial <device_serial>
    python main.py --tap-delay 0.06
"""

import argparse
import json
import os
import time
import numpy as np

from device_connector import DeviceConnector
from vision import detect_grid_and_coords
from ocr import recognise_board
from solver import solve, is_valid_board
from executor import execute_solution


def _print_board(board: np.ndarray, title: str = "Board") -> None:
    """Pretty-print a 9×9 board to the console."""
    print(f"\n{'─' * 25}")
    print(f"  {title}")
    print(f"{'─' * 25}")
    for r in range(9):
        if r > 0 and r % 3 == 0:
            print("  ------+-------+------")
        row_str = ""
        for c in range(9):
            if c > 0 and c % 3 == 0:
                row_str += " | "
            val = board[r, c]
            row_str += f" {val}" if val != 0 else " ."
        print(f"  {row_str}")
    print(f"{'─' * 25}\n")


def main():
    parser = argparse.ArgumentParser(description="Android Sudoku Solver")
    parser.add_argument("--serial", type=str, default=None,
                        help="ADB serial of the target device (auto-detect if omitted)")
    parser.add_argument("--tap-delay", type=float, default=0.04,
                        help="Delay (seconds) between cell taps (default: 0.04)")
    parser.add_argument("--button-delay", type=float, default=0.15,
                        help="Delay after tapping a number button (default: 0.15)")
    args = parser.parse_args()

    t_total = time.perf_counter()

    # ── Step 1: Connect ──────────────────────────────────────────────
    print("\n╔══════════════════════════════════════╗")
    print("║      ANDROID SUDOKU SOLVER           ║")
    print("╚══════════════════════════════════════╝\n")

    print("[1/5] Connecting to device …")
    device = DeviceConnector(device_serial=args.serial)

    # ── Step 2: Capture ──────────────────────────────────────────────
    print("[2/5] Capturing screen …")
    t0 = time.perf_counter()
    screenshot = device.capture_screen()
    print(f"       Screenshot captured in {time.perf_counter() - t0:.2f}s  "
          f"({screenshot.shape[1]}×{screenshot.shape[0]})")

    # ── Step 3: Vision — detect grid + extract coords ────────────────
    print("[3/5] Detecting grid & extracting coordinates …")
    t0 = time.perf_counter()
    screen_w, screen_h = device.get_screen_size()

    coords_file = os.path.join(os.path.dirname(__file__), "coords.json")
    if os.path.exists(coords_file):
        # ── Manual calibration available ──
        print("       Using manual coordinates from coords.json")
        with open(coords_file, "r") as f:
            cal = json.load(f)
        cell_coords = [[(c[0], c[1]) for c in row] for row in cal["cell_coords"]]
        button_coords = {int(k): (v[0], v[1]) for k, v in cal["button_coords"].items()}

        # Still need to extract cell images via auto-detection for OCR
        cells, _, _, _ = detect_grid_and_coords(
            screenshot, screen_w, screen_h,
        )
    else:
        # ── Fully automatic ──
        cells, cell_coords, button_coords, _ = detect_grid_and_coords(
            screenshot, screen_w, screen_h,
        )

    print(f"       Grid detected in {time.perf_counter() - t0:.2f}s")
    print(f"       Button coords: {button_coords}")

    # ── Step 4: OCR — read digits ────────────────────────────────────
    print("[4/5] Recognising digits (OCR) …")
    t0 = time.perf_counter()
    initial_board = recognise_board(cells)
    ocr_time = time.perf_counter() - t0

    givens = int(np.count_nonzero(initial_board))
    _print_board(initial_board, title=f"Initial Board  ({givens} givens, OCR {ocr_time:.2f}s)")

    if not is_valid_board(initial_board):
        print("⚠  WARNING: The detected board has conflicts.  OCR may have "
              "mis-read some digits.  Proceeding anyway — check the output!\n")

    # ── Step 5: Solve ────────────────────────────────────────────────
    print("[5/5] Solving …")
    solved_board = initial_board.copy()
    t0 = time.perf_counter()
    success = solve(solved_board)
    solve_time = time.perf_counter() - t0

    if not success:
        print("✗  Solver failed — the board is likely unsolvable (OCR error?).")
        return

    _print_board(solved_board, title=f"Solved Board  ({solve_time * 1000:.1f} ms)")

    # ── Step 6: Execute quick-fill ───────────────────────────────────
    print("[6/6] Executing quick-fill on device …\n")
    t0 = time.perf_counter()
    execute_solution(
        device,
        initial_board,
        solved_board,
        cell_coords,
        button_coords,
        tap_delay=args.tap_delay,
        button_delay=args.button_delay,
    )
    exec_time = time.perf_counter() - t0

    total_time = time.perf_counter() - t_total
    empties = 81 - givens

    print(f"\n{'═' * 40}")
    print(f"  ✓  Puzzle solved and filled!")
    print(f"     Givens:      {givens}")
    print(f"     Filled:      {empties} cells")
    print(f"     OCR time:    {ocr_time:.2f} s")
    print(f"     Solve time:  {solve_time * 1000:.1f} ms")
    print(f"     Tap time:    {exec_time:.2f} s")
    print(f"     Total time:  {total_time:.2f} s")
    print(f"{'═' * 40}\n")


if __name__ == "__main__":
    main()
