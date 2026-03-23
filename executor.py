"""
executor.py — Quick-fill touch automation.

The target Sudoku app uses "quick fill" mode:
  1. Tap a number button (1–9) at the bottom of the screen.
  2. Then tap all empty cells on the grid that need that number.
  3. Repeat for every number.

This module compares initial_board vs solved_board to figure out
which cells were originally empty, groups them by digit, and executes
the tap sequences.
"""

import time
import numpy as np
from device_connector import DeviceConnector


def execute_solution(
    device: DeviceConnector,
    initial_board: np.ndarray,
    solved_board: np.ndarray,
    cell_coords: list[list[tuple[int, int]]],
    button_coords: dict[int, tuple[int, int]],
    tap_delay: float = 0.08,
    button_delay: float = 0.30,
) -> None:
    """
    Fill the Sudoku board on the device using quick-fill mode.

    Parameters
    ----------
    device        : DeviceConnector instance
    initial_board : 9×9 numpy array (0 = empty, 1-9 = given)
    solved_board  : 9×9 numpy array (fully solved)
    cell_coords   : 9×9 nested list of (x, y) screen coords for each cell
    button_coords : dict  digit → (x, y) for the number buttons
    tap_delay     : seconds to wait between cell taps (default: 0.08)
    button_delay  : seconds to wait after tapping a number button (default: 0.30)
    """
    total_taps = 0

    for digit in range(1, 10):
        # Collect all cells that were empty AND need this digit
        targets: list[tuple[int, int]] = []
        for r in range(9):
            for c in range(9):
                if initial_board[r, c] == 0 and solved_board[r, c] == digit:
                    targets.append(cell_coords[r][c])

        if not targets:
            continue

        # Step 1 & 2 — Select the number button and tap all target cells
        bx, by = button_coords[digit]
        print(f"  [Executor] Tapping button {digit} & {len(targets)} cell(s) …")
        sequence = [(bx, by)]
        
        for tx, ty in targets:
            sequence.append((tx, ty))
                
        device.tap_sequence(sequence)
        total_taps += len(targets)

    print(f"  [Executor] Done — {total_taps} cells filled.")
