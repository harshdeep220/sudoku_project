"""
vision.py — OpenCV-based grid detection, perspective warp, cell extraction,
             and screen coordinate mapping.

Pipeline
--------
1. Pre-process screenshot (grayscale → blur → adaptive threshold).
2. Find the largest square contour (the Sudoku grid).
3. Perspective-warp to a clean 450×450 top-down image.
4. Slice into 81 cell images with aggressive 15 % margin crop to
   eliminate grid-line artefacts.
5. Normalise each cell to WHITE-text-on-BLACK-background.
6. Map each cell centre back to absolute device-screen coordinates.
7. Locate the 9 number-input buttons at the bottom of the screen.
8. (Debug) save a debug_board.png showing all 81 cropped cells.
"""

import os
import cv2
import numpy as np


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

WARP_SIZE = 450                        # warped board side length (px)
CELL_SIZE = WARP_SIZE // 9             # 50 px per cell
CELL_MARGIN_RATIO = 0.18               # 18 % shaved off each side — aggressive crop

DEBUG_DIR = os.path.dirname(__file__)  # debug_board.png saved here


# ------------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------------

def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR screenshot to a binary image optimised for contour detection.

    Steps: grayscale → Gaussian blur (7×7) → adaptive threshold (inverted).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2,
    )
    return thresh


# ------------------------------------------------------------------
# Grid detection
# ------------------------------------------------------------------

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points as: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]      # top-left     (smallest x+y)
    rect[2] = pts[np.argmax(s)]      # bottom-right (largest  x+y)
    rect[1] = pts[np.argmin(d)]      # top-right    (smallest x-y)
    rect[3] = pts[np.argmax(d)]      # bottom-left  (largest  x-y)
    return rect


def find_grid_corners(thresh: np.ndarray) -> np.ndarray | None:
    """
    Find the four corners of the largest quadrilateral contour.

    Returns
    -------
    np.ndarray (4, 2) of corner coordinates in the *original* image space,
    or None if no suitable contour was found.
    """
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            return _order_corners(corners)

    return None


# ------------------------------------------------------------------
# Perspective warp
# ------------------------------------------------------------------

def warp_grid(image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a perspective warp to extract a top-down view of the grid.

    Returns (warped, M) where M is the 3×3 perspective matrix.
    """
    dst = np.array([
        [0, 0],
        [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1],
        [0, WARP_SIZE - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (WARP_SIZE, WARP_SIZE))
    return warped, M


# ------------------------------------------------------------------
# Cell extraction  (aggressive margin + polarity normalisation)
# ------------------------------------------------------------------

def _normalise_polarity(cell: np.ndarray) -> np.ndarray:
    """
    Ensure a grayscale cell is WHITE text on BLACK background.

    Strategy: sample the mean intensity of a small border strip.
    If the border (background) is bright → invert so digits are white.
    """
    h, w = cell.shape
    # Sample a 3-pixel-wide frame around the edges
    border_pixels = np.concatenate([
        cell[0:3, :].ravel(),          # top 3 rows
        cell[h - 3:h, :].ravel(),      # bottom 3 rows
        cell[:, 0:3].ravel(),          # left 3 cols
        cell[:, w - 3:w].ravel(),      # right 3 cols
    ])
    if border_pixels.mean() > 127:
        # Background is bright → invert
        return cv2.bitwise_not(cell)
    return cell


def extract_cells(warped: np.ndarray, save_debug: bool = True) -> list[np.ndarray]:
    """
    Slice the warped 450×450 grid image into 81 individual cell images.

    Each cell has an aggressive 18 % margin crop on all four sides to
    completely remove grid-line artefacts.  The result is normalised to
    white-text-on-black-background.

    Parameters
    ----------
    warped     : BGR warped grid image (WARP_SIZE × WARP_SIZE).
    save_debug : if True, save ``debug_board.png`` showing all 81 cells.

    Returns
    -------
    list of 81 grayscale np.ndarray images (row-major order).
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cells: list[np.ndarray] = []

    margin = int(CELL_SIZE * CELL_MARGIN_RATIO)  # ~9 px on a 50-px cell

    for row in range(9):
        for col in range(9):
            y1 = row * CELL_SIZE + margin
            y2 = (row + 1) * CELL_SIZE - margin
            x1 = col * CELL_SIZE + margin
            x2 = (col + 1) * CELL_SIZE - margin
            cell = gray[y1:y2, x1:x2]

            # Normalise polarity → white text on black bg
            cell = _normalise_polarity(cell)
            cells.append(cell)

    # ── Debug visualisation ──────────────────────────────────────────
    if save_debug:
        _save_debug_board(cells)

    return cells


def _save_debug_board(cells: list[np.ndarray]) -> None:
    """
    Stitch 81 cell images into a 9×9 grid and save as ``debug_board.png``.
    Each cell is resized to 50×50 for uniform display.
    """
    DISPLAY_SIZE = 50
    GAP = 2          # pixel gap between cells (white line for visibility)

    board_w = 9 * DISPLAY_SIZE + 10 * GAP
    board_h = 9 * DISPLAY_SIZE + 10 * GAP
    canvas = np.full((board_h, board_w), 128, dtype=np.uint8)  # grey background

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, 9)
        resized = cv2.resize(cell, (DISPLAY_SIZE, DISPLAY_SIZE),
                             interpolation=cv2.INTER_AREA)
        y = GAP + r * (DISPLAY_SIZE + GAP)
        x = GAP + c * (DISPLAY_SIZE + GAP)
        canvas[y:y + DISPLAY_SIZE, x:x + DISPLAY_SIZE] = resized

    path = os.path.join(DEBUG_DIR, "debug_board.png")
    cv2.imwrite(path, canvas)
    print(f"[Vision] Debug board saved → {path}")


# ------------------------------------------------------------------
# Coordinate mapping — cells
# ------------------------------------------------------------------

def compute_cell_screen_coords(
    corners: np.ndarray,
    M: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """
    Map the centre of each of the 81 cells from *warped* space back
    to *device-screen* coordinates.
    """
    M_inv = np.linalg.inv(M)
    coords: list[list[tuple[int, int]]] = []

    for row in range(9):
        row_coords: list[tuple[int, int]] = []
        for col in range(9):
            wx = col * CELL_SIZE + CELL_SIZE / 2
            wy = row * CELL_SIZE + CELL_SIZE / 2

            pt = np.array([wx, wy, 1.0])
            screen_pt = M_inv @ pt
            screen_pt /= screen_pt[2]

            sx, sy = int(round(screen_pt[0])), int(round(screen_pt[1]))
            row_coords.append((sx, sy))
        coords.append(row_coords)

    return coords


# ------------------------------------------------------------------
# Coordinate mapping — number buttons
# ------------------------------------------------------------------

def find_number_button_coords(
    image: np.ndarray,
    grid_bottom_y: int,
    screen_width: int,
    screen_height: int,
) -> dict[int, tuple[int, int]]:
    """
    Estimate (X, Y) screen coordinates for the 1–9 number buttons
    located at the **very bottom** of the screen.

    The buttons sit in roughly the bottom 8–12 % of the screen,
    *below* the Undo / Erase / Hint icon row.

    Parameters
    ----------
    image         : BGR screenshot  (unused but kept for API compatibility)
    grid_bottom_y : Y-coordinate of the grid's bottom edge on screen
    screen_width  : device screen width
    screen_height : device screen height

    Returns
    -------
    dict mapping digit (1–9) → (x, y) screen coordinate of button centre.
    """
    # Target the centre of the bottom ~10 % of the screen.
    # On a 2400-px-tall screen this gives Y ≈ 2280, right in the
    # number-button row that sits below the Undo/Hint icons.
    button_y = int(screen_height * 0.93)

    # Horizontal positions: 9 evenly-spaced across the width
    margin = screen_width * 0.02          # small side margin
    usable = screen_width - 2 * margin
    step = usable / 9

    buttons: dict[int, tuple[int, int]] = {}
    for i in range(9):
        digit = i + 1
        bx = int(margin + step * i + step / 2)
        buttons[digit] = (bx, button_y)

    print(f"[Vision] Number-button Y = {button_y}  "
          f"(screen {screen_width}×{screen_height})")

    return buttons


def find_fast_pencil_coords(
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """
    Estimate the (X, Y) screen coordinates of the "Fast Pencil" button.

    This button sits in the toolbar row between the grid and the 1–9
    number buttons (alongside Undo, Erase, Pencil, Hint).  It is
    typically the 3rd of 5 equally-spaced icons, i.e. centred horizontally,
    at roughly 83–87 % of the screen height.

    Returns
    -------
    (x, y) — absolute screen coordinates of the Fast Pencil button centre.
    """
    fp_x = screen_width // 2                   # centre of screen
    fp_y = int(screen_height * 0.855)          # toolbar row
    print(f"[Vision] Fast Pencil coords = ({fp_x}, {fp_y})")
    return (fp_x, fp_y)


# ------------------------------------------------------------------
# Public convenience function
# ------------------------------------------------------------------

def detect_grid_and_coords(
    screenshot: np.ndarray,
    screen_width: int,
    screen_height: int,
) -> tuple[list[np.ndarray], list[list[tuple[int, int]]], dict[int, tuple[int, int]], tuple[int, int]]:
    """
    Full vision pipeline: detect grid → warp → extract cells → compute coords.

    Returns
    -------
    (cells, cell_coords, button_coords, fast_pencil_coords)
        cells              : list of 81 grayscale cell images
        cell_coords        : 9×9 list of (x, y) screen coordinates for cell centres
        button_coords      : dict  digit (1-9) → (x, y) for number buttons
        fast_pencil_coords : (x, y) for the Fast Pencil activation button

    Raises
    ------
    RuntimeError  if the grid cannot be detected.
    """
    thresh = preprocess(screenshot)
    corners = find_grid_corners(thresh)

    if corners is None:
        raise RuntimeError(
            "Could not detect the Sudoku grid.  "
            "Make sure the full 9×9 board is visible on screen."
        )

    warped, M = warp_grid(screenshot, corners)
    cells = extract_cells(warped, save_debug=True)
    cell_coords = compute_cell_screen_coords(corners, M)

    grid_bottom_y = int(max(corners[:, 1]))
    button_coords = find_number_button_coords(
        screenshot, grid_bottom_y, screen_width, screen_height,
    )
    fast_pencil = find_fast_pencil_coords(screen_width, screen_height)

    return cells, cell_coords, button_coords, fast_pencil
