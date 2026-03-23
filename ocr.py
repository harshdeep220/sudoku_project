"""
ocr.py — Digit recognition for extracted Sudoku cells.

Uses the lightweight CNN model trained by  train_digit_model.py.
Each cell image is classified as either *empty* (0) or a digit 1–9.

IMPORTANT: This module expects cells to already be normalised to
WHITE text on BLACK background (vision.py handles this).
"""

import os
import numpy as np
import cv2
from tensorflow import keras

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "digit_model.keras")
IMG_SIZE = 28                          # must match the training size
EMPTY_THRESHOLD = 0.03                 # max white-pixel ratio to count as empty
                                       # lowered from 0.05 → stricter empty check


# ------------------------------------------------------------------
# Model loading (cached singleton)
# ------------------------------------------------------------------

_model: keras.Model | None = None


def _get_model() -> keras.Model:
    """Load the digit-recognition CNN (cached after first call)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Digit model not found at {MODEL_PATH}.\n"
                "Run  python train_digit_model.py  first."
            )
        _model = keras.models.load_model(MODEL_PATH)
        print(f"[OCR] Model loaded from {MODEL_PATH}")
    return _model


# ------------------------------------------------------------------
# Cell helpers
# ------------------------------------------------------------------

def _is_empty(cell: np.ndarray) -> bool:
    """
    Determine whether a cell image is empty (no digit).

    The cell is already white-on-black (normalised by vision.py).
    We simply threshold and check the white-pixel ratio.
    """
    # Otsu threshold — robust to varying contrast
    _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.count_nonzero(binary) / binary.size
    return white_ratio < EMPTY_THRESHOLD


def _prepare_cell(cell: np.ndarray) -> np.ndarray:
    """
    Prepare a cell image for CNN inference.

    Steps
    -----
    1. Otsu threshold → clean binary (white digit on black bg).
    2. Find the bounding box of the digit and crop it.
    3. Pad to square, add small border, resize to 28×28.
    4. Normalise to [0, 1].
    """
    # Binary threshold — cell is already white-on-black
    _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding rect of non-zero pixels (the digit)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)

    # Crop the digit region
    digit_crop = binary[y:y + h, x:x + w]

    # Pad to square, preserving aspect ratio
    side = max(w, h)
    padded = np.zeros((side, side), dtype=np.uint8)
    px = (side - w) // 2
    py = (side - h) // 2
    padded[py:py + h, px:px + w] = digit_crop

    # Add a small border (matches training data convention)
    border = max(2, side // 6)
    padded = cv2.copyMakeBorder(padded, border, border, border, border,
                                cv2.BORDER_CONSTANT, value=0)

    # Resize to model input
    resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Normalise
    normalised = resized.astype(np.float32) / 255.0
    return normalised.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def recognise_board(cells: list[np.ndarray]) -> np.ndarray:
    """
    Classify all 81 cell images and return the board state.

    Parameters
    ----------
    cells : list of 81 grayscale cell images (row-major, WHITE on BLACK).

    Returns
    -------
    np.ndarray of shape (9, 9), dtype int.
        0 = empty cell, 1–9 = recognised digit.
    """
    model = _get_model()
    board = np.zeros((9, 9), dtype=int)

    # Batch non-empty cells for faster inference
    batch_indices: list[tuple[int, int]] = []
    batch_images: list[np.ndarray] = []

    for idx, cell in enumerate(cells):
        row, col = divmod(idx, 9)
        if _is_empty(cell):
            board[row][col] = 0
        else:
            prepared = _prepare_cell(cell)
            batch_images.append(prepared)
            batch_indices.append((row, col))

    if batch_images:
        batch = np.vstack(batch_images)                   # (N, 28, 28, 1)
        predictions = model.predict(batch, verbose=0)     # (N, 9)
        predicted_labels = np.argmax(predictions, axis=1) # 0-indexed

        for (row, col), label in zip(batch_indices, predicted_labels):
            board[row][col] = int(label) + 1              # convert to 1–9

    return board
