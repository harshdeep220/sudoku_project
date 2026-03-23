"""
train_digit_model.py — Generate synthetic printed-digit data and train a CNN.

Instead of MNIST (handwritten), this script renders digits 1–9 using
system sans-serif fonts via Pillow, applies minor augmentations, and
trains a lightweight CNN.  Run once to produce  model/digit_model.keras.

Usage
-----
    python train_digit_model.py
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

IMG_SIZE = 28                   # model input size (28×28 grayscale)
SAMPLES_PER_DIGIT = 2000       # images to generate per digit (1-9)
NUM_CLASSES = 9                 # digits 1 … 9
EPOCHS = 12
BATCH_SIZE = 64
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.keras")

# Fonts to try — the script will use whichever are available on the system.
# On Windows "arial.ttf" is almost always present.  On Linux/Mac we
# fall back to the Pillow built-in bitmap font.
CANDIDATE_FONTS = [
    "arial.ttf",
    "arialbd.ttf",          # Arial Bold
    "calibri.ttf",
    "verdana.ttf",
    "tahoma.ttf",
    "segoeui.ttf",          # Segoe UI
    "DejaVuSans.ttf",       # Linux common
    "LiberationSans-Regular.ttf",
]

# ------------------------------------------------------------------
# Font discovery
# ------------------------------------------------------------------

def _discover_fonts(sizes: list[int] = [18, 20, 22, 24]) -> list[ImageFont.FreeTypeFont]:
    """Return a list of usable TrueType font objects at various sizes."""
    fonts: list[ImageFont.FreeTypeFont] = []
    for name in CANDIDATE_FONTS:
        for size in sizes:
            try:
                f = ImageFont.truetype(name, size)
                fonts.append(f)
            except (OSError, IOError):
                pass
    if not fonts:
        # Ultimate fallback: Pillow default (bitmap) at one "size"
        print("[train] WARNING: No TrueType fonts found, using Pillow default.")
        fonts.append(ImageFont.load_default())
    return fonts


# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------

def _render_digit(digit: int, font: ImageFont.FreeTypeFont,
                  canvas_size: int = IMG_SIZE) -> np.ndarray:
    """
    Render a single digit centred on a canvas_size × canvas_size
    grayscale image, returned as a float32 array in [0, 1].

    Parameters
    ----------
    digit       : int 1–9
    font        : Pillow font object
    canvas_size : output image size in pixels

    Returns
    -------
    np.ndarray of shape (canvas_size, canvas_size), dtype float32, values [0,1]
    """
    img = Image.new("L", (canvas_size, canvas_size), color=0)  # black bg
    draw = ImageDraw.Draw(img)

    text = str(digit)

    # Get the bounding box of the rendered text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Centre the text on the canvas
    x = (canvas_size - text_w) / 2 - bbox[0]
    y = (canvas_size - text_h) / 2 - bbox[1]

    draw.text((x, y), text, fill=255, font=font)

    return np.array(img, dtype=np.float32) / 255.0


def _augment(image: np.ndarray) -> np.ndarray:
    """
    Apply minor augmentations to a 2-D grayscale image:
      • Random 1–2 pixel shift in X and Y
      • Small random scaling (±10 %)
      • Additive Gaussian noise
    """
    h, w = image.shape

    # --- Random shift (translate) ---
    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    M_shift = np.float32([[1, 0, dx], [0, 1, dy]])

    # Build a combined affine: scale + translate
    scale = random.uniform(0.88, 1.12)
    cx, cy = w / 2, h / 2
    M_scale = np.float32([
        [scale, 0, cx * (1 - scale) + dx],
        [0, scale, cy * (1 - scale) + dy],
    ])

    import cv2
    image = cv2.warpAffine(image, M_scale, (w, h),
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # --- Gaussian noise ---
    noise = np.random.normal(0, 0.03, image.shape).astype(np.float32)
    image = np.clip(image + noise, 0, 1)

    return image


def generate_dataset(samples_per_digit: int = SAMPLES_PER_DIGIT):
    """
    Create X_train, y_train arrays of synthetically rendered digits.

    Labels are 0-indexed:  digit 1 → label 0 … digit 9 → label 8.

    Returns
    -------
    X : np.ndarray  (N, 28, 28, 1)  float32
    y : np.ndarray  (N,)            int
    """
    fonts = _discover_fonts()
    print(f"[train] Using {len(fonts)} font variants for data generation.")

    images, labels = [], []

    for digit in range(1, 10):                       # 1 … 9
        for _ in range(samples_per_digit):
            font = random.choice(fonts)
            img = _render_digit(digit, font)
            img = _augment(img)
            images.append(img)
            labels.append(digit - 1)                 # 0-indexed label

    X = np.array(images, dtype=np.float32)[..., np.newaxis]   # (N,28,28,1)
    y = np.array(labels, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


# ------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------

def build_model() -> keras.Model:
    """
    Small CNN for printed-digit classification (9 classes: 1–9).

    Architecture:
        Conv2D(32) → MaxPool → Conv2D(64) → MaxPool →
        Flatten → Dense(128) → Dropout → Dense(9, softmax)
    """
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Synthetic Digit Model Trainer")
    print("=" * 60)

    # 1 — Generate data
    print("\n[1/3] Generating synthetic training data …")
    X, y = generate_dataset()
    print(f"       Dataset shape: X={X.shape}  y={y.shape}")

    # 80/20 train-val split
    split = int(0.8 * len(y))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"       Train: {len(y_train)}  |  Val: {len(y_val)}")

    # 2 — Build & train
    print("\n[2/3] Building and training CNN …")
    model = build_model()
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # 3 — Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n[3/3] Model saved to {MODEL_PATH}")
    print("=" * 60)
    print("  Done!  You can now run  python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
