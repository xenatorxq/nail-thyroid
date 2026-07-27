"""
utils.py
--------
Shared utility functions for the AI-Based Smart Nail Screening System.

Provides:
  - Logger factory.
  - Image loading / display helpers.
  - Prediction formatting.
  - Log file management.
  - Duration formatting.
  - Filesystem helpers.

DISCLAIMER: This system is for educational and research purposes only.
"""

import csv
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger with console and optional file output.

    Calling this function multiple times with the same `name` returns the
    same logger instance without adding duplicate handlers.

    Parameters
    ----------
    name : str
        Logger name (used as prefix in log messages).
    log_file : str, optional
        Path to a log file. If provided, log messages are also written there.
    level : int
        Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image_as_array(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV and return as an RGB NumPy array.

    Parameters
    ----------
    image_path : str
        Path to the image file (JPEG or PNG).

    Returns
    -------
    np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If OpenCV cannot decode the file.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(
            f"OpenCV failed to decode image: {image_path}"
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_pil_image(
    pil_image: Image.Image,
    max_size: Tuple[int, int] = (400, 400),
) -> Image.Image:
    """
    Resize a PIL image proportionally so it fits within max_size.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Input image.
    max_size : Tuple[int, int]
        Maximum (width, height) in pixels.

    Returns
    -------
    PIL.Image.Image
        Resized image, maintaining aspect ratio.
    """
    pil_image.thumbnail(max_size, Image.LANCZOS)
    return pil_image


def array_to_pil(array: np.ndarray) -> Image.Image:
    """
    Convert a NumPy array (uint8, RGB) to a PIL Image.

    Parameters
    ----------
    array : np.ndarray
        RGB image array.

    Returns
    -------
    PIL.Image.Image
    """
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def display_image_matplotlib(
    image_path: str,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """
    Display (or save) an image with a title using Matplotlib.

    Parameters
    ----------
    image_path : str
        Path to the image.
    title : str
        Title to show above the image.
    save_path : str, optional
        If provided, save the figure instead of displaying it.
    """
    img_array = load_image_as_array(image_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_array)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Prediction display helpers
# ---------------------------------------------------------------------------

def format_prediction_text(
    class_label: str,
    confidence_pct: str,
    is_abnormal: bool,
) -> str:
    """
    Format a prediction into a human-readable multi-line string for display
    in the desktop application result panel.

    Parameters
    ----------
    class_label : str
        The predicted class label.
    confidence_pct : str
        Confidence percentage string, e.g. "94.82%".
    is_abnormal : bool
        True if the prediction indicates possible thyroid features.

    Returns
    -------
    str
        Formatted prediction text ready for the UI.
    """
    status_icon = "⚠" if is_abnormal else "✓"
    lines = [
        f"Prediction:  {status_icon}  {class_label}",
        f"Confidence:  {confidence_pct}",
    ]
    return "\n".join(lines)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.

    Examples:
      62.5  → "1m 02s"
      3700  → "1h 01m 40s"
      45.0  → "45s"

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted duration string.
    """
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Log / CSV record saving
# ---------------------------------------------------------------------------

def save_prediction_log(
    log_path: str,
    image_path: str,
    class_label: str,
    confidence: float,
    class_index: int,
) -> None:
    """
    Append a prediction record to a CSV log file.

    The CSV schema is:
        timestamp, image_name, class_index, class_label, confidence_pct

    Parameters
    ----------
    log_path : str
        Path to the CSV log file (created if it does not exist).
    image_path : str
        Path of the analysed image.
    class_label : str
        Predicted class label string.
    confidence : float
        Model confidence in [0.0, 1.0].
    class_index : int
        Predicted class integer index.
    """
    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_exists = os.path.isfile(log_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_name = os.path.basename(image_path)
    confidence_pct = f"{confidence * 100:.2f}%"

    with open(log_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(
                ["timestamp", "image_name", "class_index", "class_label", "confidence"]
            )
        writer.writerow(
            [timestamp, image_name, class_index, class_label, confidence_pct]
        )


def read_prediction_log(log_path: str) -> list[dict]:
    """
    Read all records from the prediction CSV log file.

    Parameters
    ----------
    log_path : str
        Path to the CSV log file.

    Returns
    -------
    list[dict]
        List of log records as dictionaries.  Empty list if file not found.
    """
    if not os.path.isfile(log_path):
        return []

    records: list[dict] = []
    with open(log_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            records.append(dict(row))
    return records


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """
    Create a directory (and any intermediate parents) if it does not exist.

    Parameters
    ----------
    path : str
        Directory path.

    Returns
    -------
    str
        The same path (absolute).
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    Assumes this file lives at  <project_root>/src/utils.py.

    Returns
    -------
    pathlib.Path
    """
    return Path(__file__).resolve().parent.parent


def get_model_path() -> str:
    """Return the default saved-model path."""
    return str(get_project_root() / "saved_model" / "model.keras")


def get_log_csv_path() -> str:
    """Return the default prediction log CSV path."""
    return str(get_project_root() / "assets" / "prediction_log.csv")


def model_exists() -> bool:
    """
    Return True if the default saved model file exists on disk.

    Returns
    -------
    bool
    """
    return os.path.isfile(get_model_path())


def list_images_in_directory(directory: str) -> list[str]:
    """
    Recursively list all JPEG/PNG image paths within a directory.

    Parameters
    ----------
    directory : str
        Root directory to search.

    Returns
    -------
    list[str]
        Sorted list of absolute image file paths.
    """
    supported = {".jpg", ".jpeg", ".png"}
    image_paths: list[str] = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if Path(filename).suffix.lower() in supported:
                image_paths.append(os.path.join(root, filename))

    return sorted(image_paths)
