"""
preprocess.py
-------------
Image preprocessing pipeline for the AI-Based Smart Nail Screening System.

Handles:
  - Loading images from disk (JPEG and PNG).
  - Resizing to the model's expected input size.
  - RGB colour-space normalisation.
  - Building tf.data datasets from directory trees.
  - Applying data augmentation during training.

DISCLAIMER: This module is for educational and research purposes only.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_HEIGHT: int = 224
IMG_WIDTH: int = 224
IMG_SIZE: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Low-level image helpers (OpenCV-based)
# ---------------------------------------------------------------------------

def load_image_cv2(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV and convert it to RGB.

    OpenCV reads images in BGR order by default; this function converts to
    RGB so that it matches the channel order expected by the Keras model.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        Image array with shape (H, W, 3) in uint8 [0, 255].

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If OpenCV fails to decode the image.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(
            f"OpenCV could not decode image: {image_path}. "
            "Check that the file is a valid JPEG or PNG."
        )

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def resize_image(image: np.ndarray, size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Resize an image array to the target size using bilinear interpolation.

    Parameters
    ----------
    image : np.ndarray
        Input image array (H, W, C).
    size : Tuple[int, int]
        Target (width, height). Defaults to (224, 224).

    Returns
    -------
    np.ndarray
        Resized image with shape (size[1], size[0], C).
    """
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] to [0.0, 1.0].

    Note: MobileNetV3Small with `include_preprocessing=True` applies its own
    internal normalisation to [-1, 1]. Passing [0, 1] float inputs is
    therefore safe — the backbone's built-in layer handles the final rescale.

    Parameters
    ----------
    image : np.ndarray
        Image array with dtype uint8 or float32.

    Returns
    -------
    np.ndarray
        Float32 array in the range [0.0, 1.0].
    """
    return image.astype(np.float32) / 255.0


def preprocess_image_for_prediction(image_path: str) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image at inference time.

    Steps:
      1. Load via OpenCV (BGR → RGB).
      2. Resize to IMG_SIZE.
      3. Normalise to [0, 1].
      4. Add batch dimension → shape (1, H, W, 3).

    Parameters
    ----------
    image_path : str
        Path to the input image.

    Returns
    -------
    np.ndarray
        Preprocessed image batch of shape (1, 224, 224, 3), dtype float32.
    """
    image = load_image_cv2(image_path)
    image = resize_image(image, IMG_SIZE)
    image = normalize_image(image)
    image = np.expand_dims(image, axis=0)  # (H, W, C) → (1, H, W, C)
    return image


def load_image_pil(image_path: str) -> Image.Image:
    """
    Load an image using Pillow (fallback / UI preview helper).

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    PIL.Image.Image
        Loaded PIL Image object in RGB mode.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If Pillow cannot identify the image.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(
            f"Pillow could not identify image format: {image_path}"
        ) from exc

    return img


def is_supported_image(path: str) -> bool:
    """
    Return True if the file extension is in SUPPORTED_EXTENSIONS.

    Parameters
    ----------
    path : str
        File path or filename.

    Returns
    -------
    bool
    """
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Dataset builders (tf.data)
# ---------------------------------------------------------------------------

def _get_augmentation_layer() -> tf.keras.Sequential:
    """
    Return a Sequential data-augmentation layer stack.

    Augmentations applied only during training to improve generalisation:
      - Random horizontal flip.
      - Random rotation (±15°).
      - Random zoom (±10%).
      - Random brightness/contrast.

    Returns
    -------
    tf.keras.Sequential
        The augmentation pipeline.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),          # ±~15°
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_dataset(
    dataset_dir: str,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
    augment_train: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Build train and validation tf.data.Dataset objects from an image directory.

    Expected directory layout:
    ::

        dataset_dir/
          normal/        ← Class 0
          thyroid/       ← Class 1

    Parameters
    ----------
    dataset_dir : str
        Root directory containing class sub-folders.
    batch_size : int
        Number of images per batch.
    validation_split : float
        Fraction of data to hold out for validation (e.g. 0.2 = 20%).
    seed : int
        Random seed for reproducible splits.
    augment_train : bool
        Whether to apply data augmentation to the training set.

    Returns
    -------
    train_ds : tf.data.Dataset
        Batched, shuffled training dataset.
    val_ds : tf.data.Dataset
        Batched validation dataset (no shuffling, no augmentation).
    class_names : list[str]
        Alphabetically ordered list of class names inferred from sub-folders.

    Raises
    ------
    FileNotFoundError
        If dataset_dir does not exist or contains no valid images.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Create 'dataset/normal/' and 'dataset/thyroid/' and populate "
            "them with labelled fingernail images."
        )

    # ------------------------------------------------------------------
    # Use Keras utility to load from directory
    # ------------------------------------------------------------------
    load_kwargs = dict(
        directory=dataset_dir,
        labels="inferred",
        label_mode="int",             # sparse integer labels (0 / 1)
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=seed,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
    )

    train_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        validation_split=validation_split,
        subset="training",
        **load_kwargs,
    )
    val_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        validation_split=validation_split,
        subset="validation",
        **load_kwargs,
    )

    class_names: list[str] = train_ds.class_names

    # ------------------------------------------------------------------
    # Normalise pixel values to [0, 1]
    # ------------------------------------------------------------------
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # ------------------------------------------------------------------
    # Augmentation (training only)
    # ------------------------------------------------------------------
    if augment_train:
        augment = _get_augmentation_layer()
        train_ds = train_ds.map(
            lambda x, y: (augment(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # ------------------------------------------------------------------
    # Performance optimisations
    # ------------------------------------------------------------------
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def count_dataset_images(dataset_dir: str) -> dict[str, int]:
    """
    Count the number of images per class in the dataset directory.

    Parameters
    ----------
    dataset_dir : str
        Root dataset directory containing class sub-folders.

    Returns
    -------
    dict[str, int]
        Mapping of class_name → image_count.
    """
    counts: dict[str, int] = {}
    dataset_path = Path(dataset_dir)

    if not dataset_path.is_dir():
        return counts

    for class_dir in sorted(dataset_path.iterdir()):
        if class_dir.is_dir():
            n = sum(
                1 for f in class_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            counts[class_dir.name] = n

    return counts
