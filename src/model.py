"""
model.py
--------
Model architecture definition for the AI-Based Smart Nail Screening System.

This module defines the CNN model using MobileNetV3Small as a backbone
with transfer learning. The model is designed for binary image classification:
  - Class 0: Normal Nail
  - Class 1: Possible Thyroid Dysfunction Features

DISCLAIMER: This model is for educational and research purposes only.
It does NOT constitute a medical diagnosis. Always consult a healthcare
professional for any medical concerns.
"""

import os
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_HEIGHT: int = 224
IMG_WIDTH: int = 224
IMG_CHANNELS: int = 3
NUM_CLASSES: int = 2
LEARNING_RATE: float = 1e-3

CLASS_NAMES: list[str] = ["Normal Nail", "Possible Thyroid Dysfunction Features"]


class NailScreeningModel:
    """
    Encapsulates the MobileNetV3Small-based transfer learning model for
    nail screening classification.

    Attributes
    ----------
    num_classes : int
        Number of output classes (default: 2).
    input_shape : Tuple[int, int, int]
        Input image dimensions (H, W, C).
    learning_rate : float
        Adam optimizer learning rate.
    model : tf.keras.Model
        The compiled Keras model (set after build()).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_shape: Tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model: Optional[tf.keras.Model] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> tf.keras.Model:
        """
        Build and compile the transfer-learning model.

        Architecture
        ------------
        1. MobileNetV3Small pretrained on ImageNet (frozen).
        2. GlobalAveragePooling2D — collapses spatial dimensions.
        3. Dense(256, relu) — task-specific feature extraction.
        4. Dropout(0.4) — regularisation.
        5. Dense(128, relu) — additional capacity.
        6. Dropout(0.3)
        7. Dense(num_classes, softmax) — probability over classes.

        Returns
        -------
        tf.keras.Model
            The compiled Keras model.
        """
        # ------------------------------------------------------------------
        # 1. Pre-trained backbone (frozen)
        # ------------------------------------------------------------------
        base_model = MobileNetV3Small(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet",
            include_preprocessing=True,  # built-in preprocessing (-1 to 1)
        )
        base_model.trainable = False  # freeze all backbone layers

        # ------------------------------------------------------------------
        # 2. Custom classification head
        # ------------------------------------------------------------------
        inputs = tf.keras.Input(shape=self.input_shape, name="input_image")
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dense(256, activation="relu", name="fc1")(x)
        x = layers.Dropout(0.4, name="dropout1")(x)
        x = layers.Dense(128, activation="relu", name="fc2")(x)
        x = layers.Dropout(0.3, name="dropout2")(x)
        outputs = layers.Dense(
            self.num_classes, activation="softmax", name="predictions"
        )(x)

        self.model = tf.keras.Model(inputs, outputs, name="NailScreeningModel")

        # ------------------------------------------------------------------
        # 3. Compile
        # ------------------------------------------------------------------
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        return self.model

    def unfreeze_top_layers(self, num_layers: int = 20) -> None:
        """
        Optionally unfreeze the top N layers of the backbone for fine-tuning.

        This should only be called AFTER the initial training phase
        to avoid destroying pretrained features.

        Parameters
        ----------
        num_layers : int
            Number of layers from the end of the backbone to unfreeze.
        """
        if self.model is None:
            raise RuntimeError("Call build() before unfreeze_top_layers().")

        base_model = self.model.layers[1]  # index 1 is the MobileNetV3Small layer
        base_model.trainable = True

        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False

        # Re-compile with a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate / 10),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def summary(self) -> None:
        """Print a model summary to stdout."""
        if self.model is None:
            raise RuntimeError("Call build() before summary().")
        self.model.summary()

    def save(self, path: str) -> None:
        """
        Save the Keras model in the native .keras format.

        Parameters
        ----------
        path : str
            Full file path (should end with `.keras`).
        """
        if self.model is None:
            raise RuntimeError("Call build() before save().")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"[Model] Saved to: {path}")

    @staticmethod
    def load(path: str) -> tf.keras.Model:
        """
        Load a saved Keras model from disk.

        Parameters
        ----------
        path : str
            Full file path to the `.keras` model file.

        Returns
        -------
        tf.keras.Model
            The loaded model.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        model = tf.keras.models.load_model(path)
        print(f"[Model] Loaded from: {path}")
        return model
