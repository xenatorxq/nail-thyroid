"""
train.py
--------
Training script for the AI-Based Smart Nail Screening System.

Usage (from the project root):
    python src/train.py

This script:
  1. Validates dataset structure.
  2. Builds the tf.data training / validation pipelines.
  3. Constructs the MobileNetV3Small transfer-learning model.
  4. Trains for the initial (frozen-backbone) phase.
  5. Optionally fine-tunes the top backbone layers.
  6. Saves the best model to saved_model/model.keras.
  7. Plots and saves training curves to assets/.

DISCLAIMER: This system is for educational and research purposes only.
Results do NOT constitute a medical diagnosis.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Ensure the src/ directory is importable when running as a script
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from model import NailScreeningModel, CLASS_NAMES
from preprocess import build_dataset, count_dataset_images
from utils import setup_logger, format_duration

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR: str = str(_PROJECT_ROOT / "dataset")
SAVED_MODEL_DIR: str = str(_PROJECT_ROOT / "saved_model")
ASSETS_DIR: str = str(_PROJECT_ROOT / "assets")
MODEL_PATH: str = os.path.join(SAVED_MODEL_DIR, "model.keras")
LOG_PATH: str = os.path.join(ASSETS_DIR, "training.log")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
INITIAL_EPOCHS: int = 20       # frozen-backbone phase
FINE_TUNE_EPOCHS: int = 10     # fine-tuning phase (top layers unfrozen)
FINE_TUNE: bool = True         # set False to skip fine-tuning
PATIENCE: int = 5              # EarlyStopping patience
VALIDATION_SPLIT: float = 0.20 # 80/20 train-val split

logger = setup_logger("train", LOG_PATH)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _validate_dataset(dataset_dir: str) -> None:
    """
    Validate that the dataset directory exists and contains images in both
    class sub-folders.

    Parameters
    ----------
    dataset_dir : str
        Root directory that should contain 'normal/' and 'thyroid/' sub-dirs.

    Raises
    ------
    SystemExit
        On any fatal validation failure, with a descriptive error message.
    """
    required_classes = {"normal", "thyroid"}
    dataset_path = Path(dataset_dir)

    if not dataset_path.is_dir():
        logger.error(
            "Dataset directory not found: %s\n"
            "Create the structure:\n"
            "  dataset/normal/   ← normal nail images\n"
            "  dataset/thyroid/  ← thyroid dysfunction nail images",
            dataset_dir,
        )
        sys.exit(1)

    found_classes = {d.name for d in dataset_path.iterdir() if d.is_dir()}
    missing = required_classes - found_classes
    if missing:
        logger.error(
            "Missing class sub-folders in %s: %s", dataset_dir, missing
        )
        sys.exit(1)

    counts = count_dataset_images(dataset_dir)
    for cls, count in counts.items():
        if count == 0:
            logger.error(
                "No images found in dataset/%s/. Add JPEG or PNG images.", cls
            )
            sys.exit(1)
        logger.info("  %-12s: %d images", cls, count)


def _build_callbacks(checkpoint_path: str) -> list:
    """
    Build the Keras callback list used during training.

    Callbacks:
      - ModelCheckpoint: saves the best model by val_accuracy.
      - EarlyStopping: stops training when val_loss stops improving.
      - ReduceLROnPlateau: lowers learning rate on plateau.
      - TensorBoard: logs for optional TensorBoard visualisation.

    Parameters
    ----------
    checkpoint_path : str
        Path where the best model checkpoint is saved.

    Returns
    -------
    list
        List of tf.keras.callbacks.Callback instances.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(Path(ASSETS_DIR) / "tensorboard_logs"),
            histogram_freq=1,
        ),
    ]
    return callbacks


def _plot_history(
    history: tf.keras.callbacks.History,
    fine_tune_history: Optional[tf.keras.callbacks.History] = None,
    save_path: str = "",
) -> None:
    """
    Plot and save training / validation accuracy and loss curves.

    If a fine-tuning history is provided the curves are concatenated and a
    vertical dashed line marks the transition epoch.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Training history from the frozen-backbone phase.
    fine_tune_history : optional History
        Training history from the fine-tuning phase.
    save_path : str
        File path to save the PNG figure.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    transition_epoch = len(acc)

    if fine_tune_history is not None:
        acc += fine_tune_history.history["accuracy"]
        val_acc += fine_tune_history.history["val_accuracy"]
        loss += fine_tune_history.history["loss"]
        val_loss += fine_tune_history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "AI Nail Screening — Training Curves",
        fontsize=14,
        fontweight="bold",
    )

    # --- Accuracy ---
    axes[0].plot(epochs_range, acc, label="Train Accuracy", color="steelblue")
    axes[0].plot(
        epochs_range, val_acc, label="Val Accuracy", color="darkorange", linestyle="--"
    )
    if fine_tune_history is not None:
        axes[0].axvline(
            x=transition_epoch,
            color="red",
            linestyle=":",
            label=f"Fine-tune start (ep {transition_epoch})",
        )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Loss ---
    axes[1].plot(epochs_range, loss, label="Train Loss", color="steelblue")
    axes[1].plot(
        epochs_range, val_loss, label="Val Loss", color="darkorange", linestyle="--"
    )
    if fine_tune_history is not None:
        axes[1].axvline(
            x=transition_epoch,
            color="red",
            linestyle=":",
            label=f"Fine-tune start (ep {transition_epoch})",
        )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (SparseCategoricalCrossentropy)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves saved to: %s", save_path)
    else:
        plt.show()

    plt.close(fig)


def _print_final_metrics(
    history: tf.keras.callbacks.History,
    fine_tune_history: Optional[tf.keras.callbacks.History] = None,
) -> None:
    """
    Print a concise summary of the best metrics achieved during training.

    Parameters
    ----------
    history : History
        Primary training history.
    fine_tune_history : optional History
        Fine-tuning history.
    """
    all_val_acc = history.history["val_accuracy"]
    all_val_loss = history.history["val_loss"]

    if fine_tune_history is not None:
        all_val_acc += fine_tune_history.history["val_accuracy"]
        all_val_loss += fine_tune_history.history["val_loss"]

    best_val_acc = max(all_val_acc)
    best_epoch = int(np.argmax(all_val_acc)) + 1
    final_val_loss = all_val_loss[np.argmax(all_val_acc)]

    separator = "=" * 55
    logger.info(separator)
    logger.info("  TRAINING COMPLETE — BEST METRICS")
    logger.info(separator)
    logger.info("  Best Validation Accuracy : %.4f  (%.2f%%)", best_val_acc, best_val_acc * 100)
    logger.info("  Validation Loss at Best  : %.4f", final_val_loss)
    logger.info("  Best Epoch               : %d", best_epoch)
    logger.info(separator)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train() -> None:
    """
    End-to-end training routine.

    1. Dataset validation.
    2. Dataset pipeline construction.
    3. Model build.
    4. Phase 1 — frozen backbone training.
    5. Phase 2 — optional fine-tuning.
    6. Model save.
    7. Curve plotting.
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # GPU / CPU info
    # ------------------------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPU detected: %s", [g.name for g in gpus])
        # Allow memory growth to prevent TF from grabbing all VRAM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("No GPU detected. Training on CPU (this may be slow).")

    # ------------------------------------------------------------------
    # 1. Dataset validation
    # ------------------------------------------------------------------
    logger.info("Validating dataset at: %s", DATASET_DIR)
    _validate_dataset(DATASET_DIR)

    # ------------------------------------------------------------------
    # 2. Build tf.data pipelines
    # ------------------------------------------------------------------
    logger.info("Building tf.data pipelines (batch_size=%d) ...", BATCH_SIZE)
    train_ds, val_ds, class_names = build_dataset(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=42,
        augment_train=True,
    )
    logger.info("Classes detected: %s", class_names)
    logger.info("Expected class order: %s", CLASS_NAMES)

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    logger.info("Building MobileNetV3Small transfer-learning model ...")
    screening_model = NailScreeningModel()
    model = screening_model.build()
    screening_model.summary()

    # ------------------------------------------------------------------
    # 4. Phase 1 — frozen-backbone training
    # ------------------------------------------------------------------
    logger.info("=" * 55)
    logger.info("PHASE 1: Training classification head (backbone frozen)")
    logger.info("Epochs: %d", INITIAL_EPOCHS)
    logger.info("=" * 55)

    callbacks = _build_callbacks(MODEL_PATH)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 5. Phase 2 — fine-tuning (optional)
    # ------------------------------------------------------------------
    fine_tune_history: Optional[tf.keras.callbacks.History] = None

    if FINE_TUNE:
        logger.info("=" * 55)
        logger.info("PHASE 2: Fine-tuning top 20 backbone layers")
        logger.info("Additional epochs: %d", FINE_TUNE_EPOCHS)
        logger.info("=" * 55)

        screening_model.unfreeze_top_layers(num_layers=20)

        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
            initial_epoch=len(history.history["accuracy"]),
            callbacks=callbacks,
            verbose=1,
        )

    # ------------------------------------------------------------------
    # 6. Save the best model (already saved by ModelCheckpoint)
    # ------------------------------------------------------------------
    logger.info("Best model saved to: %s", MODEL_PATH)

    # ------------------------------------------------------------------
    # 7. Final metrics + plot
    # ------------------------------------------------------------------
    _print_final_metrics(history, fine_tune_history)

    curve_path = os.path.join(ASSETS_DIR, "training_curves.png")
    _plot_history(history, fine_tune_history, save_path=curve_path)

    elapsed = format_duration(time.time() - start_time)
    logger.info("Total training time: %s", elapsed)
    logger.info("Done. Run the application with:  python src/ui.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
