"""
predict.py
----------
Inference module for the AI-Based Smart Nail Screening System.

Provides a high-level Predictor class that:
  - Loads a saved Keras model.
  - Preprocesses a single image.
  - Returns the predicted class label and confidence score.

Usage (standalone):
    python src/predict.py --image path/to/nail.jpg

DISCLAIMER: This system is for educational and research purposes only.
Predictions do NOT constitute a medical diagnosis. Always consult a
qualified healthcare professional for any health concerns.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Ensure src/ is on the path when running as a script
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from model import CLASS_NAMES, NailScreeningModel
from preprocess import is_supported_image, preprocess_image_for_prediction
from utils import setup_logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH: str = str(_PROJECT_ROOT / "saved_model" / "model.keras")
LOG_PATH: str = str(_PROJECT_ROOT / "assets" / "predictions.log")

logger = setup_logger("predict", LOG_PATH)

DISCLAIMER: str = (
    "\n"
    "━" * 60 + "\n"
    "⚠  DISCLAIMER\n"
    "━" * 60 + "\n"
    "This system is intended only for preliminary AI-based screening\n"
    "and is NOT a medical diagnosis. The result should not be used\n"
    "as a substitute for professional medical advice, diagnosis, or\n"
    "treatment. Always consult a qualified healthcare professional.\n"
    "━" * 60
)


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Container for a single inference result.

    Attributes
    ----------
    image_path : str
        Path to the analysed image.
    class_index : int
        Predicted class index (0 = Normal, 1 = Thyroid features).
    class_label : str
        Human-readable class label.
    confidence : float
        Model confidence in [0.0, 1.0].
    confidence_pct : str
        Confidence formatted as a percentage string, e.g. "94.82%".
    probabilities : np.ndarray
        Full softmax probability vector for all classes.
    inference_time_ms : float
        Inference wall-clock time in milliseconds.
    is_abnormal : bool
        True when the predicted class is NOT 'Normal Nail'.
    """

    image_path: str
    class_index: int
    class_label: str
    confidence: float
    confidence_pct: str
    probabilities: np.ndarray
    inference_time_ms: float
    is_abnormal: bool

    def __str__(self) -> str:
        separator = "─" * 50
        lines = [
            separator,
            "  AI NAIL SCREENING — RESULT",
            separator,
            f"  Image      : {os.path.basename(self.image_path)}",
            f"  Prediction : {self.class_label}",
            f"  Confidence : {self.confidence_pct}",
            f"  Inference  : {self.inference_time_ms:.1f} ms",
            separator,
        ]
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, self.probabilities)):
            marker = "◀" if i == self.class_index else "  "
            lines.append(f"  [{i}] {name:<40s} {prob * 100:5.2f}% {marker}")
        lines.append(separator)
        lines.append(DISCLAIMER)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class Predictor:
    """
    High-level inference wrapper around a trained Keras model.

    Parameters
    ----------
    model_path : str
        Path to the saved `.keras` model file.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._model: Optional[tf.keras.Model] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the Keras model from disk (called on first predict)."""
        if self._model is None:
            logger.info("Loading model from: %s", self.model_path)
            self._model = NailScreeningModel.load(self.model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> PredictionResult:
        """
        Run inference on a single nail image.

        Parameters
        ----------
        image_path : str
            Absolute or relative path to a JPEG or PNG image.

        Returns
        -------
        PredictionResult
            Dataclass containing the label, confidence, probabilities,
            and timing information.

        Raises
        ------
        FileNotFoundError
            If the image does not exist.
        ValueError
            If the file extension is not supported.
        RuntimeError
            If model inference fails unexpectedly.
        """
        # Validate file
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not is_supported_image(image_path):
            supported = ", ".join([".jpg", ".jpeg", ".png"])
            raise ValueError(
                f"Unsupported file type: '{Path(image_path).suffix}'. "
                f"Supported formats: {supported}"
            )

        # Load model (lazy)
        self._load_model()

        # Preprocess
        try:
            image_batch = preprocess_image_for_prediction(image_path)
        except Exception as exc:
            raise RuntimeError(
                f"Image preprocessing failed for '{image_path}': {exc}"
            ) from exc

        # Inference with timing
        t0 = time.perf_counter()
        try:
            raw_output: np.ndarray = self._model.predict(  # type: ignore[union-attr]
                image_batch, verbose=0
            )
        except Exception as exc:
            raise RuntimeError(
                f"Model inference failed: {exc}"
            ) from exc
        t1 = time.perf_counter()

        inference_ms = (t1 - t0) * 1000.0

        # Parse output
        probabilities: np.ndarray = raw_output[0]          # shape (num_classes,)
        class_index: int = int(np.argmax(probabilities))
        confidence: float = float(probabilities[class_index])
        class_label: str = CLASS_NAMES[class_index]
        confidence_pct: str = f"{confidence * 100:.2f}%"

        result = PredictionResult(
            image_path=image_path,
            class_index=class_index,
            class_label=class_label,
            confidence=confidence,
            confidence_pct=confidence_pct,
            probabilities=probabilities,
            inference_time_ms=inference_ms,
            is_abnormal=(class_index != 0),
        )

        logger.info(
            "Predicted '%s' (%.2f%%) for image: %s",
            class_label,
            confidence * 100,
            os.path.basename(image_path),
        )

        return result

    def batch_predict(self, image_paths: list[str]) -> list[PredictionResult]:
        """
        Run inference on a list of images.

        Parameters
        ----------
        image_paths : list[str]
            List of image file paths.

        Returns
        -------
        list[PredictionResult]
            Results in the same order as input paths.
        """
        results: list[PredictionResult] = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as exc:
                logger.error("Skipping '%s': %s", path, exc)

        return results


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AI-Based Smart Nail Screening System — Inference CLI\n\n"
            "DISCLAIMER: Results are NOT a medical diagnosis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        required=True,
        metavar="PATH",
        help="Path to the nail image (JPEG or PNG).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
        help=f"Path to the saved Keras model (default: {DEFAULT_MODEL_PATH}).",
    )
    return parser


def main() -> None:
    """Entry point for the command-line predictor."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    predictor = Predictor(model_path=args.model)

    try:
        result = predictor.predict(args.image)
        print(result)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
