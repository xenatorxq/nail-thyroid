# AI-Based Smart Nail Screening System for Thyroid Dysfunction

> **DISCLAIMER:** This project is for **educational and research purposes only**.  
> It does **NOT** diagnose, treat, or prevent any medical condition.  
> Every prediction is an AI-based screening estimate — always consult a qualified healthcare professional.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Tech Stack](#tech-stack)
4. [Installation](#installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Training the Model](#training-the-model)
7. [Running Predictions (CLI)](#running-predictions-cli)
8. [Running the Desktop Application](#running-the-desktop-application)
9. [Module Reference](#module-reference)
10. [Output Artefacts](#output-artefacts)
11. [Extending the Project](#extending-the-project)
12. [Known Limitations](#known-limitations)
13. [Future Improvements](#future-improvements)
14. [Academic References](#academic-references)

---

## Overview

This capstone project implements a binary image classification system that analyses
fingernail photographs and flags potential visual features associated with thyroid
dysfunction. The model does **not** diagnose disease; it serves as an AI-assisted
preliminary screening aid to support — not replace — clinical evaluation.

### Classification Classes

| Index | Label | Description |
|-------|-------|-------------|
| 0 | Normal Nail | No detectable visual abnormalities |
| 1 | Possible Thyroid Dysfunction Features | Visual cues (ridging, brittleness, colour change, koilonychia, etc.) warrant clinical follow-up |

---

## Project Structure

```
project/
│
├── dataset/
│   ├── normal/          ← Normal fingernail images (JPEG / PNG)
│   └── thyroid/         ← Images with possible thyroid-related nail features
│
├── models/              ← Reserved for experimental / versioned model checkpoints
│
├── assets/              ← Training curves, logs, UI screenshots
│   ├── training_curves.png
│   ├── training.log
│   ├── predictions.log
│   └── prediction_log.csv
│
├── saved_model/
│   └── model.keras      ← Best model saved by ModelCheckpoint
│
├── src/
│   ├── model.py         ← MobileNetV3Small architecture + NailScreeningModel class
│   ├── preprocess.py    ← Image loading, resizing, normalisation, tf.data pipelines
│   ├── train.py         ← End-to-end training script (frozen + fine-tuning phases)
│   ├── predict.py       ← Inference module + CLI entry point
│   ├── ui.py            ← Tkinter desktop application
│   └── utils.py         ← Shared helpers: logger, formatters, filesystem utils
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Library | Version |
|-------|---------|---------|
| Language | Python | 3.12+ |
| Deep Learning | TensorFlow / Keras | ≥ 2.16 |
| Image Processing | OpenCV | ≥ 4.9 |
| Image I/O | Pillow | ≥ 10.3 |
| Numerical | NumPy | ≥ 1.26 |
| Visualisation | Matplotlib | ≥ 3.9 |
| GUI | Tkinter | stdlib |
| Optional | Scikit-learn, Pandas | ≥ 1.5 / ≥ 2.2 |

---

## Installation

### 1. Prerequisites

- Python **3.12** or newer  
- `pip` or `conda`  
- (Optional) NVIDIA GPU with CUDA 12.x for faster training

### 2. Clone / Download

```bash
git clone https://github.com/xenatorxq/nail-thyroid.git
cd nail-thyroid/src
```

### 3. Create a Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# --- OR using conda ---
conda create -n nail-screening python=3.12 -y
conda activate nail-screening
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** Uncomment the `tensorflow-metal` line in
> `requirements.txt` before installing for GPU acceleration.

> **Linux Tkinter:** Install the system package if the GUI fails to launch:
> ```bash
> sudo apt-get install python3-tk   # Debian / Ubuntu
> sudo dnf install python3-tkinter  # Fedora / RHEL
> ```

### 5. Verify Installation

```bash
python - <<'EOF'
import tensorflow as tf, cv2, PIL, numpy, matplotlib, tkinter
print("TensorFlow :", tf.__version__)
print("OpenCV     :", cv2.__version__)
print("Pillow     :", PIL.__version__)
print("NumPy      :", numpy.__version__)
print("Matplotlib :", matplotlib.__version__)
print("Tkinter    : OK")
print("\nAll dependencies verified ✓")
EOF
```

---

## Dataset Preparation

### Directory Layout

```
dataset/
├── normal/          ← One image per file, any resolution
│   ├── normal_001.jpg
│   ├── normal_002.png
│   └── ...
└── thyroid/
    ├── thyroid_001.jpg
    ├── thyroid_002.png
    └── ...
```

### Guidelines

| Criterion | Recommendation |
|-----------|---------------|
| Minimum images per class | ≥ 100 (≥ 500 strongly recommended) |
| Supported formats | `.jpg`, `.jpeg`, `.png` |
| Minimum resolution | 224 × 224 px (larger is fine — auto-resized) |
| Class balance | Aim for ≤ 2 : 1 ratio between classes |
| Background | Plain / neutral preferred; avoid clutter |
| Lighting | Natural or controlled; avoid harsh shadows |

> **Note:** The model resizes every image to **224 × 224** and normalises pixel
> values to [0, 1] automatically. You do not need to pre-process images manually.

### Class Folder Mapping

The `tf.keras.utils.image_dataset_from_directory` function infers labels
**alphabetically** from sub-folder names:

| Folder | Inferred Index |
|--------|---------------|
| `normal` | 0 |
| `thyroid` | 1 |

This matches `CLASS_NAMES = ["Normal Nail", "Possible Thyroid Dysfunction Features"]`
defined in `src/model.py`.

---

## Training the Model

```bash
# From the project/ directory
python src/train.py
```

### What Happens

1. **Dataset validation** — checks class folders and counts images.
2. **Pipeline construction** — builds augmented `tf.data` datasets.
3. **Phase 1 (frozen backbone)** — trains only the classification head for `INITIAL_EPOCHS` epochs.
4. **Phase 2 (fine-tuning)** — unfreezes the top 20 MobileNetV3Small layers and continues training at a lower learning rate.
5. **Best model saved** — `ModelCheckpoint` saves `saved_model/model.keras` whenever `val_accuracy` improves.
6. **Training curves** — saved to `assets/training_curves.png`.

### Key Hyperparameters (edit in `train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 32 | Images per gradient step |
| `INITIAL_EPOCHS` | 20 | Frozen-backbone training epochs |
| `FINE_TUNE_EPOCHS` | 10 | Additional fine-tuning epochs |
| `FINE_TUNE` | `True` | Enable / disable fine-tuning phase |
| `PATIENCE` | 5 | EarlyStopping patience |
| `VALIDATION_SPLIT` | 0.20 | Fraction of data held out for validation |
| `LEARNING_RATE` | 1e-3 | Initial Adam learning rate |

### Sample Output

```
2024-07-27 10:15:00  INFO     train — Validating dataset at: /project/dataset
2024-07-27 10:15:00  INFO     train —   normal  : 320 images
2024-07-27 10:15:00  INFO     train —   thyroid : 280 images
...
Epoch 20/20
19/19 [==============================] - 12s 614ms/step
  loss: 0.1842 - accuracy: 0.9312 - val_loss: 0.2104 - val_accuracy: 0.9067
...
=======================================================
  TRAINING COMPLETE — BEST METRICS
=======================================================
  Best Validation Accuracy : 0.9200  (92.00%)
  Validation Loss at Best  : 0.1987
  Best Epoch               : 18
=======================================================
```

---

## Running Predictions (CLI)

```bash
python src/predict.py --image path/to/nail_image.jpg
```

Optional flag:
```bash
python src/predict.py --image path/to/image.jpg --model saved_model/model.keras
```

### Sample Output

```
──────────────────────────────────────────────────
  AI NAIL SCREENING — RESULT
──────────────────────────────────────────────────
  Image      : nail_sample.jpg
  Prediction : Possible Thyroid Dysfunction Features
  Confidence : 94.82%
  Inference  : 38.4 ms
──────────────────────────────────────────────────
  [0] Normal Nail                                 5.18%   
  [1] Possible Thyroid Dysfunction Features      94.82% ◀
──────────────────────────────────────────────────

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠  DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This system is intended only for preliminary AI-based screening
and is NOT a medical diagnosis. ...
```

---

## Running the Desktop Application

```bash
python src/ui.py
```

### Application Walkthrough

1. **Upload Image** — Opens a file dialog. Select a JPEG or PNG nail photograph.
2. **Image Preview** — The selected image is displayed in the left panel.
3. **Analyze** — Runs the model. Results appear in the right panel within seconds.
4. **Results Panel** shows:
   - Predicted class with confidence score
   - Visual confidence progress bar
   - Per-class probability breakdown
   - Inference time
   - Mandatory disclaimer box
5. **Clear** — Resets the application for a new image.
6. **Exit** — Gracefully closes the window.

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open image |
| `Ctrl+L` | Clear |
| `Ctrl+Q` | Exit |

---

## Module Reference

### `src/model.py` — `NailScreeningModel`

| Method | Description |
|--------|-------------|
| `build()` | Constructs and compiles the Keras model |
| `unfreeze_top_layers(n)` | Unfreezes top N backbone layers for fine-tuning |
| `summary()` | Prints model architecture |
| `save(path)` | Saves model in `.keras` format |
| `NailScreeningModel.load(path)` | Static method — loads a saved model |

### `src/preprocess.py`

| Function | Description |
|----------|-------------|
| `load_image_cv2(path)` | Load image with OpenCV (BGR→RGB) |
| `resize_image(img, size)` | Resize via bilinear interpolation |
| `normalize_image(img)` | Scale pixels to [0, 1] |
| `preprocess_image_for_prediction(path)` | Full pipeline → batched array |
| `build_dataset(dir, ...)` | Build train/val `tf.data` datasets |
| `count_dataset_images(dir)` | Count images per class |

### `src/predict.py` — `Predictor`

| Method | Description |
|--------|-------------|
| `predict(image_path)` | Single-image inference → `PredictionResult` |
| `batch_predict(paths)` | List of images → list of results |
| `PredictionResult` | Dataclass: label, confidence, probabilities, timing |

### `src/utils.py`

| Function | Description |
|----------|-------------|
| `setup_logger(name, log_file)` | Create / retrieve a named logger |
| `load_image_as_array(path)` | Load image as RGB NumPy array |
| `resize_pil_image(img, max_size)` | Proportional PIL resize |
| `format_prediction_text(...)` | UI-ready prediction string |
| `format_duration(seconds)` | Human-readable duration |
| `save_prediction_log(...)` | Append result to CSV log |
| `read_prediction_log(path)` | Read all CSV log records |
| `ensure_dir(path)` | Create directory if missing |
| `model_exists()` | Check if default model file exists |
| `list_images_in_directory(dir)` | Recursively list image paths |

---

## Output Artefacts

| File | Description |
|------|-------------|
| `saved_model/model.keras` | Best trained model (Keras native format) |
| `assets/training_curves.png` | Accuracy & loss curves |
| `assets/training.log` | Full training log (timestamped) |
| `assets/ui.log` | Desktop application event log |
| `assets/prediction_log.csv` | CSV record of every desktop prediction |
| `assets/tensorboard_logs/` | TensorBoard event files |

To open TensorBoard:
```bash
tensorboard --logdir assets/tensorboard_logs
# Visit http://localhost:6006
```

---

## Extending the Project

### Add More Classes

1. Create a new sub-folder in `dataset/` (e.g., `dataset/onychomycosis/`).
2. Update `NUM_CLASSES` and `CLASS_NAMES` in `src/model.py`.
3. Retrain.

### Improve Accuracy

- Collect more labelled images (ideally ≥ 500 per class from clinical sources).
- Adjust `FINE_TUNE_EPOCHS` or `num_layers` in `unfreeze_top_layers()`.
- Try a larger backbone (MobileNetV2, EfficientNetB0).

### Export for Mobile (TensorFlow Lite)

```python
import tensorflow as tf
model = tf.keras.models.load_model("saved_model/model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("saved_model/model.tflite", "wb") as f:
    f.write(tflite_model)
```

### Generate a Confusion Matrix

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Collect y_true and y_pred from val_ds ...
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
print(confusion_matrix(y_true, y_pred))
```

---

## Known Limitations

- **Dataset dependency:** Model accuracy is entirely determined by the quality and
  size of the training dataset. A small or unbalanced dataset will produce
  unreliable results.
- **Visual features only:** The model analyses pixel patterns, not clinical
  biomarkers (TSH, T3, T4 levels). Visual nail features are non-specific.
- **Class imbalance:** If one class dominates the dataset, the model may become
  biased. Use class weighting or oversample the minority class.
- **Lighting / camera variation:** Nail photographs taken under different lighting
  or with different cameras may reduce generalisation.

---

## Future Improvements

| Priority | Improvement |
|----------|-------------|
| High | Collect a larger, clinically verified dataset with IRB approval |
| High | Add Grad-CAM visualisations to highlight discriminative nail regions |
| Medium | Implement 5-fold cross-validation for robust performance estimates |
| Medium | Integrate ONNX export for deployment-agnostic inference |
| Medium | Add a REST API layer (FastAPI) for web / mobile integration |
| Low | Build a web-based UI (React + Flask) as an alternative to Tkinter |
| Low | Support video-stream analysis (webcam feed with real-time inference) |
| Low | Multi-class extension (koilonychia, onycholysis, Beau's lines, etc.) |

---

## Academic References

1. World Health Organization. *Global Prevalence of Thyroid Disorders.* WHO, 2023.
2. Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018.
3. Howard, A., et al. "Searching for MobileNetV3." ICCV 2019.
4. LeCun, Y., et al. "Deep Learning." *Nature*, 521(7553):436–444, 2015.
5. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
6. Tan, M., Le, Q.V. "EfficientNet: Rethinking Model Scaling for CNNs." ICML 2019.

---

*For educational and research
purposes only. Not intended for clinical use.*
