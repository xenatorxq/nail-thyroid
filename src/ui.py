"""
ui.py
-----
Tkinter desktop application for the AI-Based Smart Nail Screening System.

Window layout:
  ┌─────────────────────────────────────────────────────┐
  │  Title bar                                           │
  ├──────────────────────────┬──────────────────────────┤
  │  Image Preview Panel     │  Results Panel           │
  │  (left)                  │  (right)                 │
  │                          │  - Prediction label      │
  │                          │  - Confidence bar        │
  │                          │  - Class probabilities   │
  │                          │  - Disclaimer            │
  ├──────────────────────────┴──────────────────────────┤
  │  Button bar: Upload | Analyze | Clear | Exit        │
  ├─────────────────────────────────────────────────────┤
  │  Status bar                                         │
  └─────────────────────────────────────────────────────┘

DISCLAIMER: This system is for educational and research purposes only.
Results do NOT constitute a medical diagnosis. Always consult a
qualified healthcare professional for any health concerns.
"""

import os
import sys
import threading
from pathlib import Path
from tkinter import (
    filedialog,
    messagebox,
    ttk,
)
from typing import Optional
import tkinter as tk

from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from predict import Predictor, PredictionResult
from utils import (
    format_prediction_text,
    get_log_csv_path,
    get_model_path,
    model_exists,
    resize_pil_image,
    save_prediction_log,
    setup_logger,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = setup_logger("ui", str(_PROJECT_ROOT / "assets" / "ui.log"))

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
APP_TITLE = "AI Nail Screening System"
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 650

# Colour palette
BG_COLOR = "#1e1e2e"          # dark background
PANEL_BG = "#2a2a3d"          # slightly lighter for panels
ACCENT = "#7c6af7"            # purple accent
ACCENT_HOVER = "#9a8bff"
TEXT_COLOR = "#e0e0f0"
SUBTEXT_COLOR = "#9090b0"
SUCCESS_COLOR = "#50fa7b"     # green — Normal result
WARNING_COLOR = "#ffb86c"     # orange — Abnormal result
ERROR_COLOR = "#ff5555"
BUTTON_BG = "#3a3a5c"
BUTTON_ACTIVE = "#4a4a7c"
DISCLAIMER_BG = "#2a2020"
DISCLAIMER_FG = "#ffb86c"

# Fonts (cross-platform)
FONT_TITLE = ("Helvetica", 18, "bold")
FONT_HEADING = ("Helvetica", 13, "bold")
FONT_BODY = ("Helvetica", 11)
FONT_SMALL = ("Helvetica", 9)
FONT_MONO = ("Courier", 10)

IMAGE_PREVIEW_SIZE = (350, 350)

DISCLAIMER_TEXT = (
    "⚠  DISCLAIMER\n"
    "This system is intended only for preliminary AI-based screening\n"
    "and is NOT a medical diagnosis. Results should not be used as a\n"
    "substitute for professional medical advice, diagnosis, or treatment.\n"
    "Always consult a qualified healthcare professional."
)

FILETYPES = [
    ("Image files", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"),
    ("JPEG files", "*.jpg *.jpeg"),
    ("PNG files", "*.png"),
    ("All files", "*.*"),
]


# ---------------------------------------------------------------------------
# Application class
# ---------------------------------------------------------------------------

class NailScreeningApp(tk.Tk):
    """
    Main Tkinter application class for the AI Nail Screening System.

    Inherits from tk.Tk so the class IS the root window.
    """

    def __init__(self) -> None:
        super().__init__()

        self._image_path: Optional[str] = None
        self._pil_image: Optional[Image.Image] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._predictor: Optional[Predictor] = None
        self._is_analyzing: bool = False

        self._setup_window()
        self._build_menu()
        self._build_layout()
        self._set_status("Ready. Upload a fingernail image to begin.", color=SUBTEXT_COLOR)

        # Pre-load model in background so first analysis is instant
        self._preload_model_async()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """Configure root window geometry, title, and style."""
        self.title(APP_TITLE)
        self.configure(bg=BG_COLOR)
        self.resizable(True, True)
        self.minsize(760, 540)

        # Centre the window on screen
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - WINDOW_WIDTH) // 2
        y = (screen_h - WINDOW_HEIGHT) // 2
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

        # Tkinter style
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=PANEL_BG,
            background=ACCENT,
            thickness=14,
        )
        style.configure("TFrame", background=BG_COLOR)

        self.protocol("WM_DELETE_WINDOW", self._on_exit)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        """Build the application menu bar."""
        menubar = tk.Menu(self, bg=PANEL_BG, fg=TEXT_COLOR, tearoff=False)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=False, bg=PANEL_BG, fg=TEXT_COLOR)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image…", command=self._upload_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Clear", command=self._clear, accelerator="Ctrl+L")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit, accelerator="Ctrl+Q")

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=False, bg=PANEL_BG, fg=TEXT_COLOR)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Disclaimer", command=self._show_disclaimer_dialog)

        # Keyboard shortcuts
        self.bind_all("<Control-o>", lambda _: self._upload_image())
        self.bind_all("<Control-l>", lambda _: self._clear())
        self.bind_all("<Control-q>", lambda _: self._on_exit())

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Build all top-level layout frames and sub-widgets."""
        # Title bar
        self._build_title_bar()

        # Main content: left panel (image) + right panel (results)
        content_frame = tk.Frame(self, bg=BG_COLOR)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        self._build_image_panel(content_frame)
        self._build_results_panel(content_frame)

        # Button bar
        self._build_button_bar()

        # Status bar
        self._build_status_bar()

    def _build_title_bar(self) -> None:
        """Top title header strip."""
        header = tk.Frame(self, bg=ACCENT, height=52)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🔬  " + APP_TITLE,
            font=FONT_TITLE,
            bg=ACCENT,
            fg="#ffffff",
            padx=16,
        ).pack(side=tk.LEFT, pady=8)

        tk.Label(
            header,
            text="Educational & Research Use Only",
            font=FONT_SMALL,
            bg=ACCENT,
            fg="#ddd8ff",
            padx=16,
        ).pack(side=tk.RIGHT, pady=8)

    def _build_image_panel(self, parent: tk.Frame) -> None:
        """Left panel — image preview."""
        left = tk.LabelFrame(
            parent,
            text="  Image Preview  ",
            font=FONT_BODY,
            bg=PANEL_BG,
            fg=TEXT_COLOR,
            bd=1,
            relief=tk.GROOVE,
            padx=8,
            pady=8,
        )
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6), pady=4)

        # Placeholder canvas for the image
        self._image_canvas = tk.Canvas(
            left,
            bg="#12121e",
            bd=0,
            highlightthickness=1,
            highlightbackground=ACCENT,
            width=IMAGE_PREVIEW_SIZE[0],
            height=IMAGE_PREVIEW_SIZE[1],
        )
        self._image_canvas.pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder()

        # Image path label
        self._image_path_var = tk.StringVar(value="No image selected")
        tk.Label(
            left,
            textvariable=self._image_path_var,
            font=FONT_SMALL,
            bg=PANEL_BG,
            fg=SUBTEXT_COLOR,
            wraplength=320,
        ).pack(pady=(6, 0))

    def _draw_placeholder(self) -> None:
        """Draw a placeholder graphic on the image canvas."""
        self._image_canvas.delete("all")
        w = self._image_canvas.winfo_reqwidth()
        h = self._image_canvas.winfo_reqheight()
        cx, cy = w // 2, h // 2

        # Dashed border rectangle
        self._image_canvas.create_rectangle(
            20, 20, w - 20, h - 20,
            outline=SUBTEXT_COLOR,
            dash=(6, 4),
            width=1,
        )
        self._image_canvas.create_text(
            cx, cy - 16,
            text="🖼",
            font=("Helvetica", 36),
            fill=SUBTEXT_COLOR,
        )
        self._image_canvas.create_text(
            cx, cy + 28,
            text="Click  Upload Image  to begin",
            font=FONT_BODY,
            fill=SUBTEXT_COLOR,
        )

    def _build_results_panel(self, parent: tk.Frame) -> None:
        """Right panel — prediction results."""
        right = tk.LabelFrame(
            parent,
            text="  Analysis Result  ",
            font=FONT_BODY,
            bg=PANEL_BG,
            fg=TEXT_COLOR,
            bd=1,
            relief=tk.GROOVE,
            padx=12,
            pady=10,
        )
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=4)

        # --- Prediction label ---
        tk.Label(
            right, text="Prediction", font=FONT_HEADING,
            bg=PANEL_BG, fg=TEXT_COLOR,
        ).pack(anchor=tk.W, pady=(4, 2))

        self._prediction_var = tk.StringVar(value="—")
        self._prediction_label = tk.Label(
            right,
            textvariable=self._prediction_var,
            font=("Helvetica", 13, "bold"),
            bg=PANEL_BG,
            fg=SUBTEXT_COLOR,
            wraplength=340,
            justify=tk.LEFT,
        )
        self._prediction_label.pack(anchor=tk.W, pady=(0, 8))

        # --- Confidence ---
        tk.Label(
            right, text="Confidence", font=FONT_HEADING,
            bg=PANEL_BG, fg=TEXT_COLOR,
        ).pack(anchor=tk.W, pady=(0, 2))

        self._confidence_var = tk.StringVar(value="—")
        tk.Label(
            right,
            textvariable=self._confidence_var,
            font=("Helvetica", 22, "bold"),
            bg=PANEL_BG,
            fg=ACCENT,
        ).pack(anchor=tk.W)

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            right,
            variable=self._progress_var,
            maximum=100,
            mode="determinate",
            style="Horizontal.TProgressbar",
            length=320,
        )
        self._progress_bar.pack(anchor=tk.W, pady=(4, 12))

        # --- Class probabilities ---
        tk.Label(
            right, text="Class Probabilities", font=FONT_HEADING,
            bg=PANEL_BG, fg=TEXT_COLOR,
        ).pack(anchor=tk.W, pady=(0, 4))

        self._prob_frame = tk.Frame(right, bg=PANEL_BG)
        self._prob_frame.pack(anchor=tk.W, fill=tk.X)
        self._prob_labels: list[tk.Label] = []
        self._prob_bars: list[ttk.Progressbar] = []

        class_names = ["Normal Nail", "Possible Thyroid Dysfunction Features"]
        for cls_name in class_names:
            row = tk.Frame(self._prob_frame, bg=PANEL_BG)
            row.pack(fill=tk.X, pady=2)
            tk.Label(
                row, text=cls_name, font=FONT_SMALL,
                bg=PANEL_BG, fg=TEXT_COLOR, width=36, anchor=tk.W,
            ).pack(side=tk.LEFT)
            pbar = ttk.Progressbar(
                row, maximum=100, mode="determinate",
                style="Horizontal.TProgressbar", length=140,
            )
            pbar.pack(side=tk.LEFT, padx=(4, 4))
            lbl = tk.Label(row, text="0.00%", font=FONT_SMALL, bg=PANEL_BG, fg=SUBTEXT_COLOR)
            lbl.pack(side=tk.LEFT)
            self._prob_bars.append(pbar)
            self._prob_labels.append(lbl)

        # Spacer
        tk.Frame(right, bg=PANEL_BG, height=10).pack()

        # --- Inference time ---
        self._inference_time_var = tk.StringVar(value="")
        tk.Label(
            right,
            textvariable=self._inference_time_var,
            font=FONT_SMALL,
            bg=PANEL_BG,
            fg=SUBTEXT_COLOR,
        ).pack(anchor=tk.W)

        # --- Disclaimer box ---
        disclaimer_frame = tk.Frame(right, bg=DISCLAIMER_BG, bd=1, relief=tk.GROOVE)
        disclaimer_frame.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            disclaimer_frame,
            text=DISCLAIMER_TEXT,
            font=FONT_SMALL,
            bg=DISCLAIMER_BG,
            fg=DISCLAIMER_FG,
            wraplength=340,
            justify=tk.LEFT,
            padx=8,
            pady=8,
        ).pack()

    def _build_button_bar(self) -> None:
        """Button bar with Upload, Analyze, Clear, Exit."""
        bar = tk.Frame(self, bg=BG_COLOR)
        bar.pack(fill=tk.X, padx=12, pady=(0, 6))

        btn_kwargs = dict(
            font=FONT_BODY,
            bg=BUTTON_BG,
            fg=TEXT_COLOR,
            activebackground=BUTTON_ACTIVE,
            activeforeground=TEXT_COLOR,
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            bd=0,
        )

        self._upload_btn = tk.Button(
            bar, text="📂  Upload Image", command=self._upload_image, **btn_kwargs
        )
        self._upload_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._analyze_btn = tk.Button(
            bar,
            text="🔍  Analyze",
            command=self._analyze,
            bg=ACCENT,
            fg="#ffffff",
            activebackground=ACCENT_HOVER,
            activeforeground="#ffffff",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            bd=0,
            state=tk.DISABLED,
        )
        self._analyze_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._clear_btn = tk.Button(
            bar, text="🗑  Clear", command=self._clear, **btn_kwargs
        )
        self._clear_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            bar, text="✕  Exit", command=self._on_exit, **btn_kwargs
        ).pack(side=tk.RIGHT)

    def _build_status_bar(self) -> None:
        """Bottom status bar."""
        bar = tk.Frame(self, bg="#14141f", height=28)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        self._status_var = tk.StringVar(value="")
        self._status_label = tk.Label(
            bar,
            textvariable=self._status_var,
            font=FONT_SMALL,
            bg="#14141f",
            fg=SUBTEXT_COLOR,
            anchor=tk.W,
            padx=10,
        )
        self._status_label.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            bar,
            text="Educational & Research Use Only  |  Not a Medical Device",
            font=FONT_SMALL,
            bg="#14141f",
            fg=SUBTEXT_COLOR,
            padx=10,
        ).pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------------
    # Business logic
    # ------------------------------------------------------------------

    def _preload_model_async(self) -> None:
        """Load the Keras model on a background thread to avoid UI freeze."""
        def _load() -> None:
            if not model_exists():
                self._set_status(
                    "⚠  No trained model found. Run  python src/train.py  first.",
                    color=WARNING_COLOR,
                )
                return
            try:
                self._predictor = Predictor(model_path=get_model_path())
                # Trigger lazy model load
                _ = self._predictor._load_model()  # noqa: SLF001 (intentional)
                self._set_status("Model loaded and ready.", color=SUCCESS_COLOR)
            except Exception as exc:
                logger.error("Model pre-load failed: %s", exc)
                self._set_status(f"Model load error: {exc}", color=ERROR_COLOR)

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def _upload_image(self) -> None:
        """Open a file dialog to select a nail image."""
        path = filedialog.askopenfilename(
            title="Select Nail Image",
            filetypes=FILETYPES,
            initialdir=str(Path.home()),
        )
        if not path:
            return

        self._image_path = path
        self._set_status(f"Image loaded: {os.path.basename(path)}", color=TEXT_COLOR)
        self._image_path_var.set(path)
        self._load_preview(path)
        self._reset_results()
        self._analyze_btn.config(state=tk.NORMAL)

    def _load_preview(self, image_path: str) -> None:
        """Display the selected image in the preview canvas."""
        try:
            pil_img = Image.open(image_path).convert("RGB")
            # Resize proportionally to fit canvas
            canvas_w = self._image_canvas.winfo_width() or IMAGE_PREVIEW_SIZE[0]
            canvas_h = self._image_canvas.winfo_height() or IMAGE_PREVIEW_SIZE[1]
            pil_img.thumbnail((canvas_w - 8, canvas_h - 8), Image.LANCZOS)

            self._photo_image = ImageTk.PhotoImage(pil_img)
            self._image_canvas.delete("all")
            cx = (canvas_w) // 2
            cy = (canvas_h) // 2
            self._image_canvas.create_image(cx, cy, anchor=tk.CENTER, image=self._photo_image)
        except Exception as exc:
            logger.error("Preview load failed: %s", exc)
            self._set_status(f"Cannot display image: {exc}", color=ERROR_COLOR)

    def _analyze(self) -> None:
        """Run inference on the uploaded image (non-blocking via thread)."""
        if self._is_analyzing:
            return
        if not self._image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        if self._predictor is None:
            messagebox.showerror(
                "Model Not Loaded",
                "The model is not loaded yet.\n\n"
                "If you have not trained the model, run:\n"
                "    python src/train.py",
            )
            return

        self._is_analyzing = True
        self._analyze_btn.config(state=tk.DISABLED, text="⏳  Analyzing…")
        self._set_status("Analyzing image, please wait…", color=ACCENT)

        def _run() -> None:
            try:
                result: PredictionResult = self._predictor.predict(self._image_path)  # type: ignore[arg-type]
                self.after(0, lambda: self._display_result(result))
            except Exception as exc:
                logger.error("Inference error: %s", exc)
                self.after(0, lambda: self._show_inference_error(str(exc)))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _display_result(self, result: PredictionResult) -> None:
        """Update the results panel with prediction data (called from main thread)."""
        self._is_analyzing = False
        self._analyze_btn.config(state=tk.NORMAL, text="🔍  Analyze")

        color = WARNING_COLOR if result.is_abnormal else SUCCESS_COLOR

        # Prediction text
        icon = "⚠" if result.is_abnormal else "✓"
        self._prediction_var.set(f"{icon}  {result.class_label}")
        self._prediction_label.config(fg=color)

        # Confidence
        self._confidence_var.set(result.confidence_pct)
        self._progress_var.set(result.confidence * 100)

        # Per-class probability bars
        for i, (bar, label) in enumerate(zip(self._prob_bars, self._prob_labels)):
            prob = float(result.probabilities[i])
            bar["value"] = prob * 100
            label.config(text=f"{prob * 100:.2f}%")

        # Inference time
        self._inference_time_var.set(
            f"Inference time: {result.inference_time_ms:.1f} ms"
        )

        status_text = (
            f"Result: {result.class_label}  |  Confidence: {result.confidence_pct}  "
            f"|  For educational purposes only — NOT a medical diagnosis"
        )
        self._set_status(status_text, color=color)

        # Log to CSV
        try:
            save_prediction_log(
                log_path=get_log_csv_path(),
                image_path=result.image_path,
                class_label=result.class_label,
                confidence=result.confidence,
                class_index=result.class_index,
            )
        except Exception as exc:
            logger.warning("Could not write prediction log: %s", exc)

    def _show_inference_error(self, message: str) -> None:
        """Show inference error and reset UI."""
        self._is_analyzing = False
        self._analyze_btn.config(state=tk.NORMAL, text="🔍  Analyze")
        self._set_status(f"Error during analysis: {message}", color=ERROR_COLOR)
        messagebox.showerror("Analysis Error", f"An error occurred:\n\n{message}")

    def _clear(self) -> None:
        """Reset the application to its initial state."""
        self._image_path = None
        self._pil_image = None
        self._photo_image = None

        self._draw_placeholder()
        self._reset_results()
        self._image_path_var.set("No image selected")
        self._analyze_btn.config(state=tk.DISABLED)
        self._set_status("Cleared. Upload a new image to begin.", color=SUBTEXT_COLOR)

    def _reset_results(self) -> None:
        """Clear all result widgets back to their default empty state."""
        self._prediction_var.set("—")
        self._prediction_label.config(fg=SUBTEXT_COLOR)
        self._confidence_var.set("—")
        self._progress_var.set(0.0)
        self._inference_time_var.set("")
        for bar, label in zip(self._prob_bars, self._prob_labels):
            bar["value"] = 0.0
            label.config(text="0.00%")

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _set_status(self, message: str, color: str = SUBTEXT_COLOR) -> None:
        """
        Update the status bar message.

        Parameters
        ----------
        message : str
            Status message to display.
        color : str
            Foreground colour (hex or named colour).
        """
        self._status_var.set(message)
        self._status_label.config(fg=color)

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def _show_about(self) -> None:
        """Display the About dialog."""
        messagebox.showinfo(
            "About — " + APP_TITLE,
            "AI-Based Smart Nail Screening System\n"
            "for Thyroid Dysfunction Detection\n\n"
            "Version : 1.0\n"
            "Purpose : Educational / Research\n"
            "Model   : MobileNetV3Small (Transfer Learning)\n"
            "Classes : Normal Nail | Possible Thyroid Dysfunction Features\n\n"
            "⚠  This is NOT a certified medical device.\n"
            "Results are AI-based screening estimates only.\n"
            "Always consult a qualified healthcare professional.",
        )

    def _show_disclaimer_dialog(self) -> None:
        """Display the full disclaimer in a dialog."""
        messagebox.showwarning(
            "Disclaimer",
            "DISCLAIMER — IMPORTANT\n\n"
            "This system is designed for EDUCATIONAL and RESEARCH "
            "purposes ONLY.\n\n"
            "• It does NOT diagnose any medical condition.\n"
            "• It does NOT replace professional medical advice.\n"
            "• Predictions are AI-based screening estimates.\n"
            "• Accuracy depends on training data quality.\n\n"
            "Always consult a licensed healthcare professional\n"
            "for any health concerns related to your thyroid or nails.",
        )

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def _on_exit(self) -> None:
        """Gracefully exit the application."""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            logger.info("Application closed by user.")
            self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Tkinter desktop application."""
    app = NailScreeningApp()
    app.mainloop()


if __name__ == "__main__":
    main()
