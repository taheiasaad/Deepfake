# -*- coding: utf-8 -*-
"""
DeepFakeGuard — Streamlit Deployment App
=========================================
An end-to-end Streamlit interface for the DeepFakeGuard face-forgery
detector (MTCNN face detection + EfficientNetB0 binary classifier
trained with focal loss), matching the exact preprocessing pipeline
used during training so that inference results are reliable.

Features
--------
1. Image Analysis   — upload a single image, detect the face, classify
                       REAL / FAKE, show confidence + probability,
                       draw the detected face box(es), download a JSON report.
2. Video Analysis    — upload a video, uniformly sample frames, run
                       per-frame face detection + classification,
                       aggregate a video-level verdict, show a
                       frame-by-frame probability timeline, download
                       a JSON report and a per-frame CSV.
3. Batch Image Analysis — upload several images at once and get a
                       results table + downloadable CSV.
4. Model management  — load the trained .keras/.h5 model from a local
                       path or by uploading the file directly; adjust
                       the decision threshold and the number of frames
                       sampled per video from the sidebar.
5. About / How it works — architecture, methodology and a responsible
                       use disclaimer.

Run
---
    streamlit run deepfakeguard_streamlit_app.py

Requirements (see requirements.txt)
------------------------------------
    streamlit>=1.32
    tensorflow>=2.15
    opencv-python-headless
    mtcnn
    numpy
    pandas
    pillow
    matplotlib

Place the trained model at ``models/final_model.keras`` relative to this
file, or upload it from the sidebar at runtime.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call in the script.
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepFakeGuard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Heavy imports (OpenCV / TensorFlow / MTCNN) — imported inside guarded
# blocks so a missing package produces a clear in-app message instead of
# crashing the whole script before Streamlit can render anything.
# ----------------------------------------------------------------------------
IMPORT_ERROR: Optional[str] = None

try:
    import cv2
except Exception as e:  # pragma: no cover
    cv2 = None
    IMPORT_ERROR = f"OpenCV (opencv-python-headless) failed to import: {e}"

try:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as effnet_preprocess,  # noqa: F401  (kept alive for the model's internal Lambda layer)
    )
except Exception as e:  # pragma: no cover
    tf = None
    IMPORT_ERROR = (IMPORT_ERROR + " | " if IMPORT_ERROR else "") + f"TensorFlow failed to import: {e}"

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    Image = None
    IMPORT_ERROR = (IMPORT_ERROR + " | " if IMPORT_ERROR else "") + f"Pillow failed to import: {e}"

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # timeline chart will gracefully fall back to st.line_chart

# ----------------------------------------------------------------------------
# Constants — MUST mirror the training pipeline exactly, or inference
# results will silently diverge from what the model was trained on.
# ----------------------------------------------------------------------------
IMG_SIZE = 224
MIN_FACE_SIZE_PX = 40
FACE_MARGIN = 0.20
DEFAULT_THRESHOLD = 0.50
DEFAULT_NUM_FRAMES = 20
DEFAULT_MODEL_PATH = os.path.join("models", "final_model.keras")
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm", "m4v"]

# ----------------------------------------------------------------------------
# Styling — palette and type inspired by the DeepFakeGuard brand banner
# (near-black background, bold condensed orange wordmark, thin orange rule).
# ----------------------------------------------------------------------------
BRAND_ORANGE = "#FF6A3D"
BRAND_ORANGE_SOFT = "#FF8C61"
BRAND_BG = "#08080B"
BRAND_SURFACE = "#131318"
BRAND_BORDER = "#232329"
BRAND_TEXT = "#F5F5F7"
BRAND_TEXT_MUTED = "#9CA3AF"

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Anton&family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, p, span, div {{
        font-family: 'Poppins', sans-serif;
    }}

    /* ---- Hero header ---- */
    .dfg-hero {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 2.1rem 2.4rem;
        border-radius: 18px;
        background: radial-gradient(circle at 82% 25%, rgba(255,106,61,0.10) 0%, rgba(0,0,0,0) 55%), {BRAND_BG};
        border: 1px solid {BRAND_BORDER};
        margin-bottom: 1.6rem;
        overflow: hidden;
    }}
    .dfg-hero-text h1 {{
        font-family: 'Anton', sans-serif;
        font-weight: 400;
        font-size: 3rem;
        letter-spacing: 1px;
        color: {BRAND_ORANGE};
        margin: 0;
        line-height: 1;
        text-transform: uppercase;
    }}
    .dfg-hero-text p {{
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        color: {BRAND_TEXT};
        margin: 0.6rem 0 0.9rem 0;
        line-height: 1.35;
    }}
    .dfg-hero-rule {{
        width: 68px;
        height: 3px;
        background: {BRAND_ORANGE};
        border-radius: 2px;
    }}
    .dfg-hero-icon {{ flex-shrink: 0; opacity: 0.95; }}

    @media (max-width: 700px) {{
        .dfg-hero {{ flex-direction: column; text-align: center; }}
        .dfg-hero-text h1 {{ font-size: 2.1rem; }}
        .dfg-hero-icon {{ width: 70px; }}
    }}

    /* ---- Verdict banners ---- */
    .verdict-real, .verdict-fake, .verdict-error {{
        padding: 1rem 1.2rem;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.6rem;
        font-family: 'Poppins', sans-serif;
    }}
    .verdict-real  {{ background: #0f2318; color: #4ade80; border: 1px solid #22c55e; }}
    .verdict-fake  {{ background: #2a1414; color: #f87171; border: 1px solid #ef4444; }}
    .verdict-error {{ background: #2a2210; color: #fbbf24; border: 1px solid #eab308; }}

    .dfg-caption {{ color: {BRAND_TEXT_MUTED}; font-size: 0.85rem; }}

    /* ---- Tabs: orange active underline to match the brand accent ---- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [aria-selected="true"] {{ color: {BRAND_ORANGE} !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {BRAND_ORANGE} !important; }}
</style>
"""

DFG_SHIELD_SVG = f"""
<svg width="96" height="112" viewBox="0 0 110 130" xmlns="http://www.w3.org/2000/svg">
  <path d="M55 5 L100 20 L100 65 C100 95 80 115 55 125 C30 115 10 95 10 65 L10 20 Z"
        fill="none" stroke="#4b5563" stroke-width="2.5"/>
  <circle cx="55" cy="58" r="28" fill="none" stroke="#6b7280" stroke-width="1.4" opacity="0.85"/>
  <circle cx="55" cy="58" r="18" fill="none" stroke="#6b7280" stroke-width="1" opacity="0.5"/>
  <line x1="27" y1="58" x2="83" y2="58" stroke="{BRAND_ORANGE}" stroke-width="2"/>
  <circle cx="55" cy="58" r="3" fill="{BRAND_ORANGE}"/>
  <path d="M30 32 L30 24 L38 24" fill="none" stroke="{BRAND_ORANGE}" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M80 32 L80 24 L72 24" fill="none" stroke="{BRAND_ORANGE}" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M30 84 L30 92 L38 92" fill="none" stroke="{BRAND_ORANGE}" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M80 84 L80 92 L72 92" fill="none" stroke="{BRAND_ORANGE}" stroke-width="2.5" stroke-linecap="round"/>
</svg>
"""

# ==============================================================================
# Cached resource loaders
# ==============================================================================

@st.cache_resource(show_spinner="Loading face detector (MTCNN)...")
def load_face_detector():
    """Instantiate MTCNN once per session."""
    from mtcnn import MTCNN  # imported lazily so a missing package is caught here
    return MTCNN()


def _write_temp_copy(file_bytes: bytes, suffix: str) -> str:
    """Write bytes to a fresh temp file with the given suffix and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        # fdopen already closed fd on success; if write failed, best-effort cleanup
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise
    return path


def build_dfg_architecture(img_size: int = IMG_SIZE):
    """Rebuild the exact DeepFakeGuard architecture from the training script
    (EfficientNetB0 + GAP + Dropout + sigmoid head, with an internal Lambda
    that re-applies EfficientNet's preprocess_input). Used as a fallback when
    the provided file only contains weights (saved via `model.save_weights`)
    rather than a full model (saved via `model.save`)."""
    from tensorflow.keras import layers, models as keras_models
    from tensorflow.keras.applications import EfficientNetB0

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Lambda(lambda img: effnet_preprocess(img * 255.0))(inputs)
    base = EfficientNetB0(include_top=False, weights=None, input_tensor=x)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras_models.Model(inputs, outputs, name="deepfakeguard_balanced")


def _load_weights_robust(model: Any, weights_path: str) -> None:
    """
    Load weights into ``model``, handling the fact that Keras 3 has TWO
    different on-disk HDF5 weight layouts:
      - the legacy per-layer-group format (what plain `model.save_weights(x.h5)`
        produced pre-Keras-3, read via a "layer_names"-based reader), and
      - the newer native format, which Keras 3 only recognizes when the
        filename ends in ".weights.h5".

    Keras picks the reader purely from the filename suffix, so a perfectly
    valid new-format file simply named "final_model.h5" gets read by the
    legacy reader, which finds no matching structure and raises
    "... found 0 saved layers". We work around this by retrying with a
    temporary copy renamed to end in ".weights.h5" (and vice versa) whenever
    the first attempt fails.
    """
    last_err: Optional[Exception] = None
    try:
        model.load_weights(weights_path)
        return
    except Exception as e:
        last_err = e

    alt_path: Optional[str] = None
    try:
        if weights_path.endswith(".weights.h5"):
            alt_path = weights_path[: -len(".weights.h5")] + ".h5"
        elif weights_path.endswith(".h5"):
            alt_path = weights_path[: -len(".h5")] + ".weights.h5"
        else:
            alt_path = weights_path + ".weights.h5"

        with open(weights_path, "rb") as src, open(alt_path, "wb") as dst:
            dst.write(src.read())
        model.load_weights(alt_path)
        return
    except Exception as e2:
        last_err = RuntimeError(
            f"Tried both HDF5 weight layouts and neither loaded.\n"
            f"- As given ('{os.path.basename(weights_path)}'): {last_err}\n"
            f"- As the alternate format: {e2}"
        )
    finally:
        if alt_path and os.path.exists(alt_path):
            try:
                os.remove(alt_path)
            except OSError:
                pass

    raise last_err


def _try_load_weights_only(path: str) -> Any:
    """Build the known architecture fresh and load weights into it — this is
    what actually succeeds when the file is a weights-only .h5/.keras save."""
    model = build_dfg_architecture()
    _load_weights_robust(model, path)
    return model


def _try_load_keras_path(path: str) -> Any:
    """Attempt to load a Keras model from a concrete file path, trying a
    few argument combinations for compatibility across Keras/TF versions
    (the model's first layer is a Lambda re-applying EfficientNet's
    preprocess_input, which some Keras versions treat as 'unsafe' by default)."""
    last_err: Optional[Exception] = None
    for kwargs in (
        {"compile": False, "safe_mode": False},
        {"compile": False},
        {},
    ):
        try:
            return tf.keras.models.load_model(path, **kwargs)
        except TypeError:
            # This Keras/TF version doesn't accept one of the kwargs above — try the next combo.
            continue
        except Exception as e:
            last_err = e
            continue

    # The file loaded fine as HDF5/zip but has no model config → it's a
    # weights-only checkpoint. Rebuild the architecture and load into it.
    if last_err is not None and "No model config found" in str(last_err):
        try:
            return _try_load_weights_only(path)
        except Exception as weights_err:
            raise RuntimeError(
                f"File contains weights only, and rebuilding the known architecture "
                f"to load them also failed: {weights_err}"
            ) from weights_err

    raise last_err if last_err is not None else RuntimeError(f"Could not load model from {path}")


@st.cache_resource(show_spinner="Loading DeepFakeGuard model...")
def load_dfg_model(model_name: str, model_path: Optional[str] = None, file_bytes: Optional[bytes] = None):
    """
    Load the trained Keras model either from a filesystem path (model_path)
    or from raw bytes uploaded through the sidebar (file_bytes).

    ``model_name`` is only used to recover the real file extension (.keras
    vs .h5) — this matters because Keras dispatches to a different loader
    based on the extension, and a `.h5` file saved with a `.keras` name (or
    vice versa) will fail with a confusing "file not found / not a valid
    .keras zip" error even though the bytes are perfectly fine. To guard
    against exactly that mismatch, if the first attempt fails we retry once
    with the *other* extension.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is not available in this environment.")

    ext = os.path.splitext(model_name)[1].lower()
    if ext not in (".keras", ".h5"):
        ext = ".h5"
    alt_ext = ".h5" if ext == ".keras" else ".keras"

    temp_paths_to_clean: List[str] = []
    try:
        if file_bytes is not None:
            primary_path = _write_temp_copy(file_bytes, ext)
            temp_paths_to_clean.append(primary_path)
        else:
            if not model_path:
                raise FileNotFoundError("No model path was provided.")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            primary_path = model_path

        try:
            return _try_load_keras_path(primary_path)
        except Exception as first_err:
            # Retry once with the alternate extension, in case the file's
            # real format doesn't match its extension/name.
            if file_bytes is not None:
                alt_path = _write_temp_copy(file_bytes, alt_ext)
                temp_paths_to_clean.append(alt_path)
            else:
                alt_path = os.path.splitext(model_path)[0] + alt_ext
                if not os.path.exists(alt_path):
                    raise RuntimeError(
                        f"Could not load the model from '{primary_path}'. Error: {first_err}"
                    ) from first_err
            try:
                return _try_load_keras_path(alt_path)
            except Exception as second_err:
                raise RuntimeError(
                    f"Could not load the model.\n"
                    f"- Tried as '{ext}': {first_err}\n"
                    f"- Tried as '{alt_ext}': {second_err}\n"
                    "The file may be corrupted, incomplete, or saved in a format this "
                    "TensorFlow/Keras version can't read."
                ) from second_err
    finally:
        for p in temp_paths_to_clean:
            try:
                os.remove(p)
            except OSError:
                pass


def _read_json_maybe(path: Optional[str] = None, data_bytes: Optional[bytes] = None) -> Optional[Any]:
    """Best-effort JSON read from a path or raw bytes. Returns None on any failure
    (missing file, invalid JSON, etc.) instead of raising — these side files
    (config/threshold/metadata) are all optional extras."""
    try:
        if data_bytes is not None:
            return json.loads(data_bytes.decode("utf-8"))
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _extract_threshold_value(threshold_data: Any) -> Optional[float]:
    """Pull a float threshold out of threshold.json regardless of its exact shape
    (a bare number, {"threshold": 0.43}, {"FAKE_THRESHOLD": 0.43}, etc.)."""
    if threshold_data is None:
        return None
    if isinstance(threshold_data, (int, float)):
        return float(threshold_data)
    if isinstance(threshold_data, dict):
        for key in ("threshold", "FAKE_THRESHOLD", "fake_threshold", "optimal_threshold", "value", "best_threshold"):
            if key in threshold_data:
                try:
                    return float(threshold_data[key])
                except (TypeError, ValueError):
                    pass
        if len(threshold_data) == 1:
            try:
                return float(next(iter(threshold_data.values())))
            except (TypeError, ValueError):
                pass
    return None


@st.cache_resource(show_spinner="Loading DeepFakeGuard model bundle...")
def load_dfg_bundle(
    cache_key: str,
    dir_path: Optional[str] = None,
    weights_bytes: Optional[bytes] = None,
    weights_name: Optional[str] = None,
    config_bytes: Optional[bytes] = None,
    threshold_bytes: Optional[bytes] = None,
    metadata_bytes: Optional[bytes] = None,
) -> Tuple[Any, Optional[float], Dict[str, Any]]:
    """
    Load a model bundle that may be spread across several files, matching an
    export layout of: final_model.h5 (or .keras) + config.json + threshold.json
    + metadata.json — either read from a directory on disk, or from bytes
    uploaded individually through the sidebar.

    Returns (model, threshold_or_None, metadata_dict).

    ``cache_key`` only participates in the cache lookup (so re-uploading the
    exact same bytes is a cache hit) and is otherwise unused.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is not available in this environment.")

    cleanup_paths: List[str] = []
    try:
        if dir_path is not None:
            weights_path = None
            for candidate_name in ("final_model.h5", "final_model.keras"):
                candidate = os.path.join(dir_path, candidate_name)
                if os.path.exists(candidate):
                    weights_path = candidate
                    break
            if weights_path is None:
                raise FileNotFoundError(
                    f"No 'final_model.h5' or 'final_model.keras' found inside: {dir_path}"
                )
            config_data = _read_json_maybe(path=os.path.join(dir_path, "config.json"))
            threshold_data = _read_json_maybe(path=os.path.join(dir_path, "threshold.json"))
            metadata_data = _read_json_maybe(path=os.path.join(dir_path, "metadata.json"))
        else:
            if weights_bytes is None:
                raise FileNotFoundError("No model weights file was provided.")
            ext = os.path.splitext(weights_name or "final_model.h5")[1].lower()
            if ext not in (".h5", ".keras"):
                ext = ".h5"
            weights_path = _write_temp_copy(weights_bytes, ext)
            cleanup_paths.append(weights_path)
            config_data = _read_json_maybe(data_bytes=config_bytes)
            threshold_data = _read_json_maybe(data_bytes=threshold_bytes)
            metadata_data = _read_json_maybe(data_bytes=metadata_bytes)

        # --- try to reconstruct the exact architecture from config.json first ---
        model = None
        if config_data is not None:
            try:
                json_str = config_data if isinstance(config_data, str) else json.dumps(config_data)
                try:
                    model = tf.keras.models.model_from_json(
                        json_str, custom_objects={"effnet_preprocess": effnet_preprocess}
                    )
                except TypeError:
                    model = tf.keras.models.model_from_json(json_str)
            except Exception:
                model = None  # fall through to the known architecture below

        # --- fall back to the known DeepFakeGuard architecture ---
        if model is None:
            model = build_dfg_architecture()

        _load_weights_robust(model, weights_path)

        threshold_value = _extract_threshold_value(threshold_data)
        metadata_dict = metadata_data if isinstance(metadata_data, dict) else {}

        return model, threshold_value, metadata_dict
    finally:
        for p in cleanup_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ==============================================================================
# Core inference pipeline — mirrors the training script's functions exactly
# ==============================================================================

def _detect_faces(image_rgb: np.ndarray, detector) -> List[Dict[str, Any]]:
    """Run MTCNN and return every detected face as a normalized dict."""
    try:
        detections = detector.detect_faces(image_rgb)
    except Exception:
        return []

    faces = []
    for d in detections:
        x, y, w, h = d["box"]
        x1, y1 = max(0, x), max(0, y)
        faces.append({
            "x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h,
            "width": w, "height": h,
            "confidence": float(d["confidence"]),
        })
    return faces


def _select_primary_face(faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Face selection policy (identical to the training pipeline):
      1. Prefer the LARGEST face among those >= MIN_FACE_SIZE_PX.
      2. Otherwise fall back to the HIGHEST CONFIDENCE face.
      3. If no faces at all, return None.
    """
    if not faces:
        return None
    eligible = [f for f in faces if f["width"] >= MIN_FACE_SIZE_PX and f["height"] >= MIN_FACE_SIZE_PX]
    if eligible:
        return max(eligible, key=lambda f: f["width"] * f["height"])
    return max(faces, key=lambda f: f["confidence"])


def crop_face_with_margin(image_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                           margin: float = FACE_MARGIN) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    mx, my = int(box_w * margin), int(box_h * margin)
    nx1, ny1 = max(0, x1 - mx), max(0, y1 - my)
    nx2, ny2 = min(w, x2 + mx), min(h, y2 + my)
    return image_rgb[ny1:ny2, nx1:nx2]


def preprocess_face(image_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     target_size: int = IMG_SIZE) -> Optional[np.ndarray]:
    crop = crop_face_with_margin(image_rgb, x1, y1, x2, y2)
    if crop.size == 0:
        return None
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)


def draw_face_boxes(image_rgb: np.ndarray, faces: List[Dict[str, Any]],
                     primary: Optional[Dict[str, Any]]) -> np.ndarray:
    """Return a copy of the image with all detected faces boxed; the
    primary (selected) face is highlighted in green, others in gray."""
    vis = image_rgb.copy()
    for f in faces:
        is_primary = primary is not None and f is primary
        color = (34, 197, 94) if is_primary else (156, 163, 175)  # RGB
        thickness = 3 if is_primary else 2
        cv2.rectangle(vis, (f["x1"], f["y1"]), (f["x2"], f["y2"]), color, thickness)
        label = f"{'PRIMARY ' if is_primary else ''}{f['confidence']*100:.0f}%"
        cv2.putText(vis, label, (f["x1"], max(0, f["y1"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return vis


@dataclass
class ImageAnalysisResult:
    error: Optional[str] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    raw_fake_probability: Optional[float] = None
    threshold_used: Optional[float] = None
    num_faces_detected: int = 0
    faces: List[Dict[str, Any]] = field(default_factory=list)
    primary_face: Optional[Dict[str, Any]] = None
    face_crop: Optional[np.ndarray] = None
    boxed_image: Optional[np.ndarray] = None

    def to_report_dict(self) -> Dict[str, Any]:
        d = {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "raw_fake_probability": self.raw_fake_probability,
            "threshold_used": self.threshold_used,
            "num_faces_detected": self.num_faces_detected,
            "primary_face_box": (
                {k: self.primary_face[k] for k in ("x1", "y1", "x2", "y2", "confidence")}
                if self.primary_face else None
            ),
            "error": self.error,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        return d


def analyze_image_array(image_rgb: np.ndarray, model, detector, threshold: float) -> ImageAnalysisResult:
    faces = _detect_faces(image_rgb, detector)
    primary = _select_primary_face(faces)

    if primary is None:
        return ImageAnalysisResult(error="No face was detected in this image.", faces=faces)

    face_crop = preprocess_face(image_rgb, primary["x1"], primary["y1"], primary["x2"], primary["y2"])
    if face_crop is None or face_crop.size == 0:
        return ImageAnalysisResult(
            error="A face was detected but the crop was empty/invalid.",
            faces=faces, primary_face=primary,
        )

    batch = np.expand_dims(face_crop.astype(np.float32) / 255.0, axis=0)
    prob = float(model.predict(batch, verbose=0)[0][0])
    prediction = "FAKE" if prob >= threshold else "REAL"
    confidence = prob if prediction == "FAKE" else 1.0 - prob
    boxed = draw_face_boxes(image_rgb, faces, primary)

    return ImageAnalysisResult(
        prediction=prediction,
        confidence=round(confidence, 4),
        raw_fake_probability=round(prob, 4),
        threshold_used=round(float(threshold), 4),
        num_faces_detected=len(faces),
        faces=faces,
        primary_face=primary,
        face_crop=face_crop,
        boxed_image=boxed,
    )


def extract_frames(video_path: str, num_frames: int = DEFAULT_NUM_FRAMES) -> List[Tuple[int, np.ndarray]]:
    """Uniformly sample up to `num_frames` frames from a video file.
    Returns a list of (frame_index, rgb_frame) tuples."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = sorted(set(np.linspace(0, total_frames - 1, min(num_frames, total_frames)).astype(int)))
    extracted = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            extracted.append((int(idx), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return extracted


def analyze_video(video_path: str, model, detector, threshold: float,
                   num_frames: int = DEFAULT_NUM_FRAMES,
                   progress_callback=None) -> Dict[str, Any]:
    frames = extract_frames(video_path, num_frames=num_frames)
    if not frames:
        return {"error": "Could not read/extract frames from this video."}

    per_frame_rows = []
    for i, (frame_idx, frame_rgb) in enumerate(frames):
        faces = _detect_faces(frame_rgb, detector)
        primary = _select_primary_face(faces)
        row = {"frame_index": frame_idx, "face_detected": primary is not None,
               "fake_probability": None, "prediction": None}

        if primary is not None:
            face_crop = preprocess_face(frame_rgb, primary["x1"], primary["y1"], primary["x2"], primary["y2"])
            if face_crop is not None and face_crop.size > 0:
                batch = np.expand_dims(face_crop.astype(np.float32) / 255.0, axis=0)
                prob = float(model.predict(batch, verbose=0)[0][0])
                row["fake_probability"] = round(prob, 4)
                row["prediction"] = "FAKE" if prob >= threshold else "REAL"

        per_frame_rows.append(row)
        if progress_callback is not None:
            progress_callback((i + 1) / len(frames))

    per_frame_df = pd.DataFrame(per_frame_rows)
    valid = per_frame_df.dropna(subset=["fake_probability"])

    if valid.empty:
        return {
            "error": "No faces were detected in any of the sampled frames.",
            "per_frame": per_frame_df,
        }

    avg_prob = float(valid["fake_probability"].mean())
    prediction = "FAKE" if avg_prob >= threshold else "REAL"
    confidence = avg_prob if prediction == "FAKE" else 1.0 - avg_prob

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "average_fake_probability": round(avg_prob, 4),
        "threshold_used": round(float(threshold), 4),
        "frames_analyzed": len(frames),
        "frames_with_face": int(len(valid)),
        "per_frame": per_frame_df,
    }


# ==============================================================================
# UI helper functions
# ==============================================================================

def render_verdict_banner(prediction: Optional[str], confidence: Optional[float], error: Optional[str] = None):
    if error:
        st.markdown(f'<div class="verdict-error">⚠️ {error}</div>', unsafe_allow_html=True)
        return
    if prediction == "FAKE":
        st.markdown(
            f'<div class="verdict-fake">🚨 Predicted FAKE — {confidence*100:.1f}% confidence</div>',
            unsafe_allow_html=True,
        )
    elif prediction == "REAL":
        st.markdown(
            f'<div class="verdict-real">✅ Predicted REAL — {confidence*100:.1f}% confidence</div>',
            unsafe_allow_html=True,
        )


def render_probability_timeline(per_frame_df: pd.DataFrame, threshold: float):
    plot_df = per_frame_df.dropna(subset=["fake_probability"])
    if plot_df.empty:
        st.info("No frame had a detectable face, so no timeline can be shown.")
        return

    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        colors = ["#ef4444" if p >= threshold else "#22c55e" for p in plot_df["fake_probability"]]
        ax.bar(plot_df["frame_index"].astype(str), plot_df["fake_probability"], color=colors)
        ax.axhline(threshold, color="#374151", linestyle="--", linewidth=1, label=f"Threshold ({threshold:.2f})")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Fake probability")
        ax.set_ylim(0, 1)
        ax.set_title("Per-frame fake probability")
        ax.legend(loc="upper right")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.line_chart(plot_df.set_index("frame_index")["fake_probability"])


def json_download_button(data: Dict[str, Any], label: str, file_name: str):
    payload = json.dumps(data, indent=2, default=str)
    st.download_button(label=label, data=payload, file_name=file_name, mime="application/json")


def to_serializable_numpy_image(image_rgb: np.ndarray):
    """Convert an RGB numpy array to something Streamlit's st.image can render."""
    return image_rgb


# ==============================================================================
# Sidebar — model management & inference settings
# ==============================================================================

def render_sidebar() -> Tuple[Any, Any, float, int, Optional[str], Dict[str, Any]]:
    st.sidebar.markdown("## 🛡️ DeepFakeGuard")
    st.sidebar.caption("Explainable deepfake detection — MTCNN + EfficientNetB0")

    st.sidebar.markdown("### Model")
    model_source = st.sidebar.radio(
        "Load model from", ["Local path", "Upload file"], horizontal=True,
        help="Use 'Local path' if your model file(s) already sit on the same machine as this app — "
             "it's the most reliable option and avoids any temp-file copying.",
    )

    model = None
    model_error = None
    threshold_from_file: Optional[float] = None
    metadata: Dict[str, Any] = {}

    if model_source == "Local path":
        model_path = st.sidebar.text_input(
            "Model file OR model folder", value=DEFAULT_MODEL_PATH,
            help="Point this at final_model.h5 / final_model.keras directly, OR at the FOLDER "
                 "containing final_model.h5 + config.json + threshold.json + metadata.json — "
                 "the app will detect which one it is automatically.",
        )
        if st.sidebar.button("🔄 Load / reload model", use_container_width=True):
            load_dfg_model.clear()
            load_dfg_bundle.clear()
        if model_path:
            try:
                if os.path.isdir(model_path):
                    model, threshold_from_file, metadata = load_dfg_bundle(
                        cache_key=f"dir:{model_path}", dir_path=model_path,
                    )
                else:
                    model = load_dfg_model(os.path.basename(model_path), model_path=model_path)
            except Exception as e:
                model_error = str(e)
    else:
        uploaded_files = st.sidebar.file_uploader(
            "Upload model file(s)", type=["keras", "h5", "json"], accept_multiple_files=True,
            help="Upload just final_model.h5/.keras on its own, or select all four files together "
                 "(final_model.h5, config.json, threshold.json, metadata.json).",
        )
        if uploaded_files:
            weights_file = next(
                (f for f in uploaded_files if f.name.lower().endswith((".h5", ".keras"))), None
            )
            config_file = next((f for f in uploaded_files if f.name.lower() == "config.json"), None)
            threshold_file = next((f for f in uploaded_files if f.name.lower() == "threshold.json"), None)
            metadata_file = next((f for f in uploaded_files if f.name.lower() == "metadata.json"), None)

            names = ", ".join(f"`{f.name}`" for f in uploaded_files)
            st.sidebar.caption(f"Selected: {names}")

            if weights_file is None:
                model_error = "Upload at least the model weights file (final_model.h5 or final_model.keras)."
            else:
                try:
                    model, threshold_from_file, metadata = load_dfg_bundle(
                        cache_key="upload:" + "|".join(sorted(f.name for f in uploaded_files)),
                        weights_bytes=weights_file.getvalue(),
                        weights_name=weights_file.name,
                        config_bytes=config_file.getvalue() if config_file else None,
                        threshold_bytes=threshold_file.getvalue() if threshold_file else None,
                        metadata_bytes=metadata_file.getvalue() if metadata_file else None,
                    )
                except Exception as e:
                    model_error = str(e)

    if model is not None:
        st.sidebar.success("Model loaded ✅")
    elif model_error:
        st.sidebar.error(f"Model not loaded:\n{model_error}")
    else:
        st.sidebar.warning("No model loaded yet.")

    detector = None
    detector_error = None
    try:
        detector = load_face_detector()
        st.sidebar.success("Face detector ready ✅")
    except Exception as e:
        detector_error = str(e)
        st.sidebar.error(f"Face detector not available:\n{detector_error}")

    st.sidebar.markdown("### Inference settings")

    # If threshold.json supplied a value, seed the slider with it the first
    # time we see it (the user can still move the slider freely afterwards).
    if threshold_from_file is not None and st.session_state.get("dfg_last_auto_threshold") != threshold_from_file:
        st.session_state["dfg_threshold_widget"] = float(threshold_from_file)
        st.session_state["dfg_last_auto_threshold"] = threshold_from_file
        st.sidebar.info(f"📄 threshold.json → using {threshold_from_file:.3f} as the default threshold.")
    st.session_state.setdefault("dfg_threshold_widget", DEFAULT_THRESHOLD)

    threshold = st.sidebar.slider(
        "Decision threshold (fake probability ≥ threshold → FAKE)",
        min_value=0.05, max_value=0.95, step=0.01, key="dfg_threshold_widget",
    )
    num_frames = st.sidebar.slider(
        "Frames sampled per video", min_value=5, max_value=60, value=DEFAULT_NUM_FRAMES, step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "⚠️ This tool provides a probabilistic estimate, not proof of manipulation. "
        "See the 'About' tab for limitations."
    )

    return model, detector, threshold, num_frames, model_error, metadata


# ==============================================================================
# Tabs
# ==============================================================================

def render_image_tab(model, detector, threshold: float):
    st.subheader("🖼️ Image Analysis")
    st.write("Upload a photo containing a face. The app will detect the face, "
             "crop it the same way the model was trained on, and classify it.")

    uploaded = st.file_uploader("Upload an image", type=SUPPORTED_IMAGE_TYPES, key="image_uploader")
    if uploaded is None:
        return

    try:
        pil_img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not read this image file: {e}")
        return

    image_rgb = np.array(pil_img)

    col_in, col_out = st.columns(2)
    with col_in:
        st.image(image_rgb, caption="Uploaded image", use_container_width=True)

    if model is None or detector is None:
        st.warning("Load a model and make sure the face detector is ready (see sidebar) before analyzing.")
        return

    if st.button("🔍 Analyze image", type="primary", key="analyze_image_btn"):
        with st.spinner("Detecting face and running the classifier..."):
            try:
                result = analyze_image_array(image_rgb, model, detector, threshold)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.code(traceback.format_exc())
                return

        with col_out:
            if result.boxed_image is not None:
                st.image(result.boxed_image, caption="Detected face(s)", use_container_width=True)
            elif result.error:
                st.image(image_rgb, caption="No usable face found", use_container_width=True)

        render_verdict_banner(result.prediction, result.confidence, result.error)

        if not result.error:
            m1, m2, m3 = st.columns(3)
            m1.metric("Raw fake probability", f"{result.raw_fake_probability*100:.1f}%")
            m2.metric("Confidence", f"{result.confidence*100:.1f}%")
            m3.metric("Faces detected", result.num_faces_detected)

            st.progress(min(1.0, result.raw_fake_probability), text="Fake probability")

            if result.face_crop is not None:
                with st.expander("Show the exact crop fed to the model (224×224)"):
                    st.image(result.face_crop, width=224)

            json_download_button(
                result.to_report_dict(),
                "⬇️ Download JSON report",
                f"dfg_image_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
        else:
            st.info("Tip: make sure the face is clearly visible, well-lit and reasonably large in the frame "
                     f"(at least {MIN_FACE_SIZE_PX}px).")


def render_video_tab(model, detector, threshold: float, num_frames: int):
    st.subheader("🎥 Video Analysis")
    st.write("Upload a short video. The app uniformly samples frames, classifies each detected face, "
             "and aggregates a video-level verdict by averaging the per-frame fake probability.")

    uploaded = st.file_uploader("Upload a video", type=SUPPORTED_VIDEO_TYPES, key="video_uploader")
    if uploaded is None:
        return

    st.video(uploaded)

    if model is None or detector is None:
        st.warning("Load a model and make sure the face detector is ready (see sidebar) before analyzing.")
        return

    if st.button("🔍 Analyze video", type="primary", key="analyze_video_btn"):
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        progress_bar = st.progress(0.0, text="Analyzing frames...")
        try:
            result = analyze_video(
                tmp_path, model, detector, threshold, num_frames=num_frames,
                progress_callback=lambda frac: progress_bar.progress(frac, text=f"Analyzing frames... {int(frac*100)}%"),
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.code(traceback.format_exc())
            return
        finally:
            progress_bar.empty()
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        per_frame_df = result.get("per_frame")

        if result.get("error") and "prediction" not in result:
            render_verdict_banner(None, None, error=result["error"])
        else:
            render_verdict_banner(result["prediction"], result["confidence"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg. fake probability", f"{result['average_fake_probability']*100:.1f}%")
            m2.metric("Confidence", f"{result['confidence']*100:.1f}%")
            m3.metric("Frames analyzed", result["frames_analyzed"])
            m4.metric("Frames with a face", result["frames_with_face"])

        if per_frame_df is not None and not per_frame_df.empty:
            st.markdown("#### Frame-by-frame breakdown")
            render_probability_timeline(per_frame_df, threshold)
            st.dataframe(per_frame_df, use_container_width=True, hide_index=True)

            csv_bytes = per_frame_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download per-frame CSV", data=csv_bytes,
                file_name=f"dfg_video_frames_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        report = {k: v for k, v in result.items() if k != "per_frame"}
        if per_frame_df is not None:
            report["per_frame"] = per_frame_df.to_dict(orient="records")
        json_download_button(
            report, "⬇️ Download JSON report",
            f"dfg_video_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )


def render_batch_tab(model, detector, threshold: float):
    st.subheader("📊 Batch Image Analysis")
    st.write("Upload several images at once to classify them all and export a summary table.")

    uploaded_files = st.file_uploader(
        "Upload multiple images", type=SUPPORTED_IMAGE_TYPES, accept_multiple_files=True, key="batch_uploader",
    )
    if not uploaded_files:
        return

    if model is None or detector is None:
        st.warning("Load a model and make sure the face detector is ready (see sidebar) before analyzing.")
        return

    if st.button("🔍 Analyze all images", type="primary", key="analyze_batch_btn"):
        rows = []
        progress_bar = st.progress(0.0, text="Analyzing images...")
        for i, uploaded in enumerate(uploaded_files):
            try:
                pil_img = Image.open(uploaded).convert("RGB")
                image_rgb = np.array(pil_img)
                result = analyze_image_array(image_rgb, model, detector, threshold)
                rows.append({
                    "file_name": uploaded.name,
                    "prediction": result.prediction,
                    "confidence": result.confidence,
                    "raw_fake_probability": result.raw_fake_probability,
                    "num_faces_detected": result.num_faces_detected,
                    "error": result.error,
                })
            except Exception as e:
                rows.append({
                    "file_name": uploaded.name, "prediction": None, "confidence": None,
                    "raw_fake_probability": None, "num_faces_detected": 0, "error": str(e),
                })
            progress_bar.progress((i + 1) / len(uploaded_files), text=f"Analyzing images... {i+1}/{len(uploaded_files)}")
        progress_bar.empty()

        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        valid = results_df.dropna(subset=["prediction"])
        if not valid.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Images processed", len(results_df))
            c2.metric("Predicted FAKE", int((valid["prediction"] == "FAKE").sum()))
            c3.metric("Predicted REAL", int((valid["prediction"] == "REAL").sum()))

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download results CSV", data=csv_bytes,
            file_name=f"dfg_batch_results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


def render_about_tab(model_error: Optional[str], metadata: Optional[Dict[str, Any]] = None):
    st.subheader("ℹ️ About DeepFakeGuard")
    st.markdown(
        """
**Pipeline**
1. **Face detection** — MTCNN locates every face in the input; the largest face at least
   40×40px is selected (falling back to the highest-confidence detection if none qualify).
2. **Preprocessing** — the selected face is cropped with a 20% margin and resized to 224×224,
   exactly as during training.
3. **Classification** — an EfficientNetB0 backbone (ImageNet-pretrained, fine-tuned) with a
   global-average-pooling head and a sigmoid output estimates the probability that the face
   is a deepfake. The model was trained with class-balanced focal loss to handle label imbalance.
4. **Decision** — a face is labeled **FAKE** if its probability is ≥ the threshold set in the
   sidebar (the threshold was originally tuned on a validation set to maximize F1, and can be
   adjusted here to trade off precision vs. recall).
5. **Video verdict** — for videos, frames are uniformly sampled, each is scored independently,
   and the video-level probability is the mean of the per-frame probabilities.

**Limitations**
- Predictions are probabilistic, not proof of manipulation.
- Generalization to manipulation techniques or datasets not seen during training may be weaker
  than in-domain performance.
- Heavy compression, blur, extreme angles, or very small/occluded faces can reduce accuracy.
- No face detected → no verdict can be produced for that image/frame.
        """
    )
    if metadata:
        with st.expander("📄 Model metadata (from metadata.json)"):
            st.json(metadata)
    if model_error:
        st.error(f"Current model status: {model_error}")
    st.markdown(
        '<p class="dfg-caption">Use this tool responsibly. A "FAKE" result is a signal for '
        "further investigation, not a legal or definitive determination.</p>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# Main entry point
# ==============================================================================

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="dfg-hero">
            <div class="dfg-hero-text">
                <h1>DeepFakeGuard</h1>
                <p>AI-Powered Deepfake Detection<br>for Images &amp; Videos</p>
                <div class="dfg-hero-rule"></div>
            </div>
            <div class="dfg-hero-icon">{DFG_SHIELD_SVG}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if IMPORT_ERROR:
        st.error(
            "Some required packages failed to import, so this app cannot run yet:\n\n"
            f"{IMPORT_ERROR}\n\n"
            "Install the missing packages (see requirements.txt) and restart the app."
        )
        st.stop()

    model, detector, threshold, num_frames, model_error, metadata = render_sidebar()

    tab_image, tab_video, tab_batch, tab_about = st.tabs(
        ["🖼️ Image Analysis", "🎥 Video Analysis", "📊 Batch Analysis", "ℹ️ About"]
    )

    with tab_image:
        render_image_tab(model, detector, threshold)
    with tab_video:
        render_video_tab(model, detector, threshold, num_frames)
    with tab_batch:
        render_batch_tab(model, detector, threshold)
    with tab_about:
        render_about_tab(model_error, metadata)


if __name__ == "__main__":
    main()
