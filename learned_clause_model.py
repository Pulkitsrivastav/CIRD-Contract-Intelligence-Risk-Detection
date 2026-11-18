"""Locally trained TF-IDF classifier used as a conservative second opinion.

The model is trained only from the bundled, labelled NDA clause examples. It
never sends contract text outside the user's machine. Predictions are advisory:
low-confidence and ambiguous predictions are abstained from, and ML-only matches
are routed to human review rather than automatically marked Present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "offline_clause_classifier.joblib"
LABELS = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}
MIN_CONFIDENCE = 0.45
MIN_MARGIN = 0.22

_MODEL = None
_MODEL_ERROR = None


def _load_model():
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERROR:
        return None
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained offline classifier not found: {MODEL_PATH}")
        _MODEL = joblib.load(MODEL_PATH)
        return _MODEL
    except Exception as exc:  # retain rule-only operation if model loading fails
        _MODEL_ERROR = f"Local learned classifier unavailable: {exc}"
        return None


def predict_clause(text: str) -> dict[str, Any] | None:
    """Return a high-confidence clause-family prediction, otherwise abstain."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) < 45:
        return None
    model = _load_model()
    if model is None:
        return None
    try:
        probabilities = model.predict_proba([cleaned])[0]
        classes = list(model.classes_)
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        label, confidence = ranked[0]
        second = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        margin = float(confidence) - second
        if label not in LABELS or confidence < MIN_CONFIDENCE or margin < MIN_MARGIN:
            return None
        return {
            "clause_id": str(label),
            "confidence": round(float(confidence), 4),
            "margin": round(margin, 4),
            "model": "TF-IDF word/character n-gram Logistic Regression",
        }
    except Exception:
        return None


def model_status() -> dict[str, Any]:
    model = _load_model()
    if model is None:
        return {"available": False, "message": _MODEL_ERROR or "Local model unavailable"}
    metadata_path = ROOT / "data" / "offline_model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {"available": True, **metadata}
