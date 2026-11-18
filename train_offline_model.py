"""Re-train CIRD's local clause-family classifier from its bundled labelled data.

Run: python train_offline_model.py
No network access or API key is used.
"""
from __future__ import annotations
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clause_training_examples.jsonl"
MODEL = ROOT / "data" / "offline_clause_classifier.joblib"
META = ROOT / "data" / "offline_model_metadata.json"

rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
texts = [row["text"] for row in rows]
labels = [row["label"] for row in rows]
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)
features = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode")),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                              sublinear_tf=True, max_features=45000)),
])
model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(max_iter=1800, C=3.0, class_weight="balanced", random_state=42)),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
metrics = {
    "training_examples": len(X_train),
    "held_out_examples": len(X_test),
    "held_out_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "held_out_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    "classes": sorted(set(labels)),
    "seed": 42,
    "method": "TF-IDF word and character n-grams + Logistic Regression",
    "training_data_note": "Bundled manually labelled NDA clause paraphrase examples plus unrelated OTHER examples. Synthetic educational data; not a representative legal corpus.",
}
joblib.dump(model, MODEL, compress=3)
META.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"Trained local classifier: {len(X_train)} training examples; {len(X_test)} held-out examples")
print(f"Held-out accuracy: {metrics['held_out_accuracy']:.1%}")
print(classification_report(y_test, predictions, zero_division=0))
print(f"Saved: {MODEL}")
