"""Unified inference over every trained model + CLI fallback interface.

    python -m src.models.predict "my wifi is not working in Block C"
    python -m src.models.predict "fee deducted twice" --all          # every model side by side
    python -m src.models.predict --file tickets.txt --model tfidf_svm

The "best" model is picked by VALIDATION macro-F1 (never by test score, which would leak
test information into model selection).
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np

from src.utils import get_logger, load_config, load_json, resolve

log = get_logger("predict")

MODEL_DISPLAY = {
    "tfidf_nb": "Naive Bayes (TF-IDF)",
    "tfidf_lr": "Logistic Regression (TF-IDF)",
    "tfidf_svm": "Linear SVM (TF-IDF)",
    "sbert_lr": "SBERT + Logistic Regression",
    "distilbert": "DistilBERT (fine-tuned)",
    "bow_nb": "Naive Bayes (BoW)",
    "bow_lr": "Logistic Regression (BoW)",
}
MAIN_MODELS = ["tfidf_nb", "tfidf_lr", "tfidf_svm", "sbert_lr", "distilbert"]
COMPARISON_MODELS = ["bow_nb", "bow_lr"]

DEPARTMENT_OF = {
    "IT_WiFi": "IT Services",
    "LMS_Portal": "LMS / ERP Support Team",
    "Academics_Exams": "Examination Cell & Academic Office",
    "Fees_Accounts": "Accounts Office",
    "Hostel_Facilities": "Hostel & Estate Office",
}


def priority(text: str, cfg: dict | None = None) -> tuple[str, list[str]]:
    """Simple, transparent rule-based priority (not learned): keyword triggers."""
    cfg = cfg or load_config()
    t = text.lower()
    hits_hi = [k for k in cfg["priority"]["high_keywords"] if re.search(rf"\b{re.escape(k)}\b", t)]
    if hits_hi or text.count("!") >= 3:
        return "High", hits_hi or ["multiple exclamation marks"]
    hits_md = [k for k in cfg["priority"]["medium_keywords"] if re.search(rf"\b{re.escape(k)}\b", t)]
    if hits_md:
        return "Medium", hits_md
    return "Low", []


def _metrics_path(name: str, cfg: dict) -> Path:
    art = resolve(cfg["paths"]["artifacts_dir"])
    if name == "sbert_lr":
        return art / "sbert_lr" / "metrics.json"
    if name == "distilbert":
        return art / "distilbert_helpdesk" / "metrics.json"
    return art / "classical" / f"{name}_metrics.json"


def _ensure_distilbert_weights(path: Path, cfg: dict) -> None:
    """The 255 MB weights file is too big for GitHub (100 MB limit), so a cloud deployment
    downloads it once from a GitHub Release asset. Locally the file already exists and this does nothing."""
    weights = path / "model.safetensors"
    if weights.exists():
        return
    url = os.environ.get("DISTILBERT_WEIGHTS_URL") or cfg.get("deploy", {}).get("distilbert_weights_url")
    if not url:
        raise FileNotFoundError(weights)
    import urllib.request

    log.info("Downloading DistilBERT weights from %s ...", url)
    tmp = weights.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(weights)
    except Exception as e:  # network/404: run without DistilBERT instead of crashing the app
        tmp.unlink(missing_ok=True)
        raise FileNotFoundError(f"could not download {url}: {e}") from e


class HelpdeskPredictor:
    """Loads whichever trained models exist and predicts with any/all of them."""

    def __init__(self, cfg: dict | None = None, models: list[str] | None = None):
        self.cfg = cfg or load_config()
        self.labels: list[str] = self.cfg["labels"]
        self.art = resolve(self.cfg["paths"]["artifacts_dir"])
        self.models: dict[str, object] = {}
        from src.utils import MODEL_LOAD_LOCK

        # Load everything eagerly and sequentially under one lock (thread-safe for Streamlit),
        # instead of lazily during the first prediction.
        with MODEL_LOAD_LOCK:
            for name in models or MAIN_MODELS:
                try:
                    self._load(name)
                except FileNotFoundError as e:
                    log.warning("Model '%s' not available (%s)", name, e)
        if not self.models:
            raise FileNotFoundError("No trained models found in artifacts/ - run the training scripts first.")

    # ---------------------------------------------------------------- loading
    def _load(self, name: str) -> None:
        if name in ("tfidf_nb", "tfidf_lr", "tfidf_svm", "bow_nb", "bow_lr"):
            path = self.art / "classical" / f"{name}.joblib"
            if not path.exists():
                raise FileNotFoundError(path)
            self.models[name] = joblib.load(path)
        elif name == "sbert_lr":
            path = self.art / "sbert_lr" / "lr.joblib"
            if not path.exists():
                raise FileNotFoundError(path)
            from src.features.embeddings import SbertEncoder
            from src.preprocessing.text_preprocessor import TextPreprocessor

            meta = load_json(self.art / "sbert_lr" / "metrics.json")
            enc = SbertEncoder(meta["encoder"])
            _ = enc.model  # load the encoder now, not during the first prediction
            self.models[name] = {"clf": joblib.load(path), "enc": enc, "prep": TextPreprocessor(mode="transformer")}
        elif name == "distilbert":
            path = self.art / "distilbert_helpdesk"
            if not (path / "config.json").exists():
                raise FileNotFoundError(path)
            _ensure_distilbert_weights(path, self.cfg)
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from src.preprocessing.text_preprocessor import TextPreprocessor

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForSequenceClassification.from_pretrained(str(path), low_cpu_mem_usage=False).to(dev).eval()
            self.models[name] = {"tok": AutoTokenizer.from_pretrained(str(path)), "model": model, "device": dev,
                                 "max_len": load_json(path / "label_map.json")["max_len"],
                                 "prep": TextPreprocessor(mode="transformer")}
        else:
            raise ValueError(f"Unknown model {name}")

    @property
    def available(self) -> list[str]:
        return list(self.models)

    def val_scores(self) -> dict[str, float]:
        out = {}
        for name in self.models:
            try:
                out[name] = load_json(_metrics_path(name, self.cfg))["val"]["macro_f1"]
            except (FileNotFoundError, KeyError):
                pass
        return out

    @property
    def best_model(self) -> str:
        scores = self.val_scores()
        return max(scores, key=scores.get) if scores else self.available[0]

    # ---------------------------------------------------------------- inference
    def predict_proba(self, texts: list[str], name: str) -> np.ndarray:
        """Probabilities with columns in config label order."""
        m = self.models[name]
        if name == "sbert_lr":
            X = m["enc"].encode(m["prep"].transform(texts))
            p, classes = m["clf"].predict_proba(X), list(m["clf"].classes_)
        elif name == "distilbert":
            import torch

            clean = m["prep"].transform(texts)
            probs = []
            with torch.no_grad():
                for i in range(0, len(clean), 32):
                    enc = m["tok"](clean[i:i + 32], truncation=True, max_length=m["max_len"], padding=True,
                                   return_tensors="pt").to(m["device"])
                    probs.append(torch.softmax(m["model"](**enc).logits, dim=-1).cpu().numpy())
            p = np.vstack(probs)
            id2label = m["model"].config.id2label
            classes = [id2label[i] for i in range(p.shape[1])]
        else:
            p, classes = m.predict_proba(texts), list(m.classes_)
        order = [classes.index(l) for l in self.labels]
        return p[:, order]

    def predict(self, texts: list[str] | str, models: list[str] | None = None) -> dict[str, dict]:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if t and t.strip() else " " for t in texts]
        out = {}
        for name in models or self.available:
            t0 = time.perf_counter()
            p = self.predict_proba(texts, name)
            idx = p.argmax(axis=1)
            out[name] = {"labels": [self.labels[i] for i in idx], "confidence": p.max(axis=1).round(4).tolist(),
                         "proba": p, "latency_ms_per_ticket": 1000 * (time.perf_counter() - t0) / len(texts)}
        return out


def explain_linear(pipeline, text: str, label: str, top_n: int = 8) -> list[tuple[str, float]]:
    """Per-ticket explanation for a TF-IDF + Logistic Regression pipeline.

    contribution(word) = tfidf_weight(word in this ticket) x coefficient(word, class).
    Positive values pushed the ticket towards `label`.
    """
    prep, vec, clf = pipeline.named_steps["prep"], pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    X = vec.transform(prep.transform([text])).tocsr()
    k = list(clf.classes_).index(label)
    names = vec.get_feature_names_out()
    contrib = [(str(names[j]), float(X[0, j] * clf.coef_[k, j])) for j in X.indices if not str(names[j]).startswith("char:")]
    return sorted(contrib, key=lambda kv: kv[1], reverse=True)[:top_n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="ticket text")
    ap.add_argument("--file", help="text file, one ticket per line")
    ap.add_argument("--model", default=None, help=f"one of {MAIN_MODELS + COMPARISON_MODELS} (default: best on val)")
    ap.add_argument("--all", action="store_true", help="show every available model")
    ap.add_argument("--entities", action="store_true", help="also run entity extraction")
    args = ap.parse_args()
    if not args.text and not args.file:
        ap.error("give ticket text or --file")
    texts = [args.text] if args.text else [l.strip() for l in open(args.file, encoding="utf-8") if l.strip()]

    pred = HelpdeskPredictor(models=[args.model] if args.model and not args.all else None)
    names = pred.available if args.all else [args.model or pred.best_model]
    res = pred.predict(texts, names)
    extractor = None
    if args.entities:
        from src.ner.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
    for i, t in enumerate(texts):
        prio, why = priority(t, pred.cfg)
        print(f"\nTicket: {t}")
        for n in names:
            lab, conf = res[n]["labels"][i], res[n]["confidence"][i]
            print(f"  {MODEL_DISPLAY[n]:<32} -> {lab:<18} ({conf:.1%})  route to: {DEPARTMENT_OF[lab]}")
        print(f"  Priority: {prio}" + (f"  (triggers: {', '.join(why)})" if why else ""))
        if extractor:
            for e in extractor.extract(t):
                print(f"    [{e['label']}] {e['text']}")


if __name__ == "__main__":
    main()
