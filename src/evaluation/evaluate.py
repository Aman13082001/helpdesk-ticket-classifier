"""Evaluate ALL trained models on the SAME held-out test set + evaluate NER on the gold sample.

Outputs
  reports/model_comparison.csv            accuracy, macro/weighted P/R/F1, latency, size, train time
  reports/figures/model_comparison.png
  reports/figures/cm_<model>.png (+ _norm) confusion matrices
  reports/classification_reports.json     per-class precision/recall/F1 per model
  reports/test_predictions.csv            every model's prediction + confidence (for error analysis)
  reports/ner_evaluation.json / .csv      entity-level P/R/F1 per label (strict + lenient)
  reports/figures/top_features_lr.png     explainability

Usage:  python -m src.evaluation.evaluate
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import Counter

import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data.load_dataset import load_processed
from src.evaluation.metrics import compute_metrics, per_class_report
from src.evaluation.visualize import plot_confusion, plot_model_comparison, plot_top_features
from src.models.predict import COMPARISON_MODELS, MAIN_MODELS, MODEL_DISPLAY, HelpdeskPredictor, _metrics_path
from src.utils import dir_size_mb, ensure_dir, get_logger, load_config, load_json, resolve, save_json

log = get_logger("evaluate")


def model_artifact(name: str, cfg: dict) -> str:
    art = cfg["paths"]["artifacts_dir"]
    return {"sbert_lr": f"{art}/sbert_lr", "distilbert": f"{art}/distilbert_helpdesk"}.get(name, f"{art}/classical/{name}.joblib")


def model_size_mb(name: str, cfg: dict) -> float:
    """On-disk size; for SBERT+LR this includes the frozen encoder from the HF cache."""
    size = dir_size_mb(model_artifact(name, cfg))
    if name == "sbert_lr":
        from pathlib import Path

        repo = "models--" + cfg["embeddings"]["model_name"].replace("/", "--")
        cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / repo
        if cache.exists():
            # Windows without symlinks stores real files in snapshots/, otherwise in blobs/
            size += sum(f.stat().st_size for f in cache.rglob("*") if f.is_file() and not f.is_symlink()) / 1e6
    return size


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# ----------------------------------------------------------------------------- NER
def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def evaluate_ner(extractor, gold: dict) -> dict:
    """Entity-level P/R/F1 per label.

    strict : same label and identical surface text (case-insensitive)
    lenient: same label and one span contains the other (boundary disagreements forgiven,
             e.g. 'Dr. Priya Nair' vs 'Priya Nair')
    Only labels that appear in the gold annotation guideline are scored; other spaCy labels
    (ORG, GPE, ...) are reported separately as 'unscored'.
    """
    scored = set(gold["labels"]) | {l for t in gold["tickets"] for _, l in t["entities"]}
    counts = {m: {l: Counter() for l in scored} for m in ("strict", "lenient")}
    errors, unscored = [], Counter()
    for t in gold["tickets"]:
        preds = extractor.extract(t["text"])
        unscored.update(p["label"] for p in preds if p["label"] not in scored)
        preds = [(_norm(p["text"]), p["label"]) for p in preds if p["label"] in scored]
        golds = [(_norm(g), l) for g, l in t["entities"]]
        for mode in ("strict", "lenient"):
            match = (lambda g, p: g == p) if mode == "strict" else (lambda g, p: g in p or p in g)
            unmatched = list(preds)
            for g, gl in golds:
                hit = next((p for p in unmatched if p[1] == gl and match(g, p[0])), None)
                if hit:
                    counts[mode][gl]["tp"] += 1
                    unmatched.remove(hit)
                else:
                    counts[mode][gl]["fn"] += 1
                    if mode == "lenient":
                        errors.append({"text": t["text"], "type": "missed", "entity": g, "label": gl,
                                       "predicted_as": [p for p in preds if g in p[0] or p[0] in g]})
            for p, pl in unmatched:
                counts[mode][pl]["fp"] += 1
                if mode == "lenient":
                    errors.append({"text": t["text"], "type": "spurious", "entity": p, "label": pl})
    result = {}
    for mode, per in counts.items():
        rows, TP, FP, FN = {}, 0, 0, 0
        for l, c in sorted(per.items()):
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            rows[l] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
                       "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}
        P, R = TP / (TP + FP) if TP + FP else 0, TP / (TP + FN) if TP + FN else 0
        result[mode] = {"per_label": rows, "micro": {"precision": round(P, 3), "recall": round(R, 3),
                                                      "f1": round(2 * P * R / (P + R), 3) if P + R else 0.0}}
    result["errors"] = errors
    result["unscored_predicted_labels"] = dict(unscored)
    result["n_tickets"] = len(gold["tickets"])
    # ticket-level qualitative criterion from the success criteria: a ticket counts as
    # "correct" if every gold entity is found (lenient) and nothing spurious is added
    bad = {e["text"] for e in errors}
    result["tickets_fully_correct"] = sum(1 for t in gold["tickets"] if t["text"] not in bad)
    return result


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-ner", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    labels = cfg["labels"]
    reports, figs = ensure_dir(cfg["paths"]["reports_dir"]), ensure_dir(cfg["paths"]["figures_dir"])
    test = load_processed("test", cfg)
    texts, y = test["text"].tolist(), test["category"].tolist()
    log.info("Test set: %d tickets", len(texts))

    pred = HelpdeskPredictor(cfg, models=MAIN_MODELS + COMPARISON_MODELS)
    out = test[["ticket_id", "text", "category", "gen_kind"]].copy() if "gen_kind" in test else test[["ticket_id", "text", "category"]].copy()
    rows, reports_json = [], {}
    for name in pred.available:
        pred.predict(texts[:4], [name])  # warm-up (lazy model loading, CUDA init) - not timed
        t0 = time.perf_counter()
        res = pred.predict(texts, [name])[name]
        batch_ms = 1000 * (time.perf_counter() - t0) / len(texts)
        singles = []
        for t in texts[:25]:  # realistic single-ticket latency as seen in the UI
            s = time.perf_counter()
            pred.predict([t], [name])
            singles.append(1000 * (time.perf_counter() - s))
        m = compute_metrics(y, res["labels"], labels)
        reports_json[name] = per_class_report(y, res["labels"], labels)
        cm = confusion_matrix(y, res["labels"], labels=labels)
        plot_confusion(cm, labels, f"{MODEL_DISPLAY[name]} - counts", figs / f"cm_{name}.png")
        plot_confusion(cm, labels, f"{MODEL_DISPLAY[name]} - row-normalised", figs / f"cm_{name}_norm.png", normalize=True)
        info = load_json(_metrics_path(name, cfg))
        rows.append({"model": name, "display_name": MODEL_DISPLAY[name], "group": "main" if name in MAIN_MODELS else "bow_comparison",
                     **m, "val_macro_f1": info["val"]["macro_f1"],
                     "latency_ms_per_ticket": round(batch_ms, 3), "single_ticket_latency_ms": round(statistics.median(singles), 2),
                     "train_time_sec": info.get("train_time_sec"), "size_mb": round(model_size_mb(name, cfg), 2),
                     "inference_device": "cuda" if name in ("sbert_lr", "distilbert") and _cuda() else "cpu"})
        out[f"pred_{name}"], out[f"conf_{name}"] = res["labels"], res["confidence"]
        log.info("%-10s acc=%.4f macroF1=%.4f  %.2f ms/ticket", name, m["accuracy"], m["macro_f1"], batch_ms)

    # ---- unseen-phrasing test set (templates never seen in training) ----
    unseen_path = resolve(cfg["paths"]["processed_dir"]) / "test_unseen.csv"
    if unseen_path.exists():
        un = load_processed("test_unseen", cfg)
        un_out = un[["ticket_id", "text", "category", "gen_kind", "template_id"]].copy()
        log.info("Unseen-template test set: %d tickets", len(un))
        for row in rows:
            name = row["model"]
            res = pred.predict(un["text"].tolist(), [name])[name]
            m = compute_metrics(un["category"], res["labels"], labels)
            row["unseen_accuracy"], row["unseen_macro_f1"] = m["accuracy"], m["macro_f1"]
            row["generalisation_gap"] = round(row["macro_f1"] - m["macro_f1"], 4)
            reports_json[f"{name}__unseen"] = per_class_report(un["category"], res["labels"], labels)
            cm = confusion_matrix(un["category"], res["labels"], labels=labels)
            plot_confusion(cm, labels, f"{MODEL_DISPLAY[name]} - UNSEEN templates", figs / f"cm_{name}_unseen.png", normalize=True)
            un_out[f"pred_{name}"], un_out[f"conf_{name}"] = res["labels"], res["confidence"]
            log.info("%-10s UNSEEN acc=%.4f macroF1=%.4f", name, m["accuracy"], m["macro_f1"])
        un_out.to_csv(reports / "test_unseen_predictions.csv", index=False, encoding="utf-8")

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    df.to_csv(reports / "model_comparison.csv", index=False)
    out.to_csv(reports / "test_predictions.csv", index=False, encoding="utf-8")
    save_json(reports_json, reports / "classification_reports.json")
    plot_model_comparison(df, figs / "model_comparison.png")
    top = resolve(reports / "top_features_lr.json")
    if top.exists():
        plot_top_features(load_json(top), figs / "top_features_lr.png")
    cols = ["display_name", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "val_macro_f1",
            "latency_ms_per_ticket", "single_ticket_latency_ms", "train_time_sec", "size_mb"]
    cols += [c for c in ("unseen_accuracy", "unseen_macro_f1", "generalisation_gap") if c in df]
    log.info("\n%s", df[cols].to_string(index=False))

    if not args.skip_ner:
        from src.ner.entity_extractor import EntityExtractor

        ner = evaluate_ner(EntityExtractor(cfg), load_json(cfg["paths"]["ner_gold"]))
        save_json(ner, reports / "ner_evaluation.json")
        pd.DataFrame(ner["lenient"]["per_label"]).T.assign(strict_f1=pd.DataFrame(ner["strict"]["per_label"]).T["f1"]) \
            .to_csv(reports / "ner_evaluation.csv")
        log.info("NER micro (strict) %s | (lenient) %s | tickets fully correct: %d/%d", ner["strict"]["micro"],
                 ner["lenient"]["micro"], ner["tickets_fully_correct"], ner["n_tickets"])


if __name__ == "__main__":
    main()
