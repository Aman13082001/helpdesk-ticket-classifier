"""SBERT sentence embeddings + Logistic Regression ("SBERT + LR").

This sits between the classical and the fine-tuned transformer models: the transformer is
used as a FROZEN feature extractor (no gradient updates), and a cheap linear classifier is
trained on top.  It shows how much a pretrained semantic representation helps without the
cost of fine-tuning.

Usage:  python -m src.models.train_embedding_lr
"""
from __future__ import annotations

import argparse
import time

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data.load_dataset import load_processed
from src.evaluation.metrics import compute_metrics
from src.features.embeddings import SbertEncoder, cached_embeddings, plot_embeddings_2d
from src.utils import ensure_dir, get_logger, load_config, save_json, set_seed

log = get_logger("train_embedding_lr")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-plot", action="store_true", help="skip the UMAP/t-SNE figure")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    ecfg, labels = cfg["embeddings"], cfg["labels"]
    out_dir = ensure_dir(f"{cfg['paths']['artifacts_dir']}/sbert_lr")
    cache = f"{cfg['paths']['artifacts_dir']}/embeddings"

    enc = SbertEncoder(ecfg["model_name"], ecfg["batch_size"])
    train, val = load_processed("train", cfg), load_processed("val", cfg)
    t0 = time.perf_counter()
    Xtr = cached_embeddings(train["text_light"].tolist(), "train", enc, cache)
    Xva = cached_embeddings(val["text_light"].tolist(), "val", enc, cache)
    embed_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    search = GridSearchCV(LogisticRegression(max_iter=3000), {"C": ecfg["lr_C_grid"]}, scoring="f1_macro", n_jobs=-1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]))
    search.fit(Xtr, train["category"])
    fit_time = time.perf_counter() - t1
    clf = search.best_estimator_
    val_metrics = compute_metrics(val["category"], clf.predict(Xva), labels)

    joblib.dump(clf, out_dir / "lr.joblib")
    info = {"model": "sbert_lr", "encoder": ecfg["model_name"], "embedding_dim": int(Xtr.shape[1]),
            "best_params": search.best_params_, "cv_macro_f1": round(float(search.best_score_), 4), "val": val_metrics,
            "train_time_sec": round(embed_time + fit_time, 2), "embed_time_sec": round(embed_time, 2)}
    save_json(info, out_dir / "metrics.json")
    log.info("SBERT+LR best=%s cvF1=%.4f valF1=%.4f", search.best_params_, info["cv_macro_f1"], val_metrics["macro_f1"])

    if not args.no_plot:
        fig = ensure_dir(cfg["paths"]["figures_dir"]) / "embeddings_2d.png"
        method = plot_embeddings_2d(Xtr, train["category"].tolist(), fig, "Training tickets - SBERT all-MiniLM-L6-v2", cfg["seed"])
        log.info("Saved %s projection -> %s", method, fig)


if __name__ == "__main__":
    main()
