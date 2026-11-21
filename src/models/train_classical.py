"""Train classical baselines on TF-IDF (and BoW for comparison).

Models
------
* Multinomial Naive Bayes - generative, assumes word independence; very fast, strong on
  short texts, a classic text baseline.
* Logistic Regression - discriminative linear model; outputs calibrated-ish probabilities
  and interpretable per-class coefficients.
* Linear SVM - maximises the margin; usually the strongest sparse-text baseline. LinearSVC
  has no probabilities, so we wrap it in CalibratedClassifierCV (Platt/sigmoid scaling) to
  get a confidence score for the UI.
* BoW + NB / BoW + LR - same classifiers on raw counts, only for the representation
  comparison in the report.

Protocol (no leakage)
---------------------
Hyper-parameters are chosen with stratified 5-fold CV **on the training split only**
(the TF-IDF vocabulary is re-fitted inside every fold by the Pipeline).  The validation
split is then used to report/compare models; the test split is untouched until evaluate.py.

Each saved model is a full inference Pipeline: TextPreprocessor -> vectorizer -> classifier,
so raw ticket text can be passed directly at prediction time.

Usage:  python -m src.models.train_classical [--fast]
"""
from __future__ import annotations

import argparse
import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.load_dataset import load_processed
from src.evaluation.metrics import compute_metrics
from src.features.tfidf_features import build_bow, build_tfidf, top_features_per_class
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.utils import ensure_dir, get_logger, load_config, save_json, set_seed

log = get_logger("train_classical")


def _grid(raw: dict) -> dict:
    """YAML lists -> tuples where sklearn expects tuples (ngram_range)."""
    return {k: [tuple(v) if isinstance(v, list) else v for v in vals] for k, vals in raw.items()}


def _with_preprocessor(pipe: Pipeline, cfg: dict) -> Pipeline:
    """Prepend the (stateless) preprocessor so the saved model accepts raw text."""
    prep = TextPreprocessor(mode="classical", replace_ids=cfg["preprocessing"]["replace_ids"],
                            keep_words=cfg["preprocessing"]["keep_words"], spacy_model=cfg["preprocessing"]["spacy_model"])
    return Pipeline([("prep", prep)] + list(pipe.steps))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="3-fold CV and a smaller grid")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    tcfg, labels = cfg["tfidf"], cfg["labels"]
    out_dir = ensure_dir(f"{cfg['paths']['artifacts_dir']}/classical")

    train, val = load_processed("train", cfg), load_processed("val", cfg)
    Xtr, ytr = train["text_clean"].tolist(), train["category"].tolist()
    Xva, yva = val["text_clean"].tolist(), val["category"].tolist()
    log.info("train=%d  val=%d", len(Xtr), len(Xva))

    folds = 3 if args.fast else tcfg["cv_folds"]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["seed"])
    mf = tcfg["max_features"]

    specs = {
        "tfidf_nb": (Pipeline([("tfidf", build_tfidf(mf)), ("clf", MultinomialNB())]), _grid(tcfg["nb_grid"])),
        "tfidf_lr": (Pipeline([("tfidf", build_tfidf(mf)), ("clf", LogisticRegression(max_iter=3000))]), _grid(tcfg["lr_grid"])),
        "tfidf_svm": (Pipeline([("tfidf", build_tfidf(mf)), ("clf", LinearSVC())]), _grid(tcfg["svm_grid"])),
    }
    if args.fast:
        specs = {k: (p, {kk: vv[:2] for kk, vv in g.items()}) for k, (p, g) in specs.items()}

    summary = {}
    for name, (pipe, grid) in specs.items():
        t0 = time.perf_counter()
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
        search.fit(Xtr, ytr)
        best = search.best_estimator_
        if name == "tfidf_svm":
            # refit the best LinearSVC inside a calibrator so it can output probabilities
            C = search.best_params_["clf__C"]
            best = Pipeline([("tfidf", best.named_steps["tfidf"]),
                             ("clf", CalibratedClassifierCV(LinearSVC(C=C), cv=folds, method="sigmoid"))])
            best.fit(Xtr, ytr)
        train_time = time.perf_counter() - t0
        val_metrics = compute_metrics(yva, best.predict(Xva), labels)
        full = _with_preprocessor(best, cfg)
        joblib.dump(full, out_dir / f"{name}.joblib")
        info = {"model": name, "best_params": {k: str(v) for k, v in search.best_params_.items()},
                "cv_macro_f1": round(float(search.best_score_), 4), "val": val_metrics,
                "train_time_sec": round(train_time, 2), "n_candidates": len(search.cv_results_["params"]),
                "n_features": int(len(best.named_steps["tfidf"].get_feature_names_out()))}
        save_json(info, out_dir / f"{name}_metrics.json")
        summary[name] = info
        log.info("%-10s best=%s cvF1=%.4f valF1=%.4f (%.1fs)", name, info["best_params"], info["cv_macro_f1"],
                 val_metrics["macro_f1"], train_time)

    # --- BoW comparison models (fixed, sensible defaults; for the representation table) ---
    for name, clf in (("bow_nb", MultinomialNB(alpha=0.3)), ("bow_lr", LogisticRegression(C=1.0, max_iter=3000))):
        t0 = time.perf_counter()
        pipe = Pipeline([("tfidf", build_bow(mf, (1, 2))), ("clf", clf)]).fit(Xtr, ytr)
        info = {"model": name, "best_params": {"note": "fixed defaults (comparison only)"},
                "val": compute_metrics(yva, pipe.predict(Xva), labels), "train_time_sec": round(time.perf_counter() - t0, 2),
                "n_features": int(len(pipe.named_steps["tfidf"].get_feature_names_out()))}
        joblib.dump(_with_preprocessor(pipe, cfg), out_dir / f"{name}.joblib")
        save_json(info, out_dir / f"{name}_metrics.json")
        summary[name] = info
        log.info("%-10s valF1=%.4f", name, info["val"]["macro_f1"])

    lr = joblib.load(out_dir / "tfidf_lr.joblib")
    save_json(top_features_per_class(lr, 60),f"{cfg['paths']['reports_dir']}/top_features_lr.json")
    log.info("Saved models to %s and top LR features to reports/top_features_lr.json", out_dir)


if __name__ == "__main__":
    main()
