"""Static figures for the report (saved to reports/figures/).

Run standalone for the EDA + entity figures:
    python -m src.evaluation.visualize
Confusion matrices / model comparison are produced by evaluate.py using the helpers here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger, load_config, save_json

log = get_logger("visualize")
PALETTE = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#8ab17d", "#6d597a", "#f4a261"]


def _save(fig, path: str | Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("saved %s", Path(path).name)


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path: str | Path, normalize: bool = False) -> None:
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=11)
    thresh = data.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{data[i, j]:.2f}" if normalize else int(data[i, j]), ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_model_comparison(df: pd.DataFrame, path: str | Path) -> None:
    """Grouped bars: accuracy / macro-F1 / weighted-F1 per model + latency panel."""
    df = df.sort_values("macro_f1")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    y = np.arange(len(df))
    bars = [("accuracy", "Accuracy (test)"), ("macro_f1", "Macro-F1 (test)")]
    if "unseen_macro_f1" in df:
        bars.append(("unseen_macro_f1", "Macro-F1 (UNSEEN templates)"))
    else:
        bars.append(("weighted_f1", "Weighted-F1 (test)"))
    for k, (col, lab) in enumerate(bars):
        ax1.barh(y + (k - 1) * 0.27, df[col], height=0.27, label=lab, color=PALETTE[k])
    ax1.set_yticks(y, df["display_name"])
    lo = max(0.0, df[[c for c, _ in bars]].min().min() - 0.08)
    ax1.set_xlim(lo, 1.0); ax1.set_xlabel("Score")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    ax1.set_title("Model comparison: in-distribution test vs unseen phrasing")
    ax2.barh(y, df["latency_ms_per_ticket"], color=PALETTE[3])
    ax2.set_yticks(y, [""] * len(df)); ax2.set_xscale("log"); ax2.set_xlabel("ms / ticket (batch, log scale)")
    ax2.set_title("Inference latency")
    _save(fig, path)


def plot_class_distribution(df: pd.DataFrame, path: str | Path) -> None:
    counts = df.groupby(["split", "category"]).size().unstack(0)[["train", "val", "test"]]
    ax = counts.plot(kind="bar", figsize=(8, 4.5), color=PALETTE[:3])
    ax.set_title("Class distribution per split"); ax.set_ylabel("tickets"); ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    _save(ax.figure, path)


def plot_text_length(df: pd.DataFrame, path: str | Path) -> None:
    lens = df["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, (cat, g) in enumerate(df.assign(n=lens).groupby("category")):
        ax.hist(g["n"], bins=range(0, 70, 2), alpha=0.5, label=cat, color=PALETTE[k])
    ax.set_xlabel("words per ticket"); ax.set_ylabel("tickets"); ax.set_title("Ticket length distribution")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_top_terms(df: pd.DataFrame, path: str | Path, n: int = 10) -> dict:
    """Top terms per class by mean TF-IDF weight (fitted on TRAIN only)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(min_df=3, sublinear_tf=True)
    X = vec.fit_transform(df["text_clean"])
    names = vec.get_feature_names_out()
    cats = sorted(df["category"].unique())
    fig, axes = plt.subplots(1, len(cats), figsize=(3.2 * len(cats), 4.2))
    out = {}
    for ax, cat, color in zip(axes, cats, PALETTE):
        mean = np.asarray(X[(df["category"] == cat).values].mean(axis=0)).ravel()
        idx = mean.argsort()[::-1][:n]
        out[cat] = [str(names[i]) for i in idx]
        ax.barh(range(n)[::-1], mean[idx], color=color)
        ax.set_yticks(range(n)[::-1], names[idx], fontsize=9); ax.set_title(cat, fontsize=10); ax.set_xticks([])
    fig.suptitle("Top TF-IDF terms per category (train)")
    _save(fig, path)
    return out


def plot_top_features(top: dict[str, list], path: str | Path, n: int = 10) -> None:
    cats = list(top)
    fig, axes = plt.subplots(1, len(cats), figsize=(3.2 * len(cats), 4.2))
    for ax, cat, color in zip(axes, cats, PALETTE):
        # word features only: char n-grams are redundant fragments of the same words
        feats = [f for f in top[cat] if not f[0].startswith("char:")][:n]
        ax.barh(range(len(feats))[::-1], [w for _, w in feats], color=color)
        ax.set_yticks(range(len(feats))[::-1], [f for f, _ in feats], fontsize=8); ax.set_title(cat, fontsize=10)
    fig.suptitle("Most positive Logistic Regression word/bigram coefficients per class (explainability)")
    _save(fig, path)


def plot_entity_frequency(label_counts: dict[str, int], path: str | Path) -> None:
    items = sorted(label_counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([k for k, _ in items], [v for _, v in items], color=PALETTE[0])
    ax.set_xlabel("entity mentions (test set)"); ax.set_title("Extracted entity types")
    _save(fig, path)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = load_config()
    from src.data.load_dataset import load_processed

    figs = ensure_dir(cfg["paths"]["figures_dir"])
    parts = []
    for s in ("train", "val", "test"):
        parts.append(load_processed(s, cfg).assign(split=s))
    df = pd.concat(parts, ignore_index=True)
    plot_class_distribution(df, figs / "class_distribution.png")
    plot_text_length(df, figs / "text_length_hist.png")
    top_terms = plot_top_terms(df[df["split"] == "train"].reset_index(drop=True), figs / "top_terms_per_class.png")
    save_json(top_terms, f"{cfg['paths']['reports_dir']}/top_terms_per_class.json")

    from src.ner.entity_extractor import EntityExtractor, entity_stats

    test = df[df["split"] == "test"].reset_index(drop=True)
    ents = EntityExtractor(cfg).extract_many(test["text"].tolist())
    stats = entity_stats(ents, test["category"].tolist())
    save_json(stats, f"{cfg['paths']['reports_dir']}/entity_stats_test.json")
    plot_entity_frequency(stats["label_counts"], figs / "entity_frequency.png")


if __name__ == "__main__":
    main()
