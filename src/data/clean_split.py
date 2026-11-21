"""Inspect -> clean -> stratified 70/15/15 split -> cache preprocessed text.

Steps
1. report raw size, missing values, duplicates, class balance
2. drop empty / missing text, strip whitespace, drop exact duplicate texts
3. hold out ~15% of the phrasing templates per class as an extra "test_unseen" set, then a
   stratified train/val/test split of the rest (fixed seed) - test sets are never used for
   training/tuning
4. add two cached columns so later scripts do not recompute spaCy:
     text_clean  (classical mode, for BoW/TF-IDF)
     text_light  (transformer mode, for SBERT / DistilBERT)
   The preprocessor is stateless (learns nothing), so this cannot leak test information;
   all *fitted* statistics (TF-IDF vocabulary/IDF, classifiers) are fitted on train only.

Usage:  python -m src.data.clean_split
"""
from __future__ import annotations

import argparse
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_dataset import load_synthetic
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.utils import ensure_dir, get_logger, load_config, resolve, save_json

log = get_logger("clean_split")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"raw_rows": len(df), "missing_text": int(df["text"].isna().sum())}
    df = df.copy()
    df["text"] = df["text"].astype("string").str.strip()
    stats["empty_text"] = int((df["text"].fillna("") == "").sum()) - stats["missing_text"]
    df = df[df["text"].fillna("") != ""]
    df = df.dropna(subset=["category"])
    stats["exact_duplicate_rows"] = int(df.duplicated(subset=["text", "category"]).sum())
    conflicting = df.groupby("text")["category"].nunique()
    stats["texts_with_conflicting_labels"] = int((conflicting > 1).sum())
    # keep the first occurrence of each text (also resolves conflicting duplicates)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    stats["clean_rows"] = len(df)
    return df, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cfg = load_config()
    seed, sp = cfg["seed"], cfg["split"]

    raw = load_synthetic(cfg)
    log.info("Raw data: %d rows, columns=%s", len(raw), list(raw.columns))
    log.info("Raw class counts:\n%s", raw["category"].value_counts().to_string())
    df, stats = clean(raw)
    log.info("Cleaning stats: %s", stats)

    # --- unseen-template hold-out: whole phrasing templates are removed BEFORE the split, so
    # no ticket written from them is ever seen in training. Measures generalisation to new
    # phrasings (the main weakness of template-generated data).
    splits = [("train", None), ("val", None), ("test", None)]
    if "template_id" in df and sp.get("unseen_template_frac", 0) > 0:
        rng = random.Random(seed)
        held = []
        for lab in cfg["labels"]:
            tids = sorted({t for t in df["template_id"] if t.startswith(f"core:{lab}:")}, key=lambda t: int(t.split(":")[-1]))
            held += rng.sample(tids, max(1, round(len(tids) * sp["unseen_template_frac"])))
        mask = df["template_id"].isin(held)
        unseen, df = df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)
        stats["unseen_templates"] = sorted(held)
        log.info("Held out %d templates -> %d tickets in test_unseen", len(held), len(unseen))
        splits.append(("test_unseen", unseen))

    train, temp = train_test_split(df, test_size=sp["val"] + sp["test"], stratify=df["category"], random_state=seed)
    val, test = train_test_split(temp, test_size=sp["test"] / (sp["val"] + sp["test"]),
                                 stratify=temp["category"], random_state=seed)

    pre_c = TextPreprocessor(mode="classical", replace_ids=cfg["preprocessing"]["replace_ids"],
                             keep_words=cfg["preprocessing"]["keep_words"], spacy_model=cfg["preprocessing"]["spacy_model"])
    pre_t = TextPreprocessor(mode="transformer")
    out_dir = ensure_dir(cfg["paths"]["processed_dir"])
    parts = {"train": train, "val": val, "test": test}
    parts.update({n: p for n, p in splits if p is not None})
    for name, part in parts.items():
        part = part.copy()
        part["text_clean"] = pre_c.transform(part["text"].tolist())
        part["text_light"] = pre_t.transform(part["text"].tolist())
        part.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        stats[f"{name}_rows"] = len(part)
        stats[f"{name}_class_counts"] = part["category"].value_counts().to_dict()
        log.info("%-5s %5d rows -> %s", name, len(part), out_dir / f"{name}.csv")

    save_json(stats, resolve(cfg["paths"]["reports_dir"]) / "data_stats.json")
    log.info("Saved stats -> reports/data_stats.json")


if __name__ == "__main__":
    main()
