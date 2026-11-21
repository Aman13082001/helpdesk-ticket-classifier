"""Dataset loaders.

(a) the synthetic CSV produced by generate_dataset.py
(b) ANY user-provided CSV with a text column and a label column - e.g. a Kaggle
    IT-helpdesk / customer-support dataset - mapped onto our 5 categories.

Plugging in a real dataset
--------------------------
    python -m src.data.load_dataset --csv path/to/real.csv --text-col body \
        --label-col queue --label-map label_map.json --out data/raw/real_mapped.csv

where label_map.json maps source labels to ours, e.g.
    {"Network": "IT_WiFi", "Hardware": "IT_WiFi", "Billing": "Fees_Accounts",
     "Facilities": "Hostel_Facilities"}
Rows whose label is not in the map are dropped (and counted in the log). Then point
`paths.raw_data` in config.yaml to the new file and run clean_split.py as usual.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.utils import get_logger, load_config, resolve

log = get_logger("load_dataset")
SCHEMA = ["ticket_id", "text", "category", "source", "created_date"]


def load_synthetic(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve(cfg["paths"]["raw_data"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.data.generate_dataset` first.")
    return pd.read_csv(path, encoding="utf-8")


def load_external(csv: str, text_col: str = "text", label_col: str = "label",
                  label_map: dict[str, str] | None = None, allowed: list[str] | None = None) -> pd.DataFrame:
    """Load a user CSV and convert it to our schema."""
    df = pd.read_csv(csv, encoding="utf-8")
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Columns {missing} not found. Available: {list(df.columns)}")
    out = pd.DataFrame({"text": df[text_col].astype("string"), "category": df[label_col].astype(str)})
    if label_map:
        out["category"] = out["category"].map(label_map)
    if allowed:
        before = len(out)
        out = out[out["category"].isin(allowed)]
        log.info("Kept %d/%d rows whose label maps to the allowed categories", len(out), before)
    out.insert(0, "ticket_id", [f"EXT{i:06d}" for i in range(len(out))])
    out["source"] = "external"
    out["created_date"] = pd.NaT
    return out[SCHEMA].reset_index(drop=True)


def load_processed(split: str, cfg: dict | None = None) -> pd.DataFrame:
    """Load train/val/test produced by clean_split.py (includes cached cleaned text columns)."""
    cfg = cfg or load_config()
    path = resolve(cfg["paths"]["processed_dir"]) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.data.clean_split` first.")
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--label-map", default=None, help="JSON file mapping source labels -> our 5 labels")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config()
    label_map = json.load(open(args.label_map, encoding="utf-8")) if args.label_map else None
    df = load_external(args.csv, args.text_col, args.label_col, label_map, cfg["labels"])
    df.to_csv(resolve(args.out), index=False, encoding="utf-8")
    log.info("Wrote %d rows -> %s\n%s", len(df), args.out, df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
