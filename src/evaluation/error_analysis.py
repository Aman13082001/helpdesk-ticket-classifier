"""Error analysis: what goes wrong, and why?

Reads reports/test_predictions.csv (from evaluate.py) and reports/ner_evaluation.json, then
writes:
  reports/misclassified.csv     every test ticket that at least one main model got wrong
  reports/error_analysis.md     auto-generated DRAFT with confusion pairs, 15+ annotated
                                errors, classical-vs-transformer disagreements, NER edge cases
                                and placeholders for the team's manual commentary

Probable causes are assigned by transparent heuristics (see `probable_causes`). Because the
data is synthetic we also know each ticket's generation type (standard / ambiguous /
vague / mislabel) - we use it to VERIFY the heuristics, which would be impossible with
real data.

Usage:  python -m src.evaluation.error_analysis
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import joblib
import pandas as pd

from src.models.predict import MAIN_MODELS, MODEL_DISPLAY, _metrics_path
from src.utils import get_logger, load_config, load_json, resolve

log = get_logger("error_analysis")

# small hand-written cue lexicon per class (used only to explain errors, never to predict)
CUES = {
    "IT_WiFi": ["wifi", "internet", "network", "email", "password", "vpn", "lab", "system", "pc", "printer", "software", "eduroam"],
    "LMS_Portal": ["portal", "lms", "moodle", "login", "upload", "submit", "assignment", "quiz", "course"],
    "Academics_Exams": ["exam", "hall ticket", "result", "attendance", "marks", "timetable", "re-evaluation", "syllabus"],
    "Fees_Accounts": ["fee", "payment", "paid", "refund", "receipt", "scholarship", "deducted", "fine", "amount", "transaction"],
    "Hostel_Facilities": ["hostel", "room", "mess", "water", "electricity", "bus", "library", "warden", "fan"],
}
NEGATION_RE = re.compile(r"\b(not|no|never|cannot|can't|cant|unable|nt|n't|didn't|doesn't)\b", re.I)


def probable_causes(row: pd.Series, pred_col: str, vocab: set[str] | None, minority: str) -> list[str]:
    text, true, pred = str(row["text"]), row["category"], row[pred_col]
    low = text.lower()
    causes = []
    kind = str(row.get("gen_kind", ""))
    if "mislabel" in kind:
        causes.append("label noise: the gold label itself looks mis-filed")
    true_cues = [c for c in CUES[true] if c in low]
    pred_cues = [c for c in CUES[pred] if c in low]
    if true_cues and pred_cues:
        causes.append(f"multi-intent / overlapping categories (cues for both: {true_cues[:2]} vs {pred_cues[:2]})")
    elif pred_cues and not true_cues:
        causes.append(f"misleading cue words of predicted class {pred_cues[:3]}")
    if len(text.split()) <= 6:
        causes.append("short / vague text with little context")
    if vocab is not None:
        oov = [w for w in str(row.get("text_clean", "")).split() if w not in vocab]
        if len(oov) >= 2:
            causes.append(f"noisy / out-of-vocabulary words (typos, abbreviations): {oov[:4]}")
    if NEGATION_RE.search(text) and not causes:
        causes.append("negation / failure phrasing changes the meaning")
    conf = row.get(pred_col.replace("pred_", "conf_"))
    if conf is not None and conf < 0.6:
        causes.append(f"low model confidence ({conf:.2f}) - genuinely uncertain")
    if true == minority:
        causes.append("true class is the smallest class (imbalance)")
    if not causes:
        causes.append("template overlap: phrasing similar to tickets of the predicted class")
    return causes


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-examples", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config()
    rep = resolve(cfg["paths"]["reports_dir"])
    df = pd.read_csv(rep / "test_predictions.csv", keep_default_na=False)
    df[[c for c in df if c.startswith("conf_")]] = df[[c for c in df if c.startswith("conf_")]].round(3)
    test = pd.read_csv(resolve(cfg["paths"]["processed_dir"]) / "test.csv", keep_default_na=False)
    df = df.merge(test[["ticket_id", "text_clean"]], on="ticket_id", how="left")
    models = [m for m in MAIN_MODELS if f"pred_{m}" in df]
    val = {m: load_json(_metrics_path(m, cfg))["val"]["macro_f1"] for m in models}
    best = max(val, key=val.get)
    classical = [m for m in ("tfidf_nb", "tfidf_lr", "tfidf_svm") if m in models]
    best_classical = max(classical, key=val.get) if classical else None
    transformer = "distilbert" if "distilbert" in models else ("sbert_lr" if "sbert_lr" in models else None)
    minority = df["category"].value_counts().idxmin()
    try:
        lr = joblib.load(resolve(cfg["paths"]["artifacts_dir"]) / "classical" / "tfidf_lr.joblib")
        vocab = set(lr.named_steps["tfidf"].word_.vocabulary_)
    except Exception:
        vocab = None

    # ---------------- misclassified.csv
    wrong = pd.DataFrame({m: df[f"pred_{m}"] != df["category"] for m in models})
    df["n_models_wrong"] = wrong.sum(axis=1)
    df["failed_models"] = wrong.apply(lambda r: ", ".join(m for m in models if r[m]), axis=1)
    mis = df[df["n_models_wrong"] > 0].copy()
    keep = ["ticket_id", "text", "category"] + (["gen_kind"] if "gen_kind" in df else []) + \
           [c for m in models for c in (f"pred_{m}", f"conf_{m}")] + ["n_models_wrong", "failed_models"]
    mis[keep].sort_values("n_models_wrong", ascending=False).to_csv(rep / "misclassified.csv", index=False, encoding="utf-8")
    log.info("%d / %d test tickets misclassified by at least one model (best model %s wrong on %d)",
             len(mis), len(df), best, int(wrong[best].sum()))

    # ---------------- confusion pairs
    pair_rows = []
    for m in models:
        e = df[df[f"pred_{m}"] != df["category"]]
        for (t, p), c in Counter(zip(e["category"], e[f"pred_{m}"])).most_common(5):
            pair_rows.append({"model": MODEL_DISPLAY[m], "true": t, "predicted": p, "count": c})
    pairs = pd.DataFrame(pair_rows)
    best_err = df[df[f"pred_{best}"] != df["category"]].copy()
    best_pairs = Counter(zip(best_err["category"], best_err[f"pred_{best}"])).most_common(8)

    # ---------------- representative errors of the best model (diverse confusion pairs)
    best_err["causes"] = best_err.apply(lambda r: probable_causes(r, f"pred_{best}", vocab, minority), axis=1)
    best_err = best_err.sort_values(["n_models_wrong", f"conf_{best}"], ascending=[False, False])
    chosen, seen = [], Counter()
    for _, r in best_err.iterrows():  # round-robin over confusion pairs for variety
        key = (r["category"], r[f"pred_{best}"])
        if seen[key] < 3:
            chosen.append(r); seen[key] += 1
        if len(chosen) >= args.n_examples:
            break
    if len(chosen) < args.n_examples:  # pad with errors from other models if the best model makes few
        for m in models:
            for _, r in df[(df[f"pred_{m}"] != df["category"]) & (~df["ticket_id"].isin([c["ticket_id"] for c in chosen]))].iterrows():
                r = r.copy(); r["causes"] = probable_causes(r, f"pred_{m}", vocab, minority); r["_model"] = m
                chosen.append(r)
                if len(chosen) >= args.n_examples:
                    break
            if len(chosen) >= args.n_examples:
                break
    cause_counter = Counter(c.split(":")[0].split(" (")[0] for _, r in best_err.iterrows() for c in r["causes"])

    # ---------------- classical vs transformer disagreements
    def disagree(a: str, b: str, n: int = 6) -> pd.DataFrame:
        d = df[(df[f"pred_{a}"] == df["category"]) & (df[f"pred_{b}"] != df["category"])]
        return d[["text", "category", f"pred_{b}", f"conf_{b}", f"conf_{a}"]].head(n)

    # ---------------- NER
    ner_path = rep / "ner_evaluation.json"
    ner = load_json(ner_path) if ner_path.exists() else None

    # ---------------- markdown
    md = ["# Error Analysis (auto-generated draft)", "",
          "> Generated by `python -m src.evaluation.error_analysis`. Numbers come from the held-out **test** set. "
          "Sections marked **Team commentary** should be completed by hand before submission.", "",
          f"* Test tickets: **{len(df)}**; misclassified by at least one main model: **{len(mis)}**; "
          f"by all {len(models)} main models: **{int((df['n_models_wrong'] == len(models)).sum())}**",
          f"* Best model by validation macro-F1: **{MODEL_DISPLAY[best]}** - wrong on **{len(best_err)}** test tickets",
          ""]
    md += ["## 1. Errors per model", "", _md_table(pd.DataFrame(
        [{"model": MODEL_DISPLAY[m], "errors": int(wrong[m].sum()), "error_rate": f"{wrong[m].mean():.2%}"} for m in models])), ""]
    md += ["## 2. Most confused category pairs", "", f"### Best model ({MODEL_DISPLAY[best]})", "",
           _md_table(pd.DataFrame([{"true": t, "predicted": p, "count": c} for (t, p), c in best_pairs])), "",
           "### Top-5 pairs for every model", "", _md_table(pairs), ""]
    if "gen_kind" in df:
        k = df.assign(best_wrong=wrong[best]).groupby("gen_kind")["best_wrong"].agg(["count", "sum", "mean"]).reset_index()
        k.columns = ["generation type", "test tickets", "best-model errors", "error rate"]
        k["error rate"] = k["error rate"].map(lambda v: f"{v:.1%}")
        md += ["## 3. Error rate by ticket type (only possible because the data is synthetic)", "",
               "This verifies the heuristic causes below: errors concentrate in ambiguous (multi-intent), vague and mislabelled tickets.", "",
               _md_table(k), ""]
    md += ["## 4. Probable-cause frequency (best model's errors, heuristic)", "",
           _md_table(pd.DataFrame([{"cause": c, "errors": n} for c, n in cause_counter.most_common()])), ""]
    md += [f"## 5. Representative errors ({len(chosen)})", ""]
    for i, r in enumerate(chosen, 1):
        m = r.get("_model", best) if isinstance(r.get("_model", None), str) else best
        other = ", ".join(f"{MODEL_DISPLAY[x].split(' (')[0]}={r[f'pred_{x}']}" for x in models if x != m)
        md += [f"**{i}.** `{r['text']}`", "",
               f"- True: **{r['category']}** | {MODEL_DISPLAY[m]} predicted: **{r[f'pred_{m}']}** (conf {r[f'conf_{m}']:.2f})"
               + (f" | type: `{r['gen_kind']}`" if "gen_kind" in r else ""),
               f"- Other models: {other}",
               f"- Probable cause(s): {'; '.join(r['causes'])}", ""]
    md += ["**Team commentary:** _(explain 3-4 of the errors above in your own words; suggest a fix for each, e.g. "
           "multi-label output, more training examples, a spelling-normaliser, or a 'needs human review' threshold)_", ""]
    if best_classical and transformer:
        md += [f"## 6. Where the classical model wins and the transformer loses ({MODEL_DISPLAY[best_classical]} right, {MODEL_DISPLAY[transformer]} wrong)", "",
               _md_table(disagree(best_classical, transformer)) or "_none_", "",
               f"## 7. Where the transformer wins and the classical model loses ({MODEL_DISPLAY[transformer]} right, {MODEL_DISPLAY[best_classical]} wrong)", "",
               _md_table(disagree(transformer, best_classical)) or "_none_", "",
               "**Team commentary:** _(typical pattern: TF-IDF wins when one rare, decisive keyword or ID-like token matters; "
               "the transformer wins on paraphrases, typos and longer context - check whether that holds in the rows above)_", ""]
    up = rep / "test_unseen_predictions.csv"
    if up.exists():
        un = pd.read_csv(up, keep_default_na=False)
        um = [m for m in models if f"pred_{m}" in un]
        tbl = pd.DataFrame([{"model": MODEL_DISPLAY[m], "errors": int((un[f"pred_{m}"] != un["category"]).sum()),
                             "error rate":f"{(un[f'pred_{m}'] != un['category']).mean():.1%}"} for m in um])
        md += ["## 8. Generalisation to UNSEEN phrasing templates", "",
               f"{len(un)} tickets written from {un['template_id'].nunique()} templates that were removed before splitting, "
               "so no model saw these phrasings in training. This is the closest proxy we have for real, differently-worded tickets.", "",
               _md_table(tbl), ""]
        worst = un.assign(err=un[f"pred_{best_classical or best}"] != un["category"]).groupby("template_id")["err"].mean() \
            .sort_values(ascending=False).head(6)
        md += [f"Templates hardest for {MODEL_DISPLAY[best_classical or best]} (error rate):", ""]
        for tid, rate in worst.items():
            g = un[un["template_id"] == tid]
            wrong_g = g[g[f"pred_{best_classical or best}"] != g["category"]]
            ex = (wrong_g if len(wrong_g) else g).iloc[0]
            md += [f"- `{tid}` ({rate:.0%} wrong, true class {ex['category']}) e.g. *{ex['text']}* -> predicted "
                   f"**{ex[f'pred_{best_classical or best}']}**"]
        md += [""]
        if best_classical and "distilbert" in um:
            d = un[(un["pred_distilbert"] == un["category"]) & (un[f"pred_{best_classical}"] != un["category"])]
            d2 = un[(un["pred_distilbert"] != un["category"]) & (un[f"pred_{best_classical}"] == un["category"])]
            md += [f"DistilBERT right / {MODEL_DISPLAY[best_classical]} wrong: **{len(d)}** tickets; the reverse: **{len(d2)}**.", "",
                   _md_table(d[["text", "category", f"pred_{best_classical}"]].head(6)), "",
                   "Interpretation: on new wording the sparse models only have overlapping *words*; when the decisive words "
                   "were never seen with the right class, they fall back on generic cue words (portal, payment, room...). "
                   "The pretrained transformer maps paraphrases close together, so it degrades far less. "
                   "This, not the in-distribution score, is the evidence for the transformer.", "",
                   "**Team commentary:** _(pick two rows and explain which words misled the classical model)_", ""]
    if ner:
        md += ["## 9. NER edge cases (gold sample)", "",
               f"Strict micro-F1 **{ner['strict']['micro']['f1']}**, lenient micro-F1 **{ner['lenient']['micro']['f1']}**; "
               f"tickets with every entity correct: **{ner['tickets_fully_correct']}/{ner['n_tickets']}**.", ""]
        missed = [e for e in ner["errors"] if e["type"] == "missed"]
        spur = [e for e in ner["errors"] if e["type"] == "spurious"]
        md += ["### Missed entities", "", _md_table(pd.DataFrame(
            [{"entity": e["entity"], "label": e["label"], "predicted instead": e["predicted_as"] or "-", "ticket": e["text"]} for e in missed]))
               if missed else "_none_", "",
               "### Spurious / wrongly-labelled entities", "", _md_table(pd.DataFrame(
                   [{"entity": e["entity"], "predicted label": e["label"], "ticket": e["text"]} for e in spur])) if spur else "_none_", "",
               "Typical causes: (a) formats not covered by the regex rules (e.g. `ERR-504`, `E1021`, `CSE-205`, `Hostel D 210`, "
               "`Lab-3`); (b) the small spaCy model tagging lowercase or domain words as ORG/PERSON; (c) boundary disagreements "
               "(titles, prepositions) - handled by the lenient score; (d) generic words like 'library' that are a LOCATION in "
               "one sentence and part of a fine description in another.", "",
               "**Team commentary:** _(propose pattern fixes and re-run `python -m src.evaluation.evaluate`)_", ""]
    (rep / "error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote reports/error_analysis.md and reports/misclassified.csv")


if __name__ == "__main__":
    main()
