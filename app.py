"""Streamlit UI for the University Helpdesk Ticket Classifier.

    streamlit run app.py

Tabs: Single Ticket | Batch Upload | Dashboard | Model Performance | About / Architecture
"""
from __future__ import annotations

import io
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from src.models.predict import DEPARTMENT_OF, MODEL_DISPLAY, HelpdeskPredictor, explain_linear, priority
from src.utils import load_config, load_json, resolve

st.set_page_config(page_title="Helpdesk Ticket Classifier", page_icon="🎫", layout="wide")
CFG = load_config()
REPORTS, FIGS = resolve(CFG["paths"]["reports_dir"]), resolve(CFG["paths"]["figures_dir"])
MAX_CHARS = 2000
EXAMPLES = {
    "Wi-Fi": "Hi, wifi is not working in Block C since yesterday, cannot attend the online class. pls help 🙏",
    "Portal": "cannot login to the student portal, it shows ERR_403 even with the correct password. ID KRM2023MCA045",
    "Exam": "My hall ticket for the 3rd semester exams is not generated yet and the exam is on 28/11/2026!",
    "Fees": "Fee payment failed but ₹45,000 was deducted, transaction TXN4829301845. Accounts Office not responding.",
    "Hostel": "There has been no water supply in Hostel B-212 since 12 Nov, please fix it urgently.",
    "Ambiguous": "fee paid but portal shows unpaid and I cannot access my courses on Moodle",
    # Deliberate failure case for the demo: "warden office" co-occurs with IT tickets in training
    # ("hostel internet down, warden says contact IT"), so TF-IDF models lean to IT_WiFi while
    # DistilBERT reads the whole sentence and answers Hostel_Facilities.
    "Spurious cue": "No water supply in Hostel B-212 since 12 Nov, the warden office is not responding. Urgent!",
}


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading models ...")
def get_predictor() -> HelpdeskPredictor | None:
    try:
        return HelpdeskPredictor(CFG)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading spaCy NER ...")
def get_extractor():
    from src.ner.entity_extractor import EntityExtractor

    return EntityExtractor(CFG)


@st.cache_data(show_spinner=False)
def entities_for(texts: tuple[str, ...]) -> list[list[dict]]:
    return get_extractor().extract_many(list(texts))


@st.cache_data(show_spinner=False)
def load_demo_tickets() -> pd.DataFrame:
    """Sample tickets used for the format example and the downloadable templates."""
    path = resolve("data/demo_tickets.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({"text": list(EXAMPLES.values())[:5], "category": ["IT_WiFi", "LMS_Portal", "Academics_Exams",
                                                                          "Fees_Accounts", "Hostel_Facilities"]})


def count_ragged_rows(text: str) -> list[int]:
    """Line numbers of rows with MORE fields than the header (usually unquoted commas in the ticket text)."""
    import csv

    rows = csv.reader(io.StringIO(text))
    header = next(rows, [])
    return [i for i, row in enumerate(rows, start=2) if len(row) > len(header)]


def read_uploaded_csv(data: bytes) -> tuple[pd.DataFrame | None, str | None]:
    """Read an uploaded CSV, trying UTF-8 first and falling back to Excel's Windows encoding.

    Returns (dataframe, error). Rows with extra fields make the file invalid, because pandas
    would silently shift the columns and classify the wrong text.
    """
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        ragged = count_ragged_rows(text)
        if ragged:
            shown = ", ".join(map(str, ragged[:10])) + (" ..." if len(ragged) > 10 else "")
            return None, (f"{len(ragged)} row(s) have more values than the header (line {shown}). This usually "
                          "means the ticket text contains commas but is not wrapped in double quotes, e.g. "
                          "`\"wifi down, pls help\",IT_WiFi`. Re-save the file from Excel/Sheets as CSV, which adds the quotes.")
        try:
            return pd.read_csv(io.StringIO(text)), None
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            return None, (f"{e}. Check that the first row is a header, every row has the same number of "
                          "columns, and text containing commas is wrapped in double quotes.")
    return None, "Unknown file encoding. Re-save the file as **CSV UTF-8**."


def guess_text_column(df: pd.DataFrame) -> str:
    """'text' if present, otherwise the string column with the longest average value."""
    if "text" in df.columns:
        return "text"
    str_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    if not str_cols:
        return df.columns[0]
    return max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())


def top_words(texts: list[str], n: int = 10) -> list[tuple[str, int]]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = ENGLISH_STOP_WORDS | {"pls", "please", "hi", "hello", "sir", "madam", "dear", "kindly", "help", "regards",
                                 "thanks", "team", "respected", "morning", "good", "asap", "need", "nt"}
    c = Counter(w for t in texts for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in stop)
    return c.most_common(n)


predictor = get_predictor()
st.title("🎫 University Helpdesk Ticket Classifier")
st.caption("Routes free-text tickets to the right department, extracts key entities and compares classical vs transformer models.")
if predictor is None:
    st.error("No trained models found in `artifacts/`. Run training first:\n\n"
             "```\npython -m src.data.generate_dataset\npython -m src.data.clean_split\npython -m src.models.train_classical\n"
             "python -m src.models.train_embedding_lr\npython -m src.models.train_transformer\n```")
    st.stop()

with st.sidebar:
    st.header("Settings")
    best = predictor.best_model
    primary = st.selectbox("Primary model", predictor.available, index=predictor.available.index(best),
                           format_func=lambda m: MODEL_DISPLAY[m] + ("  ★ best on val" if m == best else ""))
    show_all = st.checkbox("Compare all models", value=True)
    st.markdown("**Validation macro-F1**")
    st.dataframe(
        pd.DataFrame({"model": [MODEL_DISPLAY[m] for m in predictor.val_scores()],
                      "val F1": list(predictor.val_scores().values())}),
        hide_index=True,
        column_config={"model": st.column_config.TextColumn("Model", width="medium"),
                       "val F1": st.column_config.NumberColumn("Val F1", format="%.3f", width="small")},
        alt="Validation macro-F1 per model",
    )
    st.caption("Best model is chosen on the validation split, never on the test split.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Single Ticket", "Batch Upload", "Dashboard", "Model Performance", "About / Architecture"])

# ================================================================== 1. single ticket
with tab1:
    st.subheader("Classify a single ticket")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        if col.button(name, width="stretch"):
            st.session_state["ticket_text"] = text
    text = st.text_area("Ticket text", key="ticket_text", height=120,
                        placeholder="e.g. my wifi password is not working in Lab 3 ...")
    clicked = st.button("Analyse ticket", type="primary")
    if clicked and not (text and text.strip()):
        st.warning("Please enter some ticket text.")
    if text and text.strip():  # re-analyse on every rerun so results persist while browsing
        if True:
            if len(text) > MAX_CHARS:
                st.info(f"Input is {len(text)} characters - only the first {MAX_CHARS} are analysed.")
                text = text[:MAX_CHARS]
            models = predictor.available if show_all else [primary]
            res = predictor.predict([text], models)
            lab, conf = res[primary]["labels"][0], res[primary]["confidence"][0]
            prio, why = priority(text, CFG)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predicted category", lab)
            c2.metric("Confidence", f"{conf:.1%}")
            c3.metric("Route to", DEPARTMENT_OF[lab])
            c4.metric("Priority", prio, help="Rule-based: " + (", ".join(why) if why else "no urgency keywords"))
            st.progress(min(max(conf, 0.0), 1.0), text=f"{MODEL_DISPLAY[primary]} confidence")
            if conf < 0.6:
                st.warning("Low confidence - this ticket may be multi-intent or vague. Suggest human review.")

            left, right = st.columns([1.1, 1])
            with left:
                st.markdown("**Class probabilities**")
                probs = pd.DataFrame({"category": predictor.labels, "probability": res[primary]["proba"][0]})
                fig = px.bar(probs.sort_values("probability"), x="probability", y="category", orientation="h", range_x=[0, 1],
                             color="probability", color_continuous_scale="Teal", height=260)
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")
            with right:
                if show_all:
                    st.markdown("**Model comparison**")
                    st.dataframe(pd.DataFrame([{"model": MODEL_DISPLAY[m], "prediction": res[m]["labels"][0],
                                                "confidence": f"{res[m]['confidence'][0]:.1%}",
                                                "latency (ms)": round(res[m]["latency_ms_per_ticket"], 1)} for m in models]),
                                 hide_index=True, width="stretch")
                    if len({res[m]["labels"][0] for m in models}) > 1:
                        st.info("Models disagree - a sign of an ambiguous ticket.")
                if "tfidf_lr" in predictor.available:
                    expl = explain_linear(predictor.models["tfidf_lr"], text, lab)
                    if expl:
                        st.markdown(f"**Why {lab}?** (top Logistic Regression word contributions)")
                        st.write(", ".join(f"`{w}` (+{v:.2f})" for w, v in expl if v > 0) or "_no strong single-word evidence_")

            st.markdown("**Extracted entities**")
            ents = get_extractor().extract(text)
            st.markdown(get_extractor().to_html(text, ents), unsafe_allow_html=True)
            if ents:
                st.dataframe(pd.DataFrame(ents)[["label", "text", "source", "start", "end"]], hide_index=True)
            else:
                st.caption("No entities found.")

# ================================================================== 2. batch upload
with tab2:
    st.subheader("Batch classification")
    st.caption("Classify many tickets at once. Prepare your file in one of the two formats below.")

    # ---------------- file-format guide ----------------
    demo = load_demo_tickets()
    csv_col, txt_col = st.columns(2)
    with csv_col.container(border=True, height="stretch"):
        st.markdown("**:material/table: Option A: CSV file** (recommended)")
        st.table(
            {
                "text": "**Required.** The ticket text, one ticket per row",
                "category": f"Optional. True label, one of: {', '.join(f'`{l}`' for l in CFG['labels'])}. "
                            "If present, the app reports agreement with the predictions",
                "ticket_id": "Optional. Kept in the output",
                "created_date": "Optional. `YYYY-MM-DD`; enables the weekly trend on the Dashboard",
            },
            border="horizontal",
            alt="Expected CSV columns",
        )
        st.markdown(
            "- First row must be a **header** with column names\n"
            "- Column names are case-sensitive; if yours is not called `text`, pick it after upload\n"
            "- Wrap text containing commas in double quotes: `\"wifi down, pls help\"`\n"
            "- Save as **UTF-8** (Excel: *Save As → CSV UTF-8*)"
        )
        st.download_button("Download CSV template", demo.to_csv(index=False).encode("utf-8"),
                           "ticket_template.csv", "text/csv", icon=":material/download:")
    with txt_col.container(border=True, height="stretch"):
        st.markdown("**:material/description: Option B: plain .txt file**")
        st.markdown(
            "- **One ticket per line**, no header\n"
            "- Blank lines are skipped\n"
            "- Use this when you only have the ticket text (no labels, IDs or dates)"
        )
        st.code("\n".join(demo["text"].head(4)), language=None)
        st.download_button("Download TXT template", "\n".join(demo["text"]).encode("utf-8"),
                           "ticket_template.txt", "text/plain", icon=":material/download:")
    st.markdown("**Example of a valid CSV**")
    st.dataframe(demo.head(5), hide_index=True, alt="Example rows of a valid ticket CSV")
    st.caption(f"Limits: 10 MB per file · tickets longer than {MAX_CHARS:,} characters are truncated · "
               "English text works best (the models were trained on English tickets).")

    up = st.file_uploader("Upload your CSV or TXT file", type=["csv", "txt"], max_upload_size=10)
    if up is not None and st.session_state.get("batch_file") != (up.name, up.size):
        # a new file was uploaded -> forget the previous results
        st.session_state["batch_file"] = (up.name, up.size)
        st.session_state.pop("batch", None)
        st.session_state.pop("batch_ents", None)
    batch = None
    if up is not None:
        if up.name.lower().endswith(".csv"):
            raw, err = read_uploaded_csv(up.getvalue())
            if err:
                st.error(err, title="Could not read this CSV", icon=":material/error:")
            elif raw.empty:
                st.error("The file has a header but no rows.", title="Empty file", icon=":material/error:")
            else:
                guess = guess_text_column(raw)
                text_col = st.selectbox("Which column contains the ticket text?", list(raw.columns),
                                        index=list(raw.columns).index(guess))
                if guess != "text" and "text" not in raw.columns:
                    st.info(f"No column named `text` - using `{guess}` (the column with the longest text). "
                            "Change it above if that is wrong.", icon=":material/info:")
                if pd.api.types.is_numeric_dtype(raw[text_col]):
                    st.warning(f"`{text_col}` contains numbers, not ticket text. Please choose the text column.",
                               title="This does not look like text", icon=":material/warning:")
                batch = raw.copy()
                if text_col != "text" and "text" in batch.columns:
                    batch = batch.rename(columns={"text": "text_original"})  # keep, don't overwrite
                batch["text"] = raw[text_col].fillna("").astype(str)
                if "category" in batch.columns:
                    bad = sorted(set(batch["category"].dropna().astype(str)) - set(CFG["labels"]))
                    if bad:
                        st.warning(f"Unknown labels in `category`: {', '.join(bad[:8])}. Allowed: "
                                   f"{', '.join(CFG['labels'])}. Rows with unknown labels are ignored in the agreement score.",
                                   title="Unrecognised categories", icon=":material/warning:")
        else:
            content = up.getvalue().decode("utf-8", "replace")
            lines = [l.strip() for l in io.StringIO(content) if l.strip()]
            if not lines:
                st.error("The text file is empty.", title="Empty file", icon=":material/error:")
            else:
                batch = pd.DataFrame({"text": lines})
                if "," in lines[0] and lines[0].lower().startswith(("text", "ticket")):
                    st.warning("The first line looks like a CSV header. For files with columns, save them as `.csv`.",
                               title="Is this a CSV?", icon=":material/warning:")

    if batch is not None:
        n_total = len(batch)
        batch = batch[batch["text"].str.strip().astype(bool)].reset_index(drop=True)
        n_empty, n_long = n_total - len(batch), int((batch["text"].str.len() > MAX_CHARS).sum())
        n_dup = int(batch["text"].duplicated().sum())
        summary = st.container(horizontal=True)
        summary.metric("Tickets to classify", len(batch), border=True)
        summary.metric("Empty rows skipped", n_empty, border=True)
        summary.metric("Duplicate texts", n_dup, border=True)
        summary.metric(f"Truncated (> {MAX_CHARS:,} chars)", n_long, border=True)
        if batch.empty:
            st.error("No non-empty tickets found in the selected column.", title="Nothing to classify", icon=":material/error:")
            batch = None

    if batch is not None:
        bm = st.selectbox("Model for batch", predictor.available, index=predictor.available.index(primary),
                          format_func=MODEL_DISPLAY.get)
        do_ner = st.checkbox("Extract entities", value=True)
        if st.button("Run batch", type="primary"):
            bar = st.progress(0.0, text="Classifying ...")
            labels, confs = [], []
            texts = batch["text"].str.slice(0, MAX_CHARS).tolist()
            step = 64
            for i in range(0, len(texts), step):
                r = predictor.predict(texts[i:i + step], [bm])[bm]
                labels += r["labels"]; confs += r["confidence"]
                bar.progress(min(1.0, (i + step) / len(texts)), text=f"Classified {min(i + step, len(texts))}/{len(texts)}")
            batch["predicted_category"], batch["confidence"] = labels, confs
            batch["department"] = batch["predicted_category"].map(DEPARTMENT_OF)
            batch["priority"] = [priority(t, CFG)[0] for t in texts]
            if do_ner:
                bar.progress(1.0, text="Extracting entities ...")
                ents = entities_for(tuple(texts))
                batch["entities"] = ["; ".join(f"{e['label']}:{e['text']}" for e in es) for es in ents]
                st.session_state["batch_ents"] = ents
            bar.empty()
            st.session_state["batch"] = batch
        if "batch" in st.session_state:
            res = st.session_state["batch"]
            st.dataframe(res, width="stretch", height=380)
            st.download_button("Download results CSV", res.to_csv(index=False).encode("utf-8"), "ticket_predictions.csv", "text/csv")
            if "category" in res.columns:
                valid = res[res["category"].astype(str).isin(CFG["labels"])]
                if len(valid):
                    acc = (valid["category"] == valid["predicted_category"]).mean()
                    st.info(f"Agreement between your `category` column and the predictions: **{acc:.1%}** "
                            f"({len(valid)} labelled rows).", icon=":material/fact_check:")

# ================================================================== 3. dashboard
def render_dashboard() -> None:
    st.subheader("Ticket dashboard")
    if "batch" in st.session_state:
        dash = st.session_state["batch"].copy()
        ents = st.session_state.get("batch_ents")
        st.caption("Source: your last batch upload.")
    else:
        p = REPORTS / "test_predictions.csv"
        if not p.exists():
            st.info("Upload a batch or run `python -m src.evaluation.evaluate` to populate the dashboard.")
            return
        tp = pd.read_csv(p, keep_default_na=False)
        bcol = f"pred_{predictor.best_model}"
        dash = tp.rename(columns={bcol: "predicted_category", bcol.replace("pred_", "conf_"): "confidence"})
        test = pd.read_csv(resolve(CFG["paths"]["processed_dir"]) / "test.csv", keep_default_na=False)
        dash = dash.merge(test[["ticket_id", "created_date", "source"]], on="ticket_id", how="left")
        ents = None
        st.caption(f"Source: held-out test set ({len(dash)} tickets), predictions of {MODEL_DISPLAY[predictor.best_model]}.")
    if ents is None:
        ents = entities_for(tuple(dash["text"].astype(str)))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tickets", len(dash))
    k2.metric("Mean confidence", f"{dash['confidence'].mean():.1%}")
    k3.metric("Low-confidence (<60%)", int((dash["confidence"] < 0.6).sum()))
    k4.metric("Entities extracted", sum(len(e) for e in ents))

    c1, c2 = st.columns(2)
    counts = dash["predicted_category"].value_counts().reset_index()
    counts.columns = ["category", "tickets"]
    c1.plotly_chart(px.pie(counts, names="category", values="tickets", hole=0.45, title="Predicted category distribution"),
                    width="stretch")
    c2.plotly_chart(px.histogram(dash, x="confidence", color="predicted_category", nbins=25, title="Confidence histogram"),
                    width="stretch")

    c3, c4 = st.columns(2)
    lab_counts = Counter(e["label"] for es in ents for e in es)
    if lab_counts:
        ldf = pd.DataFrame(lab_counts.items(), columns=["entity type", "mentions"]).sort_values("mentions")
        c3.plotly_chart(px.bar(ldf, x="mentions", y="entity type", orientation="h", title="Entity types"), width="stretch")
    rows = [{"category": cat, "entity": f"{e['label']}: {e['text']}"} for cat, es in zip(dash["predicted_category"], ents) for e in es
            if e["label"] not in ("DATE", "TIME", "PERSON", "EMAIL", "PHONE")]
    if rows:
        edf = pd.DataFrame(rows).value_counts().reset_index(name="count")
        edf = edf.groupby("category").head(5)
        c4.plotly_chart(px.bar(edf, x="count", y="entity", color="category", orientation="h", height=450,
                               title="Top entities per category"), width="stretch")

    if "created_date" in dash and dash["created_date"].astype(str).str.len().gt(0).any():
        d = dash.assign(date=pd.to_datetime(dash["created_date"], errors="coerce")).dropna(subset=["date"])
        if len(d):
            wk = d.groupby([pd.Grouper(key="date", freq="W"), "predicted_category"]).size().reset_index(name="tickets")
            st.plotly_chart(px.line(wk, x="date", y="tickets", color="predicted_category", markers=True,
                                    title="Tickets per week by category"), width="stretch")

    st.markdown("**Top terms per predicted category**")
    cats = sorted(dash["predicted_category"].unique())
    tcols = st.columns(len(cats))
    for col, cat in zip(tcols, cats):
        tw = pd.DataFrame(top_words(dash.loc[dash["predicted_category"] == cat, "text"].astype(str).tolist()), columns=["term", "n"])
        col.markdown(f"*{cat}*")
        col.dataframe(tw, hide_index=True, width="stretch")


with tab3:
    render_dashboard()

# ================================================================== 4. model performance
with tab4:
    st.subheader("Model performance on the held-out test set")
    mc = REPORTS / "model_comparison.csv"
    if not mc.exists():
        st.info("Run `python -m src.evaluation.evaluate` to generate the comparison.")
    else:
        comp = pd.read_csv(mc)
        show = comp[["display_name", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "val_macro_f1",
                     "latency_ms_per_ticket", "single_ticket_latency_ms", "train_time_sec", "size_mb"]]
        st.dataframe(show.style.highlight_max(subset=["accuracy", "macro_f1", "weighted_f1"], color="#c7f0d8")
                     .format(precision=4), hide_index=True, width="stretch")
        long = comp.melt(id_vars="display_name", value_vars=["accuracy", "macro_f1", "weighted_f1"], var_name="metric")
        st.plotly_chart(px.bar(long, x="value", y="display_name", color="metric", barmode="group", orientation="h",
                               range_x=[max(0, long["value"].min() - 0.05), 1], height=420), width="stretch")
        st.caption("Accuracy alone is not enough: compare macro-F1 (class-balanced), latency and model size before claiming one model is 'better'.")
        m = st.selectbox("Confusion matrix / per-class report for", comp["model"].tolist(), format_func=MODEL_DISPLAY.get)
        a, b = st.columns(2)
        for col, suffix in ((a, ""), (b, "_norm")):
            f = FIGS / f"cm_{m}{suffix}.png"
            if f.exists():
                col.image(str(f))
        cr = REPORTS / "classification_reports.json"
        if cr.exists():
            rep = pd.DataFrame(load_json(cr)[m]).T
            st.dataframe(rep.style.format(precision=3), width="stretch")
    ne = REPORTS / "ner_evaluation.json"
    if ne.exists():
        st.markdown("**NER evaluation on the 50-ticket gold sample**")
        nj = load_json(ne)
        st.write(f"Strict micro-F1 **{nj['strict']['micro']['f1']}**, lenient micro-F1 **{nj['lenient']['micro']['f1']}**, "
                 f"tickets fully correct **{nj['tickets_fully_correct']}/{nj['n_tickets']}**")
        st.dataframe(pd.DataFrame(nj["lenient"]["per_label"]).T, width="stretch")
    for name, cap in (("top_features_lr.png", "Explainability - top LR features"), ("embeddings_2d.png", "SBERT embedding space")):
        if (FIGS / name).exists():
            st.image(str(FIGS / name), caption=cap)

# ================================================================== 5. about
with tab5:
    st.subheader("Architecture")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8f4f2", fontname="Helvetica"];
      raw [label="Raw tickets\\n(CSV / text)"]; clean [label="Data layer\\nclean + stratified split"];
      prep [label="Preprocessing\\nclassical | light"]; tfidf [label="BoW / TF-IDF"]; sbert [label="SBERT\\nembeddings"];
      clf [label="NB / LR / SVM"]; slr [label="SBERT + LR"]; bert [label="DistilBERT\\nfine-tuned"];
      ner [label="spaCy NER +\\nrule layer"]; ev [label="Evaluation +\\nerror analysis"]; ui [label="Streamlit UI\\n+ dashboard"];
      raw -> clean -> prep; prep -> tfidf -> clf; prep -> sbert -> slr; prep -> bert; raw -> ner;
      clf -> ev; slr -> ev; bert -> ev; ner -> ev; ev -> ui; ner -> ui; }""")
    ds = REPORTS / "data_stats.json"
    if ds.exists():
        s = load_json(ds)
        st.markdown(f"**Dataset:** synthetic, {s['raw_rows']} raw rows -> {s['clean_rows']} after cleaning "
                    f"(train {s['train_rows']} / val {s['val_rows']} / test {s['test_rows']}). See `docs/DATA_DICTIONARY.md`.")
    st.markdown("""
**Categories:** IT_WiFi · LMS_Portal · Academics_Exams · Fees_Accounts · Hostel_Facilities

**Limitations**
- Training data is **synthetic** (template-generated). Scores are optimistic vs. real tickets; templates
  share structure across splits. Validate on real tickets before deployment.
- One label per ticket - multi-intent tickets (e.g. *fee paid but portal shows unpaid*) are forced into one class.
- Priority is a transparent keyword rule, not a learned model.
- Rule-based entities only cover the formats in `ner/patterns.jsonl`; the small spaCy model mislabels some domain words.
- English only; code-mixed Hinglish tickets are out of scope.
""")
