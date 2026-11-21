"""Hybrid entity extraction: spaCy statistical NER + rule-based domain entities.

Why hybrid?  spaCy's pretrained model knows general entities (PERSON, ORG, DATE, GPE ...)
but has never seen "KRM2023MCA045" or "ERR_403".  Those domain entities have very regular
formats, so rules (regex + phrase lists) are more precise than training a new NER model,
and need no labelled data.

Layers (patterns live in ner/patterns.jsonl, so they can be edited without touching code)
1. regex patterns  (STUDENT_ID, COURSE_CODE, ERROR_CODE, AMOUNT, TXN_ID, DATE, LOCATION, EMAIL, PHONE)
2. spaCy EntityRuler phrase patterns, case-insensitive (DEPARTMENT, LOCATION, SYSTEM);
   placed BEFORE the statistical "ner" component so the model respects these spans
3. spaCy statistical NER (only labels listed in config ner.keep_spacy_labels; all-lowercase
   PERSON/ORG spans are discarded as likely false positives of the small model)

Conflict policy: when spans overlap, RULES WIN (regex > phrase rules > statistical model);
among same-priority spans the longer one wins.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import pandas as pd

from src.utils import get_logger, load_config, resolve

log = get_logger("entity_extractor")

COLORS = {
    "STUDENT_ID": "#f9c74f", "COURSE_CODE": "#90be6d", "DEPARTMENT": "#f8961e", "LOCATION": "#43aa8b",
    "ERROR_CODE": "#f94144", "AMOUNT": "#577590", "TXN_ID": "#a29bfe", "SYSTEM": "#4cc9f0", "DATE": "#ffafcc",
    "TIME": "#ffc8dd", "PERSON": "#cdb4db", "ORG": "#bde0fe", "GPE": "#caffbf", "EMAIL": "#e9c46a",
    "PHONE": "#e76f51", "MONEY": "#8ecae6", "FAC": "#b7e4c7", "PRODUCT": "#ddd", "EVENT": "#fcd5ce",
}
PRIORITY = {"regex": 0, "ruler": 1, "model": 2}


class EntityExtractor:
    def __init__(self, cfg: dict | None = None, spacy_model: str | None = None, patterns_path: str | None = None):
        import spacy

        self.cfg = cfg or load_config()
        model = spacy_model or self.cfg["ner"]["spacy_model"]
        self.keep_spacy = set(self.cfg["ner"]["keep_spacy_labels"])
        self.nlp = spacy.load(model, disable=["lemmatizer"])
        path = resolve(patterns_path or self.cfg["paths"]["patterns"])
        phrase, self.regexes = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)
                if "regex" in p:
                    flags = re.I if "i" in p.get("flags", "") else 0
                    self.regexes.append((p["label"], re.compile(p["regex"], flags)))
                else:
                    phrase.append({"label": p["label"], "pattern": p["pattern"]})
        ruler = self.nlp.add_pipe("entity_ruler", before="ner", config={"phrase_matcher_attr": "LOWER"})
        ruler.add_patterns(phrase)
        self.ruler_labels = {p["label"] for p in phrase}
        log.info("Loaded spaCy '%s' + %d regex + %d phrase patterns", model, len(self.regexes), len(phrase))

    # ------------------------------------------------------------------ core
    def _candidates(self, doc) -> list[dict]:
        text = doc.text
        cands = []
        for label, rx in self.regexes:
            for m in rx.finditer(text):
                cands.append({"text": m.group(0), "label": label, "start": m.start(), "end": m.end(), "source": "regex"})
        for ent in doc.ents:
            if ent.label_ in self.ruler_labels:
                src = "ruler"
            elif ent.label_ in self.keep_spacy:
                # heuristic: the small model tags lowercase phrases ("hostel wifi") as PERSON/ORG;
                # real names/organisations are almost always capitalised in tickets
                if ent.label_ in ("PERSON", "ORG") and ent.text == ent.text.lower():
                    continue
                src = "model"
            else:
                continue
            cands.append({"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char, "source": src})
        return cands

    @staticmethod
    def _resolve(cands: list[dict]) -> list[dict]:
        """Greedy non-overlapping selection: rules first, then longer spans."""
        cands = sorted(cands, key=lambda c: (PRIORITY[c["source"]], -(c["end"] - c["start"]), c["start"]))
        chosen: list[dict] = []
        for c in cands:
            if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in chosen):
                chosen.append(c)
        return sorted(chosen, key=lambda c: c["start"])

    def extract(self, text: str) -> list[dict]:
        if not text or not text.strip():
            return []
        return self._resolve(self._candidates(self.nlp(text)))

    def extract_many(self, texts: list[str], batch_size: int = 128) -> list[list[dict]]:
        texts = [t if isinstance(t, str) else "" for t in texts]
        return [self._resolve(self._candidates(doc)) for doc in self.nlp.pipe(texts, batch_size=batch_size)]

    # ------------------------------------------------------------------ display
    def to_html(self, text: str, ents: list[dict] | None = None) -> str:
        """displaCy HTML with our merged entities (for Streamlit)."""
        from spacy import displacy

        ents = self.extract(text) if ents is None else ents
        data = {"text": text, "ents": [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in ents], "title": None}
        return displacy.render(data, style="ent", manual=True, page=False, options={"colors": COLORS})


def entity_stats(all_ents: list[list[dict]], categories: list[str] | None = None, top_n: int = 8) -> dict:
    """Counts per label, top entity strings overall and per category."""
    label_counts = Counter(e["label"] for ents in all_ents for e in ents)
    top = defaultdict(Counter)
    per_cat = defaultdict(Counter)
    for i, ents in enumerate(all_ents):
        for e in ents:
            top[e["label"]][e["text"].lower()] += 1
            if categories is not None:
                per_cat[categories[i]][f"{e['label']}: {e['text'].lower()}"] += 1
    return {
        "label_counts": dict(label_counts.most_common()),
        "top_entities_per_label": {k: v.most_common(top_n) for k, v in top.items()},
        "top_entities_per_category": {k: v.most_common(top_n) for k, v in per_cat.items()},
        "tickets_with_entities": sum(1 for e in all_ents if e),
        "n_tickets": len(all_ents),
    }


def entities_to_frame(all_ents: list[list[dict]], ids: list | None = None) -> pd.DataFrame:
    rows = [{"row": ids[i] if ids else i, **e} for i, ents in enumerate(all_ents) for e in ents]
    return pd.DataFrame(rows, columns=["row", "text", "label", "start", "end", "source"])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract entities from a ticket")
    ap.add_argument("text", nargs="?", default="Hi, I am KRM2023MCA045 (MCA-301). Paid ₹45,000 via UPI, TXN4829301845, "
                                                "on 12 Nov but the student portal shows ERR_403. Room Hostel B-212. - Dr. Priya Nair")
    a = ap.parse_args()
    for e in EntityExtractor().extract(a.text):
        print(f"{e['label']:<12} {e['text']:<28} [{e['source']}]")
