"""Configurable, scikit-learn compatible text preprocessor.

Two modes
---------
* ``classical``  - aggressive cleaning for BoW / TF-IDF models: lowercase, placeholder
  replacement, punctuation/emoji removal, spaCy tokenisation, stop-word removal (keeping
  negations), lemmatisation.  Sparse models only see surface tokens, so collapsing
  variants ("payments"/"paid"/"pay" -> "pay") and removing noise shrinks the vocabulary and
  improves generalisation.
* ``transformer`` - LIGHT cleaning only (whitespace, URLs/e-mails/phones, emoji, repeated
  characters).  BERT-style models were pre-trained on natural sentences: stop-words,
  word order, inflections and punctuation are *signal* for them, and their WordPiece
  tokenizer already handles rare words, so aggressive cleaning would hurt.

Stop-word policy
----------------
We remove spaCy's English stop-words EXCEPT negations and failure words
(not, no, cannot, unable, failed, never, ...).  "payment failed" vs "payment done" or
"not received" vs "received" differ only by these words, so dropping them would destroy
meaning.

Why lemmatisation (not stemming)?
---------------------------------
Lemmas are real dictionary words ("studies" -> "study", "was" -> "be") which keeps
features interpretable for the explainability table and the viva, whereas Porter stemming
produces non-words ("univers", "facil") and over-merges ("university"/"universe").
The stemming comparison is shown in notebooks/02_experiments.ipynb.

Each step is a separate method so it can be unit-tested and demonstrated on its own.
The class is stateless (``fit`` learns nothing), therefore applying it to all splits
before the TF-IDF fit cannot leak test information.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_KEEP_WORDS = ("not", "no", "nor", "cannot", "unable", "failed", "never", "without", "neither", "none")

# --- regexes (order matters: specific entities before generic punctuation removal) ---
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:edu|com|in|org|net)(?:\.in)?/\S*)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b")
STUDENT_ID_RE = re.compile(r"\bKRM\d{4}[A-Z]{2,6}\d{3}\b", re.I)
TXN_RE = re.compile(r"\b(?:TXN|UTR)-?\d{6,12}\b", re.I)
ERROR_RE = re.compile(r"\b(?:ERR_[A-Z0-9]+|E-\d{3,5}|HTTP\s?\d{3}|error code \d{3,4})\b", re.I)
COURSE_RE = re.compile(r"\b(?:MCA|BCA|MBA|BTCS|CS|ENCS|ETCS|ETMMTI)-?\d{3}\b", re.I)
AMOUNT_RE = re.compile(r"(?:₹\s?|\bRs\.?\s?|\bINR\s?)\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s?rupees\b", re.I)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF️]+")
REPEAT_CHAR_RE = re.compile(r"([a-z])\1{2,}")       # "pleaseee" -> "please"
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1+")         # "!!!" -> "!"
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
CONTRACTIONS = {
    r"\bcan'?t\b": "cannot", r"\bwon'?t\b": "will not", r"\bdon'?t\b": "do not", r"\bdoesn'?t\b": "does not",
    r"\bdidn'?t\b": "did not", r"\bisn'?t\b": "is not", r"\baren'?t\b": "are not", r"\bwasn'?t\b": "was not",
    r"\bhaven'?t\b": "have not", r"\bhasn'?t\b": "has not", r"\bcouldn'?t\b": "could not", r"\bnt\b": "not",
    r"\bn't\b": " not", r"\bi'm\b": "i am", r"\bpls\b": "please", r"\bplz\b": "please", r"\bacct\b": "account",
    r"\bu\b": "you", r"\br\b": "are", r"\bbcoz\b": "because", r"\bregd\b": "regarding", r"\bthx\b": "thanks",
    r"\bsem\b": "semester", r"\bdept\b": "department", r"\bwid\b": "with", r"\bwi-?fi\b": "wifi",
    r"\breciept\b": "receipt", r"\bassgnmt\b": "assignment", r"\blappy\b": "laptop", r"\bmsg\b": "message",
}
CONTRACTION_RES = [(re.compile(k), v) for k, v in CONTRACTIONS.items()]

PROTECTED = {"lms", "vpn", "erp", "upi", "neft", "cgpa", "spss", "sms", "bus", "teams", "gps", "os", "ps", "mess"}

# placeholders are single alphabetic words so spaCy keeps them as one token
PLACEHOLDERS = {
    "url": "urltoken", "email": "emailtoken", "phone": "phonetoken", "student_id": "studentidtoken",
    "txn": "txnidtoken", "error": "errorcodetoken", "course": "coursecodetoken", "amount": "amounttoken",
}


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """sklearn-style preprocessor: ``TextPreprocessor(mode="classical").transform(texts)``."""

    def __init__(self, mode: str = "classical", replace_ids: bool = True, remove_stopwords: bool = True,
                 lemmatize: bool = True, keep_words: Iterable[str] = DEFAULT_KEEP_WORDS,
                 spacy_model: str = "en_core_web_sm"):
        self.mode = mode
        self.replace_ids = replace_ids
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.keep_words = tuple(keep_words)
        self.spacy_model = spacy_model
        self._nlp = None

    # spaCy objects are big and not needed in the pickle -> reload lazily after unpickling
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_nlp"] = None
        return state

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy

            # parser + NER are not needed for lemmas -> disabling them is ~3x faster
            self._nlp = spacy.load(self.spacy_model, disable=["parser", "ner"])
        return self._nlp

    @property
    def stop_words(self) -> set[str]:
        return set(self.nlp.Defaults.stop_words) - set(self.keep_words)

    # ------------------------------------------------------------------ individual steps
    @staticmethod
    def normalize(text: str) -> str:
        """Unicode NFKC + whitespace normalisation (keeps case: entity regexes need it)."""
        text = unicodedata.normalize("NFKC", str(text))
        return re.sub(r"\s+", " ", text).strip()

    def replace_entities(self, text: str) -> str:
        """URLs/e-mails/phones always; IDs/codes/amounts only if ``replace_ids``.

        Replacing them stops thousands of unique IDs becoming noisy, never-seen-again
        vocabulary, while still telling the model "an error code was mentioned".
        """
        text = URL_RE.sub(f" {PLACEHOLDERS['url']} ", text)
        text = EMAIL_RE.sub(f" {PLACEHOLDERS['email']} ", text)
        text = PHONE_RE.sub(f" {PLACEHOLDERS['phone']} ", text)
        if self.replace_ids:
            text = STUDENT_ID_RE.sub(f" {PLACEHOLDERS['student_id']} ", text)
            text = TXN_RE.sub(f" {PLACEHOLDERS['txn']} ", text)
            text = ERROR_RE.sub(f" {PLACEHOLDERS['error']} ", text)
            text = COURSE_RE.sub(f" {PLACEHOLDERS['course']} ", text)
            text = AMOUNT_RE.sub(f" {PLACEHOLDERS['amount']} ", text)
        return text

    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand negations/abbreviations BEFORE punctuation removal so "can't" -> "cannot"."""
        text = text.replace("’", "'")
        for pattern, repl in CONTRACTION_RES:
            text = pattern.sub(repl, text)
        return text

    @staticmethod
    def squeeze_repeats(text: str) -> str:
        return REPEAT_PUNCT_RE.sub(r"\1", REPEAT_CHAR_RE.sub(r"\1", text))

    @staticmethod
    def strip_symbols(text: str) -> str:
        """Remove emoji and every non-alphanumeric character; pure digits are dropped later."""
        text = EMOJI_RE.sub(" ", text)
        return re.sub(r"\s+", " ", NON_ALNUM_RE.sub(" ", text)).strip()

    def tokens(self, docs: Iterable) -> list[list[str]]:
        """spaCy tokenisation + stop-word policy + lemmatisation for an iterable of Docs."""
        stops = self.stop_words if self.remove_stopwords else set()
        out = []
        for doc in docs:
            toks = []
            for t in doc:
                low = t.text.lower()
                # domain acronyms ending in "s" must not be "de-pluralised" (LMS -> "lm")
                w = t.lemma_.lower() if self.lemmatize and low not in PROTECTED else low
                if t.is_space or t.text.isdigit() or len(w) < 2:
                    continue
                if t.text.lower() in stops or w in stops:
                    continue
                toks.append(w)
            out.append(toks)
        return out

    # ------------------------------------------------------------------ pipelines
    def light_clean(self, text: str) -> str:
        """Transformer mode: keep natural language, only remove pure noise."""
        text = self.normalize(text)
        text = URL_RE.sub(" URL ", text)
        text = EMAIL_RE.sub(" EMAIL ", text)
        text = PHONE_RE.sub(" PHONE ", text)
        text = EMOJI_RE.sub(" ", text)
        text = REPEAT_PUNCT_RE.sub(r"\1", re.sub(r"([A-Za-z])\1{2,}", r"\1", text))
        return re.sub(r"\s+", " ", text).strip()

    def pre_tokenize(self, text: str) -> str:
        """All regex steps of classical mode (everything except spaCy)."""
        text = self.normalize(text)
        text = self.replace_entities(text)
        text = text.lower()
        text = self.expand_contractions(text)
        text = self.squeeze_repeats(text)
        return self.strip_symbols(text)

    def fit(self, X, y=None):  # stateless: nothing to learn
        return self

    def transform(self, X) -> list[str]:
        texts = ["" if t is None else str(t) for t in X]
        if self.mode == "transformer":
            return [self.light_clean(t) for t in texts]
        if self.mode != "classical":
            raise ValueError(f"mode must be 'classical' or 'transformer', got {self.mode!r}")
        pre = [self.pre_tokenize(t) for t in texts]
        docs = self.nlp.pipe(pre, batch_size=256)
        return [" ".join(toks) for toks in self.tokens(docs)]


def demo() -> None:
    """Before/after examples (run: python -m src.preprocessing.text_preprocessor)."""
    examples = [
        "Hi, WiFi is NOT working in Block C since 12 Nov!!! pls help 🙏",
        "Fee payment failed but ₹45,000 was deducted, transaction id TXN4829301845. Contact 9876543210",
        "cannot login to the student portal, it shows ERR_403. Student ID KRM2023MCA045",
        "Assignment upload on LMS for MCA-301 failing... see https://lms.krmu.edu.in/course/view.php?id=212",
        "no water supply in Hostel B-212 since yesterday, pleaseee fix it asap",
        "I can't access my acct bcoz password expired, mail me at priya.nair12@gmail.com",
    ]
    classical = TextPreprocessor(mode="classical")
    light = TextPreprocessor(mode="transformer")
    for raw, c, l in zip(examples, classical.transform(examples), light.transform(examples)):
        print(f"\nRAW        : {raw}\nCLASSICAL  : {c}\nTRANSFORMER: {l}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Show before/after preprocessing examples").parse_args()
    demo()
