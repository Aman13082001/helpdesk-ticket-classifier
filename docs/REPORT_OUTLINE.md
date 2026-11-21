# Project Report (draft, pre-filled from actual results)

**University Helpdesk Ticket Classifier: Ticket Routing and Entity Extraction with Classical and Transformer NLP**
ETMMTI303 – Language Technologies and Text Intelligence · MCA (AI & ML) Sem III · K.R. Mangalam University

> All numbers below come from the files in `reports/`. Text in _italics with brackets_ must be written by the team.
> Rewrite the prose in your own words: the viva will test understanding, not this draft.

---

## Abstract

University helpdesks receive unstructured tickets that must be routed manually. We built an end-to-end NLP system that classifies a ticket into one of five departments and extracts key entities (student IDs, course codes, error codes, amounts, transaction IDs, locations, systems). Using a 5,843-ticket synthetic dataset, we compared Naive Bayes, Logistic Regression and Linear SVM on BoW/TF-IDF features with Sentence-BERT + Logistic Regression and a fine-tuned DistilBERT.

On an in-distribution test set all models reach about 0.985 macro-F1. On a stricter test set built from phrasing templates never seen in training, DistilBERT achieves 0.911 macro-F1 versus 0.845 for the best classical model (TF-IDF + LR) and 0.729 for the SVM. This shows the transformer's advantage lies in generalising to new wording, at ~270× the model size. A hybrid rule-plus-spaCy NER reaches 0.968 micro-F1 on 50 hand-labelled tickets. A Streamlit application exposes single-ticket and batch prediction, entity highlighting and an analytics dashboard.

## 1. Introduction
* _[Motivation: volume of tickets at KRMU, delays caused by mis-routing]_
* Contributions: (1) a realistic synthetic ticket corpus with controlled ambiguity and noise; (2) a reproducible classical-vs-transformer comparison under two test regimes; (3) a hybrid NER for university entities; (4) a deployable Streamlit app.

## 2. Problem Statement
* **Input:** free-text ticket (single, or CSV/TXT batch). **Output:** category, confidence, department, priority, entities.
* **Users:** helpdesk agents, department admins, students.
* **Categories:** IT_WiFi, LMS_Portal, Academics_Exams, Fees_Accounts, Hostel_Facilities (definitions in `docs/DATA_DICTIONARY.md` §3).
* **Success criteria:** macro-F1 ≥ 0.85 on held-out test; NER correct on ≥ 80% of a 50-ticket sample; app works on unseen text. **All met.**

## 3. Literature / Background
* Text classification with BoW/TF-IDF and linear models (Manning et al., 2008; Joachims 1998, SVMs for text).
* Naive Bayes for text (McCallum & Nigam, 1998).
* Transformers: BERT (Devlin et al., 2019), DistilBERT (Sanh et al., 2019), Sentence-BERT (Reimers & Gurevych, 2019).
* NER: statistical (spaCy) vs rule-based; hybrid systems for domain entities.
* _[Add 2–3 papers on IT-ticket / customer-support classification]_

## 4. Dataset & Data Dictionary
Summarise `docs/DATA_DICTIONARY.md`:
* 6,052 raw → 5,843 clean tickets (6 missing, 6 blank, 195 duplicates removed).
* Splits: train 3,498 · val 750 · test 750 · **test_unseen 845** (20 held-out templates).
* Mild imbalance (IT_WiFi 1,490 … LMS_Portal 956). Includes 586 multi-intent, 75 vague and 75 deliberately mislabelled tickets.
* Figures: `class_distribution.png`, `text_length_hist.png`.
* **Limitation:** synthetic. The in-distribution test shares templates with training.

## 5. System Architecture
Insert the Mermaid diagram from the README (or a screenshot of the app's *About* tab). Walk through the 11 layers of the course blueprint.

## 6. Preprocessing
* Two modes: *classical* (aggressive) vs *transformer* (light). Explain why.
* Steps: Unicode/whitespace normalisation → placeholders (URL, e-mail, phone, student ID, course code, error code, amount, txn) → lowercase → contraction/abbreviation expansion → repeated-character squeeze → symbol/emoji removal → spaCy tokenisation → stop-word removal **keeping negations** → lemmatisation (with protected acronyms such as LMS, VPN, UPI).
* Example table from `python -m src.preprocessing.text_preprocessor`.
* Experiments (`notebooks/02_experiments.ipynb`, validation macro-F1):

| Variant | Val macro-F1 | Vocabulary |
|---|---|---|
| no normalisation | 0.9895 | 6,898 |
| Porter stemming | 0.9895 | 6,600 |
| spaCy lemmatisation (ours) | 0.9884 | 6,727 |

| Stop-word policy | Val macro-F1 |
|---|---|
| remove stop-words, keep negations (ours) | 0.9884 |
| remove all stop-words | 0.9884 |
| keep all words | 0.9884 |

Interpretation: on this data the choices make no measurable F1 difference (≤ 1 ticket), so we choose on **interpretability and meaning**. Lemmas are real words, and keeping *not* stops "payment not received" from collapsing into "payment received" (shown in the notebook).

## 7. Text Representations
* BoW (CountVectorizer 1–2-grams); TF-IDF word 1–2-grams with sublinear TF; optional char_wb 2–5-grams; SBERT all-MiniLM-L6-v2 (384-d); DistilBERT contextual [CLS] representation.
* Notebook results (val macro-F1):

| Representation | LR | NB |
|---|---|---|
| BoW unigram | 0.9801 | 0.9819 |
| BoW 1–2-gram | 0.9829 | 0.9884 |
| TF-IDF unigram | 0.9856 | 0.9845 |
| TF-IDF 1–2-gram | 0.9884 | 0.9884 |
| TF-IDF word + char 2–5 | **0.9895** | 0.9869 |

* Typo robustness (25% of words corrupted): word TF-IDF 0.9884 → 0.9846; word + char 0.9895 → 0.9881. Char n-grams halve the drop.
* SBERT UMAP (`embeddings_2d.png`): classes form many small, tight islands, essentially **one per template**. This visual evidence shows the data is templated and motivates the unseen-template test.

## 8. Classical Models
GridSearchCV, stratified 5-fold on train, scoring macro-F1:

| Model | Best parameters | CV macro-F1 | Val macro-F1 | Features |
|---|---|---|---|---|
| MultinomialNB | alpha=1.0, word 1–2g, min_df=1 | 0.9825 | 0.9870 | 6,727 |
| Logistic Regression | C=5, word 1–2g + char | 0.9838 | 0.9895 | 12,866 |
| Linear SVM (+ sigmoid calibration) | C=1, word 1g + char | 0.9852 | 0.9882 | 7,340 |

Explainability: `top_features_lr.png`. Sensible features (fee, refund, wifi, lms, hall ticket, mess, bus) sit next to **spurious ones** (`hi antivirus` → Fees, `moodle kindly` → Hostel). These were learnt from the mis-filed training tickets. _[Discuss: label noise directly corrupts what a linear model learns.]_

## 9. Named Entity Recognition
* Hybrid: 13 regex patterns + 51 phrase patterns (`ner/patterns.jsonl`) + spaCy `en_core_web_sm`. Rules win on overlap. All-lowercase PERSON/ORG model spans are discarded.
* Gold evaluation (50 tickets, 125 mentions): **micro P 0.976 / R 0.960 / F1 0.968**; 42/50 tickets fully correct.

| Label | P | R | F1 |
|---|---|---|---|
| AMOUNT, DATE, DEPARTMENT, EMAIL, PERSON, PHONE, STUDENT_ID, SYSTEM, TXN_ID | 1.00 | 1.00 | 1.00 |
| LOCATION | 1.00 | 0.91 | 0.96 |
| COURSE_CODE | 1.00 | 0.90 | 0.95 |
| ERROR_CODE | 1.00 | 0.71 | 0.83 |
| TIME | 0.50 | 1.00 | 0.67 |

* Entity statistics over the test set: `entity_frequency.png`, `reports/entity_stats_test.json`.

## 10. Transformer Models
* **SBERT + LR:** frozen all-MiniLM-L6-v2 embeddings, LR with C=8 (CV). Val macro-F1 0.9813.
* **DistilBERT:** `distilbert-base-uncased`, max_len 128, lr 3e-5, batch 16, weight decay 0.01, warm-up 10%, 4 epochs, fp16 on RTX 4050. Best checkpoint by val macro-F1 with early stopping (patience 2). Val macro-F1 per epoch: 0.981 → 0.987 → 0.985 → 0.987. Training took 54 s.

## 11. Experimental Setup
Seed 42 everywhere; stratified 70/15/15 split after removing held-out templates. Hyper-parameters chosen on train (CV) and validation only. Both test sets were used once for the final report. Hardware: _[CPU model]_, NVIDIA RTX 4050 Laptop (6 GB), Windows 11, Python 3.12, scikit-learn 1.9, transformers 4.57, spaCy 3.8.

## 12. Results & Comparison
Insert `reports/model_comparison.csv` (README table) and `model_comparison.png`.

**Key findings**
1. **In-distribution test (750):** all models score 0.970–0.987 macro-F1. NB, LR, SVM and DistilBERT differ by ≤ 2 tickets. Not a meaningful ranking.
2. **Unseen-template test (845):** DistilBERT **0.911**, TF-IDF+LR 0.845, SBERT+LR 0.824, BoW+LR 0.792, NB 0.766, SVM **0.729**. Generalisation gap: DistilBERT 0.076 vs SVM 0.257.
3. The SVM was best in CV but generalises worst. Its sharp margin over-fits template-specific n-grams. TF-IDF+LR with char n-grams is the most robust classical model.
4. Frozen SBERT + LR is weaker than fine-tuned DistilBERT on both sets. General-purpose sentence similarity alone does not capture which department must act; fine-tuning does.
5. Cost: DistilBERT is 269 MB vs ~1 MB for the linear models. Latency on a GPU is comparable (~1 ms/ticket batched, 4–20 ms single). On CPU the transformer will be markedly slower _[measure if you demo on CPU]_.
6. **Model selection:** by the protocol (validation macro-F1) TF-IDF+LR is "best". The unseen-template result argues for DistilBERT in deployment. _[Discuss: a better protocol would also hold out templates in validation.]_

## 13. Error Analysis
Summarise `reports/error_analysis.md`:
* Test: 25/750 tickets wrong for at least one model; 9 wrong for all five. Of TF-IDF+LR's 11 errors, **9 are mis-filed gold labels**. Every model "disagrees" with the wrong label, which is the correct behaviour.
* Remaining errors: vague tickets ("room problem", "result issue") and multi-intent tickets ("hall ticket not generated … and my email password is also not working").
* Most confused pairs: Hostel→IT (hostel Wi-Fi), Hostel→Fees (hostel refunds, fines), IT→Academics.
* Unseen templates: hardest for LR were "notifications from the LMS are not coming to my email" (→ IT, because of *email*), "network is down in the entire Block B" (→ Hostel, because of *Block*), "scholarship amount not credited for the first semester" (→ Academics, because of *semester*). DistilBERT was right where LR was wrong on 125 tickets; the reverse happened on 63.
* NER: unseen formats `ERR-504`, `E1021`, `CSE-205`, `Lab-3`, `Hostel D 210`; spaCy over-tags durations ("5 minutes", "night") as TIME.
* _[Team commentary: explain 4 errors in your own words with a fix for each]_

## 14. Application & Visualisation
Screenshots of the five tabs: Single Ticket (prediction, confidence, all-model comparison, LR word contributions, displaCy entities, priority); Batch Upload (CSV/TXT, progress bar, download); Dashboard (category pie, confidence histogram, entity types, top entities per category, weekly trend, top terms); Model Performance; About. Plus the CLI.

## 15. Limitations & Future Work
* Synthetic data, so the in-distribution scores are optimistic. Next steps: collect and annotate real KRMU tickets, then re-run (`load_dataset.py`).
* Single-label output: move to multi-label or a "needs human review" threshold for low confidence (5/750 test tickets < 0.6).
* Priority is rule-based: train a priority classifier.
* NER rules: extend patterns (hyphen variants), or train a custom spaCy NER on annotated tickets.
* Hinglish / code-mixed tickets: try multilingual models (mBERT, MuRIL).
* Active learning loop where agent corrections become new training data. SLA dashboard per department.

## 16. Conclusion
_[3–4 sentences: what you built, the main measured finding (transformer generalises better to new wording at higher cost; classical models are excellent in-distribution), and what you would do next.]_

## References
See README. _[Format in the citation style your department requires.]_

## Team Contribution

| Member | Roll no. | Contributions |
|---|---|---|
| _[name]_ | _[ ]_ | e.g. data generation, preprocessing, EDA |
| _[name]_ | _[ ]_ | e.g. classical models, evaluation |
| _[name]_ | _[ ]_ | e.g. transformers, NER |
| _[name]_ | _[ ]_ | e.g. Streamlit app, report, presentation |
