# Presentation Outline (14 slides, ~15 min + live demo)

Suggested split for a 4-person team: A = slides 1–4, B = 5–7, C = 8–10, D = 11–14 + demo.

| # | Slide | Content | Visual |
|---|---|---|---|
| 1 | Title | University Helpdesk Ticket Classifier · Topic 13 · team names | app screenshot |
| 2 | Problem | Tickets arrive as free text → slow manual routing. Users: agents, admins, students. Input → output | one ticket → category + entities |
| 3 | Dataset | 5,843 synthetic tickets, 5 classes, noise, multi-intent, mislabels; **why synthetic + limitation** | `class_distribution.png`, sample tickets |
| 4 | Architecture | 11-layer blueprint | Mermaid diagram / About tab |
| 5 | Preprocessing | two modes; placeholders; negation-aware stop-words; lemmatisation; before/after | before/after table |
| 6 | Representations | BoW vs TF-IDF vs char n-grams vs SBERT; typo-robustness result | notebook table, `embeddings_2d.png` |
| 7 | Classical models | NB / LR / SVM, GridSearchCV on train, best params; explainability (incl. spurious features from label noise) | `top_features_lr.png` |
| 8 | Transformers | SBERT + LR; DistilBERT fine-tuning setup; early stopping; 54 s on GPU | training curve (epoch F1) |
| 9 | NER | hybrid rules + spaCy, rules win; gold F1 0.968 | displaCy screenshot, per-label table |
| 10 | Results ① | in-distribution test: all ≈ 0.985, so "no winner" | `model_comparison.png` |
| 11 | Results ② ★ | **unseen templates: DistilBERT 0.91 vs LR 0.85 vs SVM 0.73**. The real evidence; size/latency trade-off | same chart + `cm_distilbert_unseen.png` vs `cm_tfidf_svm_unseen.png` |
| 12 | Error analysis | 9/11 errors = mislabelled gold; vague & multi-intent; confusion pairs; NER misses | 3 example tickets |
| 13 | Limitations & future | synthetic data, single label, rule priority, English only → real data, multi-label, active learning | – |
| 14 | Conclusion + Demo | key takeaway in one sentence; switch to the app | – |

## Live demo script (~4 min)

Before the presentation: `streamlit run app.py`, open the browser, warm it up with one prediction (models load once).

1. **Single Ticket → "Wi-Fi" example.** Show category, confidence, route-to department, priority, and all five models agreeing.
2. **"Fees" example.** Point out the entity highlights: AMOUNT ₹45,000, TXN_ID, DEPARTMENT Accounts Office. Priority is High because of *deducted*.
3. **"Ambiguous" example** ("fee paid but portal shows unpaid…"). Show lower confidence and model disagreement: *this is why we need human review for low confidence*. Open "Why?" to show the LR word contributions.
4. **"Spurious cue" example** ("No water supply in Hostel B-212 … the warden office is not responding"). TF-IDF+LR and SVM say **IT_WiFi**, while DistilBERT says **Hostel_Facilities** (99%). Open "Why?": the words *warden*, *office* and *not* push towards IT, because in training "warden" mostly appeared in tickets like "hostel internet down, warden says contact IT". This is a live example of a spurious correlation, and of why the transformer generalises better.
5. **Type a fresh ticket live** (ask the examiner for one!), e.g. *"projector in room 304 of block D not working, lecture at 10:30 am"*. This shows the system works on unseen input.
6. **Batch Upload.** Upload a prepared CSV (e.g. 30 tickets), run, show the table and the download button.
7. **Dashboard.** Category pie, confidence histogram, entities per category, weekly trend.
8. **Model Performance.** Comparison table: point at the *unseen-template* column, open the DistilBERT vs SVM unseen confusion matrices.

Backup if the app fails: CLI `python -m src.models.predict "..." --all --entities`, plus screenshots in the slides.

## Likely examiner probes during the demo
* "Type something completely different": try Hinglish ("wifi nahi chal raha hostel me") and admit it's out of scope.
* "Why is confidence 99% on a wrong label?": calibration on templated data; label noise.
* "Is BERT better?": only on unseen phrasing; show the numbers and the cost.
