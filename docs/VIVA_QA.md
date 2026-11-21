# Viva Preparation: Likely Questions and Short Answers

Every team member should be able to answer all of these *in their own words* and point to
the file where it is implemented.

## A. Problem & data

**1. What problem does your system solve?**
A university helpdesk receives free-text tickets. Routing them to the right department by hand is slow and error-prone. Our system predicts one of 5 categories (IT_WiFi, LMS_Portal, Academics_Exams, Fees_Accounts, Hostel_Facilities), gives a confidence score and highlights key entities (student ID, course code, error code, amount, location...) so staff can act faster.

**2. Why is your data synthetic? Isn't that a weakness?**
Public helpdesk datasets are corporate IT and don't contain university departments. We generated tickets from ~130 hand-written templates with slot filling and noise (`src/data/generate_dataset.py`). **Yes, it is a limitation:** templates share structure between train and test, so scores are optimistic. Real tickets would contain more varied phrasing, code-mixing (Hinglish) and longer stories. `load_dataset.py` shows how to plug in a real CSV.

**3. How did you stop the task from being trivially easy?**
Shared vocabulary across classes ("portal", "password", "payment", "room", "exam"), inline paraphrase alternatives, ~10% multi-intent tickets, ~4% vague tickets, typos/abbreviations/casing noise, and ~1.5% deliberately mis-filed labels, as found in real ticket logs.

**4. Why a 70/15/15 stratified split?**
Train to fit, validation to choose hyper-parameters / best model / early stopping, test only once for the final report. Stratification keeps the class proportions equal in every split, which matters because the classes are mildly imbalanced.

**5. What is data leakage and how did you prevent it?**
Leakage is when information from the test set influences training. We (a) split before any fitting, (b) fit TF-IDF vocabulary/IDF only on train (re-fitted inside every CV fold by the sklearn `Pipeline`), (c) tune hyper-parameters with CV on train, (d) select the best model on validation, never on test. The preprocessor is stateless (learns nothing), so caching its output for all splits is safe.

## B. Preprocessing

**6. Why lemmatisation instead of stemming?**
Lemmas are real words ("studies"→"study", "paid"→"pay"), so features stay interpretable. Porter stemming creates non-words ("facil", "univers") and can over-merge unrelated words. In `notebooks/02_experiments.ipynb` both give similar F1, so we prefer the interpretable one.

**7. Why keep negations when removing stop-words?**
"payment not received" and "payment received" mean opposite things. Standard stop-word lists contain *not, no, never, cannot*, so we remove them from the list. This is our explicit stop-word policy (`keep_words` in `config.yaml`).

**8. Why replace URLs, IDs and error codes with placeholder tokens?**
Every student ID is unique, so as raw tokens they would add thousands of useless, never-repeated features. Replacing them with `studentidtoken` / `errorcodetoken` keeps the useful signal ("an error code was mentioned", which hints at IT/LMS) without noise.

**9. Why two preprocessing modes?**
Sparse models only see surface tokens, so aggressive normalisation helps them. Transformers were pre-trained on natural sentences: word order, function words and inflections are signal for them, and WordPiece handles rare words. So BERT/SBERT get only light cleaning.

## C. Representations & classical models

**10. Difference between Bag-of-Words and TF-IDF?**
BoW counts occurrences. TF-IDF multiplies term frequency by inverse document frequency, `log(N/df)`, so words appearing in many tickets ("please", "issue") are down-weighted and distinctive words ("challan", "eduroam") up-weighted. We also use `sublinear_tf` (1+log tf).

**11. Why add character n-grams?**
Robustness to typos: "pasword" and "password" share most character 2-5-grams. Experiment 4 in `02_experiments.ipynb` measures this on typo-corrupted validation text.

**12. How does Multinomial Naive Bayes work and why is it "naive"?**
It picks the class maximising P(class)·Π P(word|class). It's "naive" because it assumes words are independent given the class. That's false, but it works well for text and is very fast. `alpha` is Laplace/Lidstone smoothing, so unseen words don't give zero probability.

**13. Logistic Regression vs Linear SVM?**
Both are linear classifiers on TF-IDF. LR minimises log-loss and gives probabilities. SVM maximises the margin (hinge loss), is often slightly stronger on sparse text, but has no probabilities. So we wrap it in `CalibratedClassifierCV` (Platt scaling) to obtain a confidence score.

**14. What is `C`?**
Inverse regularisation strength. Small C means stronger regularisation (simpler model, less overfitting), large C means weaker regularisation. We tuned it with 5-fold CV.

**15. How did you choose hyper-parameters?**
`GridSearchCV` with stratified 5-fold CV on the training split, scoring = macro-F1 (`config.yaml` → `tfidf.*_grid`). Best parameters are stored in `artifacts/classical/*_metrics.json`.

## D. Transformers

**16. What is Sentence-BERT and how did you use it?**
A BERT-style model fine-tuned (siamese network) so that sentence embeddings are comparable with cosine similarity. We used `all-MiniLM-L6-v2` (384-d) as a *frozen* feature extractor and trained Logistic Regression on top ("SBERT + LR").

**17. How does DistilBERT differ from BERT?**
DistilBERT has 6 layers instead of 12 and ~66M instead of ~110M parameters. It is trained by knowledge distillation (the student mimics BERT's outputs). It is ~40% smaller and ~60% faster while keeping ~97% of BERT's language-understanding performance. No token-type embeddings, no NSP objective.

**18. What does fine-tuning mean here?**
We add a classification layer on top of the [CLS] representation and update **all** weights with a small learning rate (3e-5) for a few epochs on our labelled tickets, using AdamW, linear warm-up/decay and weight decay.

**19. How did you prevent overfitting?**
Validation macro-F1 is evaluated every epoch, the best checkpoint is kept (`load_best_model_at_end`), early stopping uses patience 2, plus weight decay, few epochs, and pretrained initialisation. For classical models: regularisation (C/alpha) tuned with CV, `min_df`, `max_features`.

**20. Is BERT better?**
Only claim what the numbers show (see `reports/model_comparison.csv`). On the in-distribution test set all models score about 0.985 macro-F1, so there is no winner. On the **unseen-template** test set DistilBERT scores 0.911 vs 0.845 for TF-IDF+LR and 0.729 for SVM, so it generalises better to new wording. It is also ~270× larger (269 MB vs 1 MB) and needs a GPU to train quickly, while TF-IDF + linear models are tiny and interpretable. The right choice depends on accuracy vs latency vs explainability.

**20b. What is the unseen-template test set and why did you add it?**
Our tickets come from templates, so the normal test set contains new tickets written from the *same* templates as training, which is close to memorisation. We removed 20 templates (~15% per class) *before* splitting and test only on tickets from them. It simulates new wording and is the most honest estimate we have of real-world performance.

## E. NER

**21. How does entity extraction work?**
It is a hybrid. (1) Regex rules for formatted domain entities (STUDENT_ID, COURSE_CODE, ERROR_CODE, AMOUNT, TXN_ID, dates, rooms). (2) A spaCy `EntityRuler` with case-insensitive phrase lists (DEPARTMENT, SYSTEM, LOCATION), placed before the statistical NER. (3) spaCy `en_core_web_sm` for PERSON/DATE/TIME/ORG... On overlap, rules win (regex > phrase > model), and the longer span wins otherwise.

**22. Why rules and not train a custom NER model?**
Domain entities have very regular formats and we have no labelled NER training data. Rules are precise, transparent and editable (`ner/patterns.jsonl`). Their weakness is unseen formats (e.g. `ERR-504`), which our gold evaluation exposes.

**23. How did you evaluate NER?**
On 50 hand-labelled tickets (`data/ner_gold.json`), entity-level precision/recall/F1 per label. *Strict* scoring requires identical text, *lenient* scoring allows boundary differences (e.g. "Dr. Priya Nair" vs "Priya Nair").

## F. Evaluation & error analysis

**24. Why macro-F1 as the main metric?**
Classes are imbalanced. Macro-F1 averages F1 equally over classes, so a model can't look good by doing well only on the biggest class. Accuracy and weighted-F1 are reported too.

**25. What is a confusion matrix and what did yours show?**
Rows are true classes and columns are predicted classes. Off-diagonal cells show which categories are confused. On the test set our most confused pairs were Hostel→IT (hostel Wi-Fi), Hostel→Fees (hostel refunds, fines) and IT→Academics. On unseen templates, single words misled the classical model: "notifications from the LMS are not coming to my *email*" went to IT, and "network down in *Block* B" went to Hostel.

**26. What were the main error causes?**
Multi-intent tickets, vague/very short tickets, misleading cue words, typos/abbreviations, and mis-filed gold labels (see `reports/error_analysis.md`). Because the data is synthetic we could verify that errors concentrate in ambiguous/vague/mislabelled tickets.

**27. Precision vs recall in this application?**
For routing, low recall for a class means its tickets go to the wrong department and wait longer. Low precision means departments get irrelevant tickets. Low-confidence predictions can be sent to human review.

## G. Application & engineering

**28. How is the system used?**
Through the Streamlit app (`streamlit run app.py`) with single-ticket, batch CSV/TXT upload, dashboard and model-performance tabs, or through the CLI fallback `python -m src.models.predict "text"`.

**29. How is priority decided?**
A transparent keyword rule (urgent, exam tomorrow, deducted, no water...). It is not learned. A future extension is to train a priority classifier.

**30. How do you make results reproducible?**
A global seed (`config.yaml`), seeded data generation, fixed splits, config-driven paths, pinned requirement ranges, and saved models and metrics.

**31. What would you do with more time?**
Collect and annotate real tickets, add multi-label output, train a custom NER, support Hinglish, add an active-learning loop where agents correct low-confidence predictions, and add an SLA dashboard per department.
