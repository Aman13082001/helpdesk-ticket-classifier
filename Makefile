# Usage: make all   (or individual targets).  On Windows without make, use run_all.ps1
PY ?= python

.PHONY: all data train train-fast evaluate figures notebooks test app clean

all: data train evaluate figures test

data:
	$(PY) -m src.data.generate_dataset
	$(PY) -m src.data.clean_split

train:
	$(PY) -m src.models.train_classical
	$(PY) -m src.models.train_embedding_lr
	$(PY) -m src.models.train_transformer

train-fast:
	$(PY) -m src.models.train_classical --fast
	$(PY) -m src.models.train_embedding_lr
	$(PY) -m src.models.train_transformer --fast

evaluate:
	$(PY) -m src.evaluation.evaluate
	$(PY) -m src.evaluation.error_analysis

figures:
	$(PY) -m src.evaluation.visualize

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_experiments.ipynb

test:
	$(PY) -m pytest -q

app:
	$(PY) -m streamlit run app.py

clean:
	rm -rf artifacts/* reports/* data/processed/*
