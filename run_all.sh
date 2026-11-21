#!/usr/bin/env bash
# End-to-end pipeline (Linux/macOS/Git-Bash).  ./run_all.sh [--fast]
set -euo pipefail
cd "$(dirname "$0")"
FAST="${1:-}"
python -m src.data.generate_dataset
python -m src.data.clean_split
python -m src.models.train_classical $FAST
python -m src.models.train_embedding_lr
python -m src.models.train_transformer $FAST
python -m src.evaluation.evaluate
python -m src.evaluation.error_analysis
python -m src.evaluation.visualize
python -m pytest -q
echo "Done. Launch the app with:  streamlit run app.py"
