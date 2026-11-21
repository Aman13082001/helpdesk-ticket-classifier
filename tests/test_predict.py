"""Smoke tests for inference (skipped automatically if models have not been trained yet)."""
import numpy as np
import pytest

from src.models.predict import HelpdeskPredictor, priority
from src.utils import load_config


@pytest.fixture(scope="module")
def pred():
    try:
        return HelpdeskPredictor(models=["tfidf_nb", "tfidf_lr", "tfidf_svm"])
    except FileNotFoundError:
        pytest.skip("classical models not trained - run python -m src.models.train_classical")


def test_output_shape(pred):
    res = pred.predict(["wifi not working in block c", "fee deducted twice"])
    for name, r in res.items():
        assert len(r["labels"]) == 2 and r["proba"].shape == (2, 5)
        assert np.allclose(r["proba"].sum(axis=1), 1.0, atol=1e-4)
        assert all(l in pred.labels for l in r["labels"])


@pytest.mark.parametrize("text,label", [
    ("eduroam wifi is not connecting on my laptop, authentication failed", "IT_WiFi"),
    ("assignment upload on moodle fails for the course, cannot submit", "LMS_Portal"),
    ("my hall ticket for the end semester exam is not generated", "Academics_Exams"),
    ("semester fee payment failed but the amount was deducted from my bank account", "Fees_Accounts"),
    ("no water supply in my hostel room since yesterday", "Hostel_Facilities"),
])
def test_clear_cases(pred, text, label):
    assert pred.predict(text, [pred.best_model])[pred.best_model]["labels"][0] == label


def test_empty_text_does_not_crash(pred):
    assert len(pred.predict([""])["tfidf_lr"]["labels"]) == 1


def test_priority_rule():
    cfg = load_config()
    assert priority("exam tomorrow and hall ticket missing", cfg)[0] == "High"
    assert priority("still waiting for my refund", cfg)[0] == "Medium"
    assert priority("please share the syllabus", cfg)[0] == "Low"
