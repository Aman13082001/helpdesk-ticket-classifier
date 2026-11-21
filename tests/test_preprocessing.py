"""Unit tests for the preprocessing steps.  Run: pytest -q"""
import pickle

import pytest

from src.preprocessing.text_preprocessor import TextPreprocessor


@pytest.fixture(scope="module")
def pre():
    return TextPreprocessor(mode="classical")


def test_normalize_whitespace():
    assert TextPreprocessor.normalize("  hello \n\t world  ") == "hello world"


def test_placeholders(pre):
    out = pre.replace_entities("see https://lms.krmu.edu.in/x mail a.b@gmail.com call 9876543210 id KRM2023MCA045 "
                               "code ERR_403 course MCA-301 paid ₹45,000 txn TXN4829301845")
    for token in ("urltoken", "emailtoken", "phonetoken", "studentidtoken", "errorcodetoken", "coursecodetoken",
                  "amounttoken", "txnidtoken"):
        assert token in out, token


def test_contractions_and_negation_kept(pre):
    out = pre.transform(["I can't login and the payment was not received"])[0]
    assert "not" in out.split()          # negation survives stop-word removal
    assert "login" in out


def test_stopwords_removed(pre):
    out = pre.transform(["the wifi is in the block"])[0].split()
    assert "the" not in out and "is" not in out and "wifi" in out


def test_squeeze_and_symbols(pre):
    assert TextPreprocessor.squeeze_repeats("pleaseee help!!!") == "please help!"
    assert TextPreprocessor.strip_symbols("wifi 🙏 down!!") == "wifi down"


def test_lemmatization(pre):
    out = pre.transform(["payments failed for students"])[0].split()
    assert "payment" in out and "student" in out


def test_transformer_mode_is_light():
    out = TextPreprocessor(mode="transformer").transform(["Hi, WiFi is NOT working!!! pleaseee 🙏 https://x.com/a"])[0]
    assert "NOT" in out and "is" in out and "URL" in out and "🙏" not in out and "pleaseee" not in out


def test_empty_input(pre):
    assert pre.transform(["", None]) == ["", ""]


def test_picklable(pre):
    pre.transform(["load spacy"])
    clone = pickle.loads(pickle.dumps(pre))
    assert clone._nlp is None and clone.transform(["wifi not working"])[0]
