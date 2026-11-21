"""Tests for the hybrid entity extractor."""
import pytest

from src.ner.entity_extractor import EntityExtractor


@pytest.fixture(scope="module")
def ex():
    return EntityExtractor()


def labels_of(ents):
    return {(e["text"], e["label"]) for e in ents}


def test_domain_regex_entities(ex):
    ents = labels_of(ex.extract("I am KRM2023MCA045, MCA-301 upload fails with ERR_403, paid Rs 4,500 txn TXN4829301845"))
    assert ("KRM2023MCA045", "STUDENT_ID") in ents
    assert ("MCA-301", "COURSE_CODE") in ents
    assert ("ERR_403", "ERROR_CODE") in ents
    assert ("Rs 4,500", "AMOUNT") in ents
    assert ("TXN4829301845", "TXN_ID") in ents


def test_phrase_rules_case_insensitive(ex):
    ents = labels_of(ex.extract("the student portal and moodle are down, contact the accounts office"))
    assert ("student portal", "SYSTEM") in ents
    assert ("moodle", "SYSTEM") in ents
    assert ("accounts office", "DEPARTMENT") in ents


def test_locations(ex):
    ents = labels_of(ex.extract("No water in Hostel B-212 and Lab 3 of Block C"))
    assert {("Hostel B-212", "LOCATION"), ("Lab 3", "LOCATION"), ("Block C", "LOCATION")} <= ents


def test_rules_win_and_no_overlap(ex):
    ents = ex.extract("Paid ₹45,000 on 12 Nov via UPI")
    spans = sorted((e["start"], e["end"]) for e in ents)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "spans overlap"
    amount = [e for e in ents if e["label"] == "AMOUNT"]
    assert amount and amount[0]["source"] == "regex"   # regex AMOUNT beats spaCy MONEY


def test_empty(ex):
    assert ex.extract("") == [] and ex.extract("   ") == []


def test_html(ex):
    html = ex.to_html("Block C wifi down")
    assert "Block C" in html and "LOCATION" in html
