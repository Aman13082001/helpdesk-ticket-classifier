"""Metric helpers shared by training and evaluation scripts.

Macro-F1 is our headline metric: it averages F1 over classes WITHOUT weighting by class
size, so a model cannot look good by doing well only on the large IT_WiFi class while
neglecting the smaller LMS_Portal / Fees_Accounts classes.
"""
from __future__ import annotations

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, labels: list[str] | None = None) -> dict:
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(p_ma), 4), "macro_recall": round(float(r_ma), 4), "macro_f1": round(float(f_ma), 4),
        "weighted_precision": round(float(p_w), 4), "weighted_recall": round(float(r_w), 4), "weighted_f1": round(float(f_w), 4),
    }


def per_class_report(y_true, y_pred, labels: list[str] | None = None) -> dict:
    return classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
