"""Golden-value tests for the eval metric functions (hand-computed expectations)."""

import numpy as np

from transaction_intelligence.eval import metrics


def test_macro_f1_hand_computed():
    # a: P=1.0 R=0.5 F1=0.667 ; b: P=0.667 R=1.0 F1=0.8 ; macro = 0.7333
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    assert abs(metrics.macro_f1(y_true, y_pred, ["a", "b"]) - 0.73333) < 1e-4


def test_accuracy():
    assert metrics.accuracy(["a", "b", "c"], ["a", "b", "x"]) == 2 / 3
    assert metrics.accuracy([], []) == 0.0


def test_ece_single_bin():
    # both confidences land in one bin; acc 0.5, conf 0.9 -> ECE 0.4
    assert abs(metrics.expected_calibration_error([0.9, 0.9], [True, False]) - 0.4) < 1e-9


def test_ece_perfectly_calibrated_is_zero():
    # conf 1.0 and correct -> no gap
    assert metrics.expected_calibration_error([1.0, 1.0], [True, True]) == 0.0


def test_accuracy_at_coverage_curve():
    df = metrics.accuracy_at_coverage([0.9, 0.8, 0.2, 0.1], [True, True, False, False], steps=4)
    accs = df["accuracy"].tolist()
    assert accs[0] == 1.0  # top 25% (most confident) all correct
    assert abs(accs[2] - 2 / 3) < 1e-9  # top 75%
    assert accs[-1] == 0.5  # full coverage
    assert np.isclose(df["coverage"].iloc[-1], 1.0)


def test_per_class_and_confusion_shapes():
    y_true = ["a", "b", "a"]
    y_pred = ["a", "a", "a"]
    rep = metrics.per_class_report(y_true, y_pred, ["a", "b"])
    assert list(rep["label"]) == ["a", "b"]
    cm = metrics.confusion_df(y_true, y_pred, ["a", "b"])
    assert cm.loc["a", "a"] == 2 and cm.loc["b", "a"] == 1
