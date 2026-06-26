"""Tests for the TF-IDF + LogReg baseline and its Predictor conformance."""

from transaction_intelligence import taxonomy as tx
from transaction_intelligence.data.datasets import load_split
from transaction_intelligence.eval import Predictor, evaluate
from transaction_intelligence.models.baseline import train_baseline

SAMPLE = load_split("train").sample(2000, random_state=0).reset_index(drop=True)
MODEL = train_baseline(SAMPLE)


def test_is_a_predictor():
    assert isinstance(MODEL, Predictor)


def test_fits_its_training_sample():
    # Fit-sanity (in-distribution), not a generalization claim: the model should learn the sample.
    res = evaluate(MODEL, SAMPLE, "train-sample")
    assert res.subtype.accuracy > 0.8
    assert res.ece is not None


def test_proba_aligned_to_taxonomy_and_matches_predict():
    rows = SAMPLE.head(50)["descriptor"].tolist()
    proba = MODEL.predict_proba(rows)
    assert proba.shape == (50, tx.NUM_SUBTYPES)
    argmax_labels = [tx.SUBTYPES[i] for i in proba.argmax(axis=1)]
    assert argmax_labels == MODEL.predict(rows)
