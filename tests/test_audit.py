import json

import pytest

from judge_lab.audit import audit, load_cases
from judge_lab.models import Case, Vote


def make_case(expected="A", original="A", swapped="B", confidence=0.8, case_id="x"):
    return Case(case_id, Vote(expected), Vote(original), Vote(swapped), confidence)


def test_swap_normalization_and_position_bias_are_distinct():
    report = audit([make_case()], bootstrap_samples=200)
    assert report["swap_consistency"] == 1.0
    assert report["first_position_rate"] == 0.5
    assert report["pairwise_accuracy"] == 1.0


def test_same_display_slot_winner_exposes_position_bias():
    report = audit([make_case(expected="A", original="A", swapped="A")], bootstrap_samples=50)
    assert report["swap_consistency"] == 0.0
    assert report["first_position_rate"] == 1.0


def test_tie_is_an_explicit_reference_outcome():
    report = audit([make_case(expected="tie", original="tie", swapped="tie")], bootstrap_samples=50)
    assert report["reference_outcomes"]["tie"] == 1
    assert report["pairwise_accuracy"] == 1.0
    assert report["first_position_rate"] is None


def test_bootstrap_is_reproducible_and_bounded():
    cases = [make_case(case_id=str(i), original="B" if i % 2 else "A") for i in range(12)]
    first = audit(cases, bootstrap_samples=500, seed=9)
    second = audit(cases, bootstrap_samples=500, seed=9)
    assert first == second
    low, high = first["pairwise_accuracy_ci95"]
    assert 0 <= low <= high <= 1


def test_confidence_metrics_and_bins_cover_every_presentation():
    report = audit([make_case(confidence=0.9)], bootstrap_samples=20)
    assert report["confidence_brier"] == pytest.approx(0.01)
    assert sum(bucket["count"] for bucket in report["confidence_bins"]) == 2
    assert report["confidence_ece_5_bins"] >= 0


def test_jsonl_loader_validates_fields_and_duplicates(tmp_path):
    path = tmp_path / "cases.jsonl"
    row = {"case_id": "one", "expected": "A", "original": "A", "swapped": "B", "confidence": 0.8}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_cases(path)


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError, match="0 to 1"):
        Case.from_mapping({"case_id": "x", "expected": "A", "original": "A", "swapped": "B", "confidence": 1.2})


def test_empty_audit_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        audit([])
