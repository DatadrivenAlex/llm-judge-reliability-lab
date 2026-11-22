"""Metrics for paired-order LLM judge audits."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

from .models import Case, Vote


def load_cases(path: str | Path) -> list[Case]:
    """Read strict JSONL records; reject duplicate IDs and malformed lines."""
    cases: list[Case] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"line {line_no}: each record must be a JSON object")
            case = Case.from_mapping(row, line_no)
            if case.case_id in seen:
                raise ValueError(f"line {line_no}: duplicate case_id {case.case_id!r}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("input contains no cases")
    return cases


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float] | None:
    if not values:
        return None
    if samples <= 0:
        return None
    rng = random.Random(seed)
    n = len(values)
    boot = sorted(mean(rng.choices(values, k=n)) for _ in range(samples))
    lo = boot[max(0, int(0.025 * samples))]
    hi = boot[min(samples - 1, int(0.975 * samples))]
    return [lo, hi]


def audit(cases: Iterable[Case], *, bootstrap_samples: int = 1000, seed: int = 17) -> dict:
    """Summarize accuracy and paired-order behavior; confidence means P(vote is correct)."""
    rows = list(cases)
    if not rows:
        raise ValueError("at least one case is required")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples cannot be negative")

    accuracy: list[float] = []
    case_accuracy: list[float] = []
    consistency: list[float] = []
    first_position: list[float] = []
    brier: list[float] = []
    confidence_bins = [{"count": 0, "correct": 0, "confidence_sum": 0.0} for _ in range(5)]
    outcome_counts = {vote.value: 0 for vote in Vote}

    for case in rows:
        outcome_counts[case.expected.value] += 1
        swapped_identity = case.swapped.swapped_identity()
        # Average the paired presentations; confidence describes each recorded vote.
        pair_results = (
            (case.original, float(case.original is case.expected)),
            (swapped_identity, float(swapped_identity is case.expected)),
        )
        case_accuracy.append(mean(correct for _, correct in pair_results))
        for vote, correct in pair_results:
            accuracy.append(correct)
            brier.append((case.confidence - correct) ** 2)
            bucket = min(int(case.confidence * 5), 4)
            confidence_bins[bucket]["count"] += 1
            confidence_bins[bucket]["correct"] += int(correct)
            confidence_bins[bucket]["confidence_sum"] += case.confidence
        consistency.append(float(case.original is swapped_identity))
        decisive = [vote for vote in (case.original, case.swapped) if vote is not Vote.TIE]
        if decisive:
            # Votes are in displayed-slot coordinates for each presentation.
            first_position.append(sum(vote is Vote.A for vote in decisive) / len(decisive))

    binned = []
    total = sum(bucket["count"] for bucket in confidence_bins)
    ece = 0.0
    for index, bucket in enumerate(confidence_bins):
        count = bucket["count"]
        if not count:
            continue
        observed = bucket["correct"] / count
        average_confidence = bucket["confidence_sum"] / count
        lower, upper = index / 5, (index + 1) / 5
        binned.append({"range": [lower, upper], "count": count, "mean_confidence": average_confidence, "accuracy": observed})
        ece += count / total * abs(observed - average_confidence)

    return {
        "case_count": len(rows),
        "presentation_count": len(accuracy),
        "reference_outcomes": outcome_counts,
        "pairwise_accuracy": mean(accuracy),
        "pairwise_accuracy_ci95": _bootstrap_ci(case_accuracy, bootstrap_samples, seed),
        "swap_consistency": mean(consistency),
        "swap_consistency_ci95": _bootstrap_ci(consistency, bootstrap_samples, seed + 1),
        "first_position_rate": _mean_or_none(first_position),
        "first_position_ci95": _bootstrap_ci(first_position, bootstrap_samples, seed + 2),
        "confidence_brier": mean(brier),
        "confidence_ece_5_bins": ece,
        "confidence_bins": binned,
        "interpretation": (
            "Position rates near 0.5 indicate no aggregate first-slot preference; inspect the "
            "paired cases and uncertainty interval before drawing conclusions."
        ),
    }
