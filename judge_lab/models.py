"""Validated records and vote normalization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Vote(str, Enum):
    A = "A"
    B = "B"
    TIE = "tie"

    def swapped_identity(self) -> Vote:
        """Map a vote made after swapping display order back to original candidate IDs."""
        if self is Vote.A:
            return Vote.B
        if self is Vote.B:
            return Vote.A
        return Vote.TIE


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    expected: Vote
    original: Vote
    swapped: Vote
    confidence: float

    @classmethod
    def from_mapping(cls, row: dict[str, Any], line: int = 0) -> Case:
        prefix = f"line {line}: " if line else ""
        required = {"case_id", "expected", "original", "swapped", "confidence"}
        missing = required - row.keys()
        extra = row.keys() - required
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing fields {sorted(missing)}")
            if extra:
                parts.append(f"unknown fields {sorted(extra)}")
            raise ValueError(prefix + "; ".join(parts))
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(prefix + "case_id must be a non-empty string")
        try:
            expected = Vote(row["expected"])
            original = Vote(row["original"])
            swapped = Vote(row["swapped"])
        except (ValueError, TypeError) as exc:
            raise ValueError(prefix + "votes must be 'A', 'B', or 'tie'") from exc
        confidence = row["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError(prefix + "confidence must be a number from 0 to 1")
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError(prefix + "confidence must be a number from 0 to 1")
        return cls(case_id.strip(), expected, original, swapped, confidence)
