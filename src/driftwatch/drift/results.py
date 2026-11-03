"""Drift Engine Result Models and Evidence Classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class EvidenceClass(str, Enum):
    """Rigorous categorization of operational ML signals.
    
    Principles:
    - STATISTICAL_SHIFT: Feature or raw input distribution divergence (unlabeled).
    - PREDICTION_SHIFT: Model prediction distribution divergence (unlabeled).
    - ESTIMATED_PERFORMANCE: Heuristic or density-ratio performance proxy without labels.
    - VERIFIED_PERFORMANCE: True error degradation computed against verified ground-truth labels.
    """
    STATISTICAL_SHIFT = "STATISTICAL_SHIFT"
    PREDICTION_SHIFT = "PREDICTION_SHIFT"
    ESTIMATED_PERFORMANCE = "ESTIMATED_PERFORMANCE"
    VERIFIED_PERFORMANCE = "VERIFIED_PERFORMANCE"


class TestType(str, Enum):
    """Statistical drift test identifiers."""
    __test__ = False  # Prevent pytest from collecting this enum as a test class
    PSI = "PSI"
    KOLMOGOROV_SMIRNOV = "KOLMOGOROV_SMIRNOV"
    JENSEN_SHANNON = "JENSEN_SHANNON"
    WASSERSTEIN = "WASSERSTEIN"
    CHI_SQUARE = "CHI_SQUARE"
    MEAN_SHIFT = "MEAN_SHIFT"
    VARIANCE_SHIFT = "VARIANCE_SHIFT"
    MISSING_RATE = "MISSING_RATE"
    PREDICTION_DRIFT = "PREDICTION_DRIFT"


@dataclass(frozen=True)
class FeatureDriftResult:
    """Comprehensive, auditable record of an individual statistical drift evaluation.
    
    Adheres strictly to the brief requirement:
    dataset_id, reference_id, window, feature, test, statistic, p_value,
    adjusted_p_value, effect_size, threshold, decision, sample_sizes,
    timestamp, config_hash.
    """
    dataset_id: str
    reference_id: str
    window: str
    feature: str
    test: TestType
    statistic: float
    p_value: float | None
    adjusted_p_value: float | None
    effect_size: float
    threshold: float
    decision: bool  # True if drift is confirmed
    sample_size_ref: int
    sample_size_curr: int
    evidence_class: EvidenceClass
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    config_hash: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["test"] = self.test.value
        d["evidence_class"] = self.evidence_class.value
        return d
