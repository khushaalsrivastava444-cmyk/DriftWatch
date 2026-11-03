"""Data Quality Validation Rules and Reporting Structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuleSeverity(str, Enum):
    """Validation issue severity tier."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class RuleViolation:
    """Detailed record of a validation failure."""
    rule_name: str
    severity: RuleSeverity
    affected_column: str | None
    violation_count: int
    total_count: int
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregate result of a data quality validation run."""
    is_valid: bool
    violations: list[RuleViolation] = field(default_factory=list)
    quarantined_count: int = 0
    clean_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(v.severity == RuleSeverity.ERROR for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        return any(v.severity == RuleSeverity.WARNING for v in self.violations)

    def summary(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_count": sum(1 for v in self.violations if v.severity == RuleSeverity.ERROR),
            "warning_count": sum(1 for v in self.violations if v.severity == RuleSeverity.WARNING),
            "info_count": sum(1 for v in self.violations if v.severity == RuleSeverity.INFO),
            "quarantined_rows": self.quarantined_count,
            "clean_rows": self.clean_count,
        }
