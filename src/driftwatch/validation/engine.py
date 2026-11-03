"""Configurable Data Validation Engine with Quarantine Routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from driftwatch.validation.rules import RuleSeverity, RuleViolation, ValidationReport

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configurable thresholds for data validation."""
    required_columns: list[str] = field(default_factory=lambda: [
        "date", "store_id", "item_id", "category", "sell_price", "on_promotion", "sales"
    ])
    primary_key: list[str] = field(default_factory=lambda: ["store_id", "item_id", "date"])
    expected_types: dict[str, str] = field(default_factory=lambda: {
        "sales": "numeric",
        "sell_price": "numeric",
        "on_promotion": "numeric",
        "store_id": "string",
        "item_id": "string",
        "category": "string",
    })
    max_null_ratio: dict[str, float] = field(default_factory=lambda: {
        "sales": 0.0,
        "sell_price": 0.0,
        "date": 0.0,
        "store_id": 0.0,
        "item_id": 0.0,
    })
    value_ranges: dict[str, tuple[float | None, float | None]] = field(default_factory=lambda: {
        "sales": (0.0, None),         # non-negative sales
        "sell_price": (0.01, 10000.0), # realistic price
        "on_promotion": (0, 1),
    })
    allowed_categories: dict[str, set[str]] | None = None
    iqr_outlier_factor: float = 3.0  # Tukey extreme outlier multiplier
    quarantine_on_error: bool = True


class DataValidationEngine:
    """Engine executing configurable data quality rules and quarantine isolation."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate(self, df: pd.DataFrame) -> tuple[ValidationReport, pd.DataFrame, pd.DataFrame]:
        """Validate an input DataFrame against configured data quality constraints.
        
        Returns:
            Tuple of:
                - ValidationReport (metadata, rule violations)
                - Clean DataFrame (rows passing all ERROR-level checks)
                - Quarantined DataFrame (rows failing at least one ERROR-level check)
        """
        violations: list[RuleViolation] = []
        n_rows = len(df)
        if n_rows == 0:
            violations.append(
                RuleViolation(
                    rule_name="empty_dataset",
                    severity=RuleSeverity.ERROR,
                    affected_column=None,
                    violation_count=0,
                    total_count=0,
                    message="Input dataset contains zero rows.",
                )
            )
            report = ValidationReport(is_valid=False, violations=violations, clean_count=0, quarantined_count=0)
            return report, df.copy(), df.copy()

        error_masks: list[pd.Series] = []

        # 1. Schema check (Required Columns)
        missing_cols = [c for c in self.config.required_columns if c not in df.columns]
        if missing_cols:
            violations.append(
                RuleViolation(
                    rule_name="required_columns",
                    severity=RuleSeverity.ERROR,
                    affected_column=None,
                    violation_count=len(missing_cols),
                    total_count=len(self.config.required_columns),
                    message=f"Missing required columns: {missing_cols}",
                    details={"missing_columns": missing_cols},
                )
            )
            # If critical columns are missing, entire dataset is quarantined
            report = ValidationReport(is_valid=False, violations=violations, clean_count=0, quarantined_count=n_rows)
            return report, df.iloc[0:0].copy(), df.copy()

        # 2. Type checks
        for col, expected_type in self.config.expected_types.items():
            if col in df.columns:
                if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                    violations.append(
                        RuleViolation(
                            rule_name="type_check",
                            severity=RuleSeverity.ERROR,
                            affected_column=col,
                            violation_count=n_rows,
                            total_count=n_rows,
                            message=f"Column '{col}' has type '{df[col].dtype}', expected numeric.",
                        )
                    )
                    error_masks.append(pd.Series(True, index=df.index))

        # 3. Null checks
        for col, max_null_ratio in self.config.max_null_ratio.items():
            if col in df.columns:
                null_mask = df[col].isna()
                null_count = int(null_mask.sum())
                actual_ratio = null_count / n_rows
                if actual_ratio > max_null_ratio:
                    severity = RuleSeverity.ERROR if max_null_ratio == 0.0 else RuleSeverity.WARNING
                    violations.append(
                        RuleViolation(
                            rule_name="null_rate",
                            severity=severity,
                            affected_column=col,
                            violation_count=null_count,
                            total_count=n_rows,
                            message=f"Column '{col}' null ratio {actual_ratio:.2%} exceeds threshold {max_null_ratio:.2%}",
                            details={"actual_ratio": actual_ratio, "threshold": max_null_ratio},
                        )
                    )
                    if severity == RuleSeverity.ERROR:
                        error_masks.append(null_mask)

        # 4. Duplicate checks (Primary Key)
        if all(c in df.columns for c in self.config.primary_key):
            dup_mask = df.duplicated(subset=self.config.primary_key, keep=False)
            dup_count = int(dup_mask.sum())
            if dup_count > 0:
                violations.append(
                    RuleViolation(
                        rule_name="duplicate_primary_key",
                        severity=RuleSeverity.ERROR,
                        affected_column=", ".join(self.config.primary_key),
                        violation_count=dup_count,
                        total_count=n_rows,
                        message=f"Found {dup_count} duplicate primary key records on {self.config.primary_key}",
                    )
                )
                error_masks.append(dup_mask)

        # 5. Range checks
        for col, (min_val, max_val) in self.config.value_ranges.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                mask = pd.Series(False, index=df.index)
                if min_val is not None:
                    mask = mask | (df[col] < min_val)
                if max_val is not None:
                    mask = mask | (df[col] > max_val)

                count = int(mask.sum())
                if count > 0:
                    violations.append(
                        RuleViolation(
                            rule_name="value_range",
                            severity=RuleSeverity.ERROR,
                            affected_column=col,
                            violation_count=count,
                            total_count=n_rows,
                            message=f"Column '{col}' has {count} values outside valid range [{min_val}, {max_val}]",
                            details={"min_val": min_val, "max_val": max_val},
                        )
                    )
                    error_masks.append(mask)

        # 6. Category checks
        if self.config.allowed_categories:
            for col, allowed_set in self.config.allowed_categories.items():
                if col in df.columns:
                    cat_mask = ~df[col].astype(str).isin(allowed_set)
                    count = int(cat_mask.sum())
                    if count > 0:
                        unseen = sorted(df.loc[cat_mask, col].astype(str).unique().tolist())
                        violations.append(
                            RuleViolation(
                                rule_name="unexpected_categories",
                                severity=RuleSeverity.WARNING,
                                affected_column=col,
                                violation_count=count,
                                total_count=n_rows,
                                message=f"Column '{col}' contains {count} records with unexpected categories: {unseen}",
                                details={"unseen_categories": unseen},
                            )
                        )

        # 7. Tukey Outlier check
        for col in ["sales", "sell_price"]:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if len(series) >= 20:
                    q25 = series.quantile(0.25)
                    q75 = series.quantile(0.75)
                    iqr = q75 - q25
                    if iqr > 0:
                        upper_fence = q75 + (self.config.iqr_outlier_factor * iqr)
                        lower_fence = q25 - (self.config.iqr_outlier_factor * iqr)
                        outlier_mask = (df[col] < lower_fence) | (df[col] > upper_fence)
                        count = int(outlier_mask.sum())
                        if count > 0:
                            violations.append(
                                RuleViolation(
                                    rule_name="tukey_outlier",
                                    severity=RuleSeverity.WARNING,
                                    affected_column=col,
                                    violation_count=count,
                                    total_count=n_rows,
                                    message=f"Column '{col}' has {count} extreme Tukey outliers outside [{lower_fence:.2f}, {upper_fence:.2f}]",
                                    details={"lower_fence": lower_fence, "upper_fence": upper_fence},
                                )
                            )

        # 8. Date validity check
        if "date" in df.columns:
            date_nulls = pd.to_datetime(df["date"], errors="coerce").isna()
            date_null_count = int(date_nulls.sum())
            if date_null_count > 0:
                violations.append(
                    RuleViolation(
                        rule_name="invalid_dates",
                        severity=RuleSeverity.ERROR,
                        affected_column="date",
                        violation_count=date_null_count,
                        total_count=n_rows,
                        message=f"Found {date_null_count} unparseable or invalid date values.",
                    )
                )
                error_masks.append(date_nulls)

        # Build clean vs quarantined partitions
        if error_masks and self.config.quarantine_on_error:
            combined_error_mask = pd.concat(error_masks, axis=1).any(axis=1)
            quarantined_df = df[combined_error_mask].copy().reset_index(drop=True)
            clean_df = df[~combined_error_mask].copy().reset_index(drop=True)
        else:
            quarantined_df = df.iloc[0:0].copy()
            clean_df = df.copy()

        has_errors = any(v.severity == RuleSeverity.ERROR for v in violations)
        report = ValidationReport(
            is_valid=not has_errors,
            violations=violations,
            clean_count=len(clean_df),
            quarantined_count=len(quarantined_df),
            metadata={"total_checked": n_rows},
        )
        return report, clean_df, quarantined_df
