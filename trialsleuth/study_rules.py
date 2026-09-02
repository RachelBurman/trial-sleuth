"""Strict study-rule schema and deterministic rule execution."""

from __future__ import annotations

from typing import Literal, TypeAlias

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .validation import Finding, SEVERITY_ORDER

Severity: TypeAlias = Literal["High", "Medium", "Low"]


class StrictSchema(BaseModel):
    """Base model that rejects coercion and undeclared model output."""

    model_config = ConfigDict(extra="forbid", strict=True)


class RuleBase(StrictSchema):
    severity: Severity


class RequiredRule(RuleBase):
    rule_type: Literal["required"]
    field: str = Field(min_length=1, max_length=200)


class NumericRangeRule(RuleBase):
    rule_type: Literal["numeric_range"]
    field: str = Field(min_length=1, max_length=200)
    minimum: FiniteFloat | None
    maximum: FiniteFloat | None
    minimum_inclusive: bool
    maximum_inclusive: bool

    @model_validator(mode="after")
    def validate_bounds(self) -> NumericRangeRule:
        if self.minimum is None and self.maximum is None:
            raise ValueError("at least one numeric bound is required")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum cannot be greater than maximum")
        if (
            self.minimum is not None
            and self.minimum == self.maximum
            and not (self.minimum_inclusive and self.maximum_inclusive)
        ):
            raise ValueError("equal numeric bounds must both be inclusive")
        return self


class AllowedValuesRule(RuleBase):
    rule_type: Literal["allowed_values"]
    field: str = Field(min_length=1, max_length=200)
    allowed_values: list[str] = Field(min_length=1, max_length=50)
    case_sensitive: bool


class DateOrderRule(RuleBase):
    rule_type: Literal["date_order"]
    earlier_field: str = Field(min_length=1, max_length=200)
    later_field: str = Field(min_length=1, max_length=200)
    allow_equal: bool


class ConditionalRequiredRule(RuleBase):
    rule_type: Literal["conditional_required"]
    condition_field: str = Field(min_length=1, max_length=200)
    operator: Literal["equals", "not_equals", "in"]
    condition_values: list[str] = Field(min_length=1, max_length=20)
    required_field: str = Field(min_length=1, max_length=200)
    case_sensitive: bool


class VisitWindowRule(RuleBase):
    rule_type: Literal["visit_window"]
    participant_field: str = Field(min_length=1, max_length=200)
    visit_field: str = Field(min_length=1, max_length=200)
    date_field: str = Field(min_length=1, max_length=200)
    anchor_visit: str = Field(min_length=1, max_length=200)
    target_visit: str = Field(min_length=1, max_length=200)
    expected_days: int = Field(ge=0, le=3650)
    tolerance_days: int = Field(ge=0, le=3650)


RuleSpec: TypeAlias = (
    RequiredRule
    | NumericRangeRule
    | AllowedValuesRule
    | DateOrderRule
    | ConditionalRequiredRule
    | VisitWindowRule
)


class RuleSet(StrictSchema):
    rules: list[RuleSpec] = Field(max_length=20)
    unsupported_notes: list[str] = Field(max_length=20)


def _missing_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        mask = mask | series.astype("string").str.strip().eq("")
    return mask.fillna(True)


def _positions(mask: pd.Series) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(mask.fillna(False).tolist()) if bool(value))


def _parse_dates(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().mask(_missing_mask(series))
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
    except TypeError:  # pragma: no cover - compatibility with older pandas
        return pd.to_datetime(values, errors="coerce", utc=True)


def _normalised_text(series: pd.Series, case_sensitive: bool = False) -> pd.Series:
    values = series.astype("string").str.strip()
    return values if case_sensitive else values.str.casefold()


def _normalised_value(value: str, case_sensitive: bool = False) -> str:
    value = value.strip()
    return value if case_sensitive else value.casefold()


def _has_parseable_date_values(series: pd.Series) -> bool:
    present = ~_missing_mask(series)
    if not present.any():
        return True
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_datetime64_any_dtype(
        series.dtype
    ):
        return False
    return bool(_parse_dates(series)[present].notna().any())


def rule_columns(rule: RuleSpec) -> tuple[str, ...]:
    if isinstance(rule, (RequiredRule, NumericRangeRule, AllowedValuesRule)):
        return (rule.field,)
    if isinstance(rule, DateOrderRule):
        return (rule.earlier_field, rule.later_field)
    if isinstance(rule, ConditionalRequiredRule):
        return (rule.condition_field, rule.required_field)
    return (rule.participant_field, rule.visit_field, rule.date_field)


def describe_rule(rule: RuleSpec) -> str:
    """Render a rule from constrained values rather than model-authored prose."""

    if isinstance(rule, RequiredRule):
        return f"{rule.field} must be populated"
    if isinstance(rule, NumericRangeRule):
        constraints: list[str] = []
        if rule.minimum is not None:
            operator = ">=" if rule.minimum_inclusive else ">"
            constraints.append(f"{operator} {rule.minimum}")
        if rule.maximum is not None:
            operator = "<=" if rule.maximum_inclusive else "<"
            constraints.append(f"{operator} {rule.maximum}")
        return f"{rule.field} must be {' and '.join(constraints)}"
    if isinstance(rule, AllowedValuesRule):
        values = ", ".join(_normalised_value(value, True) for value in rule.allowed_values)
        return f"{rule.field} must be one of: {values}"
    if isinstance(rule, DateOrderRule):
        relation = "on or before" if rule.allow_equal else "before"
        return f"{rule.earlier_field} must be {relation} {rule.later_field}"
    if isinstance(rule, ConditionalRequiredRule):
        values = ", ".join(
            _normalised_value(value, True) for value in rule.condition_values
        )
        operator = rule.operator.replace("_", " ")
        return (
            f"{rule.required_field} is required when {rule.condition_field} "
            f"{operator} {values}"
        )
    return (
        f"{rule.target_visit} must occur {rule.expected_days} +/- {rule.tolerance_days} "
        f"days after {rule.anchor_visit}"
    )


def validate_rule_set(rule_set: RuleSet, dataframe: pd.DataFrame) -> tuple[str, ...]:
    """Validate rule semantics against the active dataset before execution."""

    errors: list[str] = []
    columns = [str(column) for column in dataframe.columns]
    duplicated_columns = {column for column in columns if columns.count(column) > 1}
    seen_rules: set[str] = set()

    for index, rule in enumerate(rule_set.rules, start=1):
        label = f"Rule {index}"
        for column in rule_columns(rule):
            if column not in columns:
                errors.append(f"{label} references missing column {column!r}.")
            elif column in duplicated_columns:
                errors.append(f"{label} references ambiguous duplicate column {column!r}.")

        signature = rule.model_dump_json(exclude={"severity"})
        if signature in seen_rules:
            errors.append(f"{label} duplicates an earlier rule.")
        seen_rules.add(signature)

        if (
            isinstance(rule, NumericRangeRule)
            and rule.field in columns
            and rule.field not in duplicated_columns
        ):
            present = ~_missing_mask(dataframe[rule.field])
            parsed = pd.to_numeric(dataframe[rule.field], errors="coerce")
            if present.any() and not parsed[present].notna().any():
                errors.append(f"{label} requires numeric values in {rule.field!r}.")

        if isinstance(rule, AllowedValuesRule):
            if any(not value.strip() for value in rule.allowed_values):
                errors.append(f"{label} contains a blank allowed value.")
            values = [
                _normalised_value(value, rule.case_sensitive)
                for value in rule.allowed_values
            ]
            if len(values) != len(set(values)):
                errors.append(f"{label} contains duplicate allowed values.")

        if isinstance(rule, DateOrderRule):
            if rule.earlier_field == rule.later_field:
                errors.append(f"{label} must compare two different date columns.")
            for column in (rule.earlier_field, rule.later_field):
                if (
                    column in columns
                    and column not in duplicated_columns
                    and not _has_parseable_date_values(dataframe[column])
                ):
                    errors.append(f"{label} requires date values in {column!r}.")

        if isinstance(rule, ConditionalRequiredRule):
            if any(not value.strip() for value in rule.condition_values):
                errors.append(f"{label} contains a blank condition value.")
            if rule.operator != "in" and len(rule.condition_values) != 1:
                errors.append(
                    f"{label} operator {rule.operator!r} requires exactly one value."
                )
            values = [
                _normalised_value(value, rule.case_sensitive)
                for value in rule.condition_values
            ]
            if len(values) != len(set(values)):
                errors.append(f"{label} contains duplicate condition values.")

        if isinstance(rule, VisitWindowRule):
            if len(set(rule_columns(rule))) != 3:
                errors.append(f"{label} requires different participant, visit, and date columns.")
            if rule.anchor_visit.casefold() == rule.target_visit.casefold():
                errors.append(f"{label} requires different anchor and target visits.")
            if all(
                column in columns and column not in duplicated_columns
                for column in rule_columns(rule)
            ):
                if not _has_parseable_date_values(dataframe[rule.date_field]):
                    errors.append(
                        f"{label} requires date values in {rule.date_field!r}."
                    )
                visits = _normalised_text(dataframe[rule.visit_field])
                anchor = _normalised_value(rule.anchor_visit)
                target = _normalised_value(rule.target_visit)
                if not visits.eq(anchor).any():
                    errors.append(f"{label} anchor visit {rule.anchor_visit!r} was not found.")
                if not visits.eq(target).any():
                    errors.append(f"{label} target visit {rule.target_visit!r} was not found.")

    return tuple(errors)


def _required_mask(rule: RequiredRule, dataframe: pd.DataFrame) -> pd.Series:
    return _missing_mask(dataframe[rule.field])


def _numeric_range_mask(rule: NumericRangeRule, dataframe: pd.DataFrame) -> pd.Series:
    source = dataframe[rule.field]
    present = ~_missing_mask(source)
    values = pd.to_numeric(source, errors="coerce")
    mask = present & values.isna()
    if rule.minimum is not None:
        below = values.lt(rule.minimum) if rule.minimum_inclusive else values.le(rule.minimum)
        mask = mask | (present & below)
    if rule.maximum is not None:
        above = values.gt(rule.maximum) if rule.maximum_inclusive else values.ge(rule.maximum)
        mask = mask | (present & above)
    return mask


def _allowed_values_mask(rule: AllowedValuesRule, dataframe: pd.DataFrame) -> pd.Series:
    source = dataframe[rule.field]
    present = ~_missing_mask(source)
    values = _normalised_text(source, rule.case_sensitive)
    allowed = [
        _normalised_value(value, rule.case_sensitive) for value in rule.allowed_values
    ]
    return present & ~values.isin(allowed)


def _date_order_mask(rule: DateOrderRule, dataframe: pd.DataFrame) -> pd.Series:
    earlier_source = dataframe[rule.earlier_field]
    later_source = dataframe[rule.later_field]
    present = ~_missing_mask(earlier_source) & ~_missing_mask(later_source)
    earlier = _parse_dates(earlier_source)
    later = _parse_dates(later_source)
    unparsable = earlier.isna() | later.isna()
    out_of_order = earlier.gt(later) if rule.allow_equal else earlier.ge(later)
    return present & (unparsable | out_of_order)


def _conditional_required_mask(
    rule: ConditionalRequiredRule, dataframe: pd.DataFrame
) -> pd.Series:
    condition_source = dataframe[rule.condition_field]
    present = ~_missing_mask(condition_source)
    values = _normalised_text(condition_source, rule.case_sensitive)
    expected = [
        _normalised_value(value, rule.case_sensitive) for value in rule.condition_values
    ]

    matches = values.isin(expected)
    if rule.operator == "not_equals":
        matches = ~matches
    return present & matches & _missing_mask(dataframe[rule.required_field])


def _visit_window_mask(rule: VisitWindowRule, dataframe: pd.DataFrame) -> pd.Series:
    participant_source = dataframe[rule.participant_field]
    visit_source = dataframe[rule.visit_field]
    dates = _parse_dates(dataframe[rule.date_field])
    participants = _normalised_text(participant_source, case_sensitive=True)
    visits = _normalised_text(visit_source)
    valid_participant = ~_missing_mask(participant_source)
    anchor_name = rule.anchor_visit.strip().casefold()
    target_name = rule.target_visit.strip().casefold()
    mask = pd.Series(False, index=dataframe.index, dtype="bool")

    for participant in participants[valid_participant].dropna().unique():
        participant_mask = valid_participant & participants.eq(participant)
        anchor_dates = dates[participant_mask & visits.eq(anchor_name)].dropna()
        target_positions = [
            position
            for position, is_target in enumerate(
                (participant_mask & visits.eq(target_name)).tolist()
            )
            if is_target
        ]
        if anchor_dates.empty:
            continue

        minimum_days = rule.expected_days - rule.tolerance_days
        maximum_days = rule.expected_days + rule.tolerance_days
        for position in target_positions:
            target_date = dates.iloc[position]
            if pd.isna(target_date):
                mask.iloc[position] = True
                continue
            differences = (target_date - anchor_dates).dt.total_seconds() / 86_400
            if not differences.between(minimum_days, maximum_days, inclusive="both").any():
                mask.iloc[position] = True

    return mask


def _rule_mask(rule: RuleSpec, dataframe: pd.DataFrame) -> pd.Series:
    if isinstance(rule, RequiredRule):
        return _required_mask(rule, dataframe)
    if isinstance(rule, NumericRangeRule):
        return _numeric_range_mask(rule, dataframe)
    if isinstance(rule, AllowedValuesRule):
        return _allowed_values_mask(rule, dataframe)
    if isinstance(rule, DateOrderRule):
        return _date_order_mask(rule, dataframe)
    if isinstance(rule, ConditionalRequiredRule):
        return _conditional_required_mask(rule, dataframe)
    return _visit_window_mask(rule, dataframe)


def execute_rule_set(rule_set: RuleSet, dataframe: pd.DataFrame) -> list[Finding]:
    """Execute a validated rule set using only fixed deterministic functions."""

    errors = validate_rule_set(rule_set, dataframe)
    if errors:
        raise ValueError("Rule set is not executable: " + " ".join(errors))

    findings: list[Finding] = []
    for rule in rule_set.rules:
        tested = describe_rule(rule)
        rows = _positions(_rule_mask(rule, dataframe))
        if not rows:
            continue
        findings.append(
            Finding(
                severity=rule.severity,
                issue_type="Study rule violation",
                affected_rows=rows,
                columns=rule_columns(rule),
                rule_tested=tested,
                explanation=f"{len(rows)} records do not satisfy: {tested}.",
            )
        )

    return sorted(
        findings,
        key=lambda finding: (
            SEVERITY_ORDER[finding.severity],
            finding.rule_tested or "",
            finding.affected_rows,
        ),
    )
