import pandas as pd
import pytest
from pydantic import ValidationError

from trialsleuth.study_rules import RuleSet, execute_rule_set, validate_rule_set


def rule_set(rules: list[dict[str, object]]) -> RuleSet:
    return RuleSet.model_validate(
        {"rules": rules, "unsupported_notes": []},
        strict=True,
    )


def test_schema_rejects_unknown_rules_extra_code_and_coerced_values() -> None:
    with pytest.raises(ValidationError):
        rule_set(
            [
                {
                    "rule_type": "python",
                    "severity": "High",
                    "code": "dataframe.dropna()",
                }
            ]
        )

    with pytest.raises(ValidationError):
        rule_set(
            [
                {
                    "rule_type": "required",
                    "severity": "High",
                    "field": "participant_id",
                    "expression": "eval('anything')",
                }
            ]
        )

    with pytest.raises(ValidationError):
        rule_set(
            [
                {
                    "rule_type": "numeric_range",
                    "severity": "Medium",
                    "field": "age",
                    "minimum": "18",
                    "maximum": 75.0,
                    "minimum_inclusive": True,
                    "maximum_inclusive": True,
                }
            ]
        )


def test_schema_rejects_invalid_numeric_bounds() -> None:
    with pytest.raises(ValidationError, match="minimum cannot be greater"):
        rule_set(
            [
                {
                    "rule_type": "numeric_range",
                    "severity": "Medium",
                    "field": "age",
                    "minimum": 75.0,
                    "maximum": 18.0,
                    "minimum_inclusive": True,
                    "maximum_inclusive": True,
                }
            ]
        )

    with pytest.raises(ValidationError, match="equal numeric bounds must both be inclusive"):
        rule_set(
            [
                {
                    "rule_type": "numeric_range",
                    "severity": "Medium",
                    "field": "age",
                    "minimum": 18.0,
                    "maximum": 18.0,
                    "minimum_inclusive": False,
                    "maximum_inclusive": True,
                }
            ]
        )

    with pytest.raises(ValidationError):
        rule_set(
            [
                {
                    "rule_type": "numeric_range",
                    "severity": "Medium",
                    "field": "age",
                    "minimum": float("nan"),
                    "maximum": 75.0,
                    "minimum_inclusive": True,
                    "maximum_inclusive": True,
                }
            ]
        )

    exact_value = rule_set(
        [
            {
                "rule_type": "numeric_range",
                "severity": "Medium",
                "field": "age",
                "minimum": 18.0,
                "maximum": 18.0,
                "minimum_inclusive": True,
                "maximum_inclusive": True,
            }
        ]
    )
    assert exact_value.rules[0].minimum == exact_value.rules[0].maximum


def test_dataset_validation_rejects_missing_columns_and_duplicate_rules() -> None:
    proposal = rule_set(
        [
            {"rule_type": "required", "severity": "High", "field": "missing_field"},
            {"rule_type": "required", "severity": "Low", "field": "missing_field"},
        ]
    )

    errors = validate_rule_set(proposal, pd.DataFrame({"age": [42]}))

    assert errors == (
        "Rule 1 references missing column 'missing_field'.",
        "Rule 2 references missing column 'missing_field'.",
        "Rule 2 duplicates an earlier rule.",
    )
    with pytest.raises(ValueError, match="not executable"):
        execute_rule_set(proposal, pd.DataFrame({"age": [42]}))


def test_dataset_validation_rejects_ambiguous_condition_values() -> None:
    proposal = rule_set(
        [
            {
                "rule_type": "conditional_required",
                "severity": "Medium",
                "condition_field": "status",
                "operator": "equals",
                "condition_values": ["Withdrawn", "Stopped"],
                "required_field": "reason",
                "case_sensitive": False,
            }
        ]
    )

    errors = validate_rule_set(
        proposal,
        pd.DataFrame({"status": ["Active"], "reason": [None]}),
    )

    assert errors == ("Rule 1 operator 'equals' requires exactly one value.",)


def test_dataset_validation_handles_duplicate_numeric_columns_without_crashing() -> None:
    proposal = rule_set(
        [
            {
                "rule_type": "numeric_range",
                "severity": "Medium",
                "field": "age",
                "minimum": 18.0,
                "maximum": 75.0,
                "minimum_inclusive": True,
                "maximum_inclusive": True,
            }
        ]
    )
    dataframe = pd.DataFrame([[20, 21]], columns=["age", "age"])

    assert validate_rule_set(proposal, dataframe) == (
        "Rule 1 references ambiguous duplicate column 'age'.",
    )


def test_dataset_validation_rejects_non_date_fields_and_unknown_visit_labels() -> None:
    dataframe = pd.DataFrame(
        {
            "participant_id": ["P1"],
            "visit": ["Baseline"],
            "status": ["Active"],
            "sex": ["F"],
        }
    )
    proposal = rule_set(
        [
            {
                "rule_type": "date_order",
                "severity": "High",
                "earlier_field": "status",
                "later_field": "sex",
                "allow_equal": True,
            },
            {
                "rule_type": "visit_window",
                "severity": "High",
                "participant_field": "participant_id",
                "visit_field": "visit",
                "date_field": "status",
                "anchor_visit": "Screening",
                "target_visit": "Week 4",
                "expected_days": 28,
                "tolerance_days": 3,
            },
        ]
    )

    errors = validate_rule_set(proposal, dataframe)

    assert "Rule 1 requires date values in 'status'." in errors
    assert "Rule 1 requires date values in 'sex'." in errors
    assert "Rule 2 requires date values in 'status'." in errors
    assert "Rule 2 anchor visit 'Screening' was not found." in errors
    assert "Rule 2 target visit 'Week 4' was not found." in errors


def test_deterministic_execution_of_row_level_rule_types() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [25, None, 80, "bad"],
            "sex": ["F", "M", "Unknown", "f"],
            "consent_date": ["2026-01-01", "2026-01-02", "not-a-date", "2026-01-05"],
            "treatment_date": ["2026-01-02", "2026-01-01", "2026-01-03", ""],
            "status": ["Active", "Withdrawn", "withdrawn", "Withdrawn"],
            "withdrawal_reason": ["", None, "Adverse event", ""],
        }
    )
    proposal = rule_set(
        [
            {"rule_type": "required", "severity": "High", "field": "age"},
            {
                "rule_type": "numeric_range",
                "severity": "Medium",
                "field": "age",
                "minimum": 18.0,
                "maximum": 75.0,
                "minimum_inclusive": True,
                "maximum_inclusive": True,
            },
            {
                "rule_type": "allowed_values",
                "severity": "Low",
                "field": "sex",
                "allowed_values": [" F ", " M "],
                "case_sensitive": False,
            },
            {
                "rule_type": "date_order",
                "severity": "High",
                "earlier_field": "consent_date",
                "later_field": "treatment_date",
                "allow_equal": True,
            },
            {
                "rule_type": "conditional_required",
                "severity": "Medium",
                "condition_field": "status",
                "operator": "equals",
                "condition_values": [" Withdrawn "],
                "required_field": "withdrawal_reason",
                "case_sensitive": False,
            },
        ]
    )

    findings = execute_rule_set(proposal, dataframe)
    rows_by_rule = {finding.rule_tested: finding.affected_rows for finding in findings}

    assert rows_by_rule["age must be populated"] == (1,)
    assert rows_by_rule["age must be >= 18.0 and <= 75.0"] == (2, 3)
    assert rows_by_rule["sex must be one of: F, M"] == (2,)
    assert rows_by_rule["consent_date must be on or before treatment_date"] == (1, 2)
    assert rows_by_rule[
        "withdrawal_reason is required when status equals Withdrawn"
    ] == (1, 3)
    assert all(finding.issue_type == "Study rule violation" for finding in findings)


def test_visit_window_is_deterministic_and_requires_a_comparable_anchor() -> None:
    dataframe = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2", "P3", "P4", "P4"],
            "visit": [
                "Baseline",
                "Week 4",
                "Baseline",
                "Week 4",
                "Week 4",
                "Baseline",
                "Week 4",
            ],
            "visit_date": [
                "2026-01-01",
                "2026-01-30",
                "2026-01-01",
                "2026-02-10",
                "2026-01-29",
                "2026-01-01",
                "not-a-date",
            ],
        }
    )
    proposal = rule_set(
        [
            {
                "rule_type": "visit_window",
                "severity": "High",
                "participant_field": "participant_id",
                "visit_field": "visit",
                "date_field": "visit_date",
                "anchor_visit": "Baseline",
                "target_visit": "Week 4",
                "expected_days": 28,
                "tolerance_days": 3,
            }
        ]
    )

    findings = execute_rule_set(proposal, dataframe)

    assert findings[0].affected_rows == (3, 6)
    assert findings[0].evidence(dataframe)["Source row"].tolist() == [5, 8]


def test_visit_window_keeps_participant_identifiers_case_sensitive() -> None:
    dataframe = pd.DataFrame(
        {
            "participant_id": ["P1", "p1", "p1"],
            "visit": ["Baseline", "Baseline", "Week 4"],
            "visit_date": ["2026-01-13", "2026-01-01", "2026-02-10"],
        }
    )
    proposal = rule_set(
        [
            {
                "rule_type": "visit_window",
                "severity": "High",
                "participant_field": "participant_id",
                "visit_field": "visit",
                "date_field": "visit_date",
                "anchor_visit": "Baseline",
                "target_visit": "Week 4",
                "expected_days": 28,
                "tolerance_days": 3,
            }
        ]
    )

    assert execute_rule_set(proposal, dataframe)[0].affected_rows == (2,)


def test_bundled_data_has_deterministic_violations_for_demo_rules() -> None:
    dataframe = pd.read_csv("sample_data/synthetic_trial_data.csv")
    proposal = rule_set(
        [
            {"rule_type": "required", "severity": "High", "field": "age"},
            {
                "rule_type": "numeric_range",
                "severity": "High",
                "field": "age",
                "minimum": 18.0,
                "maximum": 65.0,
                "minimum_inclusive": True,
                "maximum_inclusive": True,
            },
            {"rule_type": "required", "severity": "High", "field": "sex"},
            {
                "rule_type": "allowed_values",
                "severity": "Medium",
                "field": "sex",
                "allowed_values": ["Female", "Male"],
                "case_sensitive": True,
            },
            {"rule_type": "required", "severity": "High", "field": "status"},
            {
                "rule_type": "allowed_values",
                "severity": "Medium",
                "field": "status",
                "allowed_values": ["Enrolled", "Active", "Completed"],
                "case_sensitive": True,
            },
            {
                "rule_type": "visit_window",
                "severity": "High",
                "participant_field": "participant_id",
                "visit_field": "visit",
                "date_field": "visit_date",
                "anchor_visit": "Baseline",
                "target_visit": "Week 4",
                "expected_days": 28,
                "tolerance_days": 3,
            },
        ]
    )

    findings = execute_rule_set(proposal, dataframe)
    rows_by_rule = {finding.rule_tested: finding.affected_rows for finding in findings}

    assert rows_by_rule["age must be populated"] == (19,)
    assert rows_by_rule["age must be >= 18.0 and <= 65.0"] == (18,)
    assert rows_by_rule["sex must be populated"] == (15,)
    assert rows_by_rule["sex must be one of: Female, Male"] == (1,)
    assert rows_by_rule["status must be populated"] == (19,)
    assert rows_by_rule["status must be one of: Enrolled, Active, Completed"] == (10,)
    assert rows_by_rule["Week 4 must occur 28 +/- 3 days after Baseline"] == (10, 13)
