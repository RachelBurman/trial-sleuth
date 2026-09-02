import pandas as pd

from trialsleuth.validation import (
    detect_categorical_inconsistencies,
    detect_duplicate_participant_visits,
    detect_duplicate_rows,
    detect_malformed_dates,
    detect_missingness,
    detect_numeric_outliers,
    detect_out_of_order_visits,
    infer_column_roles,
    run_validations,
)


def test_infers_common_clinical_column_roles() -> None:
    dataframe = pd.DataFrame(
        columns=["USUBJID", "Visit Name", "Assessment Date", "result"]
    )

    roles = infer_column_roles(dataframe)

    assert roles.participant == "USUBJID"
    assert roles.visit == "Visit Name"
    assert roles.visit_date == "Assessment Date"


def test_date_inference_does_not_match_date_inside_another_word() -> None:
    dataframe = pd.DataFrame(columns=["participant_id", "visit", "candidate_status"])

    roles = infer_column_roles(dataframe)

    assert roles.dates == ()


def test_detects_exact_and_participant_visit_duplicates() -> None:
    dataframe = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P2", "P2"],
            "visit": ["Baseline", "Baseline", "Week 4", "Week 4"],
            "value": [10, 11, 20, 20],
        }
    )

    exact = detect_duplicate_rows(dataframe)
    participant_visit = detect_duplicate_participant_visits(dataframe)

    assert exact[0].affected_rows == (2, 3)
    assert participant_visit[0].affected_rows == (0, 1, 2, 3)


def test_missingness_includes_blank_strings() -> None:
    dataframe = pd.DataFrame({"result": [1, None, 3, None], "note": ["ok", " ", "ok", "ok"]})

    findings = detect_missingness(dataframe)
    findings_by_column = {finding.columns[0]: finding for finding in findings}

    assert findings_by_column["result"].affected_rows == (1, 3)
    assert findings_by_column["result"].severity == "High"
    assert findings_by_column["note"].affected_rows == (1,)


def test_detects_malformed_and_implausible_dates() -> None:
    dataframe = pd.DataFrame(
        {"visit_date": ["2026-01-01", "2026-02-30", "not-a-date", "2200-01-01", None]}
    )

    findings = detect_malformed_dates(dataframe)

    assert findings[0].affected_rows == (1, 2, 3)


def test_detects_visit_date_that_breaks_scheduled_order() -> None:
    dataframe = pd.DataFrame(
        {
            "subject_id": ["P1", "P1", "P1", "P2", "P2"],
            "visit": ["Screening", "Baseline", "Week 4", "Baseline", "Week 4"],
            "visit_date": [
                "2026-01-01",
                "2026-01-10",
                "2026-01-05",
                "2026-02-01",
                "2026-03-01",
            ],
        }
    )

    findings = detect_out_of_order_visits(dataframe)

    assert findings[0].affected_rows == (2,)


def test_detects_iqr_numeric_outlier_but_skips_numeric_identifier() -> None:
    dataframe = pd.DataFrame(
        {
            "participant_id": [1, 2, 3, 4, 999],
            "result": [10, 10, 11, 11, 100],
        }
    )

    findings = detect_numeric_outliers(dataframe)

    assert len(findings) == 1
    assert findings[0].columns == ("result",)
    assert findings[0].affected_rows == (4,)


def test_detects_case_and_whitespace_category_variants() -> None:
    dataframe = pd.DataFrame(
        {"status": ["Completed", " completed ", "Active", "ACTIVE", None]}
    )

    findings = detect_categorical_inconsistencies(dataframe)

    assert len(findings) == 1
    assert findings[0].affected_rows == (0, 1, 2, 3)


def test_full_validation_is_stable_and_evidence_uses_csv_row_numbers() -> None:
    dataframe = pd.DataFrame({"participant_id": ["P1", "P1"], "visit": [1, 1], "value": [4, 4]})

    findings = run_validations(dataframe)

    assert [finding.severity for finding in findings] == ["High", "Medium"]
    assert findings[0].evidence(dataframe)["Source row"].tolist() == [2, 3]
