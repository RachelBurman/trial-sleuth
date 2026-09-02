"""Deterministic data-quality checks used by the TrialSleuth dashboard."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


@dataclass(frozen=True)
class ColumnRoles:
    """Columns inferred from common clinical-trial naming conventions."""

    participant: str | None = None
    visit: str | None = None
    dates: tuple[str, ...] = ()

    @property
    def visit_date(self) -> str | None:
        return self.dates[0] if self.dates else None


@dataclass(frozen=True)
class Finding:
    """A single validation result and the source-row positions supporting it."""

    severity: str
    issue_type: str
    affected_rows: tuple[int, ...]
    explanation: str
    columns: tuple[str, ...] = ()

    @property
    def affected_count(self) -> int:
        return len(self.affected_rows)

    def summary(self) -> dict[str, object]:
        return {
            "Severity": self.severity,
            "Issue type": self.issue_type,
            "Column(s)": ", ".join(self.columns) if self.columns else "All columns",
            "Affected rows": self.affected_count,
            "Explanation": self.explanation,
        }

    def evidence(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        evidence = dataframe.iloc[list(self.affected_rows)].copy()
        evidence.insert(0, "Source row", [position + 2 for position in self.affected_rows])
        return evidence


def _normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _exact_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalised = {_normalise_name(column): column for column in columns}
    for alias in aliases:
        if alias in normalised:
            return normalised[alias]
    return None


def _is_date_column(column: str) -> bool:
    name = _normalise_name(column)
    tokens = set(re.findall(r"[a-z0-9]+", str(column).casefold()))
    date_aliases = {
        "date",
        "datetime",
        "timestamp",
        "dob",
        "visitdate",
        "assessmentdate",
        "collectiondate",
        "eventdate",
        "encounterdate",
        "startdate",
        "enddate",
        "birthdate",
        "consentdate",
        "randomizationdate",
        "randomisationdate",
        "enrollmentdate",
        "enrolmentdate",
        "treatmentdate",
        "dischargedate",
        "diagnosisdate",
        "specimendate",
        "sampledate",
        "deathdate",
        "completiondate",
        "dateofbirth",
    }
    return name in date_aliases or bool(tokens & {"date", "datetime", "timestamp", "dt"})


def infer_column_roles(dataframe: pd.DataFrame) -> ColumnRoles:
    """Infer participant, visit, and date columns from conservative name aliases."""

    columns = [str(column) for column in dataframe.columns]
    participant = _exact_column(
        columns,
        (
            "participantid",
            "subjectid",
            "patientid",
            "usubjid",
            "subjid",
            "participantnumber",
            "subjectnumber",
            "patientnumber",
            "participant",
            "subject",
            "patient",
        ),
    )
    if participant is None:
        for column in columns:
            name = _normalise_name(column)
            describes_person = any(
                token in name for token in ("participant", "subject", "patient")
            )
            describes_identifier = any(
                token in name for token in ("id", "identifier", "number", "num")
            ) or name.endswith("no")
            if describes_person and describes_identifier and not _is_date_column(column):
                participant = column
                break

    visit = _exact_column(
        columns,
        (
            "visitnumber",
            "visitnum",
            "visitno",
            "visitname",
            "visitlabel",
            "visit",
            "timepoint",
            "timepointname",
            "eventname",
            "event",
        ),
    )
    if visit is None:
        for column in columns:
            name = _normalise_name(column)
            if not _is_date_column(column) and any(
                token in name for token in ("visit", "timepoint")
            ):
                visit = column
                break

    date_columns = [column for column in columns if _is_date_column(column)]
    date_priority = (
        "visitdate",
        "assessmentdate",
        "collectiondate",
        "eventdate",
        "encounterdate",
        "date",
        "datetime",
    )
    priority = {name: index for index, name in enumerate(date_priority)}
    date_columns.sort(
        key=lambda column: (
            priority.get(_normalise_name(column), len(priority)),
            columns.index(column),
        )
    )

    return ColumnRoles(participant=participant, visit=visit, dates=tuple(date_columns))


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


def detect_duplicate_rows(dataframe: pd.DataFrame) -> list[Finding]:
    mask = dataframe.duplicated(keep=False)
    rows = _positions(mask)
    if not rows:
        return []
    return [
        Finding(
            severity="Medium",
            issue_type="Duplicate rows",
            affected_rows=rows,
            explanation=f"{len(rows)} rows are exact copies of another record.",
        )
    ]


def detect_duplicate_participant_visits(
    dataframe: pd.DataFrame, roles: ColumnRoles | None = None
) -> list[Finding]:
    roles = roles or infer_column_roles(dataframe)
    if not roles.participant or not roles.visit:
        return []

    participant = dataframe[roles.participant]
    visit = dataframe[roles.visit]
    complete = ~_missing_mask(participant) & ~_missing_mask(visit)
    mask = dataframe.duplicated(subset=[roles.participant, roles.visit], keep=False) & complete
    rows = _positions(mask)
    if not rows:
        return []
    return [
        Finding(
            severity="High",
            issue_type="Duplicate participant / visit",
            affected_rows=rows,
            columns=(roles.participant, roles.visit),
            explanation=(
                f"{len(rows)} rows reuse the same {roles.participant} and {roles.visit} combination."
            ),
        )
    ]


def detect_missingness(dataframe: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    row_count = len(dataframe)
    if row_count == 0:
        return findings

    for column in dataframe.columns:
        rows = _positions(_missing_mask(dataframe[column]))
        if not rows:
            continue
        percentage = len(rows) / row_count * 100
        severity = "High" if percentage >= 50 else "Medium" if percentage >= 20 else "Low"
        findings.append(
            Finding(
                severity=severity,
                issue_type="Missing values",
                affected_rows=rows,
                columns=(str(column),),
                explanation=f"{column} is missing in {len(rows)} rows ({percentage:.1f}%).",
            )
        )
    return findings


def detect_malformed_dates(
    dataframe: pd.DataFrame, roles: ColumnRoles | None = None
) -> list[Finding]:
    roles = roles or infer_column_roles(dataframe)
    findings: list[Finding] = []
    lower_bound = pd.Timestamp("1900-01-01", tz="UTC")
    upper_bound = pd.Timestamp("2100-12-31", tz="UTC")

    for column in roles.dates:
        present = ~_missing_mask(dataframe[column])
        parsed = _parse_dates(dataframe[column])
        mask = present & (parsed.isna() | parsed.lt(lower_bound) | parsed.gt(upper_bound))
        rows = _positions(mask)
        if rows:
            findings.append(
                Finding(
                    severity="High",
                    issue_type="Malformed or implausible date",
                    affected_rows=rows,
                    columns=(column,),
                    explanation=(
                        f"{len(rows)} values in {column} cannot be parsed as plausible calendar dates."
                    ),
                )
            )
    return findings


def _visit_order(series: pd.Series) -> pd.Series:
    order = pd.to_numeric(series, errors="coerce").astype("float64")
    labels = series.astype("string").str.strip().str.casefold()
    extracted = pd.to_numeric(
        labels.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False), errors="coerce"
    )
    order = order.fillna(extracted)

    known_labels = {
        "screening": -2.0,
        "screen": -2.0,
        "run-in": -1.0,
        "run in": -1.0,
        "baseline": 0.0,
        "randomization": 0.0,
        "randomisation": 0.0,
        "follow-up": 10_000.0,
        "follow up": 10_000.0,
        "end of study": 100_000.0,
        "termination": 100_000.0,
    }
    for label, value in known_labels.items():
        order = order.mask(order.isna() & labels.eq(label), value)
    return order


def detect_out_of_order_visits(
    dataframe: pd.DataFrame, roles: ColumnRoles | None = None
) -> list[Finding]:
    roles = roles or infer_column_roles(dataframe)
    if not roles.participant or not roles.visit or not roles.visit_date:
        return []

    participant = dataframe[roles.participant].astype("string").str.strip()
    dates = _parse_dates(dataframe[roles.visit_date])
    visit_order = _visit_order(dataframe[roles.visit])
    invalid_date = dates.isna()
    missing_participant = _missing_mask(dataframe[roles.participant])
    valid = ~(missing_participant | invalid_date | visit_order.isna())

    participant_rows: dict[str, list[int]] = {}
    for position, is_valid in enumerate(valid.tolist()):
        if is_valid:
            participant_rows.setdefault(str(participant.iloc[position]), []).append(position)

    offending: set[int] = set()
    for positions in participant_rows.values():
        rows_by_order: dict[float, list[int]] = {}
        for position in positions:
            rows_by_order.setdefault(float(visit_order.iloc[position]), []).append(position)

        latest_prior_date: pd.Timestamp | None = None
        for order in sorted(rows_by_order):
            current_positions = rows_by_order[order]
            if latest_prior_date is not None:
                offending.update(
                    position for position in current_positions if dates.iloc[position] < latest_prior_date
                )
            current_max = max(dates.iloc[position] for position in current_positions)
            if latest_prior_date is None or current_max > latest_prior_date:
                latest_prior_date = current_max

    rows = tuple(sorted(offending))
    if not rows:
        return []
    return [
        Finding(
            severity="High",
            issue_type="Visits out of chronological order",
            affected_rows=rows,
            columns=(roles.participant, roles.visit, roles.visit_date),
            explanation=(
                f"{len(rows)} visits have dates earlier than a preceding scheduled visit for the same participant."
            ),
        )
    ]


def detect_numeric_outliers(
    dataframe: pd.DataFrame, roles: ColumnRoles | None = None
) -> list[Finding]:
    roles = roles or infer_column_roles(dataframe)
    findings: list[Finding] = []
    excluded = {roles.participant, roles.visit, *roles.dates}

    for column in dataframe.select_dtypes(include="number").columns:
        column_name = str(column)
        normalised_name = _normalise_name(column_name)
        if column_name in excluded or normalised_name.endswith("id") or "identifier" in normalised_name:
            continue

        values = dataframe[column]
        clean = values.dropna()
        if len(clean) < 4:
            continue
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        rows = _positions(values.notna() & ((values < lower) | (values > upper)))
        if rows:
            findings.append(
                Finding(
                    severity="Medium",
                    issue_type="Numeric outlier",
                    affected_rows=rows,
                    columns=(column_name,),
                    explanation=(
                        f"{len(rows)} values in {column_name} fall outside the IQR range "
                        f"({lower:.2f} to {upper:.2f})."
                    ),
                )
            )
    return findings


def detect_categorical_inconsistencies(
    dataframe: pd.DataFrame, roles: ColumnRoles | None = None
) -> list[Finding]:
    roles = roles or infer_column_roles(dataframe)
    findings: list[Finding] = []
    excluded = {roles.participant, *roles.dates}

    for column in dataframe.columns:
        column_name = str(column)
        series = dataframe[column]
        if column_name in excluded or not (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue

        present = ~_missing_mask(series)
        raw = series.astype("string")
        unique_count = raw[present].nunique()
        max_categories = min(50, max(10, int(len(dataframe) * 0.2)))
        if unique_count < 2 or unique_count > max_categories:
            continue

        normalised = raw.str.strip().str.replace(r"\s+", " ", regex=True).str.casefold()
        variants: dict[str, set[str]] = {}
        for position in range(len(series)):
            if present.iloc[position]:
                variants.setdefault(str(normalised.iloc[position]), set()).add(str(raw.iloc[position]))
        inconsistent = {key: values for key, values in variants.items() if len(values) > 1}
        if not inconsistent:
            continue

        rows = _positions(present & normalised.isin(inconsistent))
        examples = [" / ".join(repr(value) for value in sorted(values)) for values in inconsistent.values()]
        example_text = "; ".join(examples[:3])
        findings.append(
            Finding(
                severity="Low",
                issue_type="Categorical inconsistency",
                affected_rows=rows,
                columns=(column_name,),
                explanation=(
                    f"{column_name} contains categories that differ only by case or whitespace: {example_text}."
                ),
            )
        )
    return findings


def run_validations(dataframe: pd.DataFrame) -> list[Finding]:
    """Run all MVP checks and return findings in stable severity/type order."""

    roles = infer_column_roles(dataframe)
    findings: list[Finding] = []
    detectors = (
        detect_duplicate_rows,
        detect_duplicate_participant_visits,
        detect_missingness,
        detect_malformed_dates,
        detect_out_of_order_visits,
        detect_numeric_outliers,
        detect_categorical_inconsistencies,
    )
    for detector in detectors:
        if detector in (detect_duplicate_rows, detect_missingness):
            findings.extend(detector(dataframe))
        else:
            findings.extend(detector(dataframe, roles))

    return sorted(
        findings,
        key=lambda finding: (
            SEVERITY_ORDER[finding.severity],
            finding.issue_type,
            finding.columns,
            finding.affected_rows,
        ),
    )
