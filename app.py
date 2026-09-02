from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from trialsleuth.export import dataframe_to_safe_csv
from trialsleuth.validation import Finding, infer_column_roles, run_validations

APP_ROOT = Path(__file__).parent
DEMO_CSV = APP_ROOT / "sample_data" / "synthetic_trial_data.csv"


st.set_page_config(
    page_title="TrialSleuth",
    page_icon=":material/fact_check:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f8f6; color: #172321; }
    [data-testid="stSidebar"] { background: #edf1ee; border-right: 1px solid #d8dfda; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8dfda;
        border-radius: 6px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricValue"] { color: #173f39; }
    [data-testid="stDataFrame"] { border: 1px solid #d8dfda; border-radius: 6px; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: 0; color: #173f39; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(contents))


def missing_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        mask = mask | series.astype("string").str.strip().eq("")
    return mask.fillna(True)


def build_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    roles = infer_column_roles(dataframe)
    profile_rows: list[dict[str, object]] = []
    for column in dataframe.columns:
        missing = int(missing_mask(dataframe[column]).sum())
        column_roles: list[str] = []
        if column == roles.participant:
            column_roles.append("Participant")
        if column == roles.visit:
            column_roles.append("Visit")
        if column in roles.dates:
            column_roles.append("Date")
        profile_rows.append(
            {
                "Column": str(column),
                "Inferred role": ", ".join(column_roles) or "-",
                "Data type": str(dataframe[column].dtype),
                "Non-null": len(dataframe) - missing,
                "Missing": missing,
                "Missing %": (missing / len(dataframe) * 100) if len(dataframe) else 0,
                "Distinct": int(dataframe[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(profile_rows)


def severity_style(value: object) -> str:
    colours = {
        "High": "background-color: #f8d7d2; color: #7d241d; font-weight: 600",
        "Medium": "background-color: #fae8bd; color: #67480a; font-weight: 600",
        "Low": "background-color: #dcece7; color: #20584f; font-weight: 600",
    }
    return colours.get(str(value), "")


with st.sidebar:
    st.header("Investigation")
    uploaded_file = st.file_uploader("Upload trial data", type=["csv"])
    st.text_area(
        "Study rules / data-dictionary notes",
        placeholder="Example: Visit 2 must occur 28 +/- 3 days after baseline.",
        height=150,
        key="study_notes",
    )
    st.caption("Notes are retained for reference only; they are not evaluated in this MVP.")

try:
    if uploaded_file is None:
        source_name = "Synthetic example"
        dataframe = read_csv_bytes(DEMO_CSV.read_bytes())
    else:
        source_name = uploaded_file.name
        dataframe = read_csv_bytes(uploaded_file.getvalue())
except (
    OSError,
    UnicodeDecodeError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
) as error:
    st.error(f"The CSV could not be read: {error}")
    st.stop()

if dataframe.empty or len(dataframe.columns) == 0:
    st.warning("The selected CSV has no data rows to investigate.")
    st.stop()

findings = run_validations(dataframe)
profile = build_profile(dataframe)
affected_rows = {row for finding in findings for row in finding.affected_rows}
high_findings = sum(finding.severity == "High" for finding in findings)
missing_cells = sum(int(missing_mask(dataframe[column]).sum()) for column in dataframe.columns)
total_cells = dataframe.size
completeness = (1 - missing_cells / total_cells) * 100 if total_cells else 100

title_column, source_column = st.columns([3, 1], vertical_alignment="bottom")
with title_column:
    st.title("TrialSleuth")
    st.caption("Clinical-trial data quality investigation")
with source_column:
    st.caption("DATA SOURCE")
    st.write(f"**{source_name}**")

metric_columns = st.columns(4)
metric_columns[0].metric("Rows", f"{len(dataframe):,}")
metric_columns[1].metric("Findings", f"{len(findings):,}")
metric_columns[2].metric("Rows affected", f"{len(affected_rows):,}")
metric_columns[3].metric(
    "Completeness",
    f"{completeness:.1f}%",
    f"{high_findings} high severity",
    delta_color="inverse",
)

findings_tab, profile_tab, data_tab = st.tabs(["Findings", "Column profile", "Source data"])

with findings_tab:
    if not findings:
        st.success("No issues were detected by the current checks.")
    else:
        filter_columns = st.columns(2)
        severity_options = [severity for severity in ("High", "Medium", "Low") if any(
            finding.severity == severity for finding in findings
        )]
        selected_severities = filter_columns[0].multiselect(
            "Severity", severity_options, default=severity_options
        )
        issue_options = sorted({finding.issue_type for finding in findings})
        selected_issues = filter_columns[1].multiselect(
            "Issue type", issue_options, default=issue_options
        )
        filtered_findings: list[Finding] = [
            finding
            for finding in findings
            if finding.severity in selected_severities and finding.issue_type in selected_issues
        ]

        summary = pd.DataFrame(finding.summary() for finding in filtered_findings)
        if summary.empty:
            st.info("No findings match the current filters.")
        else:
            st.dataframe(
                summary.style.map(severity_style, subset=["Severity"]),
                hide_index=True,
                width="stretch",
                column_config={
                    "Severity": st.column_config.TextColumn(width="small"),
                    "Issue type": st.column_config.TextColumn(width="medium"),
                    "Column(s)": st.column_config.TextColumn(width="medium"),
                    "Affected rows": st.column_config.NumberColumn(format="%d", width="small"),
                    "Explanation": st.column_config.TextColumn(width="large"),
                },
            )
            st.download_button(
                "Download findings",
                dataframe_to_safe_csv(summary),
                file_name="trial_sleuth_findings.csv",
                mime="text/csv",
                icon=":material/download:",
            )

            st.subheader("Evidence")
            for finding in filtered_findings:
                columns = ", ".join(finding.columns) if finding.columns else "All columns"
                label = f"{finding.severity} | {finding.issue_type} | {columns}"
                with st.expander(label):
                    st.write(finding.explanation)
                    evidence = finding.evidence(dataframe)
                    st.dataframe(evidence.head(200), hide_index=True, width="stretch")
                    if len(evidence) > 200:
                        st.caption(f"Showing 200 of {len(evidence)} affected rows.")

with profile_tab:
    st.dataframe(
        profile,
        hide_index=True,
        width="stretch",
        column_config={
            "Missing %": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"
            )
        },
    )
    missing_profile = profile.loc[profile["Missing"] > 0, ["Column", "Missing %"]]
    if not missing_profile.empty:
        st.subheader("Missingness by column")
        st.bar_chart(missing_profile.set_index("Column"), color="#b44d3b")

with data_tab:
    st.dataframe(dataframe, hide_index=True, width="stretch")
