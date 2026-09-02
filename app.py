from __future__ import annotations

import hashlib
import html
import json
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from trialsleuth.export import dataframe_to_safe_csv
from trialsleuth.rule_proposals import DEFAULT_MODEL, RuleProposalError, propose_rule_set
from trialsleuth.study_rules import (
    RuleSet,
    describe_rule,
    execute_rule_set,
    validate_rule_set,
)
from trialsleuth.validation import Finding, infer_column_roles, run_validations

APP_ROOT = Path(__file__).parent
DEMO_CSV = APP_ROOT / "sample_data" / "synthetic_trial_data.csv"
DEMO_STUDY_RULES = """age is required and must be between 18 and 65 inclusive.
sex is required and must be exactly Female or Male; matching is case-sensitive.
status is required and must be exactly Enrolled, Active, or Completed; matching is case-sensitive.
For each participant_id, the Week 4 visit_date must occur 28 +/- 3 days after the Baseline visit."""


st.set_page_config(
    page_title="TrialSleuth",
    page_icon=":material/fact_check:",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #1d1d1f;
        --secondary: #626266;
        --line: #d9dcda;
        --surface: #ffffff;
        --canvas: #f6f7f6;
        --accent: #087f73;
    }
    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }
    .stApp, .stApp button, .stApp input, .stApp textarea {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #f1f3f2;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--secondary);
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4 {
        color: var(--ink);
        letter-spacing: 0;
    }
    h1 { font-size: 2rem !important; line-height: 1.15 !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.1rem !important; }
    [data-testid="stCaptionContainer"] { color: var(--secondary); }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 7px;
        overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 7px; }
    .stButton > button, .stDownloadButton > button {
        min-height: 2.75rem;
        border-radius: 7px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
    }
    .sidebar-brand {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0.2rem 0 1.6rem;
    }
    .sidebar-brand span {
        color: var(--accent);
        font-weight: 800;
    }
    .source-context {
        align-items: flex-end;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        padding-bottom: 0.45rem;
    }
    .source-context .label {
        color: var(--secondary);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .source-context .value {
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 650;
        max-width: 24rem;
        overflow-wrap: anywhere;
        text-align: right;
    }
    .source-context .meta { color: var(--secondary); font-size: 0.78rem; }
    .demo-tag {
        background: #e5e8e6;
        border-radius: 999px;
        color: #4b504d;
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 750;
        margin-left: 0.35rem;
        padding: 0.15rem 0.45rem;
        text-transform: uppercase;
    }
    .metric-strip {
        background: var(--surface);
        border-bottom: 1px solid var(--line);
        border-top: 1px solid var(--line);
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin: 1rem 0 1.35rem;
    }
    .metric-item { min-width: 0; padding: 0.9rem 1.1rem; }
    .metric-item + .metric-item { border-left: 1px solid var(--line); }
    .metric-value {
        color: var(--ink);
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-value.high { color: #a63226; }
    .metric-label {
        color: var(--secondary);
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }
    .workflow-steps {
        border-bottom: 1px solid var(--line);
        color: var(--secondary);
        display: flex;
        font-size: 0.82rem;
        font-weight: 650;
        gap: 1.5rem;
        margin: 0.3rem 0 1.3rem;
        padding-bottom: 0.75rem;
    }
    .workflow-steps .active { color: var(--accent); }
    [data-baseweb="tab-list"] { gap: 0.4rem; }
    [data-baseweb="tab"] { min-height: 3rem; }
    @media (max-width: 800px) {
        .block-container { padding-top: 1rem; }
        .source-context { align-items: flex-start; margin-top: 0.5rem; }
        .source-context .value { text-align: left; }
        .metric-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .metric-item:nth-child(3) { border-left: 0; border-top: 1px solid var(--line); }
        .metric-item:nth-child(4) { border-top: 1px solid var(--line); }
        [data-testid="stHorizontalBlock"] { flex-direction: column; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 auto !important; }
    }
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


def dataset_fingerprint(dataframe: pd.DataFrame, notes: str) -> str:
    digest = hashlib.sha256()
    digest.update(notes.strip().encode("utf-8"))
    schema = [(str(column), str(dataframe[column].dtype)) for column in dataframe]
    digest.update(repr(schema).encode())
    digest.update(pd.util.hash_pandas_object(dataframe, index=True).values.tobytes())
    return digest.hexdigest()


def configured_anthropic_api_key() -> str | None:
    try:
        secret = str(st.secrets.get("ANTHROPIC_API_KEY", "")).strip()
    except (FileNotFoundError, KeyError):
        secret = ""
    if secret:
        return secret
    for variable_name in ("anthropic_api_key", "ANTHROPIC_API_KEY"):
        key = os.getenv(variable_name, "").strip()
        if key:
            return key
    return None


def proposal_summary(rule_set: RuleSet) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Severity": rule.severity,
            "Rule type": rule.rule_type.replace("_", " ").title(),
            "Proposed rule": describe_rule(rule),
        }
        for rule in rule_set.rules
    )


def verified_summary(findings: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Severity": finding.severity,
            "Rule tested": finding.rule_tested,
            "Affected records": finding.affected_count,
            "Explanation": finding.explanation,
        }
        for finding in findings
    )


def use_demo_dataset() -> None:
    st.session_state.pop("trial_data_upload", None)
    st.session_state.pop("rule_proposal", None)
    st.session_state.pop("rule_proposal_context", None)
    st.session_state.pop("verified_rule_findings", None)
    st.session_state.pop("verified_rule_context", None)


with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><span>Trial</span>Sleuth</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Data")
    uploaded_file = st.file_uploader(
        "Import CSV",
        type=["csv"],
        key="trial_data_upload",
        help="Select a clinical-trial CSV to begin a new investigation.",
    )

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

with st.sidebar:
    if uploaded_file is None:
        st.badge("Demo dataset", icon=":material/science:", color="gray")
    else:
        st.badge("Imported CSV", icon=":material/upload_file:", color="green")
    st.caption(f"{len(dataframe):,} rows · {len(dataframe.columns):,} columns")
    if uploaded_file is not None:
        st.button(
            "Use demo dataset",
            icon=":material/replay:",
            width="stretch",
            on_click=use_demo_dataset,
        )
    st.divider()
    st.caption("Record values remain local. AI proposals receive only column names and types.")

findings = run_validations(dataframe)
profile = build_profile(dataframe)
roles = infer_column_roles(dataframe)
affected_rows = {row for finding in findings for row in finding.affected_rows}
high_findings = sum(finding.severity == "High" for finding in findings)
missing_cells = sum(int(missing_mask(dataframe[column]).sum()) for column in dataframe.columns)
incomplete_columns = sum(bool(missing_mask(dataframe[column]).any()) for column in dataframe.columns)
total_cells = dataframe.size
completeness = (1 - missing_cells / total_cells) * 100 if total_cells else 100
if roles.participant and affected_rows:
    participant_values = dataframe.iloc[sorted(affected_rows)][roles.participant]
    affected_people = int(participant_values.loc[~missing_mask(participant_values)].nunique())
    impact_value = affected_people
    impact_label = "Affected participants"
else:
    impact_value = len(affected_rows)
    impact_label = "Affected rows"

title_column, source_column = st.columns([3, 2], vertical_alignment="bottom")
with title_column:
    st.title("TrialSleuth")
    st.caption("Clinical-trial data review")
with source_column:
    demo_tag = '<span class="demo-tag">Demo</span>' if uploaded_file is None else ""
    st.markdown(
        f"""
        <div class="source-context">
            <div class="label">Current dataset</div>
            <div class="value">{html.escape(source_name)}{demo_tag}</div>
            <div class="meta">{len(dataframe):,} rows · {len(dataframe.columns):,} columns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="metric-strip">
        <div class="metric-item">
            <div class="metric-value high">{high_findings:,}</div>
            <div class="metric-label">High-priority issues</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">{impact_value:,}</div>
            <div class="metric-label">{impact_label}</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">{incomplete_columns:,}</div>
            <div class="metric-label">Fields with missing data</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">{completeness:.1f}%</div>
            <div class="metric-label">Completeness</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

findings_tab, study_rules_tab, profile_tab, data_tab = st.tabs(
    [f"Issues ({len(findings)})", "Study checks", "Columns", "Data"]
)

with findings_tab:
    if not findings:
        st.success("No issues detected", icon=":material/check_circle:")
    else:
        st.subheader("Issues")
        severity_options = [severity for severity in ("High", "Medium", "Low") if any(
            finding.severity == severity for finding in findings
        )]
        filter_columns = st.columns([1.4, 1, 0.55], vertical_alignment="bottom")
        selected_severities = filter_columns[0].segmented_control(
            "Severity",
            severity_options,
            default=severity_options,
            selection_mode="multi",
            width="stretch",
        ) or []
        issue_options = sorted({finding.issue_type for finding in findings})
        selected_issue = filter_columns[1].selectbox(
            "Issue type", ["All issue types", *issue_options]
        )
        filtered_findings: list[Finding] = [
            finding
            for finding in findings
            if finding.severity in selected_severities
            and (selected_issue == "All issue types" or finding.issue_type == selected_issue)
        ]

        summary = pd.DataFrame(finding.summary() for finding in filtered_findings)
        filter_columns[2].download_button(
            "Export",
            dataframe_to_safe_csv(summary),
            file_name="trial_sleuth_findings.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            disabled=summary.empty,
        )
        if summary.empty:
            st.info("No issues match the current filters.")
        else:
            list_column, detail_column = st.columns([1, 1.15], gap="large")
            with list_column:
                st.caption(f"{len(filtered_findings):,} issues · Select a row to inspect")
                issue_list = summary.rename(
                    columns={
                        "Severity": "Priority",
                        "Issue type": "Issue",
                        "Affected rows": "Records",
                    }
                ).drop(columns=["Column(s)", "Explanation"])
                selection = st.dataframe(
                    issue_list.style.map(severity_style, subset=["Priority"]),
                    hide_index=True,
                    width="stretch",
                    height=430,
                    row_height=44,
                    column_config={
                        "Priority": st.column_config.TextColumn(width="small"),
                        "Issue": st.column_config.TextColumn(width="medium"),
                        "Records": st.column_config.NumberColumn(format="%d", width="small"),
                    },
                    key="issue_selection",
                    on_select="rerun",
                    selection_mode="single-row",
                    selection_default={"selection": {"rows": [0]}},
                )
                selected_rows = selection.selection.rows
                selected_position = selected_rows[0] if selected_rows else 0
                if selected_position >= len(filtered_findings):
                    selected_position = 0

            with detail_column:
                selected_finding = filtered_findings[selected_position]
                badge_color = {
                    "High": "red",
                    "Medium": "orange",
                    "Low": "green",
                }[selected_finding.severity]
                st.badge(
                    selected_finding.severity,
                    icon=(
                        ":material/priority_high:"
                        if selected_finding.severity == "High"
                        else None
                    ),
                    color=badge_color,
                )
                st.markdown(f"### {selected_finding.issue_type}")
                selected_columns = (
                    ", ".join(selected_finding.columns)
                    if selected_finding.columns
                    else "All fields"
                )
                st.caption(
                    f"{selected_columns} · {selected_finding.affected_count:,} affected records"
                )
                st.write(selected_finding.explanation)
                full_evidence = selected_finding.evidence(dataframe)
                evidence = full_evidence.head(200)
                highlighted_columns = [
                    column for column in selected_finding.columns if column in evidence.columns
                ]
                if not highlighted_columns and not selected_finding.columns:
                    highlighted_columns = [
                        column for column in dataframe.columns if column in evidence.columns
                    ]
                styled_evidence = evidence.style
                if highlighted_columns:
                    styled_evidence = styled_evidence.map(
                        lambda _: (
                            "background-color: #fff0ed; color: #7d241d; font-weight: 600"
                        ),
                        subset=highlighted_columns,
                    )
                st.dataframe(
                    styled_evidence,
                    hide_index=True,
                    width="stretch",
                    height=315,
                    column_config={
                        "Source row": st.column_config.NumberColumn(
                            "Source row", format="%d", width="small"
                        )
                    },
                )
                if selected_finding.affected_count > 200:
                    st.caption(f"Showing 200 of {selected_finding.affected_count:,} records")
                st.download_button(
                    "Export evidence",
                    dataframe_to_safe_csv(full_evidence),
                    file_name="trial_sleuth_evidence.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    key=f"evidence_export_{selected_position}",
                )

with study_rules_tab:
    api_key = configured_anthropic_api_key()
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    has_proposal = st.session_state.get("rule_proposal") is not None
    has_results = st.session_state.get("verified_rule_findings") is not None
    active_step = "results" if has_results else "review" if has_proposal else "define"
    st.subheader("Study checks")
    st.markdown(
        f"""
        <div class="workflow-steps">
            <span class="{'active' if active_step == 'define' else ''}">1&nbsp; Define</span>
            <span class="{'active' if active_step == 'review' else ''}">2&nbsp; Review</span>
            <span class="{'active' if active_step == 'results' else ''}">3&nbsp; Results</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    define_title, demo_rules_column = st.columns([3, 1], vertical_alignment="bottom")
    define_title.markdown("#### 1. Define checks")
    load_demo_rules = demo_rules_column.button(
        "Load demo rules",
        icon=":material/science:",
        width="stretch",
        disabled=uploaded_file is not None,
    )
    if load_demo_rules:
        st.session_state["study_notes"] = DEMO_STUDY_RULES
        st.session_state.pop("rule_proposal", None)
        st.session_state.pop("rule_proposal_context", None)
        st.session_state.pop("verified_rule_findings", None)
        st.session_state.pop("verified_rule_context", None)
    study_notes = st.text_area(
        "Protocol or data-dictionary rules",
        placeholder="Example: Week 4 must occur 28 +/- 3 days after baseline.",
        height=150,
        key="study_notes",
    )
    proposal_context = dataset_fingerprint(dataframe, study_notes)
    if api_key is None:
        st.info(
            "AI-assisted proposals require `ANTHROPIC_API_KEY`. Built-in checks remain available.",
            icon=":material/key:",
        )
    propose_clicked = st.button(
        "Generate proposed checks",
        icon=":material/auto_awesome:",
        type="primary",
        disabled=not bool(api_key),
    )

    if propose_clicked:
        if not study_notes.strip():
            st.warning("Enter at least one study rule before generating checks.")
        else:
            try:
                with st.spinner("Generating proposed checks..."):
                    generated_proposal = propose_rule_set(
                        study_notes,
                        dataframe,
                        api_key=api_key or "",
                        model=model,
                    )
                st.session_state["rule_proposal"] = generated_proposal.model_dump(mode="json")
                st.session_state["rule_proposal_context"] = proposal_context
                st.session_state.pop("verified_rule_findings", None)
                st.session_state.pop("verified_rule_context", None)
            except (RuleProposalError, ValueError) as error:
                st.error(str(error))

    proposal_data = st.session_state.get("rule_proposal")
    proposal: RuleSet | None = None
    schema_error: str | None = None
    if proposal_data is not None:
        try:
            proposal = RuleSet.model_validate(proposal_data, strict=True)
        except ValidationError:
            schema_error = "The proposal no longer conforms to the strict rule schema."

    if proposal is None:
        if schema_error:
            st.error(f"Proposal rejected. {schema_error}")
    else:
        st.divider()
        st.markdown("#### 2. Review proposed checks")
        proposal_is_current = (
            st.session_state.get("rule_proposal_context") == proposal_context
        )
        if not proposal_is_current:
            st.warning(
                "The dataset or rules changed. Generate a new proposal before running checks."
            )

        summary = proposal_summary(proposal)
        accepted_proposal = proposal
        if summary.empty:
            st.info("No supported checks were identified in these rules.")
        else:
            review_table = summary.copy()
            review_table.insert(0, "Include", True)
            proposal_digest = hashlib.sha256(
                proposal.model_dump_json().encode("utf-8")
            ).hexdigest()[:12]
            reviewed_rules = st.data_editor(
                review_table.style.map(severity_style, subset=["Severity"]),
                hide_index=True,
                width="stretch",
                disabled=["Severity", "Rule type", "Proposed rule"],
                column_config={
                    "Include": st.column_config.CheckboxColumn(width="small"),
                    "Severity": st.column_config.TextColumn(width="small"),
                    "Rule type": st.column_config.TextColumn(width="medium"),
                    "Proposed rule": st.column_config.TextColumn(width="large"),
                },
                key=f"rule_review_{proposal_context[:12]}_{proposal_digest}",
            )
            accepted_rules = [
                rule
                for rule, include in zip(proposal.rules, reviewed_rules["Include"], strict=True)
                if bool(include)
            ]
            accepted_proposal = proposal.model_copy(update={"rules": accepted_rules})
            st.caption(f"{len(accepted_rules):,} of {len(proposal.rules):,} checks selected")

        if proposal.unsupported_notes:
            with st.expander(
                f"Unsupported or ambiguous rules ({len(proposal.unsupported_notes)})",
                icon=":material/warning:",
            ):
                for note in proposal.unsupported_notes:
                    st.write(f"- {note}")

        with st.expander("Advanced details", icon=":material/code:"):
            st.json(json.loads(accepted_proposal.model_dump_json()))

        semantic_errors = validate_rule_set(accepted_proposal, dataframe)
        if semantic_errors:
            st.error("The selected checks cannot run until these issues are resolved.")
            for error in semantic_errors:
                st.write(f"- {error}")
        else:
            verification_context = hashlib.sha256(
                (proposal_context + accepted_proposal.model_dump_json()).encode("utf-8")
            ).hexdigest()
            st.caption(":material/verified: Selected checks passed schema and dataset validation")
            verify_clicked = st.button(
                "Run selected checks",
                icon=":material/fact_check:",
                type="primary",
                disabled=not proposal_is_current or not bool(accepted_proposal.rules),
            )
            if verify_clicked:
                st.session_state["verified_rule_findings"] = execute_rule_set(
                    accepted_proposal, dataframe
                )
                st.session_state["verified_rule_context"] = verification_context

        verified_findings = st.session_state.get("verified_rule_findings")
        verified_is_current = (
            not semantic_errors
            and st.session_state.get("verified_rule_context") == verification_context
        )
        if verified_findings is not None and verified_is_current:
            st.divider()
            st.markdown("#### 3. Results")
            if not verified_findings:
                st.success("No violations detected", icon=":material/check_circle:")
            else:
                verified_table = verified_summary(verified_findings)
                st.dataframe(
                    verified_table.style.map(severity_style, subset=["Severity"]),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Severity": st.column_config.TextColumn(width="small"),
                        "Rule tested": st.column_config.TextColumn(width="large"),
                        "Affected records": st.column_config.NumberColumn(
                            format="%d", width="small"
                        ),
                        "Explanation": st.column_config.TextColumn(width="large"),
                    },
                )
                st.markdown("##### Evidence")
                for finding in verified_findings:
                    label = (
                        f"{finding.severity} · {finding.affected_count} records · "
                        f"{finding.rule_tested}"
                    )
                    with st.expander(label):
                        st.write(finding.explanation)
                        evidence = finding.evidence(dataframe)
                        st.dataframe(evidence.head(200), hide_index=True, width="stretch")
                        if len(evidence) > 200:
                            st.caption(f"Showing 200 of {len(evidence)} affected rows.")

with profile_tab:
    st.subheader("Columns")
    st.caption(f"{len(profile):,} fields · Roles inferred from clinical-trial naming conventions")
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
        st.markdown("#### Missing data by field")
        st.bar_chart(missing_profile.set_index("Column"), color="#b44d3b")

with data_tab:
    data_title, data_export = st.columns([3, 1], vertical_alignment="bottom")
    data_title.subheader("Data")
    data_title.caption(f"{len(dataframe):,} rows · {len(dataframe.columns):,} columns")
    data_export.download_button(
        "Export CSV",
        dataframe_to_safe_csv(dataframe),
        file_name="trial_sleuth_data.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
    st.dataframe(dataframe, hide_index=True, width="stretch", height=540)
