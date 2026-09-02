# TrialSleuth

TrialSleuth is a lightweight Streamlit app for investigating data-quality problems in clinical-trial CSV files. It profiles an uploaded dataset and links each finding back to the affected source rows.

Study notes can also be translated by Anthropic Claude into a closed, schema-validated rule specification. The proposal is displayed for review before fixed Python validation functions execute it. Model-generated Python, SQL, expressions, and other arbitrary code are never executed.

## Checks

- Exact duplicate rows
- Duplicate participant/visit combinations when those columns can be inferred
- Missing values, including blank strings
- Malformed or implausible date values
- Visit dates that conflict with inferred visit order
- Numeric outliers using the 1.5 x IQR rule
- Categorical values that differ only by case or whitespace

AI-proposed study rules support required fields, numeric ranges, allowed categorical values, date ordering, conditional requirements, and participant visit windows. These findings are shown separately from the built-in deterministic checks.

Participant, visit, and date roles are inferred from common column names such as `participant_id`, `USUBJID`, `visit`, `timepoint`, and `visit_date`. A synthetic dataset with planted errors loads automatically for the demo.

## AI safety boundary

- Claude receives the study notes and dataset column names and data types, but never record values.
- Claude can only return the supported rule types through a strict Pydantic schema. Proposals containing unknown fields, executable code, invalid bounds, missing columns, duplicate rules, incompatible data types, or ambiguous values are rejected before execution.
- A proposal is tied to the exact dataset and study notes used to create it. Changing either invalidates the proposal and any verified findings until a new proposal is generated.
- Accepted rules run through fixed, deterministic Python validation functions. Claude does not execute rules or decide which records are erroneous.

## Run locally

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key"  # Optional; built-in checks work without it
streamlit run app.py
```

Open `http://localhost:8501`, then use the sidebar to upload a CSV or investigate the built-in example.

Set `ANTHROPIC_MODEL` to override the default `claude-haiku-4-5-20251001` proposal model. Only column names, data types, and the entered study notes are sent to Claude; record values remain local. Without an Anthropic API key, the proposal control is disabled and all existing functionality remains available. Use **Load demo study rules** with the bundled synthetic data for a repeatable end-to-end demonstration.

## Demo workflow

1. Start the app and keep the bundled **Synthetic example** dataset selected.
2. Click **Load demo study rules** in the sidebar.
3. With `ANTHROPIC_API_KEY` configured, click **Propose study rules**.
4. Open the **Study rules** tab and review the proposed rules, unsupported notes, and validated specification.
5. Click **Verify accepted rules** to execute the fixed validators and inspect the evidence-backed findings.

Uploading a CSV disables the bundled demo-rule shortcut. Built-in findings, the column profile, source-data inspection, and findings export remain available without an Anthropic key.

## Tests

```bash
pytest
```

The current suite contains 22 tests covering the Streamlit workflow, provider privacy boundary, strict rule parsing and dataset validation, deterministic rule execution, built-in checks, and spreadsheet-safe CSV export.

## Structure

- `app.py`: Streamlit dashboard and dataset profile
- `trialsleuth/validation.py`: deterministic validation functions
- `trialsleuth/study_rules.py`: strict rule schema and deterministic rule interpreter
- `trialsleuth/rule_proposals.py`: schema-constrained Anthropic proposal adapter
- `trialsleuth/export.py`: spreadsheet-safe findings export
- `sample_data/synthetic_trial_data.csv`: demo data with planted issues
- `tests/test_app.py`: Streamlit workflow smoke test
- `tests/test_rule_proposals.py`: Anthropic adapter and record-privacy test
- `tests/test_study_rules.py`: rule schema, dataset validation, and execution tests
- `tests/test_validation.py`: built-in validation tests
- `tests/test_export.py`: spreadsheet-safe export test
