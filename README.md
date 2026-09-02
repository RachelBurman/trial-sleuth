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

## Tests

```bash
pytest
```

## Structure

- `app.py`: Streamlit dashboard and dataset profile
- `trialsleuth/validation.py`: deterministic validation functions
- `trialsleuth/study_rules.py`: strict rule schema and deterministic rule interpreter
- `trialsleuth/rule_proposals.py`: schema-constrained Anthropic proposal adapter
- `trialsleuth/export.py`: spreadsheet-safe findings export
- `sample_data/synthetic_trial_data.csv`: demo data with planted issues
- `tests/test_validation.py`: focused validation tests
