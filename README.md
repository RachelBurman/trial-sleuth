# TrialSleuth

TrialSleuth is a lightweight Streamlit app for investigating data-quality problems in clinical-trial CSV files. It profiles an uploaded dataset and links each finding back to the affected source rows.

## Checks

- Exact duplicate rows
- Duplicate participant/visit combinations when those columns can be inferred
- Missing values, including blank strings
- Malformed or implausible date values
- Visit dates that conflict with inferred visit order
- Numeric outliers using the 1.5 x IQR rule
- Categorical values that differ only by case or whitespace

Participant, visit, and date roles are inferred from common column names such as `participant_id`, `USUBJID`, `visit`, `timepoint`, and `visit_date`. A synthetic dataset with planted errors loads automatically for the demo.

## Run locally

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, then use the sidebar to upload a CSV or investigate the built-in example.

## Tests

```bash
pytest
```

## Structure

- `app.py`: Streamlit dashboard and dataset profile
- `trialsleuth/validation.py`: deterministic validation functions
- `trialsleuth/export.py`: spreadsheet-safe findings export
- `sample_data/synthetic_trial_data.csv`: demo data with planted issues
- `tests/test_validation.py`: focused validation tests

Study rules and data-dictionary notes can be entered in the sidebar for reference. They are intentionally not evaluated or sent to an LLM in this MVP.
