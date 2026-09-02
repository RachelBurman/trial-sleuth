from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "app.py"


def test_demo_rules_load_and_missing_anthropic_key_is_graceful(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()

    assert not app.exception
    load_demo = next(button for button in app.button if button.label == "Load demo study rules")
    propose = next(button for button in app.button if button.label == "Propose study rules")
    assert not load_demo.disabled
    assert propose.disabled

    load_demo.click().run()

    notes = app.text_area(key="study_notes").value
    assert "age is required and must be between 18 and 65" in notes
    assert "Week 4 visit_date must occur 28 +/- 3 days after the Baseline" in notes
    assert any("ANTHROPIC_API_KEY" in notice.value for notice in app.info)
    assert any(
        "Natural-language study specification" in markdown.value
        for markdown in app.markdown
    )
    assert not app.exception
