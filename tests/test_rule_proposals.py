import json
from types import SimpleNamespace

import pandas as pd

import trialsleuth.rule_proposals as rule_proposals
from trialsleuth.study_rules import RuleSet


def test_proposal_request_sends_schema_metadata_but_not_record_values(monkeypatch) -> None:
    captured: dict[str, object] = {}
    parsed_proposal = RuleSet.model_validate(
        {
            "rules": [
                {
                    "rule_type": "required",
                    "severity": "High",
                    "field": "participant_id",
                }
            ],
            "unsupported_notes": [],
        },
        strict=True,
    )

    class FakeResponses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=parsed_proposal)

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    monkeypatch.setattr(rule_proposals, "OpenAI", FakeOpenAI)
    dataframe = pd.DataFrame(
        {
            "participant_id": ["record-secret-P001"],
            "age": [42],
        }
    )

    proposal = rule_proposals.propose_rule_set(
        "participant_id is required",
        dataframe,
        api_key="test-key",
        model="test-model",
    )

    user_context = json.loads(captured["input"][1]["content"])
    assert user_context == {
        "dataset_columns": [
            {"name": "participant_id", "data_type": "object"},
            {"name": "age", "data_type": "int64"},
        ],
        "study_notes": "participant_id is required",
    }
    assert "record-secret-P001" not in captured["input"][1]["content"]
    assert captured["text_format"] is RuleSet
    assert proposal == parsed_proposal
