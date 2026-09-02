import json
from types import SimpleNamespace

import pandas as pd
import pytest

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

    class FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(parsed_output=parsed_proposal)

    class FakeAnthropic:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.messages = FakeMessages()

    monkeypatch.setattr(rule_proposals, "Anthropic", FakeAnthropic)
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

    user_context = json.loads(captured["messages"][0]["content"])
    assert user_context == {
        "dataset_columns": [
            {"name": "participant_id", "data_type": "object"},
            {"name": "age", "data_type": "int64"},
        ],
        "study_notes": "participant_id is required",
    }
    assert "record-secret-P001" not in captured["messages"][0]["content"]
    assert captured["base_url"] == "https://api.anthropic.com"
    assert captured["system"] == rule_proposals.SYSTEM_PROMPT
    assert captured["output_format"] is RuleSet
    assert proposal == parsed_proposal


def test_proposal_permission_error_explains_runtime_access(monkeypatch) -> None:
    class PermissionErrorFromProvider(Exception):
        status_code = 403

    class FakeMessages:
        def parse(self, **kwargs):
            raise PermissionErrorFromProvider

    class FakeAnthropic:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            self.messages = FakeMessages()

    monkeypatch.setattr(rule_proposals, "Anthropic", FakeAnthropic)

    with pytest.raises(rule_proposals.RuleProposalError, match="blocked.*HTTP 403"):
        rule_proposals.propose_rule_set(
            "participant_id is required",
            pd.DataFrame({"participant_id": ["P001"]}),
            api_key="test-key",
        )


def test_proposal_ignores_anthropic_base_url_from_environment(monkeypatch) -> None:
    captured: dict[str, str] = {}
    parsed_proposal = RuleSet(rules=[], unsupported_notes=[])

    class FakeMessages:
        def parse(self, **kwargs):
            return SimpleNamespace(parsed_output=parsed_proposal)

    class FakeAnthropic:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["base_url"] = base_url
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://untrusted.invalid")
    monkeypatch.setattr(rule_proposals, "Anthropic", FakeAnthropic)

    rule_proposals.propose_rule_set(
        "participant_id is required",
        pd.DataFrame({"participant_id": ["P001"]}),
        api_key="test-key",
    )

    assert captured["base_url"] == rule_proposals.ANTHROPIC_API_URL
