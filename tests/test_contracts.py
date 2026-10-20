import pytest

from agent.contracts import ContractError, parse_drafts, parse_scores


def test_drafts_are_structured():
    raw = '{"twitter":{"post":"t"},"facebook":{"post":"f"},"linkedin":{"post":"l"},"instagram":{"post":"i"}}'
    assert parse_drafts(raw)["twitter"]["post"] == "t"


def test_malformed_judge_output_is_rejected():
    with pytest.raises(ContractError):
        parse_scores('{"persona":{"score": 11}}')
