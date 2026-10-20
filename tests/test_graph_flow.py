from types import SimpleNamespace

import pytest

from agent.contracts import ContractError
from agent.social_content_graph import (
    ContentState,
    aggregate_scores_node,
    check_refinement_node,
    finalize_content_node,
    generate_content_node,
    refine_content_node,
    score_personas_node,
)

DRAFTS = {p: {"post": f"draft for {p}"} for p in ("twitter", "facebook", "linkedin", "instagram")}
SCORES_LOW = {"strategist": {"score": 7, "feedback": "add context"}}
SCORES_HIGH = {"strategist": {"score": 9, "feedback": "accepted"}}


def response(payload):
    import json
    return SimpleNamespace(content=json.dumps(payload))


def state():
    return ContentState(topic="test")


def test_generation_scoring_acceptance(monkeypatch):
    s = state()
    monkeypatch.setattr("agent.social_content_graph.get_content_drafts", lambda _: response(DRAFTS))
    monkeypatch.setattr("agent.social_content_graph.judge_content", lambda *_: response(SCORES_HIGH))
    generate_content_node(s); score_personas_node(s); aggregate_scores_node(s)
    assert check_refinement_node(s) == "finalize"
    finalize_content_node(s)
    assert set(s.final_content) == set(DRAFTS)


def test_generation_refinement_rescore(monkeypatch):
    s = state(); calls = {"score": 0}
    monkeypatch.setattr("agent.social_content_graph.get_content_drafts", lambda _: response(DRAFTS))
    def judge(*_):
        calls["score"] += 1
        return response(SCORES_LOW if calls["score"] == 1 else SCORES_HIGH)
    monkeypatch.setattr("agent.social_content_graph.judge_content", judge)
    monkeypatch.setattr("agent.social_content_graph.refine_content", lambda *_: response({"post": "refined"}))
    generate_content_node(s); score_personas_node(s); aggregate_scores_node(s)
    assert check_refinement_node(s) == "refine"
    refine_content_node(s); score_personas_node(s); aggregate_scores_node(s)
    assert check_refinement_node(s) == "finalize"
    assert calls["score"] == 8


def test_malformed_generation_is_explicit():
    with pytest.raises(ContractError):
        from agent.contracts import parse_drafts
        parse_drafts("{}")


def test_refinement_is_bounded():
    s = state(); s.aggregated_scores = {"twitter": 1}; s.persona_scores = {"twitter": SCORES_LOW}
    assert check_refinement_node(s) == "refine"
    s.refinement_loops = {k: 1 for k in s.refinement_loops}
    assert check_refinement_node(s) == "finalize"
