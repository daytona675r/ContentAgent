"""Provider-independent contracts used at the graph boundaries."""

from __future__ import annotations

import json
from typing import Any

PLATFORMS = ("twitter", "facebook", "linkedin", "instagram")


class ContractError(ValueError):
    """Raised when a model response cannot satisfy a graph contract."""


def parse_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContractError("model response was not valid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError("model response must be a JSON object")
    return value


def parse_drafts(raw: str) -> dict[str, dict[str, str]]:
    value = parse_object(raw)
    drafts: dict[str, dict[str, str]] = {}
    for platform in PLATFORMS:
        item = value.get(platform)
        if not isinstance(item, dict) or not isinstance(item.get("post"), str) or not item["post"].strip():
            raise ContractError(f"missing generated post for {platform}")
        drafts[platform] = {"post": item["post"].strip()}
    return drafts


def parse_scores(raw: str) -> dict[str, dict[str, Any]]:
    value = parse_object(raw)
    if not value:
        raise ContractError("judge response did not contain scores")
    result: dict[str, dict[str, Any]] = {}
    for persona, score in value.items():
        if not isinstance(score, dict) or not isinstance(score.get("score"), (int, float)):
            raise ContractError(f"invalid score for {persona}")
        if not 0 <= float(score["score"]) <= 10:
            raise ContractError(f"score out of range for {persona}")
        result[persona] = {"score": float(score["score"]), "feedback": str(score.get("feedback", ""))}
    return result
