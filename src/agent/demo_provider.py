"""Deterministic provider adapter used by the public PULSE demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

PLATFORMS = ("twitter", "facebook", "linkedin", "instagram")


@dataclass
class ReplayProvider:
    """A credential-free provider with one intentional refinement pass."""

    score_calls: int = 0

    def _message(self, payload: object) -> SimpleNamespace:
        return SimpleNamespace(content=json.dumps(payload), usage_metadata={"total_tokens": 0})

    def generate(self, _: str) -> SimpleNamespace:
        return self._message(
            {platform: {"post": f"A context-first draft for {platform}."} for platform in PLATFORMS}
        )

    def judge(self, platform: str, _: str) -> SimpleNamespace:
        self.score_calls += 1
        score = 7.0 if self.score_calls <= len(PLATFORMS) else 9.0
        return self._message(
            {"strategist": {"score": score, "feedback": f"Improve {platform} context."}}
        )

    def refine(self, platform: str, _: str, __: str) -> SimpleNamespace:
        return self._message(
            {"post": f"A refined, context-first {platform} message with a clear next step."}
        )
