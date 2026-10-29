"""Typed, credential-free API for the deterministic PULSE demo."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.demo_provider import ReplayProvider
from agent.social_content_graph import ContentState, social_content_graph
from demo_mode import DEMO_CONTEXT, DEMO_NOTICE


class Scenario(BaseModel):
    id: str
    label: str
    context: str


class DemoRunRequest(BaseModel):
    scenario_id: str = Field(default="customer-context")


class DemoRunResponse(BaseModel):
    demo: bool = True
    notice: str
    scenario: Scenario
    initial_drafts: dict[str, str]
    persona_scores: dict[str, float]
    persona_feedback: str
    refinement_history: list[dict[str, str]]
    final_accepted: dict[str, str]
    token_cost: dict[str, float | int]


SCENARIOS = [
    Scenario(
        id="customer-context",
        label="Customer context to platform content",
        context=DEMO_CONTEXT,
    ),
]

app = FastAPI(title="PULSE Demo API", version="0.1.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv("PULSE_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo/scenarios", response_model=list[Scenario])
def scenarios() -> list[Scenario]:
    return SCENARIOS


@app.post("/api/demo/run", response_model=DemoRunResponse)
def run_demo(request: DemoRunRequest) -> DemoRunResponse:
    scenario = next((item for item in SCENARIOS if item.id == request.scenario_id), SCENARIOS[0])
    state = ContentState(topic=scenario.context, provider=ReplayProvider())
    try:
        result = social_content_graph.invoke(state)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"deterministic workflow failed: {exc}"
        ) from exc
    histories = result["refinement_history"]
    if not histories or not result.get("final_content"):
        raise HTTPException(
            status_code=502, detail="deterministic workflow produced no final content"
        )
    initial = {platform: entries[0] for platform, entries in histories.items() if entries}
    final = {
        platform: content["post"] if isinstance(content, dict) else content
        for platform, content in result["final_content"].items()
    }
    history = [
        {"stage": "initial", "content": "\n".join(initial.values())},
        {"stage": "refined", "content": "\n".join(final.values())},
    ]
    scores = {
        platform: sum(item["score"] for item in feedback.values()) / len(feedback)
        for platform, feedback in result["persona_scores"].items()
    }
    feedback = "; ".join(
        item["feedback"]
        for persona_scores in result["persona_scores"].values()
        for item in persona_scores.values()
    )
    return DemoRunResponse(
        notice=DEMO_NOTICE,
        scenario=scenario,
        initial_drafts=initial,
        persona_scores=scores,
        persona_feedback=feedback,
        refinement_history=history,
        final_accepted=final,
        token_cost={
            "tokens": result.get("token_count", 0),
            "estimated_usd": result.get("price_usd", 0.0),
        },
    )
