"""Typed, credential-free API for the deterministic PULSE demo."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from demo_mode import (
    DEMO_CONTEXT,
    DEMO_NOTICE,
    DEMO_OUTPUTS,
    DEMO_REFINEMENT,
    DEMO_SCORES,
)


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
    return DemoRunResponse(
        notice=DEMO_NOTICE,
        scenario=scenario,
        initial_drafts=DEMO_OUTPUTS,
        persona_scores=DEMO_SCORES,
        persona_feedback=DEMO_REFINEMENT["feedback"],
        refinement_history=[
            {"stage": "initial", "content": DEMO_REFINEMENT["initial"]},
            {"stage": "refined", "content": DEMO_REFINEMENT["accepted"]},
        ],
        final_accepted=DEMO_OUTPUTS,
        token_cost={"tokens": 842, "estimated_usd": 0.0},
    )
