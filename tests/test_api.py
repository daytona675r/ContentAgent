from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_and_scenarios() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/api/demo/scenarios")
    assert response.status_code == 200
    assert response.json()[0]["id"] == "onboarding-saas"


def test_demo_run_has_structured_workflow_trace() -> None:
    response = client.post("/api/demo/run", json={"scenario_id": "onboarding-saas"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["demo"] is True
    assert payload["initial_drafts"]
    assert payload["persona_scores"]
    assert set(payload["refinement_history"]) == {"twitter", "facebook", "linkedin", "instagram"}
    assert all(len(history) == 2 for history in payload["refinement_history"].values())
    assert payload["final_accepted"]
    assert payload["token_cost"]["estimated_usd"] == 0.0


def test_unknown_scenario_is_rejected() -> None:
    response = client.post("/api/demo/run", json={"scenario_id": "unknown"})
    assert response.status_code == 422


def test_scenarios_produce_distinct_authored_outputs() -> None:
    first = client.post("/api/demo/run", json={"scenario_id": "onboarding-saas"}).json()
    second = client.post("/api/demo/run", json={"scenario_id": "repairable-commuter"}).json()
    assert first["scenario"]["id"] != second["scenario"]["id"]
    assert first["final_accepted"]["twitter"] != second["final_accepted"]["twitter"]
    assert "fictional" in first["scenario"]["context"].lower()
