from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_and_scenarios() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/api/demo/scenarios")
    assert response.status_code == 200
    assert response.json()[0]["id"] == "customer-context"


def test_demo_run_has_structured_workflow_trace() -> None:
    response = client.post("/api/demo/run", json={"scenario_id": "customer-context"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["demo"] is True
    assert payload["initial_drafts"]
    assert payload["persona_scores"]
    assert len(payload["refinement_history"]) == 2
    assert payload["final_accepted"]
    assert payload["token_cost"]["estimated_usd"] == 0.0
