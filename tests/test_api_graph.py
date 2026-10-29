from fastapi.testclient import TestClient

from api import main


def test_api_uses_graph_and_reports_refinement_trace() -> None:
    payload = TestClient(main.app).post("/api/demo/run", json={}).json()
    assert payload["initial_drafts"]
    assert payload["final_accepted"]
    assert payload["refinement_history"][0]["stage"] == "initial"
    assert payload["refinement_history"][1]["stage"] == "refined"
    assert all(score == 9.0 for score in payload["persona_scores"].values())


def test_malformed_demo_provider_is_explicit(monkeypatch) -> None:
    def fail(_state):
        raise ValueError("malformed deterministic provider output")

    monkeypatch.setattr(main.social_content_graph, "invoke", fail)
    response = TestClient(main.app).post("/api/demo/run", json={})
    assert response.status_code == 502
    assert "deterministic workflow failed" in response.json()["detail"]
