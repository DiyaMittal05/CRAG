from fastapi.testclient import TestClient

from app.main import app, get_app_state


def _override(app_state: dict) -> None:
    app.dependency_overrides[get_app_state] = lambda: app_state


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_grounded_answer(graph, baseline_graph, llm_client):
    _override({"llm_client": llm_client, "graphs": {"crag": graph, "baseline": baseline_graph}})
    client = TestClient(app)
    try:
        response = client.post("/query", json={"question": "How do I reset my password?"})
        assert response.status_code == 200
        body = response.json()
        assert body["route"] == "vectorstore"
        assert body["answer"]
        assert body["mode"] == "crag"
        assert isinstance(body["llm_calls"], int) and body["llm_calls"] > 0
        assert isinstance(body["sources"], list)
        assert "trace_id" in body
    finally:
        app.dependency_overrides.clear()


def test_query_endpoint_baseline_mode_skips_correction(graph, baseline_graph, llm_client):
    _override({"llm_client": llm_client, "graphs": {"crag": graph, "baseline": baseline_graph}})
    client = TestClient(app)
    try:
        response = client.post(
            "/query", json={"question": "How do I reset my password?", "mode": "baseline"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "baseline"
        assert body["retries"] == 0, "baseline must never rewrite/re-retrieve"
    finally:
        app.dependency_overrides.clear()


def test_query_endpoint_rejects_empty_question(graph, baseline_graph, llm_client):
    _override({"llm_client": llm_client, "graphs": {"crag": graph, "baseline": baseline_graph}})
    client = TestClient(app)
    try:
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()
