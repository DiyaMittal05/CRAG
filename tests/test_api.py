from fastapi.testclient import TestClient

from app.main import app, get_graph


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_grounded_answer(graph):
    app.dependency_overrides[get_graph] = lambda: graph
    client = TestClient(app)
    try:
        response = client.post("/query", json={"question": "How do I reset my password?"})
        assert response.status_code == 200
        body = response.json()
        assert body["route"] == "vectorstore"
        assert body["answer"]
        assert isinstance(body["sources"], list)
        assert "trace_id" in body
    finally:
        app.dependency_overrides.clear()


def test_query_endpoint_rejects_empty_question(graph):
    app.dependency_overrides[get_graph] = lambda: graph
    client = TestClient(app)
    try:
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()
