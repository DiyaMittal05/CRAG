def test_vectorstore_route_produces_grounded_answer(graph):
    result = graph.invoke({"question": "How do I reset my password?"})

    assert result["route"] == "vectorstore"
    assert result["answer"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["escalated"], bool)
    assert result["documents"], "expected at least one relevant document to survive grading"


def test_escalation_route_short_circuits_before_retrieval(graph):
    result = graph.invoke({"question": "I want to speak to a human agent right now"})

    assert result["route"] == "escalate"
    assert result["escalated"] is True
    assert result["documents"] == []


def test_direct_answer_route_for_greeting(graph):
    result = graph.invoke({"question": "hello"})

    assert result["route"] == "direct_answer"


def test_weak_retrieval_triggers_query_rewrite_retry(graph):
    result = graph.invoke({"question": "what color is the sky"})

    assert result["route"] == "vectorstore"
    assert result["retries"] >= 1, "an off-topic question should trigger at least one self-correction retry"
