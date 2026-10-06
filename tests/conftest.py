from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.config import Settings
from app.graph.build import build_graph
from app.llm.fake import FakeLLMClient
from app.observability.tracer import Tracer
from app.vectorstore.chroma_store import build_vectorstore, get_retriever

FIXTURE_DOCS = [
    Document(
        page_content="To reset your password, click Forgot password on the login page "
        "and follow the emailed reset link, which expires after 1 hour.",
        metadata={"source": "account_access.md"},
    ),
    Document(
        page_content="Refunds are issued for duplicate charges or billing errors within "
        "30 days of the charge. We do not refund voluntary cancellations.",
        metadata={"source": "billing.md"},
    ),
    Document(
        page_content="API requests are limited to 600 requests per minute on the Pro plan "
        "and 60 requests per minute on the Free plan.",
        metadata={"source": "api_usage.md"},
    ),
]


@pytest.fixture
def test_settings(tmp_path):
    return Settings(
        use_fake_backends=True,
        chroma_persist_dir=str(tmp_path / "chroma"),
        relevance_threshold=0.2,
        max_retries=2,
        retriever_k=3,
    )


@pytest.fixture
def retriever(test_settings):
    vectorstore = build_vectorstore(test_settings, collection_name="test_support_docs")
    vectorstore.add_documents(FIXTURE_DOCS)
    return get_retriever(vectorstore, k=test_settings.retriever_k)


@pytest.fixture
def graph(test_settings, retriever):
    llm_client = FakeLLMClient()
    tracer = Tracer()
    return build_graph(llm_client, retriever, test_settings, tracer)
