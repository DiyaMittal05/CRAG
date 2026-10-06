import pytest
from pydantic import ValidationError

from app.schemas import DocumentGrade, SupportQuery


def test_document_grade_score_must_be_within_bounds():
    DocumentGrade(relevant=True, score=0.5, reasoning="ok")
    with pytest.raises(ValidationError):
        DocumentGrade(relevant=True, score=1.5, reasoning="out of bounds")


def test_support_query_requires_nonempty_question():
    with pytest.raises(ValidationError):
        SupportQuery(question="")
