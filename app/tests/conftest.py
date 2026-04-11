import os
import pytest
from app.schema.document_record import DocumentRecord, Topic, ClassificationResult, SearchRequest

os.environ.setdefault("QDRANT_HOST", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-api-key")


@pytest.fixture
def blockchain_record():
    return DocumentRecord(
        resume_id="REAL_2080",
        name="James Bentley",
        education="Computer Science degree",
        category="Blockchain",
        skills=[
            "Blockchain",
            "SQL",
            "Git",
            "Linux"
        ],
        summary="blockchain developer with good experience",
    )


@pytest.fixture
def mock_topics() -> ClassificationResult:
    """Returns a list of mock Topic instances."""
    return ClassificationResult(result=[
        Topic(name="Blockchain Development", confidence=0.95),
        Topic(name="Cryptocurrency", confidence=0.85),
        Topic(name="Distributed Ledger Technology (DLT)", confidence=0.82),
        Topic(name="Smart Contracts", confidence=0.75),
        Topic(name="Decentralization", confidence=0.72)
    ])


@pytest.fixture
def mock_search_request() -> SearchRequest:
    """Returns a list of mock Topic instances."""
    return SearchRequest(search_term="Blockchain developers", limit=3)
