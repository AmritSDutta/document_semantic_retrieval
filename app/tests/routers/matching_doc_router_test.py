from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.matching_doc_router import doc_router, get_document_service
from app.schema.document_record import SearchRequest, DocumentRecord, ClassificationResult

app = FastAPI()
app.include_router(doc_router)
mock_svc = AsyncMock()
app.dependency_overrides[get_document_service] = lambda: mock_svc
client = TestClient(app)


def test_search_docs(mock_search_request, blockchain_record):
    with patch("app.routers.matching_doc_router.do_moderation_checking", new_callable=AsyncMock) as mock_mod, \
            patch("app.routers.matching_doc_router.pii_redactor.do_pii_redaction_text",
                  new_callable=AsyncMock) as mock_pii:
        mock_pii.return_value = ["Blockchain developers"]

        mock_svc.get_matching_docs.return_value = [blockchain_record.model_dump()]

        response = client.post(
            "/docs/search",
            headers={"X-API-KEY": "1234"},
            json=mock_search_request.model_dump()
        )

        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert response.json()[0]["resume_id"] == "REAL_2080"

        # Verify the internal calls were made correctly
        mock_mod.assert_awaited_once()
        mock_pii.assert_awaited_once_with(["Blockchain developers"])
        mock_svc.get_matching_docs.assert_awaited_once()


def test_classify_doc(mock_topics):
    with patch("app.routers.matching_doc_router.do_moderation_checking", new_callable=AsyncMock) as mock_mod, \
            patch("app.routers.matching_doc_router.pii_redactor.do_pii_redaction_text",
                  new_callable=AsyncMock) as mock_pii, \
            patch("app.routers.matching_doc_router.llm.llmClassifyRequest", new_callable=AsyncMock) as mock_llm:
        mock_pii.return_value = ["Blockchain developers needed"]
        mock_llm.return_value = mock_topics

        response = client.post(
            "/docs/classify",
            headers={"X-API-KEY": "1234"},
            json={"passage": "Blockchain developers needed"}
        )

        assert response.status_code == 200
        result_model = ClassificationResult(**response.json())
        assert isinstance(result_model, ClassificationResult)
        assert response.json()['result'][0]["name"] == "Blockchain Development"
        # Verify the internal calls were made correctly
        mock_mod.assert_awaited_once()
        mock_pii.assert_awaited_once_with(["Blockchain developers needed"])
        mock_llm.assert_awaited_once()


def test_classify_and_search(mock_topics, blockchain_record):
    with patch("app.routers.matching_doc_router.do_moderation_checking", new_callable=AsyncMock) as mock_mod, \
            patch("app.routers.matching_doc_router.pii_redactor.do_pii_redaction_text",
                  new_callable=AsyncMock) as mock_pii, \
            patch("app.routers.matching_doc_router.llm.llmClassifyRequest", new_callable=AsyncMock) as mock_llm:
        mock_pii.return_value = ["Blockchain developers needed"]
        mock_llm.return_value = mock_topics
        mock_svc.get_matching_docs.return_value = [blockchain_record.model_dump()]

        response = client.post(
            "/docs/search_requirement",
            headers={"X-API-KEY": "1234"},
            json={"passage": "Blockchain developers needed"}
        )

        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert response.json()[0]["resume_id"] == "REAL_2080"

        # Verify the internal calls were made correctly
        mock_mod.assert_awaited_once()
        mock_pii.assert_awaited_once_with(["Blockchain developers needed"])
        mock_llm.assert_awaited_once()
