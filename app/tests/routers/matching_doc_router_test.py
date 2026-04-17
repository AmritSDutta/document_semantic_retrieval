from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.matching_doc_router import doc_router, get_document_service
from app.schema.document_record import ClassificationResult

app = FastAPI()
app.include_router(doc_router)
mock_svc = AsyncMock()
app.dependency_overrides[get_document_service] = lambda: mock_svc
client = TestClient(app)


class TestMatchingDocRouter:

    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        """Resets the global mock_svc before each test execution."""
        mock_svc.reset_mock()

    def test_search_docs(self, mock_search_request, blockchain_record):
        with patch("app.routers.matching_doc_router.do_moderation_checking_mistral", new_callable=AsyncMock) as mock_mod, \
                patch("app.routers.matching_doc_router.pii_redactor.do_pii_redaction_text",
                      new_callable=AsyncMock) as mock_pii:
            mock_pii.return_value = ["Blockchain developers"]
            mock_svc.get_matching_docs_by_embedding.return_value = [blockchain_record.model_dump()]

            response = client.post(
                "/docs/search",
                headers={"X-API-KEY": "1234"},
                json=mock_search_request.model_dump()
            )

            assert response.status_code == 200
            assert isinstance(response.json(), list)
            assert response.json()[0]["resume_id"] == "REAL_2080"
            mock_mod.assert_awaited_once()
            mock_pii.assert_awaited_once_with(["Blockchain developers"])
            mock_svc.get_matching_docs_by_embedding.assert_awaited_once()

    def test_classify_doc(self, mock_topics):
        with patch("app.routers.matching_doc_router.do_moderation_checking_mistral", new_callable=AsyncMock) as mock_mod, \
                patch("app.routers.matching_doc_router.pii_redactor.do_pii_redaction_text",
                      new_callable=AsyncMock) as mock_pii, \
                patch("app.routers.matching_doc_router.llm.llm_classify_request", new_callable=AsyncMock) as mock_llm:
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
            mock_mod.assert_awaited_once()
            mock_llm.assert_awaited_once()

    def test_classify_and_search(self, mock_topics, blockchain_record):
        with patch("app.routers.matching_doc_router.do_moderation_checking_mistral", new_callable=AsyncMock) as mock_mod, \
                patch("app.routers.matching_doc_router.pii_redactor.do_pii_redaction_text",
                      new_callable=AsyncMock) as mock_pii, \
                patch("app.routers.matching_doc_router.llm.llm_classify_request", new_callable=AsyncMock) as mock_llm:
            mock_pii.return_value = ["Blockchain developers needed"]
            mock_llm.return_value = mock_topics
            mock_svc.get_matching_docs.return_value = [blockchain_record.model_dump()]

            response = client.post(
                "/docs/search_requirement",
                headers={"X-API-KEY": "1234"},
                json={"passage": "Blockchain developers needed"}
            )

            assert response.status_code == 200
            assert response.json()[0]["resume_id"] == "REAL_2080"
            mock_mod.assert_awaited_once()
            mock_llm.assert_awaited_once()

    def test_train_classic_ml_model(self):
        with patch("app.service.classic_ml.berttopic_modeling.train_topic_model") as mock_train:
            response = client.post("/docs/train_classic_ml", headers={"X-API-KEY": "1234"})
            assert response.status_code == 200
            assert response.json() == "classic ml training task submitted"
            mock_train.assert_called_once()

    def test_classify_and_search_classic_ml(self, mock_topics, blockchain_record):
        with patch("app.routers.matching_doc_router.do_moderation_checking_mistral", new_callable=AsyncMock) as mock_mod, \
                patch("app.routers.matching_doc_router.pii_redactor.do_pii_redaction_text",
                      new_callable=AsyncMock) as mock_pii, \
                patch("app.service.classic_ml.berttopic_modeling.infer_topic_model") as mock_topic:
            mock_pii.return_value = ["Blockchain developers needed"]
            mock_topic.return_value = [{"topic_name": "Blockchain Development", "confidence": "0.95"}]
            mock_svc.get_matching_docs.return_value = [blockchain_record.model_dump()]

            response = client.post(
                "/docs/search_through_classic_ml",
                headers={"X-API-KEY": "1234"},
                json={"passage": "Blockchain developers needed"}
            )

            assert response.status_code == 200
            assert response.json()[0]["resume_id"] == "REAL_2080"
            mock_mod.assert_awaited_once()
            mock_topic.assert_called_once()
