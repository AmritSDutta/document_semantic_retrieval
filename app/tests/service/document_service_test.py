from unittest.mock import patch, AsyncMock

import pytest

from app.service.document_service import DocumentService


class TestDocumentService:
    @pytest.mark.asyncio
    async def test_get_matching_docs(self, mock_search_request, blockchain_record):
        with patch("app.database.document_repository.DocumentRepository") as mock_repository:
            svc = DocumentService(mock_repository)
            # special case if method is async , need to mock the return value with await
            mock_repository.get_top_k_docs = AsyncMock(return_value=[blockchain_record.model_dump()])

            response = await svc.get_matching_docs(search_term='test', how_many=1)
            assert isinstance(response, list)
            assert response[0]["resume_id"] == "REAL_2080"
            mock_repository.get_top_k_docs.assert_called_once()
