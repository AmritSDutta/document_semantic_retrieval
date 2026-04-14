import logging
from typing import List

from app.database.document_repository import DocumentRepository
from app.schema.document_record import DocumentRecord

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, repo: DocumentRepository):
        logger.info('inside def __init__(self, repo: DocumentService):')
        self.repo = repo

    async def get_matching_docs(self, search_term: str, how_many: int = 3) -> List[DocumentRecord] | None:
        logging.info(f'inside async get_matching_docs, searching: {search_term}')
        docs: List[DocumentRecord] = await self.repo.get_top_k_docs(search_term, how_many)
        return docs

    async def get_matching_docs_by_embedding(self, search_term: str,
                                             query_emb: List[float],
                                             how_many: int = 3) -> List[DocumentRecord] | None:
        logging.info(f'inside async get_matching_docs, searching: {search_term[:25]}')
        docs: List[DocumentRecord] = await self.repo.get_top_k_docs_by_embedding(search_term,query_emb, how_many)
        return docs
