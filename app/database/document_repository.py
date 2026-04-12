import logging
from typing import List

from app.schema.document_record import DocumentRecord
from app.service.embedding.EmbeddingFactory import get_query_embedding_async
from app.service.utils.time_helper import time_coro
from app.service.vector_store.VectorStoreFactory import get_vector_store

logger = logging.getLogger(__name__)


class DocumentRepository:
    def __init__(self):
        logger.info('inside def __init__(self, db: DocumentRepository)')
        self.vs = get_vector_store()

    async def get_top_k_docs(self, query: str, k: int = 3) -> List[DocumentRecord]:
        query_emb: List[float] = await get_query_embedding_async(query)
        docs: List[DocumentRecord] = await self.vs.hybrid_search(query_emb, k, query)
        return docs

    async def get_top_k_docs_by_embedding(self, query: str, query_emb: List[float], k: int = 3) -> List[DocumentRecord]:
        docs: List[DocumentRecord] = await time_coro('hybrid search', self.vs.hybrid_search(query_emb, k, query))
        return docs
