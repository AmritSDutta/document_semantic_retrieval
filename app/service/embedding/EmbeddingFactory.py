import asyncio
import logging
from typing import List

from app.config.Settings import get_settings
from app.service.embedding.MistralAIEmbeddingService import MistralAIEmbeddingService
from app.service.embedding.genai_service import GenAIEmbeddingService

logger = logging.getLogger(__name__)

_MistralAIEmbeddingService: MistralAIEmbeddingService | None = None
_GenAIEmbeddingService: GenAIEmbeddingService | None = None


async def get_query_embedding_async(text: str) -> List[float]:
    # wrap sync embedding call in a thread to avoid blocking event loop
    embedding_service = get_embedding_service()
    emb = await embedding_service.embed(text, 'retrieval_query', get_settings().EMBEDDING_DIM)
    return list(emb)


def get_embedding_service():
    settings = get_settings()
    global _MistralAIEmbeddingService, _GenAIEmbeddingService
    # Check if primary embedder is genai and if its breaker is OPEN
    if settings.EMBEDDER == "genai":

        if _GenAIEmbeddingService is None:
            logging.info(
                f'Provider: {settings.EMBEDDER}, Model: {settings.EMBEDDING_MODEL}, Dimension: {settings.EMBEDDING_DIM}')
            _GenAIEmbeddingService = GenAIEmbeddingService()
        return _GenAIEmbeddingService
    elif settings.EMBEDDER == "mistralai":
        if _MistralAIEmbeddingService is None:
            logging.info(
                f'Provider {settings.EMBEDDER}, Model: {settings.EMBEDDING_MODEL}, Dimension: {settings.EMBEDDING_DIM}')
            _MistralAIEmbeddingService = MistralAIEmbeddingService()
        return _MistralAIEmbeddingService

    raise RuntimeError("unsupported embedder")
