import asyncio
import logging
from typing import List

from app.config.Settings import get_settings
from app.service.embedding.MistralAIEmbeddingService import MistralAIEmbeddingService
from app.service.embedding.genai_service import GenAIEmbeddingService
from app.service.utils.resilience import gemini_breaker

logger = logging.getLogger(__name__)


async def get_query_embedding_async(text: str) -> List[float]:
    # wrap sync embedding call in a thread to avoid blocking event loop
    embedding_service = get_embedding_service()
    emb = await asyncio.to_thread(embedding_service.embed, text,
                                  'retrieval_query',
                                  get_settings().EMBEDDING_DIM)
    return list(emb)


def get_embedding_service():
    settings = get_settings()

    # Check if primary embedder is genai and if its breaker is OPEN
    if settings.EMBEDDER == "genai":
        if gemini_breaker.current_state == 'open':
            logging.warning("Gemini breaker is OPEN. Rerouting embedding to MistralAI fallback.")
            return MistralAIEmbeddingService()

        logging.info(
            f'embedding {settings.EMBEDDER} model to be used: {settings.EMBEDDING_MODEL}, DIMENSION: {settings.EMBEDDING_DIM}')
        return GenAIEmbeddingService()
    elif settings.EMBEDDER == "mistralai":
        logging.info(
            f'embedding {settings.EMBEDDER} model to be used: {settings.EMBEDDING_MODEL}, DIMENSION: {settings.EMBEDDING_DIM}')
        return MistralAIEmbeddingService()

    raise RuntimeError("unsupported embedder")
