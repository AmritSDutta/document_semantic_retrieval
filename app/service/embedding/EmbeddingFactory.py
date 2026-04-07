import logging

from app.config.Settings import Settings
from app.service.embedding.MistralAIEmbeddingService import MistralAIEmbeddingService
from app.service.embedding.genai_service import GenAIEmbeddingService

logger = logging.getLogger(__name__)


def get_embedding_service():
    settings = Settings()
    if settings.EMBEDDER == "genai":
        logging.info(
            f'embedding {settings.EMBEDDER} model to be used: {settings.EMBEDDING_MODEL}, DIMENSION: {settings.EMBED_DIM}')
        return GenAIEmbeddingService()
    elif settings.EMBEDDER == "mistralai":
        logging.info(
            f'embedding {settings.EMBEDDER} model to be used: {settings.EMBEDDING_MODEL}, DIMENSION: {settings.EMBED_DIM}')
        return MistralAIEmbeddingService()

    raise RuntimeError("unsupported embedder")
