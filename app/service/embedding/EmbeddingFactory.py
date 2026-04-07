import logging

from app.config.Settings import Settings
from app.service.embedding.MistralAIEmbeddingService import MistralAIEmbeddingService
from app.service.embedding.genai_service import GenAIEmbeddingService
from app.service.utils.resilience import gemini_breaker

logger = logging.getLogger(__name__)


def get_embedding_service():
    settings = Settings()
    
    # Check if primary embedder is genai and if its breaker is OPEN
    if settings.EMBEDDER == "genai":
        if gemini_breaker.current_state == 'open':
            logging.warning("Gemini breaker is OPEN. Rerouting embedding to MistralAI fallback.")
            return MistralAIEmbeddingService()
            
        logging.info(
            f'embedding {settings.EMBEDDER} model to be used: {settings.EMBEDDING_MODEL}, DIMENSION: {settings.EMBED_DIM}')
        return GenAIEmbeddingService()
    elif settings.EMBEDDER == "mistralai":
        logging.info(
            f'embedding {settings.EMBEDDER} model to be used: {settings.EMBEDDING_MODEL}, DIMENSION: {settings.EMBED_DIM}')
        return MistralAIEmbeddingService()

    raise RuntimeError("unsupported embedder")
