import json
import logging
from typing import List

from langcache import LangCache
from langcache.models import SearchStrategy, SearchResponse

from app.config.Settings import get_settings
from app.schema.document_record import DocumentRecord
from app.service.cache.RedisSemanticCahe import RedisSemanticCache
from app.service.cache.sematic_cache import SemanticCache

logger = logging.getLogger(__name__)


def get_cache_impl() -> SemanticCache:
    if get_settings().LANGCACHE_ENABLED:
        return RedisSemanticCache()
    return NoOpCache()


class NoOpCache(SemanticCache):
    def __init__(self):
        logging.info("NoOpCache initialized")

    async def save(self, prompt: str, response: List[DocumentRecord]):
        return

    async def retrieve(self, prompt: str) -> List[DocumentRecord]:
        return []
