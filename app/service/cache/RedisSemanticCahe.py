import json
import logging
from typing import List

from langcache import LangCache
from langcache.models import SearchStrategy, SearchResponse

from app.config.Settings import get_settings
from app.schema.document_record import DocumentRecord
from app.service.cache.sematic_cache import SemanticCache
logger = logging.getLogger(__name__)


class RedisSemanticCache(SemanticCache):
    def __init__(self):
        self.cache: LangCache = LangCache(
            server_url=get_settings().LANGCACHE_URL,
            cache_id=get_settings().LANGCACHE_ID,
            api_key=get_settings().LANGCACHE_API_KEY,
        )
        logging.info("Redis LANGCACHE initialized")

    async def save(self, prompt: str, response: List[DocumentRecord]):
        # SDK requires 'response' to be a str; serialize Pydantic models to JSON
        json_response = json.dumps([rec.model_dump() for rec in response])
        async with self.cache as lang_cache:
            lang_cache.set(
                prompt=prompt,
                response=json_response,
                ttl_millis=get_settings().LANGCACHE_TTL,
            )
            logging.info(f"Saved entry response with prompt: {prompt}")

    async def retrieve(self, prompt: str) -> List[DocumentRecord]:
        async with self.cache as lang_cache:
            search_response: SearchResponse = lang_cache.search(
                prompt=prompt,
                similarity_threshold=0.9,
                search_strategies=[SearchStrategy.SEMANTIC, SearchStrategy.EXACT],
            )

            if not search_response.data:
                logging.info(f"No cache match for: {prompt}")
                return []

            # Extract JSON string from the first match and deserialize
            try:
                cached_str = search_response.data[0].response
                data_list = json.loads(cached_str)
                return [DocumentRecord(**item) for item in data_list]
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logging.error(f"Error decoding cache for {prompt}: {e}")
                raise
