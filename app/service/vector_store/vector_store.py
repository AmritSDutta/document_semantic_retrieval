import re
from typing import Dict, Sequence
from fastembed.rerank.cross_encoder import TextCrossEncoder
from app.schema.document_record import DocumentRecord

_reranker = None


def get_reranker_model() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')
    return _reranker


def validate_collection_name(collection_name: str):
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', collection_name):
        raise ValueError(
            f"Invalid collection name '{collection_name}'. "
            "Must start with letter and contain only letters, numbers, and underscores."
        )


class VectorStore:

    async def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        raise NotImplementedError

    async def list_collection(self) -> list[str]:
        raise NotImplementedError

    async def hybrid_search(self, query_embedding: Sequence[float],
                            n_results: int = 3, query: str = '') -> list[DocumentRecord]:
        raise NotImplementedError

    async def check_collection_exists(self) -> bool:
        raise NotImplementedError
