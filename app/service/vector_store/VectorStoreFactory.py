import logging
from enum import Enum
from typing import Optional

from app.config.Settings import get_settings
from app.service.vector_store.milvus_vector_store import MilvusStore
from app.service.vector_store.postgres_vector_store import PGVectorStore
from app.service.vector_store.qdrant_vector_store import QdrantStore

logger = logging.getLogger(__name__)
_qdrantStore: QdrantStore | None = None
_milvusStore: MilvusStore | None = None
_pgStore: PGVectorStore | None = None


class DatabaseType(Enum):
    QDRANT = "qdrant"
    MILVUS = "milvus"
    POSTGRES = "postgres"


def get_vector_store(db_type: Optional[DatabaseType] = None):
    global _qdrantStore, _milvusStore, _pgStore
    settings = get_settings()
    logging.info(f'collection name to be used: {settings.COLLECTION_NAME},'
                 f' and vector store will be used: {db_type if db_type else settings.VECTOR_STORE}')

    effective_db_type = db_type if db_type else DatabaseType(settings.VECTOR_STORE)

    if effective_db_type == DatabaseType.QDRANT:
        if _qdrantStore is None:
            _qdrantStore = QdrantStore()
        return _qdrantStore
    elif effective_db_type == DatabaseType.MILVUS:
        if _milvusStore is None:
            _milvusStore = MilvusStore()
        return _milvusStore
    elif effective_db_type == DatabaseType.POSTGRES:
        if _pgStore is None:
            _pgStore = PGVectorStore()
        return _pgStore

    raise RuntimeError("unsupported vector store")


async def close_all_vector_stores():
    """
    Close all vector store connections gracefully.
    Should be called during application shutdown to prevent connection leaks.
    """
    global _qdrantStore, _milvusStore, _pgStore

    stores_to_close = []

    if _qdrantStore is not None:
        stores_to_close.append(("Qdrant", _qdrantStore))
    if _milvusStore is not None:
        stores_to_close.append(("Milvus", _milvusStore))
    if _pgStore is not None:
        stores_to_close.append(("PostgreSQL", _pgStore))

    for name, store in stores_to_close:
        try:
            if hasattr(store, 'close_pool'):
                await store.close_pool()
                logger.info(f"Closed {name} vector store pool")
            elif hasattr(store, 'close'):
                # For stores that have a different close method
                await store.close()
                logger.info(f"Closed {name} vector store")
        except Exception as e:
            logger.error(f"Error closing {name} vector store: {e}")

    # Reset globals
    _qdrantStore = None
    _milvusStore = None
    _pgStore = None
