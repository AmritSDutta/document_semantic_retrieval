import asyncio
import logging
from typing import Optional, Sequence, Dict

import asyncpg

from app.config.Settings import get_settings
from app.schema.document_record import DocumentRecord
from app.service.vector_store.vector_store import VectorStore, get_reranker_model, validate_collection_name

logger = logging.getLogger(__name__)


class PGVectorStore(VectorStore):
    def __init__(self):
        settings = get_settings()
        self.conn_str = settings.DB_DSN
        validate_collection_name(settings.COLLECTION_NAME)
        self.collection_name = settings.COLLECTION_NAME
        self.reranker = get_reranker_model()
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> asyncpg.Pool:
        """
        Get or create the connection pool.

        Connection pool settings:
        - min_size: Minimum number of connections to maintain
        - max_size: Maximum number of connections (prevents exhaustion)
        - command_timeout: Query timeout in seconds
        """
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:  # double-check
                    logger.info(f"Creating connection pool for {self.collection_name}")
                    self._pool = await asyncpg.create_pool(
                        dsn=self.conn_str,
                        min_size=5,
                        max_size=20,
                        command_timeout=60,
                        max_queries=50000,
                        max_inactive_connection_lifetime=300.0,
                    )
                    logger.info(f"Connection pool initialized: min=5, max=20")
        return self._pool

    async def close_pool(self):
        """
        Close the connection pool gracefully.
        """
        if self._pool is not None:
            logger.info("Closing PostgresSQL connection pool...")
            await self._pool.close()
            self._pool = None
            logger.info("Connection pool closed")

    async def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        """Standard semantic search using asyncpg."""
        results = []
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                # Convert embedding to string format for pgvector
                if hasattr(query_embedding, "tolist"):
                    emb_str = str(query_embedding.tolist())
                else:
                    emb_str = str(list(query_embedding))

                rows = await conn.fetch(
                    f"""
                    SELECT resume_id, name, category, education, skills, summary, phone, location,overall,
                           (embedding <=> $1::vector) as distance
                    FROM {self.collection_name}
                    ORDER BY distance ASC
                    LIMIT $2;
                    """,
                    emb_str, n_results
                )

                if not rows:
                    return {"results": []}

                # Prepare for reranking
                descriptions = [row["overall"] for row in rows]
                rerank_scores = self.reranker.rerank(query, descriptions)

                for row_data, r_score in zip(rows, rerank_scores):
                    # row_data is a Record object, convert to dict
                    row = dict(row_data)
                    distance = row.pop("distance")
                    row.pop("overall", None)  # Remove large overall text from response

                    dense_score = 1.0 - float(distance)
                    combined_score = dense_score + float(r_score)
                    results.append({
                        "payload": row,
                        "dense_score": dense_score,
                        "rerank_score": float(r_score),
                        "final_score": combined_score
                    })

                results.sort(key=lambda x: x["final_score"], reverse=True)
                return {"results": results}
            except Exception as e:
                logging.error(f"Postgres query error: {e}", exc_info=True)
                raise

    async def hybrid_search(self, query_embedding: Sequence[float],
                            n_results: int = 3, query: str = '') -> list[DocumentRecord]:
        """Hybrid Search using Reciprocal Rank Fusion (RRF)."""
        if hasattr(query_embedding, "tolist"):
            emb_str = str(query_embedding.tolist())
        else:
            emb_str = str(list(query_embedding))

        sql = f"""
        WITH vector_search AS (
            SELECT resume_id, row_number() OVER (ORDER BY embedding <=> $1::vector) as rank
            FROM {self.collection_name}
            ORDER BY embedding <=> $1::vector
            LIMIT $2 * 4
        ),
        text_search AS (
            SELECT resume_id, row_number() OVER (ORDER BY ts_rank(fts_vector, websearch_to_tsquery('english', $3)) DESC) as rank
            FROM {self.collection_name}
            WHERE fts_vector @@ websearch_to_tsquery('english', $3)
            LIMIT $2 * 4
        )
        SELECT 
            r.id, r.resume_id, r.name, r.category, r.education, r.skills, r.summary,r.phone, r.location,
            (COALESCE(1.0 / (60 + vs.rank), 0.0) + COALESCE(1.0 / (60 + ts.rank), 0.0)) as rrf_score
        FROM vector_search vs
        FULL OUTER JOIN text_search ts ON vs.resume_id = ts.resume_id
        JOIN {self.collection_name} r ON r.resume_id = COALESCE(vs.resume_id, ts.resume_id)
        ORDER BY rrf_score DESC
        LIMIT $2;
        """

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                rows = await conn.fetch(sql, emb_str, n_results, query)
                return [DocumentRecord(name=r['name'], resume_id=r['resume_id'], category=r['category'],
                                       education=r['education'], skills=r["skills"], summary=r["summary"],
                                       phone=r["phone"], location=r["location"]) for r in rows]
            except Exception as e:
                logging.error(f"Postgres hybrid search error: {e}", exc_info=True)
                raise

    async def list_collection(self) -> list[str]:
        """Lists available tables in the public schema."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
                return [r["table_name"] for r in rows]
            except Exception as e:
                logging.error(f"Postgres list error: {e}", exc_info=True)
                raise
