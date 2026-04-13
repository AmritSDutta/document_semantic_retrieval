import logging
from typing import Sequence, Any

import qdrant_client
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client.http.models import CollectionsResponse

from app.config.Settings import get_settings
from app.schema.document_record import DocumentRecord
from app.service.vector_store.vector_store import VectorStore, get_reranker_model, validate_collection_name

logger = logging.getLogger(__name__)


class QdrantStore(VectorStore):

    def __init__(self):
        self._late_interaction_embedding_model = None
        self._bm25_model = None
        settings = get_settings()
        validate_collection_name(settings.COLLECTION_NAME)
        self.collection_name = settings.COLLECTION_NAME

        self.qdrant_client = qdrant_client.AsyncQdrantClient(
            url=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )
        self.reranker = get_reranker_model()

    @property
    def bm25_embedding_model(self):
        if self._bm25_model is None:
            self._bm25_model = SparseTextEmbedding("Qdrant/bm25", threads=2)
        return self._bm25_model

    @property
    def late_interaction_embedding_model(self):
        if self._late_interaction_embedding_model is None:
            self._late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0", threads=4)
        return self._late_interaction_embedding_model

    async def query(self, query_embedding: Sequence[float],
                    n_results: int = 3, query: str = '') -> dict[str, list[Any]]:
        try:
            # 1. dense ranking
            hits = await self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=list(query_embedding),
                using="genai",
                limit=n_results).points

            if not hits:
                return {"results": []}

            # Extract descriptions for reranking
            descriptions = [hit.payload.get("doc", "") for hit in hits]

            # 2. Rerank using cross-encoder / LLM reranker
            rerank_scores = self.reranker.rerank(query, descriptions)  # list[float]

            # 3. Combine dense score + rerank score
            combined = []
            for hit, rerank_score in zip(hits, rerank_scores):
                combined_score = float(hit.score) + float(rerank_score)
                combined.append({
                    "payload": hit.payload,
                    "dense_score": float(hit.score),
                    "rerank_score": float(rerank_score),
                    "final_score": combined_score
                })

            # 4. Sort by final score descending
            combined.sort(key=lambda x: x["final_score"], reverse=True)
            logging.info(f'{len(combined)} combined results')
            return {"results": combined}
        except Exception as e:
            logging.error(f"Error while querying qdrant : {e}", exc_info=True)
            raise

    async def list_collection(self) -> list[str]:
        try:
            logging.info('listing collection')
            response: CollectionsResponse = await self.qdrant_client.get_collections()
            logging.info(f'returned collections: {response.collections}')
            existing_collections_list = [collection.name for collection in response.collections]
            logging.info(f'found {len(existing_collections_list)} collections')
            return existing_collections_list
        except Exception as e:
            logging.error(f"Error while listing  qdrant collections: {e}", exc_info=True)
            raise

    async def hybrid_search(self, query_embedding: Sequence[float],
                            n_results: int = 3, query: str = '') -> list[DocumentRecord]:
        try:
            from qdrant_client.http import models

            # 1. Generate query components locally using FastEmbed
            sparse_query = next(self.bm25_embedding_model.query_embed(query))
            colbert_query = next(self.late_interaction_embedding_model.query_embed(query))

            # 2. Single-call Multi-stage Search: (Dense + Sparse) -> RRF -> ColBERT Rerank
            response = await self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        prefetch=[
                            models.Prefetch(query=query_embedding, using="genai", limit=n_results * 10),
                            models.Prefetch(
                                query=models.SparseVector(
                                    indices=sparse_query.indices.tolist(),
                                    values=sparse_query.values.tolist()
                                ),
                                using="bm25",
                                limit=n_results * 10
                            ),
                        ],
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                        limit=n_results * 10  # This pool is sent to ColBERT reranking
                    )
                ],
                query=colbert_query.tolist(),  # The "Reranker"
                using="colbert",
                limit=n_results
            )
            logging.info(f'query results: {len(response.points)}')
            return [
                DocumentRecord.model_validate({
                    "resume_id": p.payload.get("ResumeID"),
                    "name": p.payload.get("Name"),
                    "education": p.payload.get("Education"),
                    "category": p.payload.get("Category"),
                    "skills": p.payload.get("Skills"),
                    "summary": p.payload.get("Summary"),
                    "phone": p.payload.get("Phone"),
                    "location": p.payload.get("Location")
                }) for p in response.points
            ]
        except Exception as e:
            logging.error(f"Error while querying qdrant : {e}", exc_info=True)
            raise

    async def close(self):
        await self.qdrant_client.close()
