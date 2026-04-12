import logging
from typing import Sequence, Dict

from pandas import DataFrame
from pymilvus import MilvusClient, Function, FunctionType, AnnSearchRequest

from app.config.Settings import get_settings
from app.schema.document_record import DocumentRecord
from app.service.vector_store.vector_store import VectorStore, get_reranker_model, validate_collection_name

logger = logging.getLogger(__name__)


class MilvusStore(VectorStore):

    def __init__(self):
        settings = get_settings()
        validate_collection_name(settings.COLLECTION_NAME)
        self.collection_name = settings.COLLECTION_NAME

        self.client = MilvusClient(
            uri=settings.MILVUS_URI,
            token=settings.MILVUS_TOKEN,
            timeout=30,
            secure=True
        )
        logging.info(f"Connected to DB: {settings.MILVUS_URI} successfully")
        self.reranker = get_reranker_model()

    async def create(self):
        raise NotImplementedError()

    async def save(self, data: DataFrame):
        raise NotImplementedError()

    async def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        try:
            # 1. Search in Milvus
            search_res = self.client.search(
                collection_name=self.collection_name,
                data=[list(query_embedding)],
                limit=n_results,
                anns_field='vector',
                output_fields=["ResumeID", "Name", "Category", "Education", "Skills", "Summary", "doc"]
            )

            hits = search_res[0]
            if not hits:
                return {"results": []}

            descriptions = [hit["entity"].get("doc", "") for hit in hits]

            # 2. Rerank
            rerank_scores = self.reranker.rerank(query, descriptions)

            # 3. Combine
            combined = []
            for hit, rerank_score in zip(hits, rerank_scores):
                dense_score = hit["distance"]
                combined_score = float(dense_score) + float(rerank_score)
                combined.append({
                    "payload": hit["entity"],
                    "dense_score": float(dense_score),
                    "rerank_score": float(rerank_score),
                    "final_score": combined_score
                })

            combined.sort(key=lambda x: x["final_score"], reverse=True)
            return {"results": combined}
        except Exception as e:
            logging.error(f"Milvus persistence error: {e}", exc_info=True)
            raise

    async def list_collection(self) -> list[str]:
        try:
            return self.client.list_collections()
        except Exception as e:
            logging.error(f"Milvus list collections error: {e}", exc_info=True)
            raise

    async def hybrid_search(self, query_embedding: Sequence[float],
                            n_results: int = 3, query: str = '') -> list[DocumentRecord]:
        try:
            logging.info(f"Hybrid search with: {query}")
            # text semantic search (dense)
            search_param_1 = {
                "data": [query_embedding],
                "anns_field": "vector",
                "param": {"nprobe": 10},
                "limit": n_results
            }
            request_1 = AnnSearchRequest(**search_param_1)

            # full-text search (sparse)
            search_param_2 = {
                "data": [query],
                "anns_field": "sparse",
                "param": {"nprobe": 10},
                "limit": n_results
            }
            request_2 = AnnSearchRequest(**search_param_2)
            reqs = [request_1, request_2]

            ranker = Function(
                name="rrf",
                input_field_names=[],  # Must be an empty list
                function_type=FunctionType.RERANK,
                params={
                    "reranker": "rrf",
                    "k": 100  # Optional
                }
            )
            res = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=reqs,
                ranker=ranker,
                output_fields=["ResumeID", "Name", "Category", "Education", "Skills", "Summary", "doc", "Phone",
                               "Location"],
                limit=n_results
            )

            # Transformation Logic
            formatted_results = []
            for hit in res[0]:
                formatted_results.append(
                    DocumentRecord(
                        name=hit.entity.get("Name"),
                        summary=hit.entity.get("Summary"),
                        resume_id=hit.entity.get("ResumeID"),
                        category=hit.entity.get("Category"),
                        education=hit.entity.get("Education"),
                        skills=hit.entity.get("Skills"),
                        phone=hit.entity.get("Phone"),
                        location=hit.entity.get("Location")
                    ))
            return formatted_results
        except Exception as e:
            logging.error(f"Milvus persistence error: {e}", exc_info=True)
            raise

    async def close(self):
        self.client.close()