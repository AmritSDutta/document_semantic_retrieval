import asyncio
import logging
from typing import List, Dict

from fastapi import APIRouter, Depends, Body, Security, BackgroundTasks
from fastapi_limiter.depends import RateLimiter
from pyrate_limiter import Limiter, Rate, Duration

from app.config.arize_confg import tracer
from app.database.document_repository import DocumentRepository
from app.routers.request_validator import sanitize_passage, do_moderation_checking, do_moderation_checking_mistral
from app.schema.document_record import DocumentRecord, SearchRequest, ClassificationResult
from app.service.document_service import DocumentService
from app.service.embedding.EmbeddingFactory import get_query_embedding_async
from app.service.llm_classifier import ClassifyLLMService
from app.service.utils.auth import get_api_key
from app.service.utils.pii_redaction import PII_Redactor
from app.service.utils.time_helper import time_coro

logger = logging.getLogger(__name__)
doc_router = APIRouter(prefix="/docs", tags=["docs"], dependencies=[Security(get_api_key)])
llm = ClassifyLLMService()
pii_redactor = PII_Redactor()


def get_document_service() -> DocumentService:
    logger.info('inside get_document_service')
    return DocumentService(DocumentRepository())


@doc_router.post("/search", status_code=200, response_model=List[DocumentRecord],
                 dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(5, Duration.SECOND * 1))))])
@tracer.chain
async def search_docs(req: SearchRequest, svc: DocumentService = Depends(get_document_service)) -> List[DocumentRecord]:
    """
    Retrieve items by category with an optional limit.
    """
    logger.info(f'### Received query -> {req.search_term.strip()[:25]} ..., limit -> {req.limit}')

    passage_redacted, query_emb = await _do_sanitization_moderation_redaction_embedding(req.search_term.strip())

    logger.info(f'Redacted req: query -> {passage_redacted[0][:25]} . . .')
    docs: List[DocumentRecord] = await svc.get_matching_docs_by_embedding(passage_redacted[0], query_emb, req.limit)
    return docs


@doc_router.post("/classify", status_code=200, response_model=ClassificationResult,
                 dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(1, Duration.SECOND * 1))))])
async def classify_doc(passage: str = Body(..., embed=True, max_length=5000)) -> ClassificationResult:
    logger.info(f'received req: passage to be classified -> {passage[:100]} ....')

    passage_redacted: List[str] = await _do_sanitization_moderation_redaction(passage)

    logger.info(f'Redacted passage to be classified -> {passage_redacted[0][:100]} ....')
    docs = await llm.llm_classify_request(passage_redacted[0])
    return ClassificationResult(result=docs.sorted_result)


@doc_router.post("/search_requirement", status_code=200, response_model=List[DocumentRecord],
                 dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(1, Duration.SECOND * 5))))])
async def classify_and_search(
        passage: str = Body(..., embed=True, max_length=5000),
        svc: DocumentService = Depends(get_document_service), limit: int = 3) -> List[DocumentRecord]:
    logger.info(f'received req: passage to be classified -> {passage[:100]} ....')

    passage_redacted: List[str] = await _do_sanitization_moderation_redaction(passage)

    logger.info(f'Redacted passage to be classified -> {passage_redacted[0][:100]} ....')
    classification_result: ClassificationResult = await llm.llm_classify_request(passage_redacted[0])
    derived_topic: str = classification_result.derive_relevant_topic().strip()
    docs: List[DocumentRecord] = await svc.get_matching_docs(derived_topic, limit)
    return docs


@doc_router.post("/train_classic_ml", status_code=200, response_model=str,
                 dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(1, Duration.MINUTE * 15))))])
async def train_classic_ml_model(background_tasks: BackgroundTasks) -> str:
    """
    Retrieve items by category with an optional limit.
    """
    from app.service.classic_ml.berttopic_modeling import train_topic_model
    logger.info(f'training started')
    background_tasks.add_task(train_topic_model)
    return 'classic ml training task submitted'


@doc_router.post("/search_through_classic_ml", status_code=200, response_model=List[DocumentRecord],
                 dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(1, Duration.SECOND * 5))))])
async def classify_and_search_classic_ml(
        passage: str = Body(..., embed=True, max_length=5000),
        svc: DocumentService = Depends(get_document_service), limit: int = 3) -> List[DocumentRecord]:
    from app.service.classic_ml.berttopic_modeling import infer_topic_model
    logger.info(f'received req: passage to be classified through classic ml-> {passage[:100]} ....')

    passage_redacted: List[str] = await _do_sanitization_moderation_redaction(passage)
    logger.info(f'Redacted passage to be classified through classic ml-> {passage_redacted[0][:100]} ....')
    result: List[Dict[str, str]] = infer_topic_model(passage_redacted[0])
    derived_topic = ','.join(set(r['topic_name'] for r in result))
    docs: List[DocumentRecord] = await svc.get_matching_docs(derived_topic, limit)
    return docs


async def _do_sanitization_moderation_redaction(passage: str) -> List[str]:
    # 1. Sanitize first (dependency for next steps)
    passage = await sanitize_passage(passage)

    # 2. Run moderation and redaction concurrently
    _, passage_redacted = await asyncio.gather(
        do_moderation_checking_mistral(passage),
        pii_redactor.do_pii_redaction_text([passage])
    )

    return passage_redacted


async def _process_chain(passage: str):
    # Chain PII and Embedding so they run while Moderation is fetching
    redacted = await pii_redactor.do_pii_redaction_text([passage])
    emb = await time_coro('embedding', get_query_embedding_async(redacted[0]))
    return redacted, emb


async def _do_sanitization_moderation_redaction_embedding(passage: str):
    passage = await sanitize_passage(passage)

    # Run all three logical tracks concurrently
    # The total time is now max(Moderation, PII + Embedding)
    results = await asyncio.gather(
        time_coro('moderation', do_moderation_checking_mistral(passage)),
        _process_chain(passage)
    )

    moderation_result, (passage_redacted, query_emb) = results
    return passage_redacted, query_emb
