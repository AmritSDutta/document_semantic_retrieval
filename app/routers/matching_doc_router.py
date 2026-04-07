import logging
from typing import List, Dict

from fastapi import APIRouter, Depends, HTTPException, Body

from app.database.document_repository import DocumentRepository
from app.database.vector_db import VectorDb, get_db
from app.routers.request_validator import sanitize_passage, do_moderation_checking
from app.schema.document_record import DocumentRecord, SearchRequest, ClassificationResult
from app.service.document_service import DocumentService
from app.service.llm_classifier import ClassifyLLMService

logger = logging.getLogger(__name__)
doc_router = APIRouter(prefix="/docs", tags=["docs"])
llm = ClassifyLLMService()


def get_document_service(db: VectorDb = Depends(get_db)) -> DocumentService:
    logger.info('inside get_document_service')
    return DocumentService(DocumentRepository(db))


@doc_router.post("/search", status_code=200, response_model=List[DocumentRecord])
async def search_docs(req: SearchRequest, svc: DocumentService = Depends(get_document_service)) -> List[DocumentRecord]:
    """
    Retrieve items by category with an optional limit.
    """
    logger.info(f'received req: query -> {req.search_term.strip()}, limit -> {req.limit}')
    docs: List[DocumentRecord] = await svc.get_matching_docs(req.search_term.strip(), req.limit)
    return docs


@doc_router.post("/classify", status_code=200, response_model=ClassificationResult)
async def classify_doc(passage: str = Body(..., embed=True, max_length=5000)) -> ClassificationResult:
    passage: str = sanitize_passage(passage)
    do_moderation_checking(passage)
    logger.info(f'received req: passage to be classified -> {passage[:100]} ....')
    try:
        docs = await llm.llmClassifyRequest(passage)
    except Exception as e:
        msg = str(e)
        # specific Gemini error error provide custom message
        if any(x in msg for x in ("UNAVAILABLE", "503", "429")):
            logger.warning(f"Upstream Gemini error: {msg}")
            raise HTTPException(
                status_code=502,
                detail="Upstream LLM temporarily unavailable or rate-limited. Please retry later.",
            )
        # all other exceptions — let them bubble normally
        logger.exception("Unexpected classification failure")
        raise
    return ClassificationResult(result=docs.sorted_result)


@doc_router.post("/search_requirement", status_code=200, response_model=List[DocumentRecord])
async def classify_doc(passage: str = Body(..., embed=True, max_length=5000),
                       svc: DocumentService = Depends(get_document_service), limit: int = 3) -> List[DocumentRecord]:
    passage: str = sanitize_passage(passage)
    do_moderation_checking(passage)
    logger.info(f'received req: passage to be classified -> {passage[:100]} ....')
    try:
        classificationResult: ClassificationResult = await llm.llmClassifyRequest(passage)
        derived_topic: str = classificationResult.derive_relevant_topic.strip()
        docs: List[DocumentRecord] = await svc.get_matching_docs(derived_topic, limit)
    except Exception as e:
        msg = str(e)
        # specific Gemini error error provide custom message
        if any(x in msg for x in ("UNAVAILABLE", "503", "429")):
            logger.warning(f"Upstream Gemini error: {msg}")
            raise HTTPException(
                status_code=502,
                detail="Upstream LLM temporarily unavailable or rate-limited. Please retry later.",
            )
        # all other exceptions — let them bubble normally
        logger.exception("Unexpected classification failure")
        raise
    return docs


@doc_router.post("/train_classic_ml", status_code=200, response_model=Dict[str, str])
async def search_docs() -> Dict[str, str]:
    """
    Retrieve items by category with an optional limit.
    """
    from app.service.classic_ml.berttopic_modeling import train_topic_model
    logger.info(f'training started')
    return train_topic_model()


@doc_router.post("/search_through_classic_ml", status_code=200, response_model=List[DocumentRecord])
async def classify_doc(passage: str = Body(..., embed=True, max_length=5000),
                       svc: DocumentService = Depends(get_document_service), limit: int = 3) -> List[DocumentRecord]:
    passage: str = sanitize_passage(passage)
    do_moderation_checking(passage)
    logger.info(f'received req: passage to be classified through classic ml-> {passage[:100]} ....')
    try:
        from app.service.classic_ml.berttopic_modeling import infer_topic_model
        result: List[Dict[str, str]] = infer_topic_model(passage)
        derived_topic = ','.join(set(r['topic_name'] for r in result))
        docs: List[DocumentRecord] = await svc.get_matching_docs(derived_topic, limit)
    except Exception as e:
        msg = str(e)
        # specific Gemini error error provide custom message
        if any(x in msg for x in ("UNAVAILABLE", "503", "429")):
            logger.warning(f"Upstream Gemini error: {msg}")
            raise HTTPException(
                status_code=502,
                detail="Upstream LLM temporarily unavailable or rate-limited. Please retry later.",
            )
        # all other exceptions — let them bubble normally
        logger.exception("Unexpected classification failure")
        raise
    return docs
