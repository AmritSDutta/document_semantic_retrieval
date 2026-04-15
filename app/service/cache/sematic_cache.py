from typing import List

from app.schema.document_record import DocumentRecord


class SemanticCache:
    async def save(self, prompt: str, response: List[DocumentRecord]):
        raise NotImplementedError()

    async def retrieve(self, prompt: str) -> List[DocumentRecord]:
        raise NotImplementedError()
