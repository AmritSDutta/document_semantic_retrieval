from typing import Sequence, List


class EmbeddingService:
    async def embed(self, texts: Sequence[str]) -> List[float]:
        raise NotImplementedError
