from typing import Sequence, List


class EmbeddingService:
    def embed(self, texts: Sequence[str]) -> List[float]:
        raise NotImplementedError
