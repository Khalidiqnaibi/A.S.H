from typing import List
from abc import ABC, abstractmethod

class IEmbedder(ABC):
    """Abstract interface for any embedding model."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError()

    @abstractmethod
    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError()