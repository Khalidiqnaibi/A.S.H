from typing import List
from langchain.schema import Document
from abc import ABC, abstractmethod

class IRetriever(ABC):
    """Abstract interface for retrieval from a knowledge base."""
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        raise NotImplementedError()