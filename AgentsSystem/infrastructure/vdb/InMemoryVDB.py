from typing import List
from langchain_core.documents import Document
from ...domain.interfaces.IVectorDB import IVectorDB

class InMemoryVDB(IVectorDB):
    """Simple in-memory vector DB for testing only."""

    def __init__(self):
        self._docs: List[Document] = []        

    def add_documents(self, path: str, base_metadata) -> None:
        #self._docs.extend(docs)
        pass

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        return self._docs[:top_k]

    def embed_text(self, text: str, base_metadata) -> List[float]:
        return [len(text)]
