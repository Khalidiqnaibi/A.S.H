from typing import List
from langchain_core.documents import Document
from domain.interfaces.IRetriever import IRetriever
from domain.interfaces.IVectorDB import IVectorDB

class VDBRetriever(IRetriever):
    """Retriever that uses any injected Vector DB."""

    def __init__(self, vdb: IVectorDB):
        self._vdb = vdb

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        return self._vdb.search(query, top_k=top_k)
