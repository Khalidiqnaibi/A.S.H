from typing import List
from langchain.schema import Document
from abc import ABC , abstractmethod
class IVectorDB(ABC):
    @abstractmethod
    def init_persistence_db(self, path: str):
        raise NotImplementedError()
    @abstractmethod
    def init_cloud_db_client(self,
                            headers,
                            host,
                            port,
                            ssl,
                            tanent,
                            database,
                            settings
                            ):
        raise NotImplementedError
    @abstractmethod
    def init_memory_db(self):
        raise NotImplementedError()
    @abstractmethod
    def load_db(self, path: str):
        raise NotImplementedError()
    @abstractmethod
    def add_documents(self, path: str, base_metadata):
        raise NotImplementedError()
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        raise NotImplementedError()
    @abstractmethod
    def embed_text(self, text: str, base_metadata):
        raise NotImplementedError()
    @abstractmethod
    def get_store(self):
        raise NotImplementedError()
    @abstractmethod
    def get_metadata_map(self):
        raise NotImplementedError()
    @abstractmethod
    def use_vdb_client(self, vdb):
        raise NotImplementedError()