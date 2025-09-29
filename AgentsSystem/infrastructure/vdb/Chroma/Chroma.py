from typing import List, Optional, Dict, Any
from langchain.schema import Document
from ....domain.interfaces.IVectorDB import IVectorDB
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, FastEmbedEmbeddings
from ...LangChain_Documents_Converter.LangChainToDocumentConverter import LangChainToDocument
import requests
import time

from .VDB_types.InMemory import InMemoryChroma
from .VDB_types.HttpChromaClient import HttpChromaClient
from .VDB_types.PersistentClient import PersistanceClient
from chromadb.config import Settings


class ChromaVDB(IVectorDB):
    def __init__(self,
                 embedding = FastEmbedEmbeddings(),
                 meta_api_url: str = "http://10.0.0.137:2020"):
        self._embedding = embedding
        self.meta_api_url = meta_api_url
        self._metadata_map = self.load_metadata_map()

    def init_persistence_db(self, path: str, settings: Settings = Settings()):
        self._vdb = Chroma(
            client=PersistanceClient(path, settings=settings).get_vdb_client(),
            embedding_function=self._embedding
        )

    def init_memory_db(self):
        self._vdb = Chroma(
            client=InMemoryChroma().get_vdb_client(),
            embedding_function=self._embedding
        )

    def init_cloud_db_client(self,
                             headers: Dict[str, str] = {},
                             host: str = "10.0.0.137",
                             port: int = 2000,
                             ssl: bool = False,
                             tanent: str = "",
                             database: str = "",
                             settings: Settings = Settings()
                             ):
        self._vdb = Chroma(
            client=HttpChromaClient(
                host=host,
                port=port,
                ssl=ssl,
                headers=headers,
                settings=settings
            ).get_vdb_client(),
            embedding_function=self._embedding,
        )

    def get_metadata_map(self):
        return self._metadata_map

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        return self._vdb.similarity_search(query=query, search_type="similarity", top_k=top_k)

    def load_db(self, path: str):
        self._vdb = Chroma(persist_directory=path)

    def embed_text(self, text: str, base_metadata):
        doc = LangChainToDocument.from_text(text)
        self._vdb.add_documents(doc)

    def add_documents(self, path: str, base_metadata: Dict[str, str]):
        documents = LangChainToDocument.from_path(path=path, base_metadata=base_metadata)
        self._vdb.add_documents(documents)

    def get_store(self):
        return self._vdb

    def use_vdb_client(self, vdb):
        self._vdb = vdb

    def save_metadata_map(self, metadata: Dict[str, Any]):
        try:
            res = requests.post(f"{self.meta_api_url}/metadata/save", json={
                "key": "default_metadata",
                "value": metadata
            })
            res.raise_for_status()
        except Exception as e:
            print("Failed to save metadata_map:", e)

    def load_metadata_map(self) -> Dict[str, Any]:
        try:
            res = requests.get(f"{self.meta_api_url}/metadata/default_metadata")
            if res.status_code == 200:
                return res.json()["value"]
            else:
                return {"documents": {}, "created_at": time.time()}
        except Exception as e:
            print("Failed to load metadata_map, starting fresh:", e)
            return {"documents": {}, "created_at": time.time()}
