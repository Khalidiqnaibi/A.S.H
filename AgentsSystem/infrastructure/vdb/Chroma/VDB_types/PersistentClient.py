import chromadb
from chromadb.config import Settings

class PersistanceClient:
    def __init__(self, path: str, settings: Settings = Settings()):
        self._vdb = chromadb.PersistentClient(
            path=path,
            settings = settings
        )
    def get_vdb_client(self):
        return self._vdb