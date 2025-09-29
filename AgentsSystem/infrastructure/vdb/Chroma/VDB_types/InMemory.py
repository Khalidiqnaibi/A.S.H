import chromadb

class InMemoryChroma:
    def __init__(self):
        self._vdb = chromadb.Client()
    def get_vdb_client(self):
        return self._vdb
