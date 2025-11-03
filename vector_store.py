import os
import faiss
import numpy as np
import pickle

VECTOR_DIM = 1024
INDEX_PATH = "data/faiss_index.bin"
META_PATH = "data/metadata.pkl"

class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = [] 
        if os.path.exists(INDEX_PATH):
            self._load()

    def _load(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.metadata = pickle.load(f)

    def _save(self):
        if self.index:
            faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def _create_index(self):
        self.index = faiss.IndexFlatIP(VECTOR_DIM)

    def add(self, embedding: np.ndarray, meta: dict):
        if self.index is None:
            self._create_index()

        emb = np.array([embedding]).astype("float32")
        self.index.add(emb)
        self.metadata.append(meta)
        self._save()

    def search(self, query_emb: np.ndarray, top_k=10, threshold=0.3):
        if self.index is None or self.index.ntotal == 0:
            return []

        query_emb = np.array([query_emb]).astype("float32")
        scores, ids = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if score >= threshold and idx < len(self.metadata):
                results.append((score, self.metadata[idx]))
        return results

    def delete(self, condition_fn):
        new_metadata = []
        new_embeddings = []
        for idx, meta in enumerate(self.metadata):
            if not condition_fn(meta):
                new_metadata.append(meta)
                emb = self.index.reconstruct(idx)
                new_embeddings.append(emb)
        self._create_index()
        if new_embeddings:
            self.index.add(np.array(new_embeddings).astype("float32"))
        self.metadata = new_metadata
        self._save()


vector_store = VectorStore()
