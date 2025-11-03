import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    _instance = None

    def __init__(self):
        if EmbeddingModel._instance is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[EmbeddingModel] Используется устройство: {self.device}")
            self.model = SentenceTransformer(
                "intfloat/multilingual-e5-large",
                device=self.device
            )
            EmbeddingModel._instance = self
        else:
            self.model = EmbeddingModel._instance.model
            self.device = EmbeddingModel._instance.device

    def get_embedding(self, text: str, is_query: bool = False) -> bytes | None:
        if not text.strip():
            return None

        prefix = "query: " if is_query else "passage: "
        text = prefix + text.strip()

        with torch.no_grad():
            embedding = self.model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        return embedding.cpu().numpy().astype(np.float32).tobytes()

    @staticmethod
    def blob_to_numpy(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)
