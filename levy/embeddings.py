from abc import ABC, abstractmethod
from typing import List
import numpy as np
import random

class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for the given text."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass

class MockEmbeddingClient(EmbeddingClient):
    """Generates random embeddings for testing infrastructure without models."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Use a seed to make "random" embeddings consistent for the same text in this simple mock
        # This is a hacky way to make simple exact replay work somewhat consistently
        # but for semantic cache to work 'correctly' in mock, we might need something smarter.
        # Actually, let's just use random. 
        # For 'semantic' testing with Mock, it won't really work unless we hash the text to a vector.
        pass

    def embed(self, text: str) -> List[float]:
        # deterministically generate a vector from string hash for basic consistency
        random.seed(text)
        vector = [random.uniform(-1, 1) for _ in range(self.dimension)]
        # Normalize
        norm = np.linalg.norm(vector)
        return (vector / norm).tolist()

    def get_dimension(self) -> int:
        return self.dimension

class SentenceTransformerClient(EmbeddingClient):
    """Wrapper around sentence-transformers."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("sentence-transformers is not installed. Please install it with `pip install sentence-transformers`.")

    def embed(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class OllamaEmbeddingClient(EmbeddingClient):
    """Client for local Ollama embeddings."""
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mxbai-embed-large"):
        # Local import to avoid circular dependency issues or just standard practice if keeping clean
        import httpx
        self.httpx = httpx
        self.base_url = base_url
        self.model = model
        self._dimension = None

    def embed(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }
        with self.httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            embedding = data["embedding"]
            if self._dimension is None:
                self._dimension = len(embedding)
            return embedding

    def get_dimension(self) -> int:
        if self._dimension is None:
            # Generate a dummy embedding to know the dimension
            self.embed("test")
        return self._dimension
