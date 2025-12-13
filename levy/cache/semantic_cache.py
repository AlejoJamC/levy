from typing import Optional, List
import numpy as np
from levy.cache.base import CacheInterface
from levy.cache.store import InMemoryStore
from levy.models import LLMRequest, CacheEntry
from levy.embeddings import EmbeddingClient

class SemanticCache(CacheInterface):
    def __init__(self, store: InMemoryStore, embedding_client: EmbeddingClient, threshold: float = 0.85):
        self.store = store
        self.embedding_client = embedding_client
        self.threshold = threshold

    def get(self, request: LLMRequest) -> Optional[CacheEntry]:
        # 1. Embed the query
        query_embedding = self.embedding_client.embed(request.prompt)
        
        # 2. Search in store
        candidates = self.store.get_all_with_embeddings()
        if not candidates:
            return None

        # 3. Compute cosine similarity
        # For prototype, we iterate. In prod, use FAISS/VectorDB.
        best_score = -1.0
        best_entry = None

        q_vec = np.array(query_embedding)
        norm_q = np.linalg.norm(q_vec)

        for entry in candidates:
            if entry.embedding is None:
                continue
            if entry.is_expired():
                # Lazy cleanup could happen here, but skipping for now
                continue

            doc_vec = np.array(entry.embedding)
            norm_doc = np.linalg.norm(doc_vec)
            
            if norm_q == 0 or norm_doc == 0:
                score = 0.0
            else:
                score = np.dot(q_vec, doc_vec) / (norm_q * norm_doc)

            if score > best_score:
                best_score = score
                best_entry = entry

        # 4. Check threshold
        if best_entry and best_score >= self.threshold:
            best_entry.access_count += 1
            # We can attach the score to metadata for transparency
            best_entry.metadata['last_similarity_score'] = float(best_score)
            return best_entry
            
        return None

    def set(self, request: LLMRequest, response_text: str, embedding: Optional[List[float]] = None) -> None:
        # Semantic cache usually relies on the SAME store as exact cache foundationally,
        # but adds the embedding vector.
        # If the entry comes from ExactCache, it might already be in store.
        # Here we explicitly ensure the embedding is present.
        
        if embedding is None:
            embedding = self.embedding_client.embed(request.prompt)
            
        # We reuse the logic from exact cache keying or create a new ID.
        # For simplicity, let's just use the exact cache key mechanism so they play nice together.
        import hashlib
        key = hashlib.sha256(request.prompt.encode('utf-8')).hexdigest()
        
        entry = CacheEntry(
            key_hash=key,
            prompt=request.prompt,
            response_text=response_text,
            embedding=embedding
        )
        self.store.set(key, entry)

    def clear(self) -> None:
        pass
