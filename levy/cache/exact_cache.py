from typing import Optional, List
from levy.cache.base import CacheInterface
from levy.cache.store import InMemoryStore
from levy.models import LLMRequest, CacheEntry
import hashlib

class ExactCache(CacheInterface):
    def __init__(self, store: InMemoryStore):
        self.store = store

    def _get_key(self, prompt: str) -> str:
        # Simple hash of the prompt
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

    def get(self, request: LLMRequest) -> Optional[CacheEntry]:
        key = self._get_key(request.prompt)
        entry = self.store.get(key)
        
        if entry:
            if entry.is_expired():
                self.store.delete(key)
                return None
            entry.access_count += 1
            return entry
        return None

    def set(self, request: LLMRequest, response_text: str, embedding: Optional[List[float]] = None) -> None:
        key = self._get_key(request.prompt)
        entry = CacheEntry(
            key_hash=key,
            prompt=request.prompt,
            response_text=response_text,
            embedding=embedding
            # TTL logic could be added here from config
        )
        self.store.set(key, entry)

    def clear(self) -> None:
        # Note: Clearing the store affects both exact and semantic if they share the store
        pass 
