from typing import Dict, List
from levy.models import CacheEntry

class InMemoryStore:
    """
    Simple in-memory storage for cache entries.
    In a real system, this would be Redis or VectorDB.
    """
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.entries: Dict[str, CacheEntry] = {}
        # Simple list for vector search iteration 
        # (not efficient for large scale, but fine for prototype)
        self.vector_index: List[CacheEntry] = []

    def get(self, key: str) -> CacheEntry | None:
        return self.entries.get(key)

    def set(self, key: str, entry: CacheEntry):
        if len(self.entries) >= self.max_size:
            # Simple eviction: remove oldest (FIFO-ish based on iteration order or random)
            # Python 3.7+ dicts preserve insertion order, so this removes the first inserted
            first_key = next(iter(self.entries))
            self.delete(first_key)
        
        self.entries[key] = entry
        if entry.embedding is not None:
             self.vector_index.append(entry)

    def delete(self, key: str):
        if key in self.entries:
            entry = self.entries.pop(key)
            if entry in self.vector_index:
                self.vector_index.remove(entry)
    
    def get_all_with_embeddings(self) -> List[CacheEntry]:
        return self.vector_index

    def clear(self):
        self.entries.clear()
        self.vector_index.clear()
