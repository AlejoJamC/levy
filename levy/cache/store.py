import threading
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
        # LEV-19-class bug: set()'s check-then-evict (len() then next(iter(...)))
        # is not atomic, and FastAPI dispatches concurrent requests to a
        # threadpool -- a concurrent insert/delete during that iter() can raise
        # "dictionary changed size during iteration". Guards every mutation.
        self._lock = threading.Lock()

    def get(self, key: str) -> CacheEntry | None:
        return self.entries.get(key)

    def set(self, key: str, entry: CacheEntry):
        with self._lock:
            if len(self.entries) >= self.max_size:
                # Simple eviction: remove oldest (FIFO-ish based on iteration order or random)
                # Python 3.7+ dicts preserve insertion order, so this removes the first inserted
                first_key = next(iter(self.entries))
                self._delete_locked(first_key)

            self.entries[key] = entry
            if entry.embedding is not None:
                 self.vector_index.append(entry)

    def delete(self, key: str):
        with self._lock:
            self._delete_locked(key)

    def _delete_locked(self, key: str):
        """Caller must hold self._lock."""
        if key in self.entries:
            entry = self.entries.pop(key)
            if entry in self.vector_index:
                self.vector_index.remove(entry)

    def get_all_with_embeddings(self) -> List[CacheEntry]:
        with self._lock:
            return list(self.vector_index)

    def clear(self):
        with self._lock:
            self.entries.clear()
            self.vector_index.clear()
