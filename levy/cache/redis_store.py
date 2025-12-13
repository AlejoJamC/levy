import json
import redis
from typing import Dict, List, Optional
from levy.cache.store import InMemoryStore
from levy.models import CacheEntry

class RedisStore:
    """
    Redis-backed storage for cache entries.
    Implements the same interface as InMemoryStore (duck typing).
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600):
        self.client = redis.from_url(redis_url)
        self.ttl = ttl
        # Note: Generic Redis is not great for vector search iteration without RediSearch.
        # For this prototype, we will stick to Key-Value storage for Exact Cache,
        # and maybe suboptimal methods for Semantic if strictly needed,
        # OR we just implement exact cache persistence here.
        # For the research scope, let's allow fetching all keys for semantic scan (slow but works for small scale).

    def get(self, key: str) -> CacheEntry | None:
        data = self.client.get(key)
        if data:
            try:
                # We assume we store as JSON
                entry_dict = json.loads(data)
                # Reconstruct CacheEntry (basic fields)
                # Note: CacheEntry has methods/defaults that might need care, 
                # but for now we just map the dict back.
                # A proper serialization schema is better, but this is a prototype.
                return CacheEntry(**entry_dict)
            except Exception:
                return None
        return None

    def set(self, key: str, entry: CacheEntry):
        # Serialize to JSON (assuming basics are serializable)
        # We need to handle 'embedding' which is a list of floats (fine),
        # but what about 'metadata'? Fine if JSON serializable.
        
        # We need to act like a dict for `CacheEntry`, so we convert to dict.
        # Check if helper method `asdict` exists or use `__dict__`? 
        # dataclasses.asdict is safer.
        from dataclasses import asdict
        data = asdict(entry)
        
        self.client.set(key, json.dumps(data), ex=self.ttl)

    def delete(self, key: str):
        self.client.delete(key)
    
    def get_all_with_embeddings(self) -> List[CacheEntry]:
        # WARNING: SLOW operations - fetch all keys and values.
        # In a real "Kafka for AI" system, use RediSearch or a VectorDB.
        keys = self.client.keys("*")
        entries = []
        if keys:
            # Batch get
            values = self.client.mget(keys)
            from dataclasses import asdict, is_dataclass
            
            for v in values:
                if v:
                    try:
                        d = json.loads(v)
                        entries.append(CacheEntry(**d))
                    except:
                        pass
        return entries

    def clear(self):
        self.client.flushdb()
