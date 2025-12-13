from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from levy.models import CacheEntry, LLMRequest

class CacheInterface(ABC):
    """Interface for a cache strategy (exact or semantic)."""
    
    @abstractmethod
    def get(self, request: LLMRequest) -> Optional[CacheEntry]:
        """Retrieve an entry if a match is found."""
        pass

    @abstractmethod
    def set(self, request: LLMRequest, response_text: str, embedding: Optional[List[float]] = None) -> None:
        """Store a new entry."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass
