from abc import ABC, abstractmethod
from typing import Any, Dict
import time
import httpx
from levy.models import LLMRequest, LLMResponse

class LLMClient(ABC):
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        pass

class MockLLMClient(LLMClient):
    """A mock client that echoes the prompt (reversed) for testing."""
    def generate(self, request: LLMRequest) -> LLMResponse:
        # Simulate network latency
        time.sleep(0.5) 
        response_text = f"Computed response for: {request.prompt[::-1]}" # Reverse string as 'computation'
        # Simple token estimation
        tokens = len(response_text.split())
        return LLMResponse(
            text=response_text,
            token_usage=tokens,
            model="mock-v1"
        )

class OpenAILLMClient(LLMClient):
    """Minimal OpenAI client using httpx."""
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def generate(self, request: LLMRequest) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        
        # Merge extra params
        payload.update(request.extra_params)

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            
            return LLMResponse(
                text=content,
                token_usage=tokens,
                model=self.model,
                metadata=data
            )
