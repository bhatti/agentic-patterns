"""
Ollama client wrapper for local model interactions.
Provides a simple interface for working with Ollama models.
"""

import os
from typing import Optional, List, Dict, Any
from loguru import logger
import httpx
from pydantic import BaseModel, Field


class OllamaResponse(BaseModel):
    """Response model for Ollama API calls."""
    model: str
    response: str
    done: bool
    context: Optional[List[int]] = None


class OllamaClient:
    """
    Client for interacting with Ollama API.
    Handles model communication, streaming, and error management.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
            default_model: Default model to use if not specified
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = default_model or os.getenv("OLLAMA_DEFAULT_MODEL", "llama3")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        
        logger.info(f"Initialized OllamaClient: {self.base_url}, default_model={self.default_model}")
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            model: Model name (uses default if not specified)
            system: System message/instruction
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, top_p, etc.)
            
        Returns:
            Generated text response
        """
        model = model or self.default_model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            if stream:
                # Handle streaming response
                return self._handle_stream(data)
            
            return data.get("response", "")
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Failed to generate text: {e}") from e
    
    def _handle_stream(self, data: Dict[str, Any]) -> str:
        """Handle streaming response (simplified - returns full text)."""
        # For simplicity, this returns the full response
        # In production, you'd want to yield chunks
        return data.get("response", "")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Chat completion using Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            stream: Whether to stream
            **kwargs: Additional parameters
            
        Returns:
            Assistant's response
        """
        model = model or self.default_model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            return data.get("message", {}).get("content", "")
            
        except httpx.HTTPError as e:
            logger.error(f"Ollama chat error: {e}")
            raise RuntimeError(f"Failed to chat: {e}") from e
    
    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except httpx.HTTPError as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def check_health(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()


# Convenience function
def get_ollama_client() -> OllamaClient:
    """Get a configured Ollama client instance."""
    return OllamaClient()

