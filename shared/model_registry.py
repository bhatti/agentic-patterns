"""
Model registry for managing different model configurations.
Provides easy access to various open-source models.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field
from loguru import logger


class ModelConfig(BaseModel):
    """Configuration for a model."""
    name: str
    ollama_name: str
    description: str
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    recommended_temperature: float = 0.7
    use_cases: list[str] = Field(default_factory=list)
    huggingface_id: Optional[str] = None


class ModelRegistry:
    """
    Registry of available open-source models.
    Provides model configurations and recommendations.
    """
    
    MODELS: Dict[str, ModelConfig] = {
        "llama3": ModelConfig(
            name="llama3",
            ollama_name="llama3",
            description="Meta's Llama 3 - General purpose, instruction-tuned",
            max_tokens=8192,
            context_window=8192,
            recommended_temperature=0.7,
            use_cases=["general", "reasoning", "code", "chat"],
            huggingface_id="meta-llama/Meta-Llama-3-8B-Instruct"
        ),
        "qwen2.5": ModelConfig(
            name="qwen2.5",
            ollama_name="qwen2.5",
            description="Qwen 2.5 - Strong multilingual and reasoning capabilities",
            max_tokens=32768,
            context_window=32768,
            recommended_temperature=0.7,
            use_cases=["multilingual", "reasoning", "math", "code"],
            huggingface_id="Qwen/Qwen2.5-7B-Instruct"
        ),
        "deepseek-r1": ModelConfig(
            name="deepseek-r1",
            ollama_name="deepseek-r1",
            description="DeepSeek R1 - Advanced reasoning and math capabilities",
            max_tokens=32768,
            context_window=32768,
            recommended_temperature=0.6,
            use_cases=["reasoning", "math", "science", "complex_problem_solving"],
            huggingface_id="deepseek-ai/DeepSeek-R1"
        ),
        "mistral": ModelConfig(
            name="mistral",
            ollama_name="mistral",
            description="Mistral 7B - Efficient and capable general-purpose model",
            max_tokens=8192,
            context_window=8192,
            recommended_temperature=0.7,
            use_cases=["general", "chat", "instruction_following"],
            huggingface_id="mistralai/Mistral-7B-Instruct-v0.2"
        ),
        "phi3": ModelConfig(
            name="phi3",
            ollama_name="phi3",
            description="Microsoft Phi-3 - Small but capable model",
            max_tokens=4096,
            context_window=4096,
            recommended_temperature=0.7,
            use_cases=["general", "code", "lightweight"],
            huggingface_id="microsoft/Phi-3-mini-4k-instruct"
        ),
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a model."""
        return cls.MODELS.get(model_name)
    
    @classmethod
    def list_models(cls) -> list[str]:
        """List all available model names."""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_models_by_use_case(cls, use_case: str) -> list[ModelConfig]:
        """Get models suitable for a specific use case."""
        return [
            config for config in cls.MODELS.values()
            if use_case in config.use_cases
        ]
    
    @classmethod
    def get_recommended_model(cls, use_case: str) -> Optional[ModelConfig]:
        """Get a recommended model for a use case."""
        models = cls.get_models_by_use_case(use_case)
        return models[0] if models else None


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Convenience function to get model configuration."""
    return ModelRegistry.get_model_config(model_name)

