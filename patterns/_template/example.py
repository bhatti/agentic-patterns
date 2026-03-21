"""
Pattern Name - Example Implementation

This example demonstrates [pattern description] using local models via Ollama.
"""

import sys
from pathlib import Path

# Add parent directory to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.ollama_client import OllamaClient
from shared.model_registry import get_model_config
from loguru import logger


def main():
    """Main example function."""
    logger.info("Starting Pattern Name example...")
    
    # Initialize Ollama client
    client = OllamaClient()
    
    # Check if Ollama is running
    if not client.check_health():
        logger.error("Ollama is not running. Please start Ollama first.")
        return
    
    # Get model configuration
    model_config = get_model_config("llama3")
    if not model_config:
        logger.warning("Model config not found, using default")
        model = "llama3"
    else:
        model = model_config.ollama_name
        logger.info(f"Using model: {model_config.description}")
    
    # Example implementation
    try:
        # TODO: Implement pattern-specific logic here
        prompt = "Your prompt here"
        
        response = client.generate(
            prompt=prompt,
            model=model,
            temperature=0.7
        )
        
        logger.info(f"Response: {response}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()

