# Setup Guide

## Initial Setup

1. **Install Python 3.10+ with pyenv** (recommended)
   ```bash
   # Install pyenv if not already installed
   # macOS: brew install pyenv
   # Linux: Follow pyenv installation guide
   
   # Install Python 3.11.0
   pyenv install 3.11.0
   
   # Set local Python version
   pyenv local 3.11.0
   
   # Verify
   python --version  # Should be 3.11.0
   ```
   
   **Note**: The project includes a `.python-version` file for automatic pyenv version selection.

2. **Install Ollama**
   - Visit [https://ollama.ai/](https://ollama.ai/)
   - Download and install Ollama for your platform
   - Verify installation: `ollama --version`

3. **Install Project Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull Required Models**
   ```bash
   # Pull a base model (recommended to start)
   ollama pull llama3
   
   # Optional: Pull additional models
   ollama pull qwen2.5
   ollama pull deepseek-r1
   ollama pull mistral
   ```

5. **Verify Ollama is Running**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Or use the health check in code
   python -c "from shared.ollama_client import OllamaClient; print(OllamaClient().check_health())"
   ```

## Environment Configuration

Create a `.env` file in the project root (optional, defaults are provided):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama3
LOG_LEVEL=INFO
```

## Testing Your Setup

Run a simple test to verify everything works:

```bash
python -c "
from shared.ollama_client import OllamaClient
client = OllamaClient()
if client.check_health():
    print('✓ Ollama is running')
    response = client.generate('Say hello in one sentence', model='llama3')
    print(f'✓ Model response: {response[:100]}...')
else:
    print('✗ Ollama is not running')
"
```

## Running Pattern Examples

1. Navigate to a pattern directory:
   ```bash
   cd patterns/pattern-name
   ```

2. Run the example:
   ```bash
   python example.py
   ```

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve` (if not running as a service)
- Check the port: Default is `11434`
- Verify firewall settings if using a remote Ollama instance

### Model Not Found
- Pull the model: `ollama pull model-name`
- List available models: `ollama list`
- Check model name matches exactly (case-sensitive)

### Import Errors
- Ensure you're in the project root or have installed the package
- Check that `shared/` directory is accessible
- Verify Python path includes the project root

### LangChain/CrewAI Issues
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.10+)
- For CrewAI, ensure you have the latest version

