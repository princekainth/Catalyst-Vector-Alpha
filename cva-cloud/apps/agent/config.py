import os

CVA_API_URL = os.getenv("CVA_API_URL", "http://localhost:8000")
CVA_CLUSTER_ID = os.getenv("CVA_CLUSTER_ID")
CVA_API_KEY = os.getenv("CVA_API_KEY")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))

# LLM Selection: 'ollama' (default) or 'openai'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# For Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "mistral-nemo:latest")

# For OpenAI (Optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# For Azure OpenAI (Optional)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# For Anthropic (Optional)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

# For Google (Optional)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_BASE_URL = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

# For Mistral (Optional)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_BASE_URL = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")

# For Cohere (Optional)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_BASE_URL = os.getenv("COHERE_BASE_URL", "https://api.cohere.ai/v1")
