import logging
from typing import Optional

import ollama
import requests
from openai import OpenAI

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    ANTHROPIC_VERSION,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    COHERE_API_KEY,
    COHERE_BASE_URL,
    GOOGLE_API_KEY,
    GOOGLE_BASE_URL,
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    MISTRAL_API_KEY,
    MISTRAL_BASE_URL,
    OLLAMA_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)


class OllamaLLMIntegration:
    def __init__(
        self,
        host: str = OLLAMA_URL,
        chat_model: str = LLM_MODEL_NAME,
        logger: Optional[logging.Logger] = None,
    ):
        self.host = host
        self.chat_model = chat_model
        self.logger = logger or logging.getLogger("OllamaLLMIntegration")
        self.provider = LLM_PROVIDER
        self._ollama_client = None
        self._openai_client = None

        if self.provider == "ollama":
            self._ollama_client = ollama.Client(host=host)
        elif self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            self._openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        elif self.provider == "azure_openai":
            if not AZURE_OPENAI_API_KEY:
                raise ValueError("AZURE_OPENAI_API_KEY is required for Azure OpenAI provider")
            if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
                raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT are required")
            base_url = (
                f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
            )
            self._openai_client = OpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=base_url,
                default_query={"api-version": AZURE_OPENAI_API_VERSION},
            )
        elif self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
        elif self.provider == "google":
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is required for Google provider")
        elif self.provider == "mistral":
            if not MISTRAL_API_KEY:
                raise ValueError("MISTRAL_API_KEY is required for Mistral provider")
        elif self.provider == "cohere":
            if not COHERE_API_KEY:
                raise ValueError("COHERE_API_KEY is required for Cohere provider")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 150,
        json_mode: bool = False,
    ) -> str:
        if self.provider == "ollama":
            options = {"temperature": temperature, "num_predict": max_tokens}
            response = self._ollama_client.generate(
                model=self.chat_model, prompt=prompt, options=options
            )
            return (response.get("response") or "").strip()

        if self.provider == "openai":
            response = self._openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()

        if self.provider == "azure_openai":
            response = self._openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()

        if self.provider == "anthropic":
            response = requests.post(
                f"{ANTHROPIC_BASE_URL.rstrip('/')}/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json={
                    "model": self.chat_model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("content", [])
            if content and isinstance(content, list):
                return (content[0].get("text") or "").strip()
            return ""

        if self.provider == "google":
            response = requests.post(
                f"{GOOGLE_BASE_URL.rstrip('/')}/models/{self.chat_model}:generateContent",
                params={"key": GOOGLE_API_KEY},
                json={
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt}]}
                    ]
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return ""
            return (parts[0].get("text") or "").strip()

        if self.provider == "mistral":
            response = requests.post(
                f"{MISTRAL_BASE_URL.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return (choices[0].get("message", {}).get("content") or "").strip()

        if self.provider == "cohere":
            response = requests.post(
                f"{COHERE_BASE_URL.rstrip('/')}/chat",
                headers={
                    "Authorization": f"Bearer {COHERE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.chat_model,
                    "message": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            return (data.get("text") or "").strip()

        raise ValueError(f"Unsupported LLM provider: {self.provider}")
