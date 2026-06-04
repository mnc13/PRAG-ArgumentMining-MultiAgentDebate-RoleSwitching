"""
LLM Client Base Classes

Defines the abstract LLMClient interface that all provider-specific clients
(OpenAI, OpenRouter, Groq, Gemini) must implement.
"""

import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for all LLM provider clients."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt to send.
            **kwargs: Optional parameters (max_tokens, temperature overrides, etc.)

        Returns:
            The model's text response as a string.
        """
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        """Return a short identifier for the provider (e.g. 'openai', 'openrouter')."""
        return "unknown"


class GeminiLLMClient(LLMClient):
    """
    Google Gemini LLM client via the google-generativeai SDK.
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-1.5-flash",
                 system_prompt: str = None, temperature: float = 0.7):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        import google.generativeai as genai
        genai.configure(api_key=self.api_key)

        generation_config = {"temperature": temperature}

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature

        if system_prompt:
            self._model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=system_prompt
            )
        else:
            self._model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )

    def generate(self, prompt: str, **kwargs) -> str:
        max_retries = 10
        base_delay = 2

        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self._model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_exception = e
                print(f"Error calling Gemini ({self.model_name}) - Attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    import time
                    sleep_time = min(base_delay * (2 ** attempt), 60)
                    time.sleep(sleep_time)
                else:
                    return f"Error generating response after {max_retries} attempts: {last_exception}"

        return f"Error generating response: {last_exception}"

    @property
    def provider_name(self) -> str:
        return "google"
