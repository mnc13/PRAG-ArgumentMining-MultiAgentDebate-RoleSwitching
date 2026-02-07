"""
Grok LLM Client (xAI)

Grok API is OpenAI-compatible, so we can use the OpenAI SDK with a different base URL.
"""

import os
from openai import OpenAI
from llm_client import LLMClient

class GrokLLMClient(LLMClient):
    def __init__(self, api_key: str = None, model_name: str = "grok-4.1-fast", 
                 system_prompt: str = None, temperature: float = 0.7):
        """
        Initialize Grok client
        
        Args:
            api_key: xAI API key (get from console.x.ai)
            model_name: Model to use (grok-4.1-fast, grok-4, grok-3)
            system_prompt: System prompt for persona
            temperature: Sampling temperature
        """
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY not found in environment variables.")
        
        # Grok API is OpenAI-compatible, just use different base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
    
    def generate(self, prompt: str) -> str:
        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1024
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Grok ({self.model_name}): {e}")
            return f"Error generating response: {e}"
    
    @property
    def provider_name(self) -> str:
        return "grok"
