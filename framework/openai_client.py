"""
OpenAI LLM Client for GPT models
"""

import os
from openai import OpenAI
from llm_client import LLMClient

class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini", 
                 system_prompt: str = None, temperature: float = 0.7):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key
            model_name: Model to use (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
            system_prompt: System prompt for persona
            temperature: Sampling temperature
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(api_key=self.api_key)
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
            print(f"Error calling OpenAI ({self.model_name}): {e}")
            return f"Error generating response: {e}"
    
    @property
    def provider_name(self) -> str:
        return "openai"
