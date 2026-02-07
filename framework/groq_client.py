"""
Groq LLM Client

Groq provides fast inference for open-source models (Llama, Qwen, Mixtral, etc.)
and also supports GPT models via OpenAI compatibility
"""

import os
from groq import Groq
from llm_client import LLMClient

class GroqLLMClient(LLMClient):
    def __init__(self, api_key: str = None, model_name: str = "llama-3.1-8b-instant", 
                 system_prompt: str = None, temperature: float = 0.7):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key (get from console.groq.com)
            model_name: Model to use (llama-3.1-8b-instant, llama-3.3-70b-versatile, 
                        qwen/qwen3-32b, openai/gpt-oss-20b)
            system_prompt: System prompt for persona
            temperature: Sampling temperature
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
    
    def generate(self, prompt: str) -> str:
        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Use non-streaming for simplicity
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048,  # Increased for longer responses
                stream=False
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq ({self.model_name}): {e}")
            return f"Error generating response: {e}"
    
    @property
    def provider_name(self) -> str:
        return "groq"
