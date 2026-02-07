"""
Ollama LLM Client for local models (Llama, Qwen, Mistral, etc.)
"""

import os
import requests
import json
from llm_client import LLMClient

class OllamaLLMClient(LLMClient):
    def __init__(self, model_name: str = "llama3.1:8b", 
                 system_prompt: str = None, temperature: float = 0.7,
                 host: str = None):
        """
        Initialize Ollama client
        
        Args:
            model_name: Model to use (llama3.1:8b, qwen2.5:7b, mistral:7b)
            system_prompt: System prompt for persona
            temperature: Sampling temperature
            host: Ollama server URL
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    def generate(self, prompt: str) -> str:
        try:
            # Combine system prompt with user prompt
            full_prompt = f"{self.system_prompt}\n\n{prompt}" if self.system_prompt else prompt
            
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=120  # 2 minute timeout for local models
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                error_msg = f"Ollama error {response.status_code}: {response.text}"
                print(f"Error calling Ollama ({self.model_name}): {error_msg}")
                return f"Error generating response: {error_msg}"
                
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Ollama. Is it running? Run 'ollama serve' in terminal."
            print(f"Error calling Ollama ({self.model_name}): {error_msg}")
            return f"Error generating response: {error_msg}"
        except Exception as e:
            print(f"Error calling Ollama ({self.model_name}): {e}")
            return f"Error generating response: {e}"
    
    @property
    def provider_name(self) -> str:
        return "ollama"
