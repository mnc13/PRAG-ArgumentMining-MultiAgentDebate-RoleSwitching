import os
import google.generativeai as genai
from dotenv import load_dotenv
from abc import ABC, abstractmethod

# Load environment variables from .env file if present
load_dotenv()

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class MockLLMClient(LLMClient):
    def generate(self, prompt: str) -> str:
        if "decompose" in prompt.lower():
            return "1. First premise of the claim.\n2. Second premise of the claim."
        elif "negotiate" in prompt.lower() or "select" in prompt.lower():
            return "Based on relevance, I select Evidence 1 and Evidence 2."
        else:
            return "Mock response from LLM."

class GeminiLLMClient(LLMClient):
    def __init__(self, api_key: str = None, model_name: str = 'gemini-2.0-flash', system_prompt: str = None, temperature: float = 0.7):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.system_prompt = system_prompt
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        try:
            # Prepend system prompt if provided
            full_prompt = f"{self.system_prompt}\n\n{prompt}" if self.system_prompt else prompt
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature
                )
            )
            return response.text
        except Exception as e:
            print(f"Error calling Gemini ({self.model_name}): {e}")
            return f"Error generating response: {e}"
