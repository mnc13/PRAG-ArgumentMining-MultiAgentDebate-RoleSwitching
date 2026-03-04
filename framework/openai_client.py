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
    
    def generate(self, prompt: str, **kwargs) -> str:
        max_retries = 15
        base_delay = 2  # seconds
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Extract max_tokens if provided, default to 1024
                max_tokens = kwargs.get('max_tokens', 1024)
                
                # Handle gpt-5-mini using the new Responses API
                if "gpt-5" in self.model_name:
                    full_input = f"{self.system_prompt}\n\n{prompt}" if self.system_prompt else prompt
                    response = self.client.responses.create(
                        model=self.model_name,
                        input=full_input,
                        timeout=120
                    )
                    return response.output_text

                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "timeout": 120
                }
                
                # Use max_completion_tokens for newer reasoning models
                if "o1" in self.model_name or "o3" in self.model_name:
                    params["max_completion_tokens"] = max_tokens
                else:
                    params["max_tokens"] = max_tokens

                response = self.client.chat.completions.create(**params)
                
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    if content:
                        return content
                    else:
                        raise ValueError(f"Empty content in OpenAI response. Raw: {response}")
                else:
                    raise ValueError(f"No choices in OpenAI response. Raw: {response}")

            except Exception as e:
                last_exception = e
                print(f"Error calling OpenAI ({self.model_name}) - Attempt {attempt + 1}/{max_retries}: {e}")
                
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    sleep_time = min(sleep_time, 60)
                    import time
                    time.sleep(sleep_time)
                else:
                    return f"Error generating response after {max_retries} attempts: {last_exception}"
        
        return f"Error generating response: {last_exception}"
    
    @property
    def provider_name(self) -> str:
        return "openai"
