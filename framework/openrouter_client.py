"""
OpenRouter LLM Client
"""

import os
import json
import requests
from llm_client import LLMClient

class OpenRouterLLMClient(LLMClient):
    def __init__(self, api_key: str = None, model_name: str = "deepseek/deepseek-r1", 
                 system_prompt: str = None, temperature: float = 0.7,
                 site_url: str = "", site_name: str = ""):
        """
        Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key
            model_name: Model to use
            system_prompt: System prompt for persona
            temperature: Sampling temperature
            site_url: Optional site URL for OpenRouter rankings
            site_name: Optional site name for OpenRouter rankings
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
        
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.site_url = site_url
        self.site_name = site_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt: str, **kwargs) -> str:
        max_retries = 15
        base_delay = 2  # seconds
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                
                messages.append({"role": "user", "content": prompt})
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                
                if self.site_url:
                    headers["HTTP-Referer"] = self.site_url
                if self.site_name:
                    headers["X-Title"] = self.site_name

                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature
                }

                # Handle explicit reasoning parameter (for deepseek-v3.2 etc)
                if kwargs.get('reasoning_enabled', False) or "deepseek" in self.model_name.lower():
                     data["reasoning"] = {"enabled": True}

                if 'max_completion_tokens' in kwargs:
                    data['max_completion_tokens'] = kwargs['max_completion_tokens']
                elif 'max_tokens' in kwargs:
                    data['max_tokens'] = kwargs['max_tokens']

                response = requests.post(
                    url=self.api_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=120  # 2 minute timeout per request
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract content
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message'].get('content')
                    if content:
                        return content
                    else:
                        raise ValueError(f"Empty content in response. Raw: {result}")
                else:
                    raise ValueError(f"No choices in response. Raw: {result}")

            except Exception as e:
                last_exception = e
                print(f"Error calling OpenRouter ({self.model_name}) - Attempt {attempt + 1}/{max_retries}: {e}")
                
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    # Cap the sleep time to something reasonable (e.g. 60s)
                    sleep_time = min(sleep_time, 60)
                    import time
                    time.sleep(sleep_time)
                else:
                    return f"Error generating response after {max_retries} attempts: {last_exception}"
        
        return f"Error generating response: {last_exception}"
    
    @property
    def provider_name(self) -> str:
        return "openrouter"
