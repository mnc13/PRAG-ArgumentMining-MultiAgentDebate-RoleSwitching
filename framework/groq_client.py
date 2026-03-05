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
                 system_prompt: str = None, temperature: float = 0.7,
                 reasoning_effort: str = None):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key (get from console.groq.com)
            model_name: Model to use (llama-3.1-8b-instant, llama-3.3-70b-versatile, 
                        qwen/qwen3-32b, openai/gpt-oss-20b)
            system_prompt: System prompt for persona
            temperature: Sampling temperature
            reasoning_effort: Optional reasoning effort for reasoning models (e.g. 'medium')
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare parameters
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "stream": False
            }

            # Handle reasoning_effort if set
            if self.reasoning_effort:
                params["reasoning_effort"] = self.reasoning_effort

            # Handle token limits - prioritize max_completion_tokens (new standard)
            if 'max_completion_tokens' in kwargs:
                params['max_completion_tokens'] = kwargs['max_completion_tokens']
            elif 'max_tokens' in kwargs:
                # Some models might require max_completion_tokens, but default to max_tokens request
                params['max_tokens'] = kwargs['max_tokens']
            else:
                # Default if nothing specified
                params['max_tokens'] = 512

            # Use non-streaming for simplicity
            completion = self.client.chat.completions.create(**params)
            
            if hasattr(completion, 'usage'):
                print(f"   [Token Usage] Input: {completion.usage.prompt_tokens}, Output: {completion.usage.completion_tokens}, Total: {completion.usage.total_tokens}")
            
                # Tracking for PRAG Extensions
                try:
                    from logging_extension import ExtensionState
                    itoks = completion.usage.prompt_tokens
                    otoks = completion.usage.completion_tokens
                    ExtensionState.current_claim_input_tokens += itoks
                    ExtensionState.current_claim_output_tokens += otoks
                    ExtensionState.current_claim_groq_tokens += completion.usage.total_tokens
                    ExtensionState.current_claim_tokens += completion.usage.total_tokens
                except Exception:
                    pass
                
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Groq ({self.model_name}): {e}")
            return f"Error generating response: {e}"
    
    @property
    def provider_name(self) -> str:
        return "groq"
