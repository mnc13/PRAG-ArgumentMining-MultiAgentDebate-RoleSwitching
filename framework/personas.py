"""
Persona Registry for Multi-Agent Debate System

Defines LLM slots and utilities for dynamic persona assignment.
"""

# Fixed LLM configurations (slots) for debate roles
AGENT_SLOTS = {
    "proponent": {
        "name": "Proponent",
        "role": "Scientific Proponent",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "temperature": 0.5,
        "expertise": ["scientific logic", "clinical analysis"],
        "system_prompt": "You are the Proponent in a clinical debate. Your goal is to argue in favor of the claim using medical evidence and technical reasoning. Maintain a scientific tone."
    },
    "opponent": {
        "name": "Opponent",
        "role": "Scientific Opponent",
        "llm_provider": "groq",
        "llm_model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "temperature": 0.5,
        "expertise": ["critical analysis", "counter-argumentation"],
        "system_prompt": "You are the Opponent in a clinical debate. Your goal is to identify weaknesses in the claim and evidence. Maintain a scientific, critical tone."
    },
    "judge": {
        "name": "Moderator",
        "role": "Scientific Moderator",
        "llm_provider": "groq",
        "llm_model": "llama-3.1-70b-versatile",
        "temperature": 0.2,
        "expertise": ["scientific oversight", "evidence synthesis"],
        "system_prompt": "You are the Moderator (Judge) of a scientific simulation. oversee the debate, ensure logical flow, and determine if sufficient evidence has been presented."
    },
    "critic": {
        "name": "Analyst",
        "role": "Scientific Analyst",
        "llm_provider": "groq",
        "llm_model": "mixtral-8x7b-32768",
        "temperature": 0.4,
        "expertise": ["logical consistency", "clinical methodology"],
        "system_prompt": "You are the Scientific Analyst. Provide neutral, methodical analysis of the arguments and testimony. Focus on technical consistency."
    },
    "expert_slot": {
        "role": "Scientific Expert",
        "llm_provider": "groq",
        "llm_model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "temperature": 0.5
    }
}

def validate_persona_config(config: dict):
    """Validate that a persona configuration has all required fields"""
    required = ["name", "role", "expertise", "system_prompt", "llm_provider", "llm_model", "temperature"]
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Persona configuration missing required fields: {missing}")

def validate_unique_models(persona_configs: list) -> bool:
    """
    Validate that all selected personas use different LLM combinations
    
    Args:
        persona_configs: List of persona configuration dictionaries
        
    Returns:
        True if all combinations are unique, raises ValueError otherwise
    """
    provider_model_combos = [
        f"{config['llm_provider']}:{config['llm_model']}"
        for config in persona_configs
    ]
    
    if len(provider_model_combos) != len(set(provider_model_combos)):
        duplicates = [pm for pm in provider_model_combos if provider_model_combos.count(pm) > 1]
        raise ValueError(f"Duplicate LLM provider:model combinations detected: {set(duplicates)}. Each agent must use a different LLM.")
    
    return True

def create_llm_client(persona_config: dict):
    """
    Factory function to create appropriate LLM client based on provider
    
    Args:
        persona_config: Persona configuration dictionary
        
    Returns:
        LLMClient instance
    """
    import os
    from llm_client import GeminiLLMClient
    from openai_client import OpenAILLMClient
    from groq_client import GroqLLMClient
    
    validate_persona_config(persona_config)
    
    provider = persona_config["llm_provider"]
    model = persona_config["llm_model"]
    system_prompt = persona_config["system_prompt"]
    temperature = persona_config["temperature"]
    
    if provider == "google":
        api_key = os.getenv("GEMINI_API_KEY")
        return GeminiLLMClient(
            api_key=api_key,
            model_name=model,
            system_prompt=system_prompt,
            temperature=temperature
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAILLMClient(
            api_key=api_key,
            model_name=model,
            system_prompt=system_prompt,
            temperature=temperature
        )
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        return GroqLLMClient(
            api_key=api_key,
            model_name=model,
            system_prompt=system_prompt,
            temperature=temperature
        )
    elif provider == "ollama":
        from ollama_client import OllamaLLMClient
        return OllamaLLMClient(
            model_name=model,
            system_prompt=system_prompt,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
