"""
Persona Registry for Multi-Agent Debate System

Defines LLM slots and utilities for dynamic persona assignment.
"""

# Fixed LLM configurations (slots) for debate roles
print("DEBUG: Loading personas.py from " + __file__)
AGENT_SLOTS = {
    "proponent": {
        "name": "Plaintiff Counsel",
        "role": "Plaintiff Counsel",
        "llm_provider": "openai",
        "llm_model": "gpt-5-mini",
        "temperature": 0.5,
        "expertise": ["legal advocacy", "evidence presentation", "clinical analysis"],
        "system_prompt": "You are the Plaintiff Counsel in a legal proceeding. Your role is to present arguments supporting the claim, interpret evidence favorably, challenge opposing arguments, and conduct examination of expert witnesses. Maintain a professional legal advocacy tone."
    },
    "opponent": {
        "name": "Defense Counsel",
        "role": "Defense Counsel",
        "llm_provider": "openrouter",
        "llm_model": "deepseek/deepseek-v3.2",
        "temperature": 0.5,
        "expertise": ["legal defense", "critical analysis", "cross-examination"],
        "system_prompt": "You are the Defense Counsel in a legal proceeding. Your role is to challenge the claim, identify weaknesses in arguments, contest evidence interpretation, and cross-examine expert witnesses. Maintain a professional legal defense tone."
    },
    "judge": {
        "name": "The Court",
        "role": "Presiding Judge",
        "llm_provider": "openrouter",
        "llm_model": "qwen/qwen3-235b-a22b-2507",
        "temperature": 0.2,
        "expertise": ["judicial oversight", "evidence synthesis", "legal neutrality"],
        "system_prompt": "You are The Court presiding over a legal proceeding. Your role is to oversee the case, ensure professional conduct from all counsels, and determine when sufficient evidence and expert testimony have been presented for deliberation."
    },
    "expert_slot": {
        "name": "Expert Witness",
        "role": "Expert Witness",
        "expertise": ["scientific domain"],
        "system_prompt": "You are a scientific expert witness. Provide technical analysis based on your expertise.",
        "llm_provider": "openrouter",
        "llm_model": "nousresearch/hermes-3-llama-3.1-405b",
        "temperature": 0.5
    },
    "critic": {
        "name": "Critic Agent",
        "role": "Independent Critic",
        "expertise": ["logical analysis", "scientific rigor", "legal argumentation"],
        "system_prompt": "You are the Independent Critic Agent. Your role is to evaluate the debate rounds for logical coherence, evidence coverage, and rebuttal quality.",
        "llm_provider": "openrouter",
        "llm_model": "deepseek/deepseek-r1",
        "temperature": 0.3
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
    reasoning_effort = persona_config.get("reasoning_effort")
    
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
            temperature=temperature,
            reasoning_effort=reasoning_effort
        )
    elif provider == "openrouter":
        from openrouter_client import OpenRouterLLMClient
        return OpenRouterLLMClient(
            model_name=model,
            system_prompt=system_prompt,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
