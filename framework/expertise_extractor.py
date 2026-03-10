from typing import List, Dict
import os
import json
from personas import AGENT_SLOTS
from openrouter_client import OpenRouterLLMClient

def extract_required_expertise(claim_text: str, premises: List[str]) -> List[Dict]:
    """
    Analyze claim and premises to identify and generate required expert personas
    
    Args:
        claim_text: The claim to analyze
        premises: Decomposed premises
        
    Returns:
        List of dynamic persona configurations (Role, Expertise, Name, System Prompt)
    """
    # Use the same provider family as the rest of the framework (OpenRouter).
    # Model choice mirrors the high-capacity judge/expert models.
    api_key = os.getenv("OPENROUTER_API_KEY")
    llm = OpenRouterLLMClient(
        api_key=api_key,
        model_name="nousresearch/hermes-3-llama-3.1-405b",
        system_prompt=None,
        temperature=0.7,
    )
    
    premises_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(premises)])
    
    prompt = f"""Analyze this claim and brainstorm 3 distinct scientific expert personas needed to evaluate it.
One should be a generalist for the domain, and two should be specialists.

Claim: {claim_text}

Decomposed Premises:
{premises_text}

For each expert, provide:
1. role: Their title (e.g., Clinical Virologist)
2. expertise: list of 3-4 specific domains (e.g., ["viral pathogenesis", "mRNA vaccines"])
3. name: A professional name (e.g., Dr. Jane Smith)
4. system_prompt: A tailored instruction for this persona. It must include:
   - Who they are and their specific focus.
   - How they should evaluate claims (e.g., "Analyze through the lens of molecular biology...").
   - Their professional tone and reasoning style.

Return ONLY a JSON array of these 3 objects.
Example: 
[
  {{
    "role": "Epidemiologist",
    "name": "Dr. Sarah Chen",
    "expertise": ["disease transmission", "public health"],
    "system_prompt": "You are Dr. Sarah Chen..."
  }},
  ...
]

Your response:"""
    
    response = llm.generate(prompt)
    
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end != 0:
            expertise_configs = json.loads(response[start:end])
        else:
            raise ValueError("No JSON array found")
    except:
        # Fallback default expertise
        expertise_configs = [
            {
                "role": "General Medical Scientist",
                "name": "Dr. Alex Taylor",
                "expertise": ["public health", "general medicine"],
                "system_prompt": f"You are Dr. Alex Taylor, a medical scientist evaluating the claim: {claim_text}"
            },
            {
                "role": "Data Scientist",
                "name": "Dr. Jordan Lee",
                "expertise": ["statistical analysis", "data interpretation"],
                "system_prompt": "You are Dr. Jordan Lee, focusing on the data and statistical significance of the claim."
            }
        ]
    
    # Always ensure we have enough for a debate (proponent, opponent, expert)
    # We will also add a critic
    critic_config = {
        "role": "Critical Analyst",
        "name": "Dr. Casey Morgan",
        "expertise": ["logic", "scientific reasoning", "argumentation"],
        "system_prompt": "You are Dr. Casey Morgan, a critical analyst. Your role is to identify logical fallacies, gaps in evidence, and ensure arguments are robust."
    }
    
    if len(expertise_configs) < 3:
        expertise_configs.append(critic_config)
    else:
        # Replace or add critic as the 4th if needed, or just append
        expertise_configs.append(critic_config)

    return expertise_configs

def assign_personas(expertise_configs: List[Dict]) -> List[Dict]:
    """
    Assign dynamic personas to LLM slots
    
    Args:
        expertise_configs: List of generated persona configurations
        
    Returns:
        List of complete persona configurations ready for MAD initialization
    """
    assigned_configs = []
    slot_keys = list(AGENT_SLOTS.keys())
    
    for i, config in enumerate(expertise_configs):
        if i >= len(slot_keys):
            break
            
        slot = AGENT_SLOTS[slot_keys[i]]
        # Merge dynamic data with slot's LLM config
        full_config = config.copy()
        full_config.update(slot)
        full_config["persona_key"] = f"dynamic_{i+1}"
def extract_single_expert(expert_type: str, claim_text: str) -> Dict:
    """
    Generate a single targeted expert persona configuration
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    llm = OpenRouterLLMClient(
        api_key=api_key,
        model_name="nousresearch/hermes-3-llama-3.1-405b",
        system_prompt=None,
        temperature=0.7,
    )
    
    prompt = f"""Generate a scientific expert persona profile.
Expertise Type Requested: {expert_type}
Topic: {claim_text}

Provide:
1. role: Specific scientific title (e.g., Clinical Virologist)
2. expertise: list of 3-4 specific scientific domains
3. system_prompt: A tailored instruction for this persona. 
   Instructions: Provide neutral, evidence-based, and highly technical scientific testimony. Use clinical language. Avoid all legal or non-clinical formatting.

Return ONLY the JSON object.
"""
    response = llm.generate(prompt)
    try:
        import json
        import re
        match = re.search(r'\{[^}]+\}', response)
        config = json.loads(match.group())
        config["name"] = config.get("role", expert_type)
        return config
    except:
        return {
            "name": expert_type,
            "role": expert_type,
            "expertise": ["general science"],
            "system_prompt": f"You are a scientific expert in {expert_type}. Provide technical analysis."
        }
