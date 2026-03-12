"""
Persona Registry for Multi-Agent Debate System  v2

Key fix:
  - System prompts no longer hardcode "legal proceeding" with no domain context.
    gpt-5-mini was defaulting to personal-injury medical cases because its most
    common legal training data is personal injury law.

  - get_agent_slots(claim_text) is a new factory function that injects a
    one-line domain hint derived from the claim into every system prompt.
    Example: "NOTE: This proceeding concerns a factual claim about sports/
    history. All discovery and arguments must focus on that domain — NOT on
    medical records, injuries, or clinical data."

  - create_llm_client() unchanged except it now accepts the enriched config
    produced by get_agent_slots().
"""

print("DEBUG: Loading personas.py from " + __file__)

import re

# ─────────────────────────────────────────────────────────────────────
# Base slot definitions (domain-neutral)
# ─────────────────────────────────────────────────────────────────────

_BASE_SLOTS = {
    "proponent": {
        "name": "Plaintiff Counsel",
        "role": "Plaintiff Counsel",
        "llm_provider": "openai",
        "llm_model": "gpt-5-mini",
        "temperature": 0.5,
        "expertise": ["legal advocacy", "evidence presentation", "factual analysis"],
        "system_prompt": (
            "You are the Plaintiff Counsel in a fact-checking legal proceeding. "
            "Your role is to present arguments SUPPORTING the claim, interpret "
            "evidence favourably, challenge opposing arguments, and examine expert "
            "witnesses. Maintain a professional legal advocacy tone. "
            "ALL discovery requests and arguments must be grounded in the specific "
            "domain of the claim — do NOT introduce unrelated topics such as medical "
            "records, injuries, or clinical data unless the claim itself is about medicine."
        )
    },
    "opponent": {
        "name": "Defense Counsel",
        "role": "Defense Counsel",
        "llm_provider": "openrouter",
        "llm_model": "deepseek/deepseek-v3.2",
        "temperature": 0.5,
        "expertise": ["legal defense", "critical analysis", "cross-examination"],
        "system_prompt": (
            "You are the Defense Counsel in a fact-checking legal proceeding. "
            "Your role is to challenge the claim, identify weaknesses in arguments, "
            "contest evidence interpretation, and cross-examine expert witnesses. "
            "Maintain a professional legal defense tone. "
            "ALL discovery requests and arguments must be grounded in the specific "
            "domain of the claim — do NOT introduce unrelated topics such as medical "
            "records, injuries, or clinical data unless the claim itself is about medicine."
        )
    },
    "judge": {
        "name": "The Court",
        "role": "Presiding Judge",
        "llm_provider": "openrouter",
        "llm_model": "qwen/qwen3-235b-a22b-2507",
        "temperature": 0.2,
        "expertise": ["judicial oversight", "evidence synthesis", "legal neutrality"],
        "system_prompt": (
            "You are The Court presiding over a fact-checking legal proceeding. "
            "Your role is to oversee the case, ensure professional conduct from all "
            "counsels, refine discovery queries to be precise and relevant, and "
            "determine when sufficient evidence and expert testimony have been "
            "presented for deliberation. Keep all proceedings focused on the "
            "specific factual claim before the court."
        )
    },
    "expert_slot": {
        "name": "Expert Witness",
        "role": "Expert Witness",
        "expertise": ["domain expert"],
        "system_prompt": (
            "You are a domain expert witness. Provide technical analysis based on "
            "your expertise relevant to the claim being adjudicated."
        ),
        "llm_provider": "openrouter",
        "llm_model": "nousresearch/hermes-3-llama-3.1-405b",
        "temperature": 0.5
    },
    "critic": {
        "name": "Critic Agent",
        "role": "Independent Critic",
        "expertise": ["logical analysis", "factual rigor", "legal argumentation"],
        "system_prompt": (
            "You are the Independent Critic Agent. Your role is to evaluate the "
            "debate rounds for logical coherence, evidence coverage, and rebuttal "
            "quality. Focus your critique on the specific factual domain of the claim."
        ),
        "llm_provider": "openrouter",
        "llm_model": "deepseek/deepseek-r1",
        "temperature": 0.3
    }
}

# ─────────────────────────────────────────────────────────────────────
# Domain detection
# ─────────────────────────────────────────────────────────────────────

def _detect_domain(claim_text: str) -> str:
    """
    Infer a short domain label from the claim text for system-prompt injection.
    Returns a plain-English phrase like "sports / college football history".
    """
    cl = claim_text.lower()

    if any(w in cl for w in ["football", "basketball", "baseball", "soccer",
                               "hockey", "tennis", "rugby", "cricket",
                               "nfl", "nba", "mlb", "nhl", "ncaa", "fifa",
                               "olympic", "championship", "season", "game",
                               "match", "score", "team", "player", "coach"]):
        return "sports / athletics history"

    if any(w in cl for w in ["covid", "vaccine", "clinical trial", "drug",
                               "medication", "hospital", "patient", "disease",
                               "treatment", "medical", "cancer", "virus",
                               "hydroxychloroquine", "fda", "randomized"]):
        return "medical / biomedical science"

    if any(w in cl for w in ["election", "president", "senator", "congress",
                               "parliament", "prime minister", "vote", "policy",
                               "law", "government", "political"]):
        return "politics / government history"

    if any(w in cl for w in ["born", "died", "founded", "established",
                               "published", "invented", "discovered",
                               "written", "directed", "awarded"]):
        return "biography / historical facts"

    if any(w in cl for w in ["film", "movie", "song", "album", "book",
                               "novel", "television", "show", "award",
                               "grammy", "oscar", "emmy"]):
        return "entertainment / media"

    if any(w in cl for w in ["located", "capital", "country", "city",
                               "population", "geography", "river", "mountain"]):
        return "geography / world facts"

    return "general encyclopaedic facts"


def _build_domain_notice(claim_text: str) -> str:
    domain = _detect_domain(claim_text)
    return (
        f"\n\nIMPORTANT — DOMAIN CONTEXT: This proceeding concerns a factual claim "
        f"about {domain}. ALL discovery requests, arguments, and evidence must "
        f"focus on {domain}. Do NOT request or reference medical records, clinical "
        f"data, DICOM files, lab results, or injury documentation unless the claim "
        f"itself explicitly concerns those topics."
    )

# ─────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────

def get_agent_slots(claim_text: str = "") -> dict:
    """
    Return AGENT_SLOTS with system prompts enriched by a domain notice
    derived from claim_text.

    Usage in mad_orchestrator.py:
        from personas import get_agent_slots
        AGENT_SLOTS = get_agent_slots(claim.text)

    If claim_text is empty, returns base slots unchanged (safe fallback).
    """
    import copy
    slots = copy.deepcopy(_BASE_SLOTS)

    if claim_text:
        notice = _build_domain_notice(claim_text)
        for key in slots:
            slots[key]["system_prompt"] += notice

    return slots


# Backward-compatible alias — code that does `from personas import AGENT_SLOTS`
# will get the base slots (no domain notice).  Prefer get_agent_slots(claim.text).
AGENT_SLOTS = _BASE_SLOTS


# ─────────────────────────────────────────────────────────────────────
# Validation helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────

def validate_persona_config(config: dict):
    required = ["name", "role", "expertise", "system_prompt",
                 "llm_provider", "llm_model", "temperature"]
    missing = [f for f in required if f not in config]
    if missing:
        raise ValueError(f"Persona config missing fields: {missing}")


def validate_unique_models(persona_configs: list) -> bool:
    combos = [f"{c['llm_provider']}:{c['llm_model']}" for c in persona_configs]
    if len(combos) != len(set(combos)):
        dupes = [x for x in combos if combos.count(x) > 1]
        raise ValueError(f"Duplicate LLM combinations: {set(dupes)}")
    return True


def create_llm_client(persona_config: dict):
    import os
    from llm_client import GeminiLLMClient
    from openai_client import OpenAILLMClient
    from groq_client import GroqLLMClient

    validate_persona_config(persona_config)

    provider         = persona_config["llm_provider"]
    model            = persona_config["llm_model"]
    system_prompt    = persona_config["system_prompt"]
    temperature      = persona_config["temperature"]
    reasoning_effort = persona_config.get("reasoning_effort")

    if provider == "google":
        return GeminiLLMClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=model, system_prompt=system_prompt, temperature=temperature)
    elif provider == "openai":
        return OpenAILLMClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=model, system_prompt=system_prompt, temperature=temperature)
    elif provider == "groq":
        return GroqLLMClient(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model, system_prompt=system_prompt, temperature=temperature,
            reasoning_effort=reasoning_effort)
    elif provider == "ollama":
        from ollama_client import OllamaLLMClient
        return OllamaLLMClient(
            model_name=model, system_prompt=system_prompt, temperature=temperature)
    elif provider == "openrouter":
        from openrouter_client import OpenRouterLLMClient
        return OpenRouterLLMClient(
            model_name=model, system_prompt=system_prompt, temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")