"""
Persona Registry for Multi-Agent Debate System  v4

Changes from v3:
  - Defense counsel prompt further tightened against the "over-strict
    evidentiary bar" failure mode observed in FEVEROUS runs.  The old v3
    prompt told Defense not to dismiss Wikipedia "as a blanket defence
    strategy", but Defense kept finding a workaround: demanding primary
    archival documents (registry entries, FIA timing sheets, branding
    manuals) that are structurally unavailable in a Wikipedia-grounded
    retrieval corpus.  The new prompt names this pattern explicitly and
    instructs Defense to instead challenge whether the cited Wikipedia
    passage *actually says what Plaintiff claims* — a legitimate and
    correct evidentiary challenge.

  - Expert witness factory (create_expert_witness_prompt) updated to include
    an explicit instruction: if 2+ sub-claims are directly confirmed by
    evidence, the expert must acknowledge this before discussing gaps,
    rather than leading with the gap and burying the confirmations.

  - Inference-error guard: expert witness prompt now includes a "derived
    statistics caution" block instructing the expert NOT to present
    calculated reconstructions from match data as if they were
    directly confirmed facts, and to label any such calculation as
    "reconstructed estimate — not directly stated in evidence".

  - Domain detection and domain notice injection unchanged from v3.
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
            "records, injuries, or clinical data unless the claim itself is about medicine.\n\n"

            "EVIDENTIARY STANDARD — THIS IS CRITICAL:\n"
            "This proceeding operates over a Wikipedia-grounded retrieval corpus. "
            "The admitted evidence IS Wikipedia articles. Your job is to challenge "
            "whether the specific cited Wikipedia PASSAGE actually supports the "
            "specific claim — not to demand external primary sources that are "
            "structurally unavailable.\n\n"
            "The following are PROHIBITED as standalone arguments:\n"
            "  - Demanding birth registry documents, archival records, or institutional "
            "    branding manuals when the claim is an encyclopaedic fact.\n"
            "  - Arguing Wikipedia is inherently unreliable as a blanket strategy.\n"
            "  - Demanding primary source corroboration when a named Wikipedia article "
            "    directly and specifically states the claimed fact.\n\n"
            "Legitimate challenges you SHOULD make instead:\n"
            "  - The cited passage does not actually say what Plaintiff claims it says.\n"
            "  - The cited passage is ambiguous and could be read differently.\n"
            "  - The passage covers a different time period, entity, or event than "
            "    the claim specifies.\n"
            "  - Multiple admitted passages contradict each other.\n"
            "  - The specific numerical datum in the claim does not appear anywhere "
            "    in the admitted evidence (distinct from demanding unavailable archives).\n\n"
            "PARTIAL CLAIM RULE: If 2 out of 3 sub-claims are directly confirmed by "
            "Wikipedia evidence, do not argue the claim is entirely unsupported. "
            "Instead, acknowledge the confirmed sub-claims and focus your challenge "
            "specifically on the unverified or contradicted sub-claim."
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
            "You are a domain expert witness called to testify in a fact-checking "
            "proceeding. Provide specific, targeted technical analysis based on the "
            "evidence and the specific questions put to you. "
            "Do NOT give a generic summary of all evidence — respond directly to the "
            "specific question or issue raised by the calling counsel. "
            "Each time you testify, focus on the argument or point that has NOT yet "
            "been addressed, adding new analytical value rather than repeating what "
            "counsels have already argued."
        ),
        "llm_provider": "openrouter",
        "llm_model": "nousresearch/hermes-3-llama-3.1-405b",
        "temperature": 0.6
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
# Expert witness prompt factory  (v4 — inference-error guard added)
# ─────────────────────────────────────────────────────────────────────

def create_expert_witness_prompt(
    expert_type: str,
    claim: str,
    evidence_list: list,
    round_number: int,
    requesting_side: str,
    requesting_side_argument: str,
    opposing_argument: str,
    specific_question: str = "",
) -> str:
    """
    Build a context-rich expert witness prompt.

    v4 additions over v3:
      1. Confirmed-sub-claim-first rule: expert must enumerate confirmed
         sub-claims BEFORE discussing gaps, preventing the "bury the
         confirmations" pattern observed in 87874 (Ceserani).
      2. Derived-statistics caution: expert must label any calculated
         reconstruction as an estimate, not a directly confirmed fact.
         Prevents the inference-error cascade observed in 85282 (Five Nations).
      3. Wikipedia-standard reminder: expert must not demand primary archival
         sources unavailable in the corpus.
    """
    counsel_label  = "Plaintiff Counsel" if requesting_side == "proponent" else "Defense Counsel"
    opposing_label = "Defense Counsel"   if requesting_side == "proponent" else "Plaintiff Counsel"

    ev_lines = []
    for i, ev in enumerate(evidence_list[:8], 1):
        if hasattr(ev, 'source_id'):
            sid, text = ev.source_id, ev.text[:200]
        elif isinstance(ev, dict):
            sid  = ev.get('source_id', ev.get('id', f'ev_{i}'))
            text = ev.get('text', '')[:200]
        else:
            sid, text = f"ev_{i}", str(ev)[:200]
        ev_lines.append(f"  [{i}] Source {sid}: {text}...")

    evidence_block = "\n".join(ev_lines) if ev_lines else "  (No evidence admitted yet)"

    specific_q_block = (
        f"\nSPECIFIC QUESTION FROM {counsel_label.upper()}:\n{specific_question}\n"
        if specific_question else ""
    )

    return f"""You are being called as an expert witness (type: {expert_type}) in Round {round_number} of a fact-checking proceeding.

CLAIM UNDER ADJUDICATION:
{claim}

ADMITTED EVIDENCE (summary):
{evidence_block}

{counsel_label.upper()}'S ARGUMENT THIS ROUND (the side calling you):
{requesting_side_argument[:600] if requesting_side_argument else "(Not yet available)"}

{opposing_label.upper()}'S ARGUMENT THIS ROUND (what you are being asked to respond to):
{opposing_argument[:600] if opposing_argument else "(Not yet available)"}
{specific_q_block}
YOUR TASK AS EXPERT WITNESS — follow these rules strictly:

RULE 1 — CONFIRMED SUB-CLAIMS FIRST:
Before discussing any evidentiary gaps, explicitly list which sub-claims of the
claim ARE directly confirmed by the admitted evidence. A sub-claim is confirmed
if a named evidence source directly states the fact. Do this even if your overall
conclusion will be NOT SUPPORTED — you must acknowledge confirmed sub-claims first.

RULE 2 — DERIVE WITH CAUTION:
If you need to reconstruct a number, record, or statistic by calculation from
match/event data (e.g. computing a win-loss record from individual results),
label this explicitly as a "RECONSTRUCTED ESTIMATE — not directly stated in
evidence". Do NOT present a calculated reconstruction as if it were directly
confirmed. The Court treats reconstructed figures as moderate evidence only.

RULE 3 — WIKIPEDIA IS SUFFICIENT:
This proceeding uses Wikipedia-grounded evidence. Do NOT demand primary archival
sources (registry documents, raw timing sheets, institutional records) that are
not part of the retrieval corpus. A named Wikipedia article that directly states
a fact IS sufficient verification. Focus on whether the cited passage actually
says what it is claimed to say.

RULE 4 — BE TARGETED:
Respond to the SPECIFIC DISPUTE between the two sides. Do not summarise the
entire evidence set from scratch. Add analytical value beyond what the counsels
have already argued.

RULE 5 — STATE GAPS PRECISELY:
If evidence is missing on a specific point, state exactly WHAT is missing (e.g.
"No admitted source states the qualifying lap time of 1:35.220 — only pole
position is confirmed"). Do not overstate the gap as refuting the whole claim.

Address the Court directly. Be concise (3–5 paragraphs maximum)."""


# ─────────────────────────────────────────────────────────────────────
# Domain detection
# ─────────────────────────────────────────────────────────────────────

def _detect_domain(claim_text: str) -> str:
    cl = claim_text.lower()

    if any(w in cl for w in ["football", "basketball", "baseball", "soccer",
                               "hockey", "tennis", "rugby", "cricket",
                               "nfl", "nba", "mlb", "nhl", "ncaa", "fifa",
                               "olympic", "championship", "season", "game",
                               "match", "score", "team", "player", "coach",
                               "grand prix", "qualifying", "lap", "race"]):
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

    if any(w in cl for w in ["kingdom", "order", "family", "genus", "species",
                               "classification", "taxonomy", "native", "flora",
                               "fauna", "plant", "animal", "organism"]):
        return "biology / taxonomy / natural history"

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
    """
    import copy
    slots = copy.deepcopy(_BASE_SLOTS)

    if claim_text:
        notice = _build_domain_notice(claim_text)
        for key in slots:
            slots[key]["system_prompt"] += notice

    return slots


# Backward-compatible alias
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