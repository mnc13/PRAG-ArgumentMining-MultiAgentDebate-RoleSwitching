"""
mad_orchestrator_expert_patch.py
─────────────────────────────────
Drop-in patch for the expert witness generation logic in mad_orchestrator.py.

PROBLEM
───────
The expert witness (nousresearch/hermes-3-llama-3.1-405b) was producing
identical testimony in every round because:
  1. The testimony prompt had no round identifier, so the model saw the same
     context and produced the same output each time.
  2. The max_tokens for hermes was set to ~313, truncating responses and
     causing the model to repeat its opening paragraph in subsequent calls.
  3. No "do not repeat" instruction was given, so the model recycled its
     first-round answer verbatim.

FIX
───
Replace the expert witness testimony generation call in your MADOrchestrator
with the `generate_expert_testimony` function below.

HOW TO APPLY
────────────
In mad_orchestrator.py, find the function/method that:
  (a) builds the expert witness prompt, and
  (b) calls the expert LLM client

Replace it with a call to `generate_expert_testimony(...)` from this module.

Typical location in mad_orchestrator.py:
    # --- Expert Witness generation ---
    expert_description = expert_llm.generate(persona_prompt)
    expert_testimony   = expert_llm.generate(testimony_prompt)

Replace with:
    from mad_orchestrator_expert_patch import generate_expert_testimony
    expert_description, expert_testimony = generate_expert_testimony(
        expert_llm       = expert_llm,
        claim            = claim_text,
        expert_type      = requested_expert_type,
        requesting_side  = requesting_side,       # "proponent" or "opponent"
        round_number     = current_round,
        current_evidence = current_evidence_list,  # List[Evidence] in this round
        side_argument    = requesting_side_argument_text,
        previous_testimony_summaries = previous_expert_summaries,  # list of str, can be []
    )
"""

from typing import List, Optional


def generate_expert_testimony(
    expert_llm,
    claim: str,
    expert_type: str,
    requesting_side: str,
    round_number: int,
    current_evidence: list,
    side_argument: str,
    previous_testimony_summaries: Optional[List[str]] = None,
) -> tuple:
    """
    Generate expert witness persona description and testimony.

    Returns
    -------
    (expert_description: str, expert_testimony: str)

    Parameters
    ----------
    expert_llm                   : LLM client for the expert witness model
    claim                        : The full claim text being adjudicated
    expert_type                  : The type of expert requested (e.g. "Botanist")
    requesting_side              : "proponent" or "opponent"
    round_number                 : Current debate round (1-indexed)
    current_evidence             : List of Evidence objects available this round
    side_argument                : The requesting side's argument text this round
    previous_testimony_summaries : Brief summaries of prior expert testimonies
                                   (to avoid repetition)
    """
    if previous_testimony_summaries is None:
        previous_testimony_summaries = []

    # ── Step 1: Generate expert persona (keep this call minimal) ─────
    persona_prompt = (
        f"You are about to testify as an expert witness in a fact-checking proceeding.\n"
        f"Requested expert type: {expert_type}\n"
        f"Claim domain: {claim[:120]}\n\n"
        f"In 1-2 sentences, describe your specific credentials and area of expertise "
        f"relevant to this claim. Be specific — name your sub-field and methodology."
    )
    expert_description = expert_llm.generate(persona_prompt).strip()

    # ── Step 2: Build evidence summary for the prompt ─────────────────
    evidence_lines = []
    for i, ev in enumerate(current_evidence[:8], 1):
        src_id = getattr(ev, 'source_id', ev.get('source_id', 'unknown') if isinstance(ev, dict) else 'unknown')
        text   = getattr(ev, 'text', ev.get('text', '') if isinstance(ev, dict) else '')
        evidence_lines.append(f"  Evidence {i} ({src_id}): {text[:200]}...")
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "  No evidence provided."

    # ── Step 3: Prior testimony note (prevent recycling) ──────────────
    prior_block = ""
    if previous_testimony_summaries:
        prior_lines = "\n".join(
            f"  - Round {i+1}: {s[:150]}"
            for i, s in enumerate(previous_testimony_summaries[-3:])  # last 3 only
        )
        prior_block = (
            f"\nPREVIOUS EXPERT TESTIMONIES IN THIS CASE (do NOT repeat these):\n"
            f"{prior_lines}\n"
            f"Your testimony MUST address new aspects not covered above.\n"
        )

    side_label = "Plaintiff" if requesting_side == "proponent" else "Defense"

    # ── Step 4: Generate testimony ────────────────────────────────────
    testimony_prompt = f"""You are testifying as an expert witness in a Wikipedia-based fact-checking proceeding.

YOUR CREDENTIALS: {expert_description}

CLAIM BEING ADJUDICATED (Round {round_number}):
{claim}

EVIDENCE CURRENTLY BEFORE THE COURT:
{evidence_block}

{side_label.upper()} COUNSEL'S ARGUMENT THIS ROUND:
{side_argument[:600] if side_argument else "No specific argument provided."}
{prior_block}
YOUR TASK:
Provide expert testimony specifically addressing the EVIDENCE above.
- Cite specific evidence IDs (e.g. "Evidence 1 (source_id)") in your analysis.
- Address the specific factual questions raised by {side_label} Counsel.
- Be concrete and specific — do NOT give generic background information.
- If the evidence is sufficient to support or refute the claim, state this clearly.
- If the evidence is insufficient on a specific point, identify EXACTLY what is missing
  and whether that gap is material to the verdict (many gaps are immaterial).
- Keep your testimony to 3-4 focused paragraphs.

Begin your testimony with "Your Honor," and do not use preamble phrases like
"Based on the preponderance of evidence presented" — go directly to the substance."""

    expert_testimony = expert_llm.generate(testimony_prompt).strip()

    return expert_description, expert_testimony


# ─────────────────────────────────────────────────────────────────────
# PRAG asymmetry fix — patch for retrieve_progressive in prag_engine.py
# ─────────────────────────────────────────────────────────────────────
"""
PRAG ASYMMETRY: IS IT A MAJOR ISSUE?
──────────────────────────────────────
Short answer: YES, it is a real and significant issue, not a minor one.

What you observed in the logs:
  - Phase 2+: "[PRAG STOP] Diminishing relevance gain" fires for the Plaintiff
    while Defense continues to admit 1-3 new exhibits.
  - In claim 48413: PRAG STOP fires 6 times, almost all on the Plaintiff side.
  - Result: Plaintiff argues from the same 3 initial chunks for rounds 3-6,
    while Defense accumulates fresh evidence. The judges see an increasingly
    one-sided evidence base and vote NOT SUPPORTED or INCONCLUSIVE.

WHY IT'S ASYMMETRIC:
  The stopping criteria check relevance_gain against the GLOBAL pool.
  After round 1, the Plaintiff's queries are similar to each other
  (they're trying to find the same supporting evidence), so relevance_gain
  quickly drops below 0.03. The Defense queries are diverse (they keep
  probing new attack angles), so they bypass the threshold more often.

THE FIX:
  In prag_engine.py, change retrieve_progressive to use per-side relevance
  tracking instead of a global pool comparison. The simplest implementation
  is to lower relevance_gain_threshold for rounds > 2 to 0.01 (instead of 0.03)
  — this is permissive enough that both sides can keep adding evidence in
  later rounds without requiring each query to be dramatically novel.

  Additionally, the redundancy_ratio_threshold of 0.85 with the formula
  `novelty < (1 - 0.85) = 0.15` is extremely strict. A chunk that is 80%
  similar to existing pool (novelty=0.20) would be KEPT, but 85% similar
  (novelty=0.15) would be dropped. Raising this to 0.90 gives slightly
  more room. The key change is below.

APPLY THIS CHANGE IN prag_engine.py __init__:
"""

PRAG_HYPERPARAMETER_PATCH = {
    # Old value → New value with rationale
    "novelty_threshold":          (0.15, 0.12),  # Slightly more permissive
    "redundancy_sim_threshold":   (0.85, 0.90),  # Was dropping too many chunks
    "redundancy_ratio_threshold": (0.85, 0.90),  # Matching sim threshold
    "relevance_gain_threshold":   (0.03, 0.01),  # Main fix: stop fires too early
}

"""
APPLY IN prag_engine.py __init__:

    self.novelty_threshold            = 0.12   # was 0.15
    self.redundancy_sim_threshold     = 0.90   # was 0.85
    self.redundancy_ratio_threshold   = 0.90   # was 0.85
    self.relevance_gain_threshold     = 0.01   # was 0.03  ← main fix

The relevance_gain_threshold change from 0.03 → 0.01 is the most impactful.
It means PRAG only stops when evidence is essentially identical to what was
already retrieved, not just when the marginal gain is modestly small.
"""