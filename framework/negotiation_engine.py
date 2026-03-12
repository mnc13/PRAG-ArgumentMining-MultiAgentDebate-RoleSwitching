"""
Negotiation Engine for Evidence Preparation  (v3 – page-aware judge)

Key fixes vs v2:
  - _calculate_weight  : judge prompt now distinguishes between
      (a) page-level relevance  – is this the right Wikipedia article?
      (b) chunk-level specificity – does THIS chunk contain the exact fact?
      A chunk from the correct page gets credit even if the specific date/
      result isn't in that 100-word window, because it BELONGS to the right
      source.  The old prompt gave 0 relevance to correct-page chunks that
      lacked the exact date string.

  - admissibility threshold kept at 0.35 for Wikipedia sources.
  - dispute threshold at 0.05.
"""

import json
import re
from typing import List, Dict
from models import Claim, Evidence
from llm_client import LLMClient


class EvidenceNegotiator:
    def __init__(self, retriever, miner_llm: LLMClient,
                 admissibility_threshold: float = 0.35,
                 dispute_threshold: float = 0.05):
        self.retriever = retriever
        self.llm = miner_llm
        self.admissibility_threshold = admissibility_threshold
        self.dispute_threshold       = dispute_threshold

        self.negotiation_state = {
            "shared_pool": [],
            "proponent_pool": [],
            "opponent_pool": [],
            "judge_state": {
                "admissible_evidence": [],
                "disputed_items": []
            }
        }

    # ─────────────────────────────────────────────────────────────────
    # Step 1 + 2  –  pool construction
    # ─────────────────────────────────────────────────────────────────

    def prepare_pools(self, claim: Claim, premises: List[str], top_k: int = 5):
        print("\n--- [Negotiator] Step 1: Premise-Grounded Shared Retrieval ---")
        shared_raw = []
        for premise in premises:
            shared_raw.extend(self.retriever.retrieve(premise, top_k=top_k))
        self.negotiation_state["shared_pool"] = self._deduplicate(shared_raw)
        print(f"   > Aggregated {len(self.negotiation_state['shared_pool'])} "
              f"shared evidence items.")

        print("\n--- [Negotiator] Step 2: Stance / Perspective Retrieval ---")
        for role in ["proponent", "opponent"]:
            display = "Plaintiff" if role == "proponent" else "Defense"
            print(f"   > Generating {display} Counsel-conditioned queries...")
            stance_q = self._generate_stance_query(claim.text, role)
            print(f"   > Stance query: {stance_q}")
            role_res = self.retriever.retrieve(stance_q, top_k=top_k)
            pool_key = f"{role}_pool"
            self.negotiation_state[pool_key] = self._deduplicate(role_res)
            print(f"   > {display} Counsel gathered "
                  f"{len(self.negotiation_state[pool_key])} items.")

    # ─────────────────────────────────────────────────────────────────
    # Step 3  –  negotiation injection
    # ─────────────────────────────────────────────────────────────────

    def negotiate_phase(self, claim: Claim):
        print("\n--- [Negotiator] Step 3: Multi-Agent Negotiation Injection ---")
        context = {
            "shared":        [e.source_id for e in self.negotiation_state["shared_pool"]],
            "proponent_only":[e.source_id for e in self.negotiation_state["proponent_pool"]],
            "opponent_only": [e.source_id for e in self.negotiation_state["opponent_pool"]],
        }
        for role, counsel in [("proponent", "Plaintiff"), ("opponent", "Defense")]:
            print(f"   > [{counsel} Counsel] Reviewing prospective evidence pools...")
            prompt = (
                f"Review these evidence discovery pools for claim: {claim.text}\n"
                f"Context: {json.dumps(context)}\n"
                f"Identify items from your pool to DISCLOSE (admit) and "
                f"opposing items to CHALLENGE."
            )
            self.llm.generate(prompt)
        print("   > Negotiation complete. Proceeding to Judicial arbitration.")

    # ─────────────────────────────────────────────────────────────────
    # Step 4  –  judicial arbitration
    # ─────────────────────────────────────────────────────────────────

    def judge_arbitration(self, claim: Claim):
        print("\n--- [The Court] Step 4: Evidence Arbitration & Admissibility ---")
        all_ev = self._deduplicate(
            self.negotiation_state["shared_pool"] +
            self.negotiation_state["proponent_pool"] +
            self.negotiation_state["opponent_pool"]
        )

        admissible, disputed = [], []

        for ev in all_ev:
            wd = self._calculate_weight(claim.text, ev.text, ev.source_id)
            weight      = wd.get("weight",      0.0)
            relevance   = wd.get("relevance",   0.0)
            credibility = wd.get("credibility", 0.0)
            reason      = wd.get("reason",      "No reason provided.")

            print(f"   > Evidence Arbitration [{ev.source_id}]: "
                  f"{{\"weight\": {weight:.3f}, \"relevance\": {relevance:.3f}, "
                  f"\"credibility\": {credibility:.3f}, \"reason\": \"{reason}\"}}")

            ev.relevance_score = weight
            entry = {
                "id": ev.source_id, "weight": weight,
                "relevance": relevance, "credibility": credibility,
                "reason": reason, "text": ev.text[:150],
            }

            if weight > self.admissibility_threshold:
                admissible.append(entry)
            elif weight > self.dispute_threshold:
                disputed.append(entry)

        admissible.sort(key=lambda x: x["weight"], reverse=True)
        self.negotiation_state["judge_state"]["admissible_evidence"] = admissible
        self.negotiation_state["judge_state"]["disputed_items"]      = disputed
        print(f"   > Admitted {len(admissible)} items. "
              f"Flagged {len(disputed)} for dispute.")

    # ─────────────────────────────────────────────────────────────────
    # Serialisation
    # ─────────────────────────────────────────────────────────────────

    def get_negotiation_json(self) -> Dict:
        return {
            "shared_pool":    [self._ev_to_dict(e) for e in self.negotiation_state["shared_pool"]],
            "proponent_pool": [self._ev_to_dict(e) for e in self.negotiation_state["proponent_pool"]],
            "opponent_pool":  [self._ev_to_dict(e) for e in self.negotiation_state["opponent_pool"]],
            "judge_state":    self.negotiation_state["judge_state"],
        }

    def _ev_to_dict(self, ev: Evidence) -> Dict:
        return {
            "source_id":       ev.source_id,
            "text":            ev.text[:300],
            "relevance_score": ev.relevance_score,
        }

    # ─────────────────────────────────────────────────────────────────
    # LLM helpers
    # ─────────────────────────────────────────────────────────────────

    def _generate_stance_query(self, claim: str, role: str) -> str:
        stance    = "supporting" if role == "proponent" else "challenging"
        counsel   = "Plaintiff"  if role == "proponent" else "Defense"
        prompt = (
            f"You are {counsel} Counsel preparing for a fact-checking proceeding.\n"
            f"Generate a concise Wikipedia search query (4-8 keywords) to find "
            f"encyclopaedic information {stance} the following claim.\n"
            f"Focus on proper nouns, dates, names, and event keywords.\n\n"
            f"Claim: {claim}\n\n"
            f"Output ONLY the search query string, nothing else."
        )
        return self.llm.generate(prompt).strip().strip('"').strip("'")

    def _calculate_weight(self, claim: str, evidence_text: str,
                           source_id: str = "") -> Dict:
        """
        PAGE-AWARE admissibility scoring.

        The key insight: evidence is a 100-word CHUNK from a Wikipedia page.
        The chunk may not contain the specific date/result but still belongs
        to exactly the right article.  We therefore score TWO things:

          page_relevance  : does this chunk come from a page whose TITLE and
                            general topic match the claim?  (inferred from the
                            [Wikipedia: Title] prefix and overall topic)

          chunk_specificity : does this specific 100-word window contain the
                              exact entities, dates, or facts in the claim?

        final weight = page_relevance * 0.5  +  chunk_specificity * 0.5
        This way a correct-page chunk that lacks the date still scores ~0.35-0.45
        instead of 0.0.
        """
        prompt = f"""You are evaluating a Wikipedia evidence chunk for a fact-checking claim.

CLAIM: {claim}

EVIDENCE CHUNK (100-word excerpt from a Wikipedia article):
{evidence_text[:1200]}

IMPORTANT: This is a SHORT EXCERPT from a larger Wikipedia article.
The chunk may not contain every specific detail of the claim even if it
comes from exactly the right article.

Evaluate TWO separate dimensions:

1. PAGE RELEVANCE (0.0–1.0): Does this chunk appear to come from a Wikipedia
   article that is about the right topic, team, person, event, or time period
   mentioned in the claim?
   - 0.8–1.0: Chunk is clearly from the specific article about this exact event/entity
   - 0.5–0.7: Chunk is from a closely related article (same team, nearby year, etc.)
   - 0.2–0.4: Chunk is from a tangentially related article
   - 0.0–0.1: Chunk is from a completely unrelated article

2. CHUNK SPECIFICITY (0.0–1.0): Does this specific excerpt contain the key
   facts, dates, names, or results needed to directly verify the claim?
   - 0.8–1.0: Excerpt directly mentions the specific fact, date, or result
   - 0.5–0.7: Excerpt mentions related facts that partially verify the claim
   - 0.2–0.4: Excerpt is on-topic but lacks specific verifying details
   - 0.0–0.1: Excerpt contains no claim-relevant details

CREDIBILITY (0.0–1.0): How reliable is this Wikipedia source?
   - Wikipedia with specific named facts/dates/statistics: 0.65–0.80
   - Wikipedia general background: 0.50–0.65
   - Vague or unsourced: 0.20–0.40

Respond ONLY in valid JSON with no extra text:
{{
    "page_relevance": 0.0-1.0,
    "chunk_specificity": 0.0-1.0,
    "credibility": 0.0-1.0,
    "reason": "One concise sentence"
}}"""

        response = self.llm.generate(prompt)
        try:
            match = re.search(r'\{[\s\S]*?\}', response)
            data  = json.loads(match.group()) if match else {}

            page_rel   = min(1.0, max(0.0, float(data.get("page_relevance",   0.5))))
            chunk_spec = min(1.0, max(0.0, float(data.get("chunk_specificity", 0.3))))
            credib     = min(1.0, max(0.0, float(data.get("credibility",       0.6))))

            # Combined relevance = weighted average of page and chunk scores
            relevance = round(0.5 * page_rel + 0.5 * chunk_spec, 3)
            weight    = round(relevance * credib, 3)

            return {
                "weight":      weight,
                "relevance":   relevance,
                "credibility": round(credib, 3),
                "reason":      data.get("reason", "No reason provided."),
            }
        except Exception as e:
            print(f"   > [Warning] Admissibility evaluation failed: {e}")
            return {
                "weight": 0.3, "relevance": 0.5,
                "credibility": 0.6, "reason": "Fallback due to evaluation error.",
            }

    # ─────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────

    def _deduplicate(self, evidence_list: List[Evidence]) -> List[Evidence]:
        seen, unique = set(), []
        for ev in evidence_list:
            if ev.source_id not in seen:
                unique.append(ev)
                seen.add(ev.source_id)
        return unique