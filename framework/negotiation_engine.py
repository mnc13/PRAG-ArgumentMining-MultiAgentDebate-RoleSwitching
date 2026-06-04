import json
import re
from typing import List, Dict
from models import Claim, Evidence
from llm_client import LLMClient


class EvidenceNegotiator:
    def __init__(self, retriever, miner_llm: LLMClient,
                 admissibility_threshold: float = 0.30,   # LOWERED from 0.35
                 dispute_threshold: float = 0.05):
        """
        admissibility_threshold lowered from 0.35 → 0.30.

        Rationale: FEVEROUS claims are verified against Wikipedia. Evidence
        chunks are 100-word windows; even a directly relevant chunk on the
        correct Wikipedia page may only score 0.25–0.35 because it talks
        about the broader article topic, not the specific infobox fact.
        The old 0.35 threshold was silently dropping good evidence, causing
        0-item admitted sets and pipeline collapse for obscure claims.

        The page-aware arbitration prompt (below) also scores page_relevance
        separately, so the new combined weight formula naturally distinguishes
        "right article, wrong chunk" (score ~0.25–0.35) from "wrong article
        entirely" (score < 0.10).
        """
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
        PAGE-AWARE admissibility scoring for Wikipedia-sourced evidence.

        FEVEROUS context: evidence is a ~100-word chunk from a Wikipedia page.
        The chunk may not contain the specific date/result but still belongs
        to exactly the right article.  We score TWO things separately:

          page_relevance    : does this chunk come from the right Wikipedia
                              article for this claim?
          chunk_specificity : does this specific excerpt contain the key facts?

        final weight = (page_relevance * 0.6 + chunk_specificity * 0.4) * credibility

        NOTE: Wikipedia is the authoritative source for FEVEROUS claims.
        Chunks from named, specific Wikipedia articles should receive
        credibility scores of 0.65–0.80, not penalised for being Wikipedia.
        """
        prompt = f"""You are evaluating a Wikipedia evidence chunk for a FEVEROUS fact-checking claim.

CLAIM: {claim}

EVIDENCE CHUNK (~100 words from a Wikipedia article):
{evidence_text[:1200]}

CONTEXT: This is a SHORT EXCERPT from a larger Wikipedia article.
Wikipedia IS the authoritative reference for this fact-checking task.
The chunk may not contain every specific detail even if it comes from the
correct article — that is normal and expected.

Evaluate TWO separate dimensions:

1. PAGE RELEVANCE (0.0–1.0)
   Does this chunk appear to come from a Wikipedia article about the right
   topic, entity, event, or time period mentioned in the claim?
   - 0.8–1.0 : Clearly from the specific article about this exact entity/event
   - 0.5–0.7 : From a closely related article (same person/team/nearby year)
   - 0.2–0.4 : From a tangentially related article
   - 0.0–0.1 : From a completely unrelated article

2. CHUNK SPECIFICITY (0.0–1.0)
   Does this specific excerpt contain the key facts, dates, names, or results
   needed to directly verify the claim?
   - 0.8–1.0 : Excerpt directly states the specific fact, date, or result
   - 0.5–0.7 : Mentions related facts that partially verify the claim
   - 0.2–0.4 : On-topic but lacks specific verifying details
   - 0.0–0.1 : No claim-relevant details

3. CREDIBILITY (0.0–1.0)
   How reliable is this Wikipedia source?
   - Named Wikipedia article with specific facts/dates/statistics : 0.70–0.85
   - Wikipedia general background article : 0.55–0.70
   - Vague, unsourced, or clearly off-topic : 0.20–0.40

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

            # Page relevance weighted more than chunk specificity — a chunk
            # from the right article is valuable even if it lacks the exact fact.
            relevance = round(0.6 * page_rel + 0.4 * chunk_spec, 3)
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