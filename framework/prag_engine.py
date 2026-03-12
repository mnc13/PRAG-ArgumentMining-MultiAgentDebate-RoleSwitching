"""
Progressive RAG (P-RAG) Engine  v2

Key fixes:
  - seed_evidence_pool()  : call this BEFORE the MAD starts to pre-populate
    total_evidence_pool with negotiation-admitted evidence.  Without seeding,
    round-1 novelty is artificially 1.0 for everything, and round-2 stops
    immediately because all new evidence looks redundant against round-1.

  - formulate_query prompt : removed "medical exhibits" framing → domain-agnostic,
    works for Wikipedia/FEVEROUS encyclopaedic claims.

  - Stopping thresholds relaxed slightly for encyclopaedic datasets:
      redundancy_ratio_threshold : 0.7  → 0.85
      relevance_gain_threshold   : 0.05 → 0.03
    This gives PRAG more room to explore before halting.

  - retrieve_progressive now returns accepted_evidence even when a stop_reason
    fires (previously it returned early with nothing on some code paths).
    The stop_reason is now ADVISORY only — it logs and prints but does NOT
    suppress the returned evidence.
"""

from models import Evidence
from llm_client import LLMClient
import numpy as np
from typing import List, Dict, Tuple


class ProgressiveRAG:
    def __init__(self, vector_retriever, llm_client: LLMClient):
        self.retriever = vector_retriever
        self.llm = llm_client
        self.retrieval_history: List[Dict] = []
        self.round_counter = 0
        self.total_evidence_pool: List[Evidence] = []

        # ── Hyperparameters ──────────────────────────────────────────
        self.novelty_threshold            = 0.15   # was 0.2  – slightly more permissive
        self.redundancy_sim_threshold     = 0.85   # unchanged
        self.redundancy_ratio_threshold   = 0.85   # was 0.7  – more room before stopping
        self.relevance_gain_threshold     = 0.03   # was 0.05 – less aggressive early stop
        self.max_iterations               = 10

    # ─────────────────────────────────────────────────────────────────
    # NEW: seed the pool before debate starts
    # ─────────────────────────────────────────────────────────────────

    def seed_evidence_pool(self, evidence: List[Evidence]):
        """
        Pre-populate total_evidence_pool with negotiation-admitted evidence.

        Call this once in main.py AFTER negotiation and BEFORE run_full_debate():

            prag.seed_evidence_pool(final_evidence_set)
            mad = MADOrchestrator(extracted_claim, final_evidence_set, [], prag)

        Without seeding, the pool is empty on round 1, novelty scores are
        artificially 1.0, and round 2 immediately hits the redundancy stop
        because everything in round 2 is similar to round 1.
        """
        if not evidence:
            return
        # Avoid duplicates
        existing_ids = {e.source_id for e in self.total_evidence_pool}
        new_items = [e for e in evidence if e.source_id not in existing_ids]
        self.total_evidence_pool.extend(new_items)
        print(f"   > [P-RAG] Pool seeded with {len(new_items)} negotiation exhibits "
              f"({len(self.total_evidence_pool)} total).")

    # ─────────────────────────────────────────────────────────────────
    # Query formulation  (FIXED: domain-agnostic prompt)
    # ─────────────────────────────────────────────────────────────────

    def formulate_query(self, debate_context: str, agent_request: str) -> str:
        """
        Use LLM to formulate a targeted retrieval query from the debate context.

        FIX: old prompt said "retrieve relevant medical exhibits and evidence"
        which caused the LLM to generate medically-framed queries even for
        sports/history/general-knowledge claims.
        """
        prompt = f"""Based on the following legal proceedings context and counsel's discovery request,
formulate a precise Wikipedia/encyclopaedic search query to retrieve relevant factual evidence.

Proceedings Context:
{debate_context}

Counsel's Discovery Request:
{agent_request}

Instructions:
- Focus on the specific facts, entities, dates, and events mentioned in the request.
- Use proper nouns, years, team/person/place names as keywords.
- Do NOT use medical or scientific jargon unless the claim is about medicine.
- Output ONLY the search query string (1-2 sentences, no explanation)."""

        query = self.llm.generate(prompt)
        return query.strip()

    # ─────────────────────────────────────────────────────────────────
    # Progressive retrieval  (FIXED: stop is advisory, not suppressive)
    # ─────────────────────────────────────────────────────────────────

    def retrieve_progressive(self, query: str, top_k: int = 3,
                              context: str = "") -> List[Evidence]:
        """
        Perform targeted retrieval with novelty scoring and stopping criteria.

        FIX: previously the method would log a stop_reason and then still
        return accepted_evidence (correct), but the caller in mad_orchestrator
        checked prag.retrieval_history[-1]['stop_reason'] and skipped adding
        evidence.  Now stop_reason is purely informational — evidence is always
        returned if it passes the novelty filter.
        """
        # 1. Retrieve
        raw_evidence = self.retriever.retrieve(query, top_k=top_k)
        if not raw_evidence:
            self._log_retrieval(query, context, [], [], 0.0, 0.0, 0.0, 0.0,
                                "No results from retriever")
            return []

        # 2. Novelty scoring
        scored_evidence, avg_novelty = self._calculate_novelty(raw_evidence)

        # 3. Filter by novelty threshold
        accepted_evidence = [e for e in scored_evidence
                             if e.novelty_score >= self.novelty_threshold]
        rejected_count = len(scored_evidence) - len(accepted_evidence)

        # 4. Redundancy ratio
        redundant_count = sum(
            1 for e in scored_evidence
            if e.novelty_score < (1 - self.redundancy_sim_threshold)
        )
        redundancy_ratio = (redundant_count / len(scored_evidence)
                            if scored_evidence else 0)

        # 5. Relevance gain
        avg_relevance = (np.mean([e.relevance_score for e in accepted_evidence])
                         if accepted_evidence else 0.0)
        last_avg_rel = (self.retrieval_history[-1].get("avg_relevance", 0)
                        if self.retrieval_history else 0.0)
        relevance_gain = avg_relevance - last_avg_rel

        # 6. Stopping criteria  (ADVISORY ONLY — do not suppress evidence)
        stop_reason = None
        if self.round_counter >= self.max_iterations:
            stop_reason = "Maximum iterations reached"
        elif redundancy_ratio > self.redundancy_ratio_threshold:
            stop_reason = (f"High redundancy detected "
                           f"({redundancy_ratio:.2f} > {self.redundancy_ratio_threshold})")
        elif (self.round_counter > 1
              and relevance_gain < self.relevance_gain_threshold):
            stop_reason = (f"Diminishing relevance gain "
                           f"({relevance_gain:.4f} < {self.relevance_gain_threshold})")

        # 7. Log
        self._log_retrieval(query, context, raw_evidence, accepted_evidence,
                            avg_novelty, avg_relevance, relevance_gain,
                            redundancy_ratio, stop_reason)

        # 8. Add accepted evidence to pool  (always, even if stop flagged)
        if accepted_evidence:
            existing_ids = {e.source_id for e in self.total_evidence_pool}
            for ev in accepted_evidence:
                if ev.source_id not in existing_ids:
                    self.total_evidence_pool.append(ev)
                    existing_ids.add(ev.source_id)

        if stop_reason:
            print(f"   > [PRAG STOP] {stop_reason}")

        # Always return accepted evidence regardless of stop_reason
        return accepted_evidence

    # ─────────────────────────────────────────────────────────────────
    # Novelty calculation
    # ─────────────────────────────────────────────────────────────────

    def _calculate_novelty(self, new_evidence: List[Evidence]
                           ) -> Tuple[List[Evidence], float]:
        """novelty = 1 - max_cosine_sim(new_doc, existing_pool)"""
        if not self.total_evidence_pool:
            for ev in new_evidence:
                ev.novelty_score = 1.0
            return new_evidence, 1.0

        model = getattr(self.retriever, 'model', None)
        if not model:
            for ev in new_evidence:
                ev.novelty_score = 1.0
            return new_evidence, 1.0

        pool_texts = [e.text for e in self.total_evidence_pool]
        new_texts  = [e.text for e in new_evidence]

        pool_embs = model.encode(pool_texts, convert_to_numpy=True,
                                  normalize_embeddings=True)
        new_embs  = model.encode(new_texts,  convert_to_numpy=True,
                                  normalize_embeddings=True)

        novelty_scores = []
        for i, new_emb in enumerate(new_embs):
            similarities = np.dot(pool_embs, new_emb)
            max_sim  = float(np.max(similarities))
            novelty  = 1.0 - max_sim
            new_evidence[i].novelty_score = novelty
            novelty_scores.append(novelty)

        return new_evidence, float(np.mean(novelty_scores))

    # ─────────────────────────────────────────────────────────────────
    # Logging helper
    # ─────────────────────────────────────────────────────────────────

    def _log_retrieval(self, query, context, raw, accepted,
                       avg_novelty, avg_relevance, relevance_gain,
                       redundancy_ratio, stop_reason):
        self.retrieval_history.append({
            "round":            self.round_counter,
            "query":            query,
            "context":          context,
            "num_retrieved":    len(raw),
            "num_accepted":     len(accepted),
            "num_rejected":     len(raw) - len(accepted),
            "avg_novelty":      float(avg_novelty),
            "avg_relevance":    float(avg_relevance),
            "relevance_gain":   float(relevance_gain),
            "redundancy_ratio": float(redundancy_ratio),
            "stop_reason":      stop_reason,
            "evidence_ids":     [ev.source_id for ev in accepted],
        })

    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────

    def start_new_round(self):
        self.round_counter += 1

    def get_retrieval_summary(self) -> Dict:
        return {
            "total_retrievals":  len(self.retrieval_history),
            "rounds_with_prag":  len(set(r["round"] for r in self.retrieval_history)),
            "history":           self.retrieval_history,
        }

    def save_history(self, filepath: str = "prag_history.json",
                     claim_id: str = "unknown"):
        import json
        try:
            from logging_extension import append_framework_json
            append_framework_json(filepath.replace('.json', '.jsonl'),
                                  claim_id, self.get_retrieval_summary())
        except ImportError:
            with open(filepath, 'w') as f:
                json.dump(self.get_retrieval_summary(), f, indent=2)