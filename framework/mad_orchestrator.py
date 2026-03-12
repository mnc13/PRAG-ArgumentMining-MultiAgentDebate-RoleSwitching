"""
Multi-Agent Debate (MAD) Orchestrator  v2

Key fixes:
  1. Uses get_agent_slots(claim.text) instead of bare AGENT_SLOTS so every
     agent's system prompt includes a domain notice derived from the claim.
     This prevents gpt-5-mini from hallucinating medical/injury discovery
     requests when the claim is about sports or history.

  2. Seeds prag.total_evidence_pool with initial_evidence at construction
     time so novelty scoring works correctly from round 1.

  3. reset_state() also re-seeds the PRAG pool (role-switching starts clean
     but still has the negotiation evidence as baseline).

  4. Context window guard: _get_debate_context() now caps each entry at
     300 chars and limits to last 6 entries to prevent deepseek-v3.2 from
     hitting the 512-token output cap due to a bloated input prompt.
"""

import json
from typing import List, Dict
from models import Claim, Evidence
from mad_system import DebateAgent, CriticAgent
from self_reflection import SelfReflection
from prag_engine import ProgressiveRAG
from personas import validate_unique_models, get_agent_slots
import numpy as np


class MADOrchestrator:
    def __init__(self, claim: Claim, initial_evidence: List[Evidence],
                 persona_configs: List[Dict], prag_engine: ProgressiveRAG):
        self.claim          = claim
        self.initial_evidence = initial_evidence
        self.evidence_pool  = list(initial_evidence)
        self.prag           = prag_engine
        self.debate_transcript: List[Dict] = []
        self.current_round  = 0

        # Validate unique models
        validate_unique_models(persona_configs)

        # Build domain-aware agent slots from the claim text
        self._agent_slots = get_agent_slots(claim.text)

        # Initialise agents
        self.agents: Dict = {}
        self._initialize_agents()

        # Self-Reflection & Critic
        self.self_reflection = SelfReflection(self.debate_transcript)
        self.critic          = CriticAgent(self._agent_slots.get('critic'))
        self.reflection_discovery_needs = {"proponent": "", "opponent": ""}
        self.last_total_reflection_score = 0.0

        # ── FIX: seed PRAG pool with negotiation evidence ─────────────
        self.prag.seed_evidence_pool(initial_evidence)

    # ─────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────

    def _initialize_agents(self):
        from expertise_extractor import extract_single_expert
        self.agents['proponent'] = DebateAgent(
            self._agent_slots['proponent'], 'proponent', self.prag)
        self.agents['opponent']  = DebateAgent(
            self._agent_slots['opponent'],  'opponent',  self.prag)
        self.agents['judge']     = DebateAgent(
            self._agent_slots['judge'],     'judge',     self.prag)
        self.agents['experts']   = []

    def reset_state(self):
        """Reset for role-switching — keeps negotiation evidence as PRAG baseline."""
        print("\n[Orchestrator] Resetting debate state for dynamic rounds...")
        self.debate_transcript.clear()
        self.current_round = 0
        self.evidence_pool = list(self.initial_evidence)
        self.last_total_reflection_score = 0.0
        self.reflection_discovery_needs  = {"proponent": "", "opponent": ""}

        self.self_reflection.reflection_history.clear()
        self.self_reflection.debate_transcript = self.debate_transcript

        # Reset PRAG counters but keep the seeded pool so novelty scoring
        # remains meaningful in the switched-role debate
        self.prag.round_counter = 0
        self.prag.retrieval_history.clear()
        self.prag.total_evidence_pool = list(self.initial_evidence)

    # ─────────────────────────────────────────────────────────────────
    # Single debate round
    # ─────────────────────────────────────────────────────────────────

    def run_debate_round(self, round_num: int) -> Dict:
        self.current_round = round_num
        self.prag.start_new_round()

        round_data = {
            "round_number":      round_num,
            "arguments":         [],
            "expert_testimonies":[],
            "new_evidence":      [],
            "prag_metrics":      [],
            "reflection_scores": {},
            "critic_evaluation": {}
        }

        print(f"\n{'='*60}")
        print(f"PROCEEDINGS PHASE {round_num}")
        print(f"{'='*60}\n")

        for side in ['proponent', 'opponent']:
            display_side = ("Plaintiff Counsel" if side == 'proponent'
                            else "Defense Counsel")

            # ── Step 1: Evidence Discovery ────────────────────────────
            print(f"--- [{display_side}] Step 1: Evidence Discovery "
                  f"(Integrative Discovery) ---")
            debate_context = self._get_debate_context()
            gap_proposal   = self.agents[side].propose_query_gap(debate_context)
            reflection_gap = self.reflection_discovery_needs.get(side, "")

            discovery_prompt = gap_proposal
            if reflection_gap:
                discovery_prompt = f"{gap_proposal} Focus also on: {reflection_gap}"

            if discovery_prompt and "None" not in discovery_prompt:
                print(f"   > [{display_side}] Discovery Need: {discovery_prompt}")
                original_query = self.prag.formulate_query(
                    debate_context, discovery_prompt)
                print(f"   > [{display_side}] Formulated Query: {original_query}")

                print(f"--- [The Court] Reviewing Discovery Request ---")
                refined_query = self.agents['judge'].refine_query(
                    original_query, debate_context)
                if refined_query != original_query:
                    print(f"   > [The Court] QUERY REFINED: {refined_query}")

                # PRAG retrieval
                new_evidence = self.prag.retrieve_progressive(
                    refined_query, top_k=3,
                    context=f"Round {round_num} - {display_side}"
                )

                if new_evidence:
                    self.evidence_pool.extend(new_evidence)
                    round_data["new_evidence"].extend([
                        {"id": e.source_id, "novelty": e.novelty_score}
                        for e in new_evidence
                    ])
                    print(f"   > [{display_side}] Admitted "
                          f"{len(new_evidence)} new exhibits.")

                if self.prag.retrieval_history:
                    latest = self.prag.retrieval_history[-1]
                    round_data["prag_metrics"].append({
                        "side":          display_side,
                        "original_query":original_query,
                        "refined_query": refined_query,
                        "novelty":       latest.get("avg_novelty"),
                        "accepted":      latest.get("num_accepted"),
                    })

            # ── Step 2: Argument Generation ───────────────────────────
            print(f"--- [{display_side}] Step 2: Generating Legal Argument ---")
            arg = self.agents[side].generate_argument(
                self.claim, self.evidence_pool, self.debate_transcript)
            self._add_to_transcript(round_data, side, arg)

        # ── Step 3: Expert Witnesses ──────────────────────────────────
        print(f"--- [The Court] Step 3: Evaluating Expert Witness Requirements ---")
        for side in ['proponent', 'opponent']:
            expert_req = self.agents[side].request_expert(self.debate_transcript)
            if not expert_req:
                continue
            display_side = "Plaintiff" if side == "proponent" else "Defense"
            print(f"   > [{display_side} Counsel] Proposed Expert Witness Type: "
                  f"{expert_req['expert_type']}")
            if self.agents['judge'].evaluate_expert_request(side, expert_req):
                print(f"   > [The Court] REQUEST GRANTED. Calling expert witness...")
                from expertise_extractor import extract_single_expert
                expert_config = extract_single_expert(
                    expert_req['expert_type'], self.claim.text)
                expert_config.update(self._agent_slots['expert_slot'])
                expert_agent = DebateAgent(expert_config, 'expert', self.prag)
                testimony = expert_agent.generate_argument(
                    self.claim, self.evidence_pool, self.debate_transcript)
                expert_entry = {
                    "agent": expert_agent.name, "role": "expert",
                    "requesting_side": side,    "text": testimony
                }
                round_data["expert_testimonies"].append(expert_entry)
                self.debate_transcript.append(expert_entry)
                print(f"\n[EXPERT TESTIMONY]: {testimony}\n")

        # ── Step 4: Self-Reflection ───────────────────────────────────
        print(f"--- [Audit] Step 4: Multi-Round Self-Reflection ---")
        for side in ['proponent', 'opponent']:
            reflection = self.self_reflection.perform_round_reflection(
                self.agents[side], side, round_num, self.claim.text)
            round_data["reflection_scores"][side] = reflection
            self.reflection_discovery_needs[side] = reflection.get(
                "discovery_need", "")

        # ── Step 5: Critic ────────────────────────────────────────────
        print(f"--- [Critic] Step 5: Round Integrity Review ---")
        critic_eval = self.critic.evaluate_round(
            round_num, self.claim.text, self.debate_transcript)
        round_data["critic_evaluation"] = critic_eval
        recs = critic_eval.get("recommendations", {})
        p_recs = recs.get('plaintiff', [])
        d_recs = recs.get('defense', [])
        print(f"   > Critic Recommendations: "
              f"{len(p_recs)} for Plaintiff, {len(d_recs)} for Defense")
        for r in p_recs:
            print(f"     * [Plaintiff Rec]: {r}")
        for r in d_recs:
            print(f"     * [Defense Rec]: {r}")

        return round_data

    # ─────────────────────────────────────────────────────────────────
    # Context helper  (FIX: capped to prevent token overflow)
    # ─────────────────────────────────────────────────────────────────

    def _get_debate_context(self) -> str:
        """
        Return a compact debate context string.

        FIX: original code used 200 chars per entry and last 4 entries.
        deepseek-v3.2 was hitting the 512-token output cap because the
        combined prompt (context + instructions) was ~3 000 tokens, leaving
        almost no room for the response.  Now capped at 300 chars per entry,
        last 6 entries — enough context without blowing the budget.
        """
        entries = self.debate_transcript[-6:]
        return "\n".join(
            f"{a['agent']}: {a['text'][:300]}..." for a in entries
        )

    # ─────────────────────────────────────────────────────────────────
    # Transcript helpers
    # ─────────────────────────────────────────────────────────────────

    def _add_to_transcript(self, round_data, role, text):
        entry = {
            "agent": self.agents[role].name,
            "role":  role,
            "text":  text,
        }
        round_data["arguments"].append(entry)
        self.debate_transcript.append(entry)
        print(f"\n{text}\n")

    # ─────────────────────────────────────────────────────────────────
    # Full debate loop
    # ─────────────────────────────────────────────────────────────────

    def run_full_debate(self, max_rounds: int = 10,
                        save_transcript: bool = True,
                        file_suffix: str = "") -> Dict:
        debate_result = {
            "claim":    self.claim.text,
            "claim_id": getattr(self.claim, 'id', 'Unknown'),
            "agents": {
                "proponent":  self.agents['proponent'].job_title,
                "opponent":   self.agents['opponent'].job_title,
                "the_court":  self.agents['judge'].job_title,
            },
            "rounds":               [],
            "convergence_metrics":  {}
        }

        last_novelty = 1.0

        for round_num in range(1, max_rounds + 1):
            round_data = self.run_debate_round(round_num)
            debate_result["rounds"].append(round_data)

            # Novelty
            current_novelties = [e['novelty'] for e in round_data["new_evidence"]]
            avg_novelty = np.mean(current_novelties) if current_novelties else 0

            # Reflection delta
            total_ref_score = sum(
                r.get('total_score', 0)
                for r in round_data["reflection_scores"].values()
            )
            delta_score = total_ref_score - self.last_total_reflection_score
            print(f"--- [Convergence] Score Delta: {delta_score:.4f} ---")

            if round_num >= 2:
                if -0.05 < delta_score < 0.05:
                    print("   > [ADAPTIVE STOP] Argument quality plateaued.")
                    debate_result["convergence_metrics"]["stop_reason"] = \
                        "Reflection plateau"
                    break
                if round_data["critic_evaluation"].get("debate_resolved", False):
                    print("   > [ADAPTIVE STOP] Critic signals resolution.")
                    debate_result["convergence_metrics"]["stop_reason"] = \
                        "Critic resolution"
                    break
                if avg_novelty < 0.1 and last_novelty < 0.1:
                    print("   > [ADAPTIVE STOP] Evidence novelty stabilized.")
                    debate_result["convergence_metrics"]["stop_reason"] = \
                        "Novelty stabilization"
                    break
                if self.agents['judge'].check_debate_completion(
                        self.debate_transcript):
                    print("   > [ADAPTIVE STOP] The Court signals sufficient evidence.")
                    debate_result["convergence_metrics"]["stop_reason"] = \
                        "Judicial signal"
                    break

            self.last_total_reflection_score = total_ref_score
            last_novelty = avg_novelty

        if save_transcript:
            self._save_outputs(debate_result, file_suffix)

        return debate_result

    # ─────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────

    def _save_outputs(self, debate_result: Dict, file_suffix: str):
        try:
            from logging_extension import append_framework_json
            append_framework_json(
                f"debate_transcript{file_suffix}.jsonl",
                self.claim, debate_result)
        except ImportError:
            with open(f"debate_transcript{file_suffix}.json", "w") as f:
                json.dump(debate_result, f, indent=2)

        self.self_reflection.save_reflection_history(
            claim_id=self.claim,
            filename=f"self_reflection{file_suffix}.json")

        self._save_judge_visibility(debate_result, file_suffix=file_suffix)

        self.prag.save_history(
            filepath=f"prag_history{file_suffix}.json",
            claim_id=self.claim)

    def _save_judge_visibility(self, debate_result: Dict, file_suffix: str = ""):
        visibility = {
            "claim":       debate_result["claim"],
            "total_rounds":len(debate_result["rounds"]),
            "prag_history":self.prag.get_retrieval_summary(),
            "query_evolution": [
                {
                    "round":    r["round_number"],
                    "side":     m["side"],
                    "original": m["original_query"],
                    "refined":  m["refined_query"],
                    "novelty":  m["novelty"],
                }
                for r in debate_result["rounds"]
                for m in r.get("prag_metrics", [])
            ]
        }
        try:
            from logging_extension import append_framework_json
            append_framework_json(
                f"judge_visibility{file_suffix}.jsonl",
                self.claim, visibility)
        except ImportError:
            with open(f"judge_visibility{file_suffix}.json", "w") as f:
                json.dump(visibility, f, indent=2)