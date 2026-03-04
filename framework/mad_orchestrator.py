"""
Multi-Agent Debate (MAD) Orchestrator

Manages multi-round debate with multiple agents
"""

import json
from typing import List, Dict
from models import Claim, Evidence
from mad_system import DebateAgent, CriticAgent
from self_reflection import SelfReflection
from prag_engine import ProgressiveRAG
from personas import validate_unique_models
import numpy as np

class MADOrchestrator:
    """
    Orchestrates multi-round debate with multiple agents
    """
    def __init__(self, claim: Claim, initial_evidence: List[Evidence], persona_configs: List[Dict], prag_engine: ProgressiveRAG):
        """
        Initialize debate orchestrator
        
        Args:
            claim: Claim to debate
            initial_evidence: Evidence from initial RAG
            persona_configs: List of persona configuration dictionaries
            prag_engine: ProgressiveRAG instance
        """
        self.claim = claim
        self.initial_evidence = initial_evidence # Store for resetting
        self.evidence_pool = list(initial_evidence)
        self.prag = prag_engine
        self.debate_transcript = []
        self.current_round = 0
        
        # Validate unique models
        validate_unique_models(persona_configs)
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents(persona_configs)
        
        # Self-Reflection & Critic
        from personas import AGENT_SLOTS
        self.self_reflection = SelfReflection(self.debate_transcript)
        self.critic = CriticAgent(AGENT_SLOTS.get('critic'))
        self.reflection_discovery_needs = {"proponent": "", "opponent": ""}
        self.last_total_reflection_score = 0.0
        
    def _initialize_agents(self, persona_configs: List[Dict]):
        """
        Initialize debate system agents
        """
        from personas import AGENT_SLOTS
        from expertise_extractor import extract_single_expert
        
        # We ignore input configs and use fixed agents for consistency
        self.agents['proponent'] = DebateAgent(AGENT_SLOTS['proponent'], 'proponent', self.prag)
        self.agents['opponent'] = DebateAgent(AGENT_SLOTS['opponent'], 'opponent', self.prag)
        self.agents['judge'] = DebateAgent(AGENT_SLOTS['judge'], 'judge', self.prag)
        
        # Experts will be summoned dynamically
        self.agents['experts'] = []

    def reset_state(self):
        """
        Reset the orchestrator state for a fresh debate (e.g., after role switching)
        """
        print("\n[Orchestrator] Resetting debate state for dynamic rounds...")
        self.debate_transcript.clear()
        self.current_round = 0
        self.evidence_pool = list(self.initial_evidence)
        self.last_total_reflection_score = 0.0
        self.reflection_discovery_needs = {"proponent": "", "opponent": ""}
        
        # Reset reflection history
        self.self_reflection.reflection_history.clear()
        self.self_reflection.debate_transcript = self.debate_transcript
        
        # Reset PRAG state
        self.prag.round_counter = 0
        self.prag.retrieval_history.clear()
        self.prag.total_evidence_pool = list(self.initial_evidence)
    
    def run_debate_round(self, round_num: int) -> Dict:
        """
        Execute one round of the scientific proceedings
        """
        self.current_round = round_num
        self.prag.start_new_round()
        
        round_data = {
            "round_number": round_num,
            "arguments": [],
            "expert_testimonies": [],
            "new_evidence": [],
            "prag_metrics": [],
            "reflection_scores": {},
            "critic_evaluation": {}
        }
        
        print(f"\n{'='*60}")
        print(f"PROCEEDINGS PHASE {round_num}")
        print(f"{'='*60}\n")
        
        # Sequentially for both sides
        for side in ['proponent', 'opponent']:
            display_side = "Plaintiff Counsel" if side == 'proponent' else "Defense Counsel"
            
            # 1. Query Proposing & Refinement Feedback Loop (Hybrid: Gap + Reflection)
            print(f"--- [{display_side}] Step 1: Evidence Discovery (Integrative Discovery) ---")
            debate_context = self._get_debate_context()
            gap_proposal = self.agents[side].propose_query_gap(debate_context)
            reflection_gap = self.reflection_discovery_needs.get(side, "")
            
            # Combine gap-driven and reflection-driven needs
            discovery_prompt = gap_proposal
            if reflection_gap:
                discovery_prompt = f"{gap_proposal} Focus also on: {reflection_gap}"
            
            if discovery_prompt and "None" not in discovery_prompt:
                print(f"   > [{display_side}] Discovery Need: {discovery_prompt}")
                original_query = self.prag.formulate_query(debate_context, discovery_prompt)
                print(f"   > [{display_side}] Formulated Query: {original_query}")
                
                # Feedback Loop: The Court refines the query
                print(f"--- [The Court] Reviewing Discovery Request ---")
                refined_query = self.agents['judge'].refine_query(original_query, debate_context)
                
                if refined_query != original_query:
                    print(f"   > [The Court] QUERY REFINED: {refined_query}")
                
                # PRAG Execution
                new_evidence = self.prag.retrieve_progressive(
                    refined_query, 
                    top_k=3, 
                    context=f"Round {round_num} - {display_side}"
                )
                
                if new_evidence:
                    self.evidence_pool.extend(new_evidence)
                    round_data["new_evidence"].extend([{"id": e.source_id, "novelty": e.novelty_score} for e in new_evidence])
                    print(f"   > [{display_side}] Admitted {len(new_evidence)} new exhibits.")
                
                # Log PRAG metrics for this side
                if self.prag.retrieval_history:
                    latest_prag = self.prag.retrieval_history[-1]
                    round_data["prag_metrics"].append({
                        "side": display_side,
                        "original_query": original_query,
                        "refined_query": refined_query,
                        "novelty": latest_prag.get("avg_novelty"),
                        "accepted": latest_prag.get("num_accepted")
                    })

            # 2. Argument Generation
            print(f"--- [{display_side}] Step 2: Generating Legal Argument ---")
            arg = self.agents[side].generate_argument(self.claim, self.evidence_pool, self.debate_transcript)
            self._add_to_transcript(round_data, side, arg)
            
        # 3. Check for Expert Witness Testimony
        print(f"--- [The Court] Step 3: Evaluating Expert Witness Requirements ---")
        for side in ['proponent', 'opponent']:
            expert_req = self.agents[side].request_expert(self.debate_transcript)
            if expert_req:
                display_side = "Plaintiff" if side == "proponent" else "Defense"
                print(f"   > [{display_side} Counsel] Proposed Expert Witness Type: {expert_req['expert_type']}")
                if expert_req and self.agents['judge'].evaluate_expert_request(side, expert_req):
                    print(f"   > [The Court] REQUEST GRANTED. Calling expert witness...")
                    from expertise_extractor import extract_single_expert
                    expert_config = extract_single_expert(expert_req['expert_type'], self.claim.text)
                    from personas import AGENT_SLOTS
                    expert_config.update(AGENT_SLOTS['expert_slot'])
                    expert_agent = DebateAgent(expert_config, 'expert', self.prag)
                    testimony = expert_agent.generate_argument(self.claim, self.evidence_pool, self.debate_transcript)
                    
                    expert_entry = {"agent": expert_agent.name, "role": "expert", "requesting_side": side, "text": testimony}
                    round_data["expert_testimonies"].append(expert_entry)
                    self.debate_transcript.append(expert_entry)
                    print(f"\n[EXPERT TESTIMONY]: {testimony}\n")

        # 4. Multi-Round Self-Reflection
        print(f"--- [Audit] Step 4: Multi-Round Self-Reflection ---")
        for side in ['proponent', 'opponent']:
            reflection = self.self_reflection.perform_round_reflection(self.agents[side], side, round_num, self.claim.text)
            round_data["reflection_scores"][side] = reflection
            self.reflection_discovery_needs[side] = reflection.get("discovery_need", "")

        # 5. Critic Agent Evaluation
        print(f"--- [Critic] Step 5: Round Integrity Review ---")
        critic_eval = self.critic.evaluate_round(round_num, self.claim.text, self.debate_transcript)
        round_data["critic_evaluation"] = critic_eval
        if critic_eval.get("recommendations"):
            recs = critic_eval["recommendations"]
            p_recs = recs.get('plaintiff', [])
            d_recs = recs.get('defense', [])
            print(f"   > Critic Recommendations: {len(p_recs)} for Plaintiff, {len(d_recs)} for Defense")
            for r in p_recs:
                print(f"     * [Plaintiff Rec]: {r}")
            for r in d_recs:
                print(f"     * [Defense Rec]: {r}")

        return round_data

    def _get_debate_context(self) -> str:
        """Helper to get text context of the debate so far"""
        return "\n".join([f"{a['agent']}: {a['text'][:200]}..." for a in self.debate_transcript[-4:]])


    def _add_to_transcript(self, round_data, role, text):
        entry = {
            "agent": self.agents[role].name,
            "role": role,
            "text": text
        }
        round_data["arguments"].append(entry)
        self.debate_transcript.append(entry)
        print(f"\n{text}\n")

    def run_full_debate(self, max_rounds: int = 10, save_transcript: bool = True, file_suffix: str = "") -> Dict:
        """
        Run proceedings with adaptive convergence rules
        """
        debate_result = {
            "claim": self.claim.text,
            "claim_id": getattr(self.claim, 'id', 'Unknown'),
            "agents": {
                "proponent": self.agents['proponent'].job_title,
                "opponent": self.agents['opponent'].job_title,
                "the_court": self.agents['judge'].job_title
            },
            "rounds": [],
            "convergence_metrics": {}
        }
        
        last_novelty = 1.0
        
        for round_num in range(1, max_rounds + 1):
            round_data = self.run_debate_round(round_num)
            debate_result["rounds"].append(round_data)
            
            # Adaptive Convergence Checks
            # 1. Evidence Novelty Stabilization
            current_novelties = [e['novelty'] for e in round_data["new_evidence"]]
            avg_novelty = np.mean(current_novelties) if current_novelties else 0
            
            # 2. Reflection Delta Check (Convergence)
            total_ref_score = sum([r.get('total_score', 0) for r in round_data["reflection_scores"].values()])
            delta_score = total_ref_score - self.last_total_reflection_score
            
            print(f"--- [Convergence] Score Delta: {delta_score:.4f} ---")
            
            if round_num >= 2:
                # Terminate if improvement plateaus
                if delta_score < 0.05 and delta_score > -0.05:
                    print(f"   > [ADAPTIVE STOP] Argument quality plateaued (delta < 5%). Proceedings concluded.")
                    debate_result["convergence_metrics"]["stop_reason"] = "Reflection plateau"
                    break
                
                # Terminate if Critic signals resolution
                if round_data["critic_evaluation"].get("debate_resolved", False):
                    print(f"   > [ADAPTIVE STOP] Critic signals all premises resolved. Deliberation begins.")
                    debate_result["convergence_metrics"]["stop_reason"] = "Critic resolution"
                    break

                # Stop if novelty is very low
                if avg_novelty < 0.1 and last_novelty < 0.1:
                    print(f"   > [ADAPTIVE STOP] Evidence novelty stabilized (< 10%). Cases closed.")
                    debate_result["convergence_metrics"]["stop_reason"] = "Novelty stabilization"
                    break
                
                # Judge's internal signal
                if self.agents['judge'].check_debate_completion(self.debate_transcript):
                    print(f"   > [ADAPTIVE STOP] The Court signals sufficient evidence. Deliberation begins.")
                    debate_result["convergence_metrics"]["stop_reason"] = "Judicial signal"
                    break
            
            self.last_total_reflection_score = total_ref_score
            last_novelty = avg_novelty
            
        # Save transcript to file for inspection
        if save_transcript:
            import json
            try:
                from logging_extension import append_framework_json
                append_framework_json(f"debate_transcript{file_suffix}.jsonl", self.claim, debate_result)
            except ImportError:
                with open(f"debate_transcript{file_suffix}.json", "w") as f:
                    json.dump(debate_result, f, indent=2)
                
            # Save reflection history
            self.self_reflection.save_reflection_history(
                claim_id=self.claim, 
                filename=f"self_reflection{file_suffix}.json"
            )
                
            # Judge Visibility JSON
            self._save_judge_visibility(debate_result, file_suffix=file_suffix)
            
            self.prag.save_history(
                filepath=f"prag_history{file_suffix}.json", 
                claim_id=self.claim
            )
        return debate_result

    def _save_judge_visibility(self, debate_result, file_suffix: str = ""):
        """Extract and save judge-specific metrics for transparency"""
        visibility = {
            "claim": debate_result["claim"],
            "total_rounds": len(debate_result["rounds"]),
            "prag_history": self.prag.get_retrieval_summary(),
            "query_evolution": []
        }
        for r in debate_result["rounds"]:
            for m in r.get("prag_metrics", []):
                visibility["query_evolution"].append({
                    "round": r["round_number"],
                    "side": m["side"],
                    "original": m["original_query"],
                    "refined": m["refined_query"],
                    "novelty": m["novelty"]
                })
        
        import json
        try:
            from logging_extension import append_framework_json
            append_framework_json(f"judge_visibility{file_suffix}.jsonl", self.claim, visibility)
        except ImportError:
            with open(f"judge_visibility{file_suffix}.json", "w") as f:
                json.dump(visibility, f, indent=2)
