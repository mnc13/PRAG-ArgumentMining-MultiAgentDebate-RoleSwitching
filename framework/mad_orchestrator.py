"""
Multi-Agent Debate (MAD) Orchestrator

Manages multi-round debate with multiple agents
"""

import json
from typing import List, Dict
from models import Claim, Evidence
from mad_system import DebateAgent
from prag_engine import ProgressiveRAG
from personas import validate_unique_models

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
        self.evidence_pool = initial_evidence
        self.prag = prag_engine
        self.debate_transcript = []
        self.current_round = 0
        
        # Validate unique models
        validate_unique_models(persona_configs)
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents(persona_configs)
        
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
            "prag_metrics": []
        }
        
        print(f"\n{'='*60}")
        print(f"PROCEEDINGS PHASE {round_num}")
        print(f"{'='*60}\n")
        
        # Sequentially for both sides
        for side in ['proponent', 'opponent']:
            display_side = "Plaintiff Counsel" if side == 'proponent' else "Defense Counsel"
            
            # 1. Query Proposing & Refinement Feedback Loop
            print(f"--- [{display_side}] Step 1: Evidence Discovery Gap Analysis ---")
            debate_context = self._get_debate_context()
            gap_proposal = self.agents[side].propose_query_gap(debate_context)
            
            if gap_proposal and "None" not in gap_proposal:
                print(f"   > [{display_side}] Proposal: {gap_proposal}")
                original_query = self.prag.formulate_query(debate_context, gap_proposal)
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
                if self.agents['judge'].evaluate_expert_request(side, expert_req):
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

    def run_full_debate(self, max_rounds: int = 10) -> Dict:
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
            
            # 2. Confidence Check (Simulation)
            # In a real implementation, we'd call self.agents['judge'].get_confidence()
            # For now, we'll use the judge's existing check_debate_completion
            
            print(f"--- [The Court] Monitoring Case Convergence ---")
            print(f"   > Exhibit Novelty: {avg_novelty:.4f}")
            
            if round_num >= 2:
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
            
            last_novelty = avg_novelty
            
        # Save results
        with open("debate_transcript.json", "w") as f:
            json.dump(debate_result, f, indent=2)
            
        # Judge Visibility JSON
        self._save_judge_visibility(debate_result)
        
        self.prag.save_history()
        return debate_result

    def _save_judge_visibility(self, debate_result):
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
        
        with open("judge_visibility.json", "w") as f:
            json.dump(visibility, f, indent=2)
