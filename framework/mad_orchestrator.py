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
        self.agents['critic'] = DebateAgent(AGENT_SLOTS['critic'], 'critic', self.prag)
        
        # Experts will be summoned dynamically
        self.agents['experts'] = []
    
    def run_debate_round(self, round_num: int) -> Dict:
        """
        Execute one round of the scientific debate
        """
        self.current_round = round_num
        self.prag.start_new_round()
        
        round_data = {
            "round_number": round_num,
            "arguments": [],
            "critiques": [],
            "expert_testimonies": [],
            "new_evidence": [],
            "prag_triggered": False
        }
        
        print(f"\n{'='*60}")
        print(f"DEBATE PHASE {round_num}")
        print(f"{'='*60}\n")
        
        # 1. Proponent argues
        print(f"--- [Proponent] Step 1: Gathering Clinical Evidence ---")
        proponent_evidence_requested = self.agents['proponent'].request_evidence(self.debate_transcript[-3:], "latest evidence")
        if proponent_evidence_requested:
            round_data["new_evidence"].extend([{"source_id": e.source_id, "text": e.text[:100]} for e in proponent_evidence_requested])
            self.evidence_pool.extend(proponent_evidence_requested) 
            print(f"   > [Proponent] Found {len(proponent_evidence_requested)} new technical sources.")

        print(f"--- [Proponent] Step 2: Generating Technical Argument ---")
        prop_arg = self.agents['proponent'].generate_argument(self.claim, self.evidence_pool, self.debate_transcript)
        self._add_to_transcript(round_data, "proponent", prop_arg)
        
        # 2. Opponent counters
        print(f"--- [Opponent] Step 1: Gathering Counter-Evidence ---")
        opponent_evidence_requested = self.agents['opponent'].request_evidence(self.debate_transcript[-3:], "latest evidence")
        if opponent_evidence_requested:
            round_data["new_evidence"].extend([{"source_id": e.source_id, "text": e.text[:100]} for e in opponent_evidence_requested])
            self.evidence_pool.extend(opponent_evidence_requested)
            print(f"   > [Opponent] Found {len(opponent_evidence_requested)} new technical sources.")

        print(f"--- [Opponent] Step 2: Generating Technical Counter-Argument ---")
        opp_arg = self.agents['opponent'].generate_argument(self.claim, self.evidence_pool, self.debate_transcript)
        self._add_to_transcript(round_data, "opponent", opp_arg)
        
        # 3. Check for Expert Summoning
        print(f"--- [Moderator] Step 3: Evaluating Scientific Expert Requirements ---")
        for side in ['proponent', 'opponent']:
            expert_req = self.agents[side].request_expert(self.debate_transcript)
            if expert_req:
                print(f"   > [{side.capitalize()}] Proposed Expert Type: {expert_req['expert_type']}")
                if self.agents['judge'].evaluate_expert_request(side, expert_req):
                    print(f"   > [Moderator] REQUEST GRANTED. Generating dynamic expert persona...")
                    from expertise_extractor import extract_single_expert
                    expert_config = extract_single_expert(expert_req['expert_type'], self.claim.text)
                    
                    from personas import AGENT_SLOTS
                    expert_config.update(AGENT_SLOTS['expert_slot'])
                    
                    expert_agent = DebateAgent(expert_config, 'expert', self.prag)
                    print(f"   > [Expert] {expert_agent.name} (Technical Expert) delivering testimony...")
                    testimony = expert_agent.generate_argument(self.claim, self.evidence_pool, self.debate_transcript)
                    
                    expert_entry = {
                        "agent": expert_agent.name,
                        "role": "expert",
                        "requesting_side": side,
                        "text": testimony
                    }
                    round_data["expert_testimonies"].append(expert_entry)
                    self.debate_transcript.append(expert_entry)
                    print(f"\n[TECHNICAL TESTIMONY]: {testimony}\n")
                else:
                    print(f"   > [Moderator] REQUEST DENIED. Proceeding with existing evidence.")

        # 4. Critic evaluate session
        print(f"[Critic] Analyzing proceedings...")
        for arg in round_data["arguments"]:
            critique = self.agents['critic'].critique_argument(arg)
            round_data["critiques"].append(critique)
        
        # 5. P-RAG (Maintain existing structure)
        if round_num % 2 == 0:
            print(f"\n[P-RAG] Archivist retrieving additional evidence...")
            debate_context = "\n".join([f"{a['agent']}: {a['text'][:200]}..." for a in self.debate_transcript[-4:]])
            new_evidence = self.agents['opponent'].request_evidence(debate_context)
            self.evidence_pool.extend(new_evidence)
            round_data["prag_triggered"] = True
        
        return round_data

    def _add_to_transcript(self, round_data, role, text):
        entry = {
            "agent": self.agents[role].name,
            "role": role,
            "text": text
        }
        round_data["arguments"].append(entry)
        self.debate_transcript.append(entry)
        print(f"\n{text}\n")

    def run_full_debate(self, max_rounds: int = 5) -> Dict:
        """
        Run debate simulation until Judge finishes or max rounds
        """
        debate_result = {
            "claim": self.claim.text,
            "claim_id": getattr(self.claim, 'id', 'Unknown'),
            "agents": {
                "proponent": self.agents['proponent'].job_title,
                "opponent": self.agents['opponent'].job_title,
                "judge": self.agents['judge'].job_title,
                "critic": self.agents['critic'].job_title
            },
            "rounds": []
        }
        
        for round_num in range(1, max_rounds + 1):
            round_data = self.run_debate_round(round_num)
            debate_result["rounds"].append(round_data)
            
            # 6. Judge decides whether to end the debate
            if round_num >= 2: # At least 2 rounds
                print(f"[Judge] Reviewing record for completion...")
                if self.agents['judge'].check_debate_completion(self.debate_transcript):
                    print(f"[Judge] THE RECORD IS CLOSED. Proceeding to final verdict.")
                    break
        
        # Save transcript
        with open("debate_transcript.json", "w") as f:
            json.dump(debate_result, f, indent=2)
        
        self.prag.save_history()
        return debate_result
