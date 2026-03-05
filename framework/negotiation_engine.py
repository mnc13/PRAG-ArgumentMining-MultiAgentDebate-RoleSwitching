"""
Negotiation Engine for Evidence Preparation

Handles premise-grounded retrieval, stance-conditioned retrieval, 
and judge-mediated arbitration.
"""

import json
from typing import List, Dict, Optional
from models import Claim, Evidence
from rag_engine import PubMedRetriever
from llm_client import LLMClient

class EvidenceNegotiator:
    def __init__(self, retriever: PubMedRetriever, miner_llm: LLMClient):
        self.retriever = retriever
        self.llm = miner_llm
        self.negotiation_state = {
            "shared_pool": [],
            "proponent_pool": [],
            "opponent_pool": [],
            "judge_state": {
                "admissible_evidence": [],
                "disputed_items": []
            }
        }

    def prepare_pools(self, claim: Claim, premises: List[str], top_k: int = 3):
        """
        1. Premise-Grounded Shared Retrieval
        2. Stance / Perspective Retrieval
        3. Evidence Pool Construction
        """
        print("\n--- [Negotiator] Step 1: Premise-Grounded Shared Retrieval ---")
        shared_pool_raw = []
        for premise in premises:
            results = self.retriever.retrieve(premise, top_k=top_k)
            shared_pool_raw.extend(results)
        
        # Deduplicate and finalize shared pool
        self.negotiation_state["shared_pool"] = self._deduplicate(shared_pool_raw)
        print(f"   > Aggregated {len(self.negotiation_state['shared_pool'])} shared evidence items.")

        print("\n--- [Negotiator] Step 2: Stance / Perspective Retrieval ---")
        for role in ["proponent", "opponent"]:
            display_role = "Plaintiff" if role == "proponent" else "Defense"
            print(f"   > Generating {display_role} Counsel-conditioned queries...")
            stance_query = self._generate_stance_query(claim.text, role)
            role_results = self.retriever.retrieve(stance_query, top_k=top_k)
            
            pool_key = f"{role}_pool"
            self.negotiation_state[pool_key] = self._deduplicate(role_results)
            print(f"   > {display_role} Counsel gathered {len(self.negotiation_state[pool_key])} perspective-specific items.")

    def negotiate_phase(self, claim: Claim):
        """
        5. Negotiation Injection - Agents discuss the pools
        """
        print("\n--- [Negotiator] Step 3: Multi-Agent Negotiation Injection ---")
        
        # Prepare context for agents
        context = {
            "shared": [e.source_id for e in self.negotiation_state["shared_pool"]],
            "proponent_only": [e.source_id for e in self.negotiation_state["proponent_pool"]],
            "opponent_only": [e.source_id for e in self.negotiation_state["opponent_pool"]]
        }
        
        # Plaintiff Counsel discloses/challenges
        import json
        print("   > [Plaintiff Counsel] Reviewing prospective evidence pools...")
        prop_input = (f"Review these evidence discovery pools for claim: {claim.text}\n"
                      f"Context: {json.dumps(context)}\n"
                      f"Identify any items from your discovery pool to DISCLOSE (admit) and any shared/defense items to CHALLENGE.")
        self.llm.generate(prop_input) # Simulate processing
        
        # Defense Counsel discloses/challenges
        print("   > [Defense Counsel] Reviewing prospective evidence pools...")
        opp_input = (f"Review these evidence discovery pools for claim: {claim.text}\n"
                     f"Context: {json.dumps(context)}\n"
                     f"Identify any items from your discovery pool to DISCLOSE (admit) and any shared/plaintiff items to CHALLENGE.")
        self.llm.generate(opp_input) # Simulate processing
        
        print("   > Negotiation complete. Proceeding to Judicial arbitration.")

    def judge_arbitration(self, claim: Claim):
        """
        4. Judicial Role - arbitration and admissibility weighting
        """
        print("\n--- [The Court] Step 4: Evidence Arbitration \u0026 Admissibility ---")
        all_candidate_evidence = []
        all_candidate_evidence.extend(self.negotiation_state["shared_pool"])
        all_candidate_evidence.extend(self.negotiation_state["proponent_pool"])
        all_candidate_evidence.extend(self.negotiation_state["opponent_pool"])
        all_candidate_evidence = self._deduplicate(all_candidate_evidence)

        admissible = []
        disputed = []

        for ev in all_candidate_evidence:
            # Calculate dynamic weight
            weight_data = self._calculate_weight(claim.text, ev.text)
            
            # Extract metrics
            weight = weight_data.get('weight', 0.0)
            relevance = weight_data.get('relevance', 0.0)
            credibility = weight_data.get('credibility', 0.0)
            reason = weight_data.get('reason', 'No reason provided.')
            
            # Log the metadata for each evidence item as requested
            print(f"   > Evidence Arbitration [{ev.source_id}]: "
                  f"{{\"weight\": {weight:.3f}, \"relevance\": {relevance:.3f}, \"credibility\": {credibility:.3f}, \"reason\": \"{reason}\"}}")
            
            ev.relevance_score = weight
            
            if weight > 0.5:
                admissible.append({
                    "id": ev.source_id,
                    "weight": weight,
                    "relevance": relevance,
                    "credibility": credibility,
                    "reason": reason,
                    "text": ev.text[:150]
                })
            elif weight > 0.1:
                # Disputed if not highly weighted but still somewhat relevant
                disputed.append({
                    "id": ev.source_id,
                    "weight": weight,
                    "relevance": relevance,
                    "credibility": credibility,
                    "reason": reason,
                    "text": ev.text[:150]
                })

        # Sort admissible by weight
        admissible.sort(key=lambda x: x['weight'], reverse=True)
        
        self.negotiation_state["judge_state"]["admissible_evidence"] = admissible
        self.negotiation_state["judge_state"]["disputed_items"] = disputed
        
        print(f"   > Admitted {len(admissible)} high-weight items. Flagged {len(disputed)} for dispute.")

    def get_negotiation_json(self) -> Dict:
        """
        Returns a JSON-serializable version of the negotiation state.
        """
        serializable_state = {
            "shared_pool": [self._ev_to_dict(e) for e in self.negotiation_state["shared_pool"]],
            "proponent_pool": [self._ev_to_dict(e) for e in self.negotiation_state["proponent_pool"]],
            "opponent_pool": [self._ev_to_dict(e) for e in self.negotiation_state["opponent_pool"]],
            "judge_state": self.negotiation_state["judge_state"]
        }
        return serializable_state

    def _ev_to_dict(self, ev: Evidence) -> Dict:
        return {
            "source_id": ev.source_id,
            "text": ev.text[:300], # Trucate for JSON overview but keep enough for context
            "relevance_score": ev.relevance_score
        }

    def _generate_stance_query(self, claim: str, role: str) -> str:
        display_role = "Plaintiff" if role == "proponent" else "Defense"
        prompt = (f"Generate a search query to find medical evidence {role == 'proponent' and 'supporting' or 'challenging'} "
                  f"the following claim for legal proceedings:\nClaim: {claim}\n"
                  f"Role: {display_role} Counsel\n"
                  f"Query (scientific keywords only):")
        return self.llm.generate(prompt).strip().strip('"')

    def _calculate_weight(self, claim: str, evidence_text: str) -> Dict:
        """
        Evaluate scientific relevance and credibility using LLM.
        Formula: weight = relevance * credibility
        """
        prompt = f"""Evaluate the scientific relevance and credibility of the following medical evidence for the claim.
        
        CLAIM: {claim}
        EVIDENCE: {evidence_text[:1000]}
        
        Provide a evaluation based on:
        1. Relevance: How directly does this evidence address the premises of the claim? (0.0 - 1.0)
        2. Credibility: Does the evidence come from a reliable scientific context or contain high-quality data? (0.0 - 1.0)
        
        Respond ONLY in valid JSON format:
        {{
            "relevance": 0.0-1.0,
            "credibility": 0.0-1.0,
            "reason": "Brief scientific justification for these scores"
        }}"""
        
        response = self.llm.generate(prompt)
        try:
            import re
            match = re.search(r'\{[\s\S]*\}', response)
            data = json.loads(match.group()) if match else {}
            
            relevance = float(data.get("relevance", 0.5))
            credibility = float(data.get("credibility", 0.5))
            
            # Joint Admissibility Weight Logic
            weight = round(relevance * credibility, 3)
            
            return {
                "weight": weight,
                "relevance": round(relevance, 3),
                "credibility": round(credibility, 3),
                "reason": data.get("reason", "No reason provided.")
            }
        except Exception as e:
            print(f"   > [Warning] Admissibility evaluation failed: {e}")
            return {
                "weight": 0.5,
                "relevance": 0.5,
                "credibility": 1.0,
                "reason": "Fallback due to evaluation error."
            }

    def _deduplicate(self, evidence_list: List[Evidence]) -> List[Evidence]:
        seen_ids = set()
        unique = []
        for ev in evidence_list:
            if ev.source_id not in seen_ids:
                unique.append(ev)
                seen_ids.add(ev.source_id)
        return unique
