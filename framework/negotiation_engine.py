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
        print("\n--- [Negotiator] Step 5: Multi-Agent Negotiation Injection ---")
        
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
        import math

        all_candidate_evidence = []
        all_candidate_evidence.extend(self.negotiation_state["shared_pool"])
        all_candidate_evidence.extend(self.negotiation_state["proponent_pool"])
        all_candidate_evidence.extend(self.negotiation_state["opponent_pool"])
        all_candidate_evidence = self._deduplicate(all_candidate_evidence)

        admissible = []
        disputed = []

        # Metrics for claim scoring
        claim_score = 0.0
        total_support_weight = 0.0
        total_refute_weight = 0.0
        neutral_weight = 0.0

        for ev in all_candidate_evidence:
            print(f"   > Evaluating Source ID: {ev.source_id}")
            # Weight = relevance * credibility (mocked via LLM or heuristic)
            weight_data = self._calculate_weight(claim.text, ev.text)
            
            # Update Evidence object (for backward compatibility)
            ev.relevance_score = weight_data['weight']
            
            # Check admissibility threshold
            if weight_data['weight'] > 0.6:
                admissible.append({
                    "id": ev.source_id,
                    "weight": weight_data['weight'],
                    "stance": weight_data['stance'],
                    "stance_confidence": weight_data['confidence'],
                    "reason": weight_data['reason'],
                    "text": ev.text[:150]
                })
            elif weight_data['weight'] > 0.2:
                # Disputed if not highly weighted but still somewhat relevant
                disputed.append({
                    "id": ev.source_id,
                    "weight": weight_data['weight'],
                    "text": ev.text[:150]
                })

        # Sort admissible by weight
        admissible.sort(key=lambda x: x['weight'], reverse=True)
        
        # Calculate Claim Score and Probability
        for item in admissible:
            s = item["stance"]
            w = item["weight"]
            claim_score += s * w

            if s == 1:
                total_support_weight += w
            elif s == -1:
                total_refute_weight += w
            else:
                neutral_weight += w

        # Logistic transformation
        try:
            probability = 1 / (1 + math.exp(-claim_score))
        except OverflowError:
            probability = 0.0 if claim_score < 0 else 1.0
            
        confidence = abs(probability - 0.5) * 2

        # Verdict Rule
        if probability > 0.65:
            verdict = "SUPPORT"
        elif probability < 0.35:
            verdict = "REFUTE"
        else:
            verdict = "NOT_ENOUGH_EVIDENCE"

        # Store results
        self.negotiation_state["judge_state"]["admissible_evidence"] = admissible
        self.negotiation_state["judge_state"]["disputed_items"] = disputed
        
        self.negotiation_state["judge_state"]["claim_metrics"] = {
            "claim_score": claim_score,
            "probability": probability,
            "confidence": confidence,
            "total_support_weight": total_support_weight,
            "total_refute_weight": total_refute_weight,
            "neutral_weight": neutral_weight,
            "evidence_count": len(admissible)
        }
        self.negotiation_state["judge_state"]["verdict"] = verdict
        
        print(f"   > Admitted {len(admissible)} high-weight items. Flagged {len(disputed)} for dispute.")
        print(f"   > Negotiation Verdict: {verdict} (Prob: {probability:.3f}, Conf: {confidence:.3f})")

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
        Evaluates evidence for stance, relevance, and confidence using LLM.
        """
        prompt = (f"Evaluate this evidence against the claim.\n"
                  f"Claim: {claim}\n"
                  f"Evidence: {evidence_text[:800]}\n\n" # Increased context slightly
                  f"Provide a strict JSON response with keys:\n"
                  f"- \"stance\": -1 (refutes), 0 (neutral), or 1 (supports)\n"
                  f"- \"weight\": scientific relevance score (0.0 to 1.0)\n"
                  f"- \"confidence\": confidence in the stance (0.0 to 1.0)\n"
                  f"- \"reason\": short explanation\n\n"
                  f"Response (JSON only):")
        
        try:
            response = self.llm.generate(prompt)
            # Basic cleanup to find JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                # Validation and clamping
                stance = int(data.get("stance", 0))
                if stance not in [-1, 0, 1]:
                    stance = 0
                    
                weight = float(data.get("weight", 0.0))
                weight = max(0.0, min(1.0, weight))
                
                confidence = float(data.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))
                
                reason = str(data.get("reason", "No reason provided."))
                
                return {
                    "weight": weight,
                    "stance": stance,
                    "confidence": confidence,
                    "reason": reason
                }
            else:
                print(f"Error parsing LLM response (no JSON found): {response[:100]}...")
        except Exception as e:
            print(f"Error in _calculate_weight: {e}")
            
        # Fallback if parsing fails
        return {
            "weight": 0.1, 
            "stance": 0, 
            "confidence": 0.0, 
            "reason": "Analysis failed."
        }

    def _deduplicate(self, evidence_list: List[Evidence]) -> List[Evidence]:
        seen_ids = set()
        unique = []
        for ev in evidence_list:
            if ev.source_id not in seen_ids:
                unique.append(ev)
                seen_ids.add(ev.source_id)
        return unique
