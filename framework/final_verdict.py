"""
Final Verdict Generator

Aggregates all evidence and generates confidence-weighted verdict with explainable output
using probabilistic log-odds aggregation.
"""

from typing import Dict, List
import json
import math

class FinalVerdict:
    """
    Generates final verdict with confidence score and reasoning
    """
    
    def __init__(self, claim, debate_result: Dict, judge_result: Dict,
                 role_switch_result: Dict, reflection_result: Dict, negotiation_metrics: Dict):
        """
        Initialize verdict generator
        
        Args:
            claim: Original Claim object
            debate_result: MAD debate transcript
            judge_result: Judge evaluation results
            role_switch_result: Role-switching consistency report
            reflection_result: Self-reflection results
            negotiation_metrics: Metrics from EvidenceNegotiator (P_neg)
        """
        self.claim = claim
        self.debate_result = debate_result
        self.judge_result = judge_result
        self.role_switch_result = role_switch_result
        self.reflection_result = reflection_result
        self.negotiation_metrics = negotiation_metrics
    
    def generate_verdict(self) -> Dict:
        """
        Generate final verdict with confidence and detailed log-odds metrics.
        """
        print("\n" + "="*60)
        print("FINAL VERDICT GENERATION (PROBABILISTIC FUSION)")
        print("="*60 + "\n")
        
        epsilon = 1e-6

        # 1. Negotiation Layer
        p_neg = self.negotiation_metrics.get('probability', 0.5)
        # Clamp P_neg
        p_neg = max(min(p_neg, 1 - epsilon), epsilon)
        l_neg = math.log(p_neg / (1 - p_neg))
        
        # 2. Panel Layer
        judge_verdicts = self.judge_result.get('judge_verdicts', [])
        judge_individual_logs = []
        l_panel = 0.0
        
        for judge in judge_verdicts:
            # Calculate Q_j
            # Scores are 0-10, so sum is 0-30. Q_j is normalized 0-1.
            q_raw = (judge.get('evidence_strength', 5) + 
                     judge.get('argument_validity', 5) + 
                     judge.get('scientific_reliability', 5)) / 30.0
            
            # Clamp Q_j
            q_j = max(min(q_raw, 1 - epsilon), epsilon)
            
            # Determine verdict sign
            v_sign = 1 if judge.get('verdict') == 'SUPPORTED' else -1
            
            # Calculate L_j
            l_j = v_sign * math.log(q_j / (1 - q_j))
            
            l_panel += l_j
            
            judge_individual_logs.append({
                "verdict": judge.get('verdict'),
                "quality_score": q_j,
                "log_odds_contribution": l_j,
                "judge_name": judge.get('judge_name'),
                "model": judge.get('model')
            })
            
        # 3. Final Fusion
        l_total = l_neg + l_panel
        p_final = 1 / (1 + math.exp(-l_total))
        
        # 4. Decision Rule
        verdict = "SUPPORT" if p_final > 0.5 else "REFUTE"
        
        # 5. Confidence
        confidence = abs(2 * p_final - 1)
        
        # Log to console
        print(f"Negotiation P_neg: {p_neg:.4f} (L_neg: {l_neg:.4f})")
        print(f"Panel L_panel: {l_panel:.4f}")
        print(f"Total Log-Odds: {l_total:.4f}")
        print(f"Final Probability: {p_final:.4f}")
        print(f"Verdict: {verdict} (Confidence: {confidence:.4f})")

        # Generate reasoning chain
        reasoning = self._generate_reasoning(verdict)
        
        # Extract key evidence
        key_evidence = self._extract_key_evidence()
        
        # Get ground truth for comparison
        ground_truth = self.claim.metadata.get('label', 'UNKNOWN') if hasattr(self.claim, 'metadata') else 'UNKNOWN'
        correct = (verdict == ground_truth) if ground_truth != 'UNKNOWN' else None
        
        result = {
            "claim_id": getattr(self.claim, 'id', 'unknown'),
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "ground_truth_label": ground_truth,
            "correct": correct,
            "final_probability_support": round(p_final, 4),
            "final_log_odds": round(l_total, 4),
            "claim_text": self.claim.text,
            
            "negotiation_probability": round(p_neg, 4),
            "negotiation_log_odds": round(l_neg, 4),
            
            "panel_log_odds": round(l_panel, 4),
            "judge_individual": judge_individual_logs,
            
            
            "reasoning": reasoning,
            "key_evidence": key_evidence,
            "metadata": {
                "role_switch_consistent": self._check_role_switch_consistency(),
                "debate_rounds": len(self.debate_result['rounds']),
                "total_evidence_used": self._count_total_evidence()
            }
        }
        
        return result
    
    def _check_role_switch_consistency(self) -> bool:
        """
        Check if role-switching showed consistency
        Returns:
            True if consistent, False otherwise
        """
        # Simple heuristic: check if analysis mentions "consistent"
        analysis = self.role_switch_result.get('analysis', '')
        
        consistency_keywords = ['consistent', 'maintained', 'coherent', 'logical']
        inconsistency_keywords = ['inconsistent', 'contradicted', 'conflicting', 'incoherent']
        
        consistent_count = sum(1 for word in consistency_keywords if word in analysis.lower())
        inconsistent_count = sum(1 for word in inconsistency_keywords if word in analysis.lower())
        
        return consistent_count > inconsistent_count
    
    def _generate_reasoning(self, final_verdict: str) -> Dict:
        """
        Generate reasoning chain for verdict
        """
        # Map verdict to winner side
        if final_verdict == 'SUPPORT':
            winner = 'proponent'
        else:
            winner = 'opponent'
        
        winner_agent_name = self.debate_result['agents'][winner]
        
        # Extract main arguments
        proponent_args = self._extract_side_arguments('proponent')
        opponent_args = self._extract_side_arguments('opponent')
        
        # Get decision factors from judicial panel
        decision_factors = []
        
        # Add majority opinion
        decision_factors.append(f"Majority Opinion: {self.judge_result.get('majority_opinion', '')[:300]}...")
        
        # Add dissenting opinion if exists
        if self.judge_result.get('dissenting_opinion'):
            decision_factors.append(f"Dissenting Opinion: {self.judge_result['dissenting_opinion'][:200]}...")
        
        # Role-switch factor
        if self._check_role_switch_consistency():
            decision_factors.append("Role-switching demonstrated consistent argumentation")
        else:
            decision_factors.append("Role-switching revealed some inconsistencies")
        
        reasoning = {
            "winner": "plaintiff_counsel" if winner == 'proponent' else "defense_counsel",
            "winner_agent": winner_agent_name,
            "judicial_verdict": final_verdict,
            "main_arguments": {
                "plaintiff_counsel": proponent_args[0][:300] + "..." if proponent_args else "N/A",
                "defense_counsel": opponent_args[0][:300] + "..." if opponent_args else "N/A"
            },
            "decision_factors": decision_factors
        }
        
        return reasoning
    
    def _extract_key_evidence(self) -> list:
        """Extract key evidence cited in debate"""
        evidence_list = []
        
        # Get evidence from debate
        for round_data in self.debate_result.get('rounds', []):
            if 'new_evidence' in round_data and round_data['new_evidence']:
                for ev in round_data['new_evidence'][:2]:  # Top 2 per round
                    evidence_list.append({
                        "source_id": ev.get('source_id') or ev.get('id', 'unknown'),
                        "relevance": ev.get('relevance_score') or ev.get('relevance', 0),
                        "novelty": ev.get('novelty', 1.0),
                        "round": round_data.get('round_number')
                    })
        
        # Limit to top 5
        return evidence_list[:5]
    
    def _extract_side_arguments(self, side: str) -> list:
        """Extract all arguments and expert testimonies for one side"""
        arguments = []
        for round_data in self.debate_result.get('rounds', []):
            # Regular arguments
            for arg in round_data.get('arguments', []):
                if arg.get('role') == side:
                    arguments.append(arg.get('text', ''))
            # Expert testimonies
            if 'expert_testimonies' in round_data:
                for expert in round_data['expert_testimonies']:
                    if expert.get('requesting_side') == side:
                        arguments.append(f"[Expert Testimony Supporting {side.capitalize()}]: {expert.get('text', '')}")
        return arguments
    
    def _count_total_evidence(self) -> int:
        """Count total evidence items used"""
        evidence_ids = set()
        
        for round_data in self.debate_result.get('rounds', []):
            if 'new_evidence' in round_data:
                for ev in round_data['new_evidence']:
                    evidence_ids.add(ev.get('source_id') or ev.get('id', ''))
        
        return len(evidence_ids)
