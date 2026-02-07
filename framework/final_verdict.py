"""
Final Verdict Generator

Aggregates all evidence and generates confidence-weighted verdict with explainable output
"""

from typing import Dict
import json

class FinalVerdict:
    """
    Generates final verdict with confidence score and reasoning
    """
    
    def __init__(self, claim, debate_result: Dict, judge_result: Dict,
                 role_switch_result: Dict, reflection_result: Dict):
        """
        Initialize verdict generator
        
        Args:
            claim: Original Claim object
            debate_result: MAD debate transcript
            judge_result: Judge evaluation results
            role_switch_result: Role-switching consistency report
            reflection_result: Self-reflection results
        """
        self.claim = claim
        self.debate_result = debate_result
        self.judge_result = judge_result
        self.role_switch_result = role_switch_result
        self.reflection_result = reflection_result
    
    def generate_verdict(self) -> Dict:
        """
        Generate final verdict with confidence and reasoning
        
        Returns:
            Complete verdict with classification, confidence, and reasoning
        """
        print("\n" + "="*60)
        print("FINAL VERDICT GENERATION")
        print("="*60 + "\n")
        
        # Determine verdict based on provisional winner
        provisional_winner = self.judge_result['provisional_winner']
        verdict = "SUPPORT" if provisional_winner == "proponent" else "REFUTE"
        
        # Calculate confidence
        confidence = self._calculate_confidence()
        
        # Generate reasoning chain
        reasoning = self._generate_reasoning(provisional_winner)
        
        # Extract key evidence
        key_evidence = self._extract_key_evidence()
        
        # Get ground truth for comparison
        ground_truth = self.claim.metadata.get('label', 'UNKNOWN') if hasattr(self.claim, 'metadata') else 'UNKNOWN'
        correct = (verdict == ground_truth) if ground_truth != 'UNKNOWN' else None
        
        result = {
            "claim": self.claim.text,
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "ground_truth_label": ground_truth,
            "correct": correct,
            "reasoning": reasoning,
            "key_evidence": key_evidence,
            "metadata": {
                "judge_scores": self.judge_result['aggregate_scores'],
                "role_switch_consistent": self._check_role_switch_consistency(),
                "self_reflection_adjustment": self.reflection_result['self_reflection']['confidence_adjustment'],
                "debate_rounds": len(self.debate_result['rounds']),
                "total_evidence_used": self._count_total_evidence()
            }
        }
        
        # Save results
        with open("final_verdict.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Verdict: {verdict}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Correct: {correct}")
        
        return result
    
    def _calculate_confidence(self) -> float:
        """
        Calculate confidence score from multiple sources
        
        Returns:
            Confidence score between 0 and 1
        """
        # 1. Base confidence from score margin (Sigmoid-like)
        proponent_score = self.judge_result['aggregate_scores']['proponent']
        opponent_score = self.judge_result['aggregate_scores']['opponent']
        total_score = proponent_score + opponent_score
        
        if total_score > 0:
            # Margin determines specific confidence (0-1)
            # A margin of 10% (0.1) should give high confidence
            margin = abs(proponent_score - opponent_score) / total_score
            margin_score = margin * 2.0  # Boost margin impact
            margin_score = min(1.0, margin_score)
        else:
            margin_score = 0.0
            
        # 2. Quality confidence (High judge scores = better debate)
        # Max possible is ~30 per judge per side. 3 judges * 2 sides * 40 = 240
        # If total score > 150, it's a high quality debate
        quality_score = min(1.0, total_score / 200.0) * 0.3
        
        base_confidence = margin_score + quality_score
        
        # 3. Adjustments
        adjustments = 0.0
        
        # Role-switching consistency
        if self._check_role_switch_consistency():
            adjustments += 0.10
        else:
            # Soften penalty for inconsistency (it's hard to be consistent sometimes)
            adjustments -= 0.05
        
        # Self-reflection (limit negative impact)
        reflection_adj = self.reflection_result['self_reflection']['confidence_adjustment']
        # Don't let self-reflection tank the score completely, cap at -0.15
        if reflection_adj < 0:
            reflection_adj = max(-0.15, reflection_adj)
            
        adjustments += reflection_adj
        
        # Final calculation
        final_confidence = base_confidence + adjustments
        
        # Ensure minimal non-zero confidence if there is a winner
        if final_confidence < 0.1 and margin > 0.02:
            final_confidence = 0.1
            
        # Clamp to [0, 1]
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return final_confidence
    
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
    
    def _generate_reasoning(self, winner: str) -> Dict:
        """
        Generate reasoning chain for verdict
        """
        winner_agent_name = self.debate_result['agents'][winner]
        
        # Extract main arguments
        proponent_args = self._extract_side_arguments('proponent')
        opponent_args = self._extract_side_arguments('opponent')
        
        # Get decision factors
        decision_factors = []
        
        # Judge reasoning
        for judge in self.judge_result['judges']:
            if judge['winner'] == winner:
                decision_factors.append(f"{judge['judge_name']}: {judge['reasoning'][:200]}...")
        
        # Role-switch factor
        if self._check_role_switch_consistency():
            decision_factors.append("Role-switching demonstrated consistent argumentation")
        else:
            decision_factors.append("Role-switching revealed some inconsistencies")
        
        # Self-reflection factor
        reflection_adj = self.reflection_result['self_reflection']['confidence_adjustment']
        if reflection_adj < 0:
            decision_factors.append(f"Self-reflection acknowledged weaknesses (confidence adjusted by {reflection_adj:+.2f})")
        else:
            decision_factors.append(f"Self-reflection reinforced arguments (confidence adjusted by {reflection_adj:+.2f})")
        
        reasoning = {
            "winner": winner,
            "winner_agent": winner_agent_name,
            "main_arguments": {
                "proponent": proponent_args[0][:300] + "..." if proponent_args else "N/A",
                "opponent": opponent_args[0][:300] + "..." if opponent_args else "N/A"
            },
            "decision_factors": decision_factors
        }
        
        return reasoning
    
    def _extract_key_evidence(self) -> list:
        """Extract key evidence cited in debate"""
        evidence_list = []
        
        # Get evidence from debate
        for round_data in self.debate_result['rounds']:
            if 'new_evidence' in round_data and round_data['new_evidence']:
                for ev in round_data['new_evidence'][:2]:  # Top 2 per round
                    evidence_list.append({
                        "source_id": ev.get('source_id', 'unknown'),
                        "relevance": ev.get('relevance_score', 0),
                        "round": round_data['round_number']
                    })
        
        # Limit to top 5
        return evidence_list[:5]
    
    def _extract_side_arguments(self, side: str) -> list:
        """Extract all arguments and expert testimonies for one side"""
        arguments = []
        for round_data in self.debate_result['rounds']:
            # Regular arguments
            for arg in round_data['arguments']:
                if arg['role'] == side:
                    arguments.append(arg['text'])
            # Expert testimonies
            if 'expert_testimonies' in round_data:
                for expert in round_data['expert_testimonies']:
                    if expert.get('requesting_side') == side:
                        arguments.append(f"[Expert Testimony Supporting {side.capitalize()}]: {expert['text']}")
        return arguments
    
    def _count_total_evidence(self) -> int:
        """Count total evidence items used"""
        evidence_ids = set()
        
        for round_data in self.debate_result['rounds']:
            if 'new_evidence' in round_data:
                for ev in round_data['new_evidence']:
                    evidence_ids.add(ev.get('source_id', ''))
        
        return len(evidence_ids)
