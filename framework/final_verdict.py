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
        
        # Determine verdict based on judicial panel final verdict
        final_judicial_verdict = self.judge_result['final_verdict']
        if final_judicial_verdict == 'SUPPORTED':
            verdict = "SUPPORT"
        elif final_judicial_verdict == 'NOT SUPPORTED':
            verdict = "REFUTE"
        else:  # INCONCLUSIVE
            # Keep as INCONCLUSIVE for threshold-based downstream evaluation
            verdict = "INCONCLUSIVE"
        
        # Calculate confidence
        confidence = self._calculate_confidence()
        
        # Generate reasoning chain
        reasoning = self._generate_reasoning(final_judicial_verdict)
        
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
                "judicial_verdict": self.judge_result['final_verdict'],
                "vote_breakdown": self.judge_result['vote_breakdown'],
                "role_switch_consistent": self._check_role_switch_consistency(),
                "self_reflection_adjustment": self.reflection_result['self_reflection']['confidence_adjustment'],
                "debate_rounds": len(self.debate_result['rounds']),
                "total_evidence_used": self._count_total_evidence()
            }
        }
        
        # Save results
        # Save results
        import json
        try:
            from logging_extension import append_framework_json
            append_framework_json("final_verdict.jsonl", self.claim, result)
        except ImportError:
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
        # 1. Base confidence from vote consensus
        vote_breakdown = self.judge_result['vote_breakdown']
        total_votes = sum(vote_breakdown.values())
        
        if total_votes > 0:
            # Get the winning verdict vote count
            final_verdict = self.judge_result['final_verdict']
            winning_votes = vote_breakdown.get(final_verdict, 0)
            
            # Consensus strength (3-0 = 1.0, 2-1 = 0.67, 1-1-1 = 0.33)
            consensus_strength = winning_votes / total_votes
            
            # Boost consensus impact
            margin_score = consensus_strength * 0.8  # Max 0.8 from consensus
        else:
            margin_score = 0.0
            
        # 2. Quality confidence from judge scores
        # Average the scores across all judges
        avg_evidence_strength = sum(v['evidence_strength'] for v in self.judge_result['judge_verdicts']) / len(self.judge_result['judge_verdicts'])
        avg_argument_validity = sum(v['argument_validity'] for v in self.judge_result['judge_verdicts']) / len(self.judge_result['judge_verdicts'])
        avg_scientific_reliability = sum(v['scientific_reliability'] for v in self.judge_result['judge_verdicts']) / len(self.judge_result['judge_verdicts'])
        
        # Normalize to 0-1 (scores are 0-10)
        quality_score = ((avg_evidence_strength + avg_argument_validity + avg_scientific_reliability) / 30) * 0.3
        
        base_confidence = margin_score + quality_score
        
        # 3. Adjustments
        adjustments = 0.0
        
        # Role-switching consistency
        is_consistent = self._check_role_switch_consistency()
        consistency_score = getattr(self, "consistency_score", 5)
        
        if consistency_score >= 7:
            rs_adj = 0.10
        elif consistency_score >= 5:
            rs_adj = 0.0
        else:
            rs_adj = -0.05
            
        adjustments += rs_adj
        
        print(f"[ROLE SWITCH] consistency_score={consistency_score}/10 | is_consistent={is_consistent} | adj={rs_adj:+.2f}")
        
        # Self-reflection (limit negative impact)
        # Defensive access to handle integrated multi-round reflection structure
        sr_data = self.reflection_result.get('self_reflection', {})
        reflection_adj = sr_data.get('confidence_adjustment', 0.0)
        
        # Don't let self-reflection tank the score completely, cap at -0.15
        if reflection_adj < 0:
            reflection_adj = max(-0.15, reflection_adj)
            
        adjustments += reflection_adj
        
        # Final calculation
        final_confidence = base_confidence + adjustments
        
        # Ensure minimal non-zero confidence if there is consensus
        if final_confidence < 0.1 and consensus_strength > 0.5:
            final_confidence = 0.1
            
        # Clamp to [0, 1]
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return final_confidence
    
    def _check_role_switch_consistency(self) -> bool:
        """
        Check if role-switching showed consistency and store the score.
        
        Returns:
            True if consistent, False otherwise
        """
        # If standard structured keys exist from the new json format
        if 'is_consistent' in self.role_switch_result and 'consistency_score' in self.role_switch_result:
            self.consistency_score = self.role_switch_result['consistency_score']
            return self.role_switch_result['is_consistent']
            
        # Fallback for older formats where analysis might just be a string that we need to parse
        analysis = self.role_switch_result.get('analysis', '')
        
        import json
        raw_json = str(analysis).strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:]
        elif raw_json.startswith("```"):
            raw_json = raw_json[3:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
            
        try:
            if isinstance(analysis, dict):
                parsed = analysis
            else:
                parsed = json.loads(raw_json)
                
            self.consistency_score = parsed.get("consistency_score", 5)
            return parsed.get("is_consistent", False)
        except Exception as e:
            # Fallback if old unstructured text or JSON parse fails
            print(f"[ROLE SWITCH] Warning: Failed to parse consistency JSON: {e}")
            self.consistency_score = 5
            return False
    
    def _generate_reasoning(self, final_verdict: str) -> Dict:
        """
        Generate reasoning chain for verdict
        """
        # Map verdict to winner side
        if final_verdict == 'SUPPORTED':
            winner = 'proponent'
        elif final_verdict == 'NOT SUPPORTED':
            winner = 'opponent'
        else:
            winner = 'proponent'  # Default for INCONCLUSIVE (Plaintiff Counsel)
        
        winner_agent_name = self.debate_result['agents'][winner]
        
        # Extract main arguments
        proponent_args = self._extract_side_arguments('proponent')
        opponent_args = self._extract_side_arguments('opponent')
        
        # Get decision factors from judicial panel
        decision_factors = []
        
        # Add majority opinion
        decision_factors.append(f"Majority Opinion: {self.judge_result['majority_opinion'][:300]}...")
        
        # Add dissenting opinion if exists
        if self.judge_result['dissenting_opinion']:
            decision_factors.append(f"Dissenting Opinion: {self.judge_result['dissenting_opinion'][:200]}...")
        
        # Role-switch factor
        if self._check_role_switch_consistency():
            decision_factors.append("Role-switching demonstrated consistent argumentation")
        else:
            decision_factors.append("Role-switching revealed some inconsistencies")
        
        # Self-reflection factor
        sr_data = self.reflection_result.get('self_reflection', {})
        reflection_adj = sr_data.get('confidence_adjustment', 0.0)
        
        if reflection_adj < 0:
            decision_factors.append(f"Self-reflection acknowledged weaknesses (confidence adjusted by {reflection_adj:+.2f})")
        else:
            decision_factors.append(f"Self-reflection reinforced arguments (confidence adjusted by {reflection_adj:+.2f})")
        
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
        for round_data in self.debate_result['rounds']:
            if 'new_evidence' in round_data and round_data['new_evidence']:
                for ev in round_data['new_evidence'][:2]:  # Top 2 per round
                    evidence_list.append({
                        "source_id": ev.get('source_id') or ev.get('id', 'unknown'),
                        "relevance": ev.get('relevance_score') or ev.get('relevance', 0),
                        "novelty": ev.get('novelty', 1.0),
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
                    evidence_ids.add(ev.get('source_id') or ev.get('id', ''))
        
        return len(evidence_ids)
