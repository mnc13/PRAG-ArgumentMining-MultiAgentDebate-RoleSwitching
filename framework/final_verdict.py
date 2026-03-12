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
        self.claim = claim
        self.debate_result = debate_result
        self.judge_result = judge_result
        self.role_switch_result = role_switch_result
        self.reflection_result = reflection_result

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_source_reliability(verdict_dict: Dict) -> float:
        """
        Read the source/scientific reliability score regardless of which key
        name was used.  judge_evaluator v1 wrote 'scientific_reliability';
        v2 writes 'source_reliability'.  Both are accepted here so that
        already-saved judge results don't break the pipeline.
        """
        return verdict_dict.get(
            'source_reliability',
            verdict_dict.get('scientific_reliability', 5)   # default 5 if neither key exists
        )

    # ─────────────────────────────────────────────────────────────────

    def generate_verdict(self) -> Dict:
        print("\n" + "="*60)
        print("FINAL VERDICT GENERATION")
        print("="*60 + "\n")
        
        final_judicial_verdict = self.judge_result['final_verdict']
        if final_judicial_verdict == 'SUPPORTED':
            verdict = "SUPPORT"
        elif final_judicial_verdict == 'NOT SUPPORTED':
            verdict = "REFUTE"
        else:
            verdict = "INCONCLUSIVE"
        
        confidence  = self._calculate_confidence()
        reasoning   = self._generate_reasoning(final_judicial_verdict)
        key_evidence= self._extract_key_evidence()
        
        ground_truth = (self.claim.metadata.get('label', 'UNKNOWN')
                        if hasattr(self.claim, 'metadata') else 'UNKNOWN')
        correct = (verdict == ground_truth) if ground_truth != 'UNKNOWN' else None
        
        result = {
            "claim":              self.claim.text,
            "verdict":            verdict,
            "confidence":         round(confidence, 3),
            "ground_truth_label": ground_truth,
            "correct":            correct,
            "reasoning":          reasoning,
            "key_evidence":       key_evidence,
            "metadata": {
                "judicial_verdict":          self.judge_result['final_verdict'],
                "vote_breakdown":            self.judge_result['vote_breakdown'],
                "role_switch_consistent":    self._check_role_switch_consistency(),
                "self_reflection_adjustment":self.reflection_result['self_reflection']['confidence_adjustment'],
                "debate_rounds":             len(self.debate_result['rounds']),
                "total_evidence_used":       self._count_total_evidence()
            }
        }
        
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
        vote_breakdown = self.judge_result['vote_breakdown']
        total_votes    = sum(vote_breakdown.values())
        
        if total_votes > 0:
            final_verdict   = self.judge_result['final_verdict']
            winning_votes   = vote_breakdown.get(final_verdict, 0)
            consensus_strength = winning_votes / total_votes
            margin_score    = consensus_strength * 0.8
        else:
            consensus_strength = 0.0
            margin_score       = 0.0
            
        verdicts = self.judge_result['judge_verdicts']
        avg_evidence_strength    = sum(v['evidence_strength']    for v in verdicts) / len(verdicts)
        avg_argument_validity    = sum(v['argument_validity']    for v in verdicts) / len(verdicts)
        avg_source_reliability   = sum(self._get_source_reliability(v) for v in verdicts) / len(verdicts)
        
        quality_score  = ((avg_evidence_strength + avg_argument_validity + avg_source_reliability) / 30) * 0.3
        base_confidence = margin_score + quality_score
        
        adjustments = 0.0
        
        is_consistent    = self._check_role_switch_consistency()
        consistency_score= getattr(self, "consistency_score", 5)
        
        if consistency_score >= 7:
            rs_adj = 0.10
        elif consistency_score >= 5:
            rs_adj = 0.0
        else:
            rs_adj = -0.05
        adjustments += rs_adj
        
        print(f"[ROLE SWITCH] consistency_score={consistency_score}/10 | "
              f"is_consistent={is_consistent} | adj={rs_adj:+.2f}")
        
        sr_data       = self.reflection_result.get('self_reflection', {})
        reflection_adj= sr_data.get('confidence_adjustment', 0.0)
        if reflection_adj < 0:
            reflection_adj = max(-0.15, reflection_adj)
        adjustments += reflection_adj
        
        final_confidence = base_confidence + adjustments
        
        if final_confidence < 0.1 and consensus_strength > 0.5:
            final_confidence = 0.1
            
        return max(0.0, min(1.0, final_confidence))
    
    def _check_role_switch_consistency(self) -> bool:
        if ('is_consistent' in self.role_switch_result
                and 'consistency_score' in self.role_switch_result):
            self.consistency_score = self.role_switch_result['consistency_score']
            return self.role_switch_result['is_consistent']
            
        analysis = self.role_switch_result.get('analysis', '')
        raw_json = str(analysis).strip()
        for prefix in ("```json", "```"):
            if raw_json.startswith(prefix):
                raw_json = raw_json[len(prefix):]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
            
        try:
            parsed = analysis if isinstance(analysis, dict) else json.loads(raw_json)
            self.consistency_score = parsed.get("consistency_score", 5)
            return parsed.get("is_consistent", False)
        except Exception as e:
            print(f"[ROLE SWITCH] Warning: Failed to parse consistency JSON: {e}")
            self.consistency_score = 5
            return False
    
    def _generate_reasoning(self, final_verdict: str) -> Dict:
        if final_verdict == 'SUPPORTED':
            winner = 'proponent'
        elif final_verdict == 'NOT SUPPORTED':
            winner = 'opponent'
        else:
            winner = 'proponent'
        
        winner_agent_name = self.debate_result['agents'][winner]
        proponent_args    = self._extract_side_arguments('proponent')
        opponent_args     = self._extract_side_arguments('opponent')
        
        decision_factors = []
        decision_factors.append(
            f"Majority Opinion: {self.judge_result['majority_opinion'][:300]}...")
        if self.judge_result['dissenting_opinion']:
            decision_factors.append(
                f"Dissenting Opinion: {self.judge_result['dissenting_opinion'][:200]}...")
        
        if self._check_role_switch_consistency():
            decision_factors.append("Role-switching demonstrated consistent argumentation")
        else:
            decision_factors.append("Role-switching revealed some inconsistencies")
        
        sr_data        = self.reflection_result.get('self_reflection', {})
        reflection_adj = sr_data.get('confidence_adjustment', 0.0)
        if reflection_adj < 0:
            decision_factors.append(
                f"Self-reflection acknowledged weaknesses (confidence adjusted by {reflection_adj:+.2f})")
        else:
            decision_factors.append(
                f"Self-reflection reinforced arguments (confidence adjusted by {reflection_adj:+.2f})")
        
        return {
            "winner":         "plaintiff_counsel" if winner == 'proponent' else "defense_counsel",
            "winner_agent":   winner_agent_name,
            "judicial_verdict": final_verdict,
            "main_arguments": {
                "plaintiff_counsel": (proponent_args[0][:300] + "...") if proponent_args else "N/A",
                "defense_counsel":   (opponent_args[0][:300] + "...") if opponent_args else "N/A",
            },
            "decision_factors": decision_factors
        }
    
    def _extract_key_evidence(self) -> list:
        evidence_list = []
        for round_data in self.debate_result['rounds']:
            for ev in round_data.get('new_evidence', [])[:2]:
                evidence_list.append({
                    "source_id": ev.get('source_id') or ev.get('id', 'unknown'),
                    "relevance": ev.get('relevance_score') or ev.get('relevance', 0),
                    "novelty":   ev.get('novelty', 1.0),
                    "round":     round_data['round_number']
                })
        return evidence_list[:5]
    
    def _extract_side_arguments(self, side: str) -> list:
        arguments = []
        for round_data in self.debate_result['rounds']:
            for arg in round_data['arguments']:
                if arg['role'] == side:
                    arguments.append(arg['text'])
            for expert in round_data.get('expert_testimonies', []):
                if expert.get('requesting_side') == side:
                    arguments.append(
                        f"[Expert Testimony Supporting {side.capitalize()}]: {expert['text']}")
        return arguments
    
    def _count_total_evidence(self) -> int:
        evidence_ids = set()
        for round_data in self.debate_result['rounds']:
            for ev in round_data.get('new_evidence', []):
                evidence_ids.add(ev.get('source_id') or ev.get('id', ''))
        return len(evidence_ids)