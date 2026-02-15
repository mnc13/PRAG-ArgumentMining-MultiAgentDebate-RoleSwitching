
import unittest
import sys
import os
from unittest.mock import MagicMock

# Ensure we can import framework modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'framework'))

from final_verdict import FinalVerdict

class MockClaim:
    def __init__(self, text, id="123", metadata=None):
        self.text = text
        self.id = id
        self.metadata = metadata or {}

class TestFinalVerdictIntegration(unittest.TestCase):
    
    def test_generate_verdict_integration(self):
        # 1. Setup Mock Data
        claim = MockClaim("Masks are effective.")
        
        debate_result = {
            'rounds': [],
            'agents': {'proponent': 'PropAgent', 'opponent': 'OppAgent'}
        }
        
        judge_result = {
            'final_verdict': 'SUPPORTED',
            'majority_opinion': 'Majority says supported.',
            'dissenting_opinion': None,
            'vote_breakdown': {'SUPPORTED': 3},
            'judge_verdicts': [
                {'judge_name': 'J1', 'model': 'm1', 'verdict': 'SUPPORTED', 'evidence_strength': 9, 'argument_validity': 9, 'scientific_reliability': 9},
                {'judge_name': 'J2', 'model': 'm2', 'verdict': 'SUPPORTED', 'evidence_strength': 8, 'argument_validity': 8, 'scientific_reliability': 8},
                {'judge_name': 'J3', 'model': 'm3', 'verdict': 'SUPPORTED', 'evidence_strength': 9, 'argument_validity': 9, 'scientific_reliability': 9}
            ]
        }
        
        # P_neg = 0.8
        negotiation_metrics = {
            'probability': 0.8,
            'claim_score': 2.5
        }
        
        role_switch_result = {'analysis': 'Consistent arguments maintained.'}
        reflection_result = {'self_reflection': {'confidence_adjustment': 0.05}}
        
        # 2. Instantiate FinalVerdict
        fv = FinalVerdict(claim, debate_result, judge_result, role_switch_result, reflection_result, negotiation_metrics)
        
        # 3. Generate Verdict
        result = fv.generate_verdict()
        
        # 4. Assertions
        print(f"\nIntegration Test Result:")
        print(f"Final Probability: {result['final_probability_support']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Verdict: {result['verdict']}")
        
        self.assertEqual(result['verdict'], 'SUPPORT')
        self.assertGreater(result['final_probability_support'], 0.95)
        self.assertIn('negotiation_log_odds', result)
        self.assertIn('judge_individual', result)
        self.assertEqual(len(result['judge_individual']), 3)

if __name__ == '__main__':
    unittest.main()
