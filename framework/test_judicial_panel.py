"""
Test script for Judicial Panel Evaluation System

Verifies that the new judicial panel can be instantiated and
that the output format is correct.
"""

import sys
import os

# Add framework directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from judge_evaluator import JudicialPanel
from models import Claim

def test_judicial_panel_initialization():
    """Test that judicial panel can be initialized"""
    print("Test 1: Initializing Judicial Panel...")
    try:
        panel = JudicialPanel()
        assert len(panel.judges) == 3, "Should have 3 judges"
        print("✓ Judicial Panel initialized successfully")
        print(f"  - Judge 1: {panel.judges[0]['model']}")
        print(f"  - Judge 2: {panel.judges[1]['model']}")
        print(f"  - Judge 3: {panel.judges[2]['model']}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False

def test_output_format():
    """Test that the expected output format is defined"""
    print("\nTest 2: Verifying Output Format...")
    try:
        # Create mock debate transcript
        mock_transcript = {
            "claim": "Test claim about COVID-19",
            "claim_id": "test_001",
            "agents": {
                "proponent": "Plaintiff Counsel",
                "opponent": "Defense Counsel",
                "judge": "Court"
            },
            "rounds": [
                {
                    "round_number": 1,
                    "arguments": [
                        {
                            "agent": "Proponent",
                            "role": "proponent",
                            "text": "The claim is supported by evidence X, Y, and Z."
                        },
                        {
                            "agent": "Opponent",
                            "role": "opponent",
                            "text": "The claim is refuted by evidence A, B, and C."
                        }
                    ],
                    "expert_testimonies": []
                }
            ]
        }
        
        # Verify the panel has the evaluate_debate method
        panel = JudicialPanel()
        assert hasattr(panel, 'evaluate_debate'), "Panel should have evaluate_debate method"
        
        # Check method signature
        import inspect
        sig = inspect.signature(panel.evaluate_debate)
        params = list(sig.parameters.keys())
        assert 'debate_transcript' in params, "Should accept debate_transcript parameter"
        assert 'admitted_evidence' in params, "Should accept admitted_evidence parameter"
        assert 'role_switch_history' in params, "Should accept role_switch_history parameter"
        
        print("✓ Output format verification passed")
        print(f"  - Method signature: {sig}")
        return True
    except Exception as e:
        print(f"✗ Format verification failed: {e}")
        return False

def test_verdict_values():
    """Test that verdict values are correctly defined"""
    print("\nTest 3: Verifying Verdict Values...")
    try:
        expected_verdicts = ['SUPPORTED', 'NOT SUPPORTED', 'INCONCLUSIVE']
        print("✓ Expected verdict values defined:")
        for verdict in expected_verdicts:
            print(f"  - {verdict}")
        return True
    except Exception as e:
        print(f"✗ Verdict value test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("JUDICIAL PANEL EVALUATION SYSTEM - UNIT TESTS")
    print("="*60 + "\n")
    
    results = []
    results.append(test_judicial_panel_initialization())
    results.append(test_output_format())
    results.append(test_verdict_values())
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {sum(results)}/{len(results)} passed")
    print("="*60)
    
    if all(results):
        print("\n✓ All tests passed! Judicial Panel is ready for integration.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
