"""
Role-Switching Mechanism for Consistency Testing

Swaps Plaintiff Counsel and Defense Counsel roles to test argument consistency
"""

import json
from typing import Dict
from mad_orchestrator import MADOrchestrator
from llm_client import GeminiLLMClient
import os

class RoleSwitcher:
    """
    Manages role-switching and consistency analysis
    """
    def __init__(self, mad_orchestrator: MADOrchestrator):
        """
        Initialize role switcher
        
        Args:
            mad_orchestrator: Original MAD orchestrator instance
        """
        self.original_mad = mad_orchestrator
        self.switched_mad = None
        
    def switch_roles(self, max_rounds: int = 3) -> Dict:
        """
        Swap Plaintiff Counsel ↔ Defense Counsel roles and re-run debate
        
        Args:
            max_rounds: Number of rounds for switched debate
            
        Returns:
            Switched debate result
        """
        print("\n" + "="*60)
        print("ROLE-SWITCHING ROUND")
        print("="*60)
        print("Swapping Plaintiff Counsel ↔ Defense Counsel roles...")
        print()
        
        # Get original agents
        original_proponent = self.original_mad.agents['proponent']
        original_opponent = self.original_mad.agents['opponent']
        
        # Swap roles
        self.original_mad.agents['proponent'] = original_opponent
        self.original_mad.agents['opponent'] = original_proponent
        
        # Update role attributes and branding
        self.original_mad.agents['proponent'].role = 'proponent'
        self.original_mad.agents['proponent'].job_title = "Plaintiff Counsel"
        self.original_mad.agents['proponent'].name = "Plaintiff Counsel"
        
        self.original_mad.agents['opponent'].role = 'opponent'
        self.original_mad.agents['opponent'].job_title = "Defense Counsel"
        self.original_mad.agents['opponent'].name = "Defense Counsel"
        
        # Reset debate state
        self.original_mad.debate_transcript = []
        self.original_mad.current_round = 0
        self.original_mad.prag.round_counter = 0
        
        # Run switched debate
        switched_result = self.original_mad.run_full_debate(max_rounds=max_rounds)
        
        # Save switched transcript
        try:
            from logging_extension import append_framework_json
            append_framework_json("debate_transcript_switched.jsonl", "unknown", switched_result)
        except ImportError:
            with open("debate_transcript_switched.json", "w") as f:
                json.dump(switched_result, f, indent=2)
        
        return switched_result
    
    def check_consistency(self, original_transcript: Dict, switched_transcript: Dict) -> Dict:
        """
        Analyze consistency between original and switched debates
        
        Args:
            original_transcript: Original debate result
            switched_transcript: Switched debate result
            
        Returns:
            Consistency analysis report
        """
        print("\n" + "="*60)
        print("CONSISTENCY ANALYSIS")
        print("="*60 + "\n")
        
        # Use OpenRouter for consistency analysis
        from personas import create_llm_client
        analyzer = create_llm_client({
            "llm_provider": "openrouter",
            "llm_model": "deepseek/deepseek-chat",
            "temperature": 0.3,
            "system_prompt": "You are an expert in logical consistency analysis and argumentation theory.",
            "name": "Consistency Analyzer",
            "role": "Consistency Analyzer",
            "expertise": ["logic", "argumentation"]
        })
        
        # Extract key arguments from both debates
        original_pro_args = [
            arg['text'] for round_data in original_transcript['rounds']
            for arg in round_data['arguments']
            if arg['role'] == 'proponent'
        ]
        
        original_opp_args = [
            arg['text'] for round_data in original_transcript['rounds']
            for arg in round_data['arguments']
            if arg['role'] == 'opponent'
        ]
        
        switched_pro_args = [
            arg['text'] for round_data in switched_transcript['rounds']
            for arg in round_data['arguments']
            if arg['role'] == 'proponent'
        ]
        
        switched_opp_args = [
            arg['text'] for round_data in switched_transcript['rounds']
            for arg in round_data['arguments']
            if arg['role'] == 'opponent'
        ]
        
        # Analyze consistency
        prompt = f"""Analyze the logical consistency of arguments when agents switch roles.

ORIGINAL PROCEEDINGS:
Plaintiff Counsel (Agent A) Arguments:
{chr(10).join(original_pro_args[:2])}

Defense Counsel (Agent B) Arguments:
{chr(10).join(original_opp_args[:2])}

SWITCHED PROCEEDINGS (Roles Swapped):
Plaintiff Counsel (Agent B - formerly Defense) Arguments:
{chr(10).join(switched_pro_args[:2])}

Defense Counsel (Agent A - formerly Plaintiff) Arguments:
{chr(10).join(switched_opp_args[:2])}

Analyze:
1. Does Agent A maintain logical consistency when switching from Plaintiff Counsel to Defense Counsel?
2. Does Agent B maintain logical consistency when switching from Defense Counsel to Plaintiff Counsel?
3. Are there contradictions in their arguments?
4. Overall consistency score (0-10)

Provide detailed analysis:"""
        
        analysis = analyzer.generate(prompt)
        
        consistency_report = {
            "claim": original_transcript['claim'],
            "original_agents": original_transcript['agents'],
            "switched_agents": switched_transcript['agents'],
            "analysis": analysis,
            "original_rounds": len(original_transcript['rounds']),
            "switched_rounds": len(switched_transcript['rounds'])
        }
        
        # Save report
        try:
            from logging_extension import append_framework_json
            append_framework_json("role_switch_report.jsonl", "unknown", consistency_report)
        except ImportError:
            with open("role_switch_report.json", "w") as f:
                json.dump(consistency_report, f, indent=2)
        
        print(f"Consistency Analysis:\n{analysis}\n")
        
        return consistency_report
