"""
Self-Reflection Module

Winner performs self-critique and re-evaluates their arguments
"""

from typing import Dict, List
import json

class SelfReflection:
    """
    Enables agents to critically review their own arguments periodically
    """
    
    def __init__(self, transcript_ref: List[Dict]):
        """
        Initialize self-reflection
        
        Args:
            transcript_ref: Reference to the debate transcript list
        """
        self.debate_transcript = transcript_ref
        self.reflection_history = []
    
    def perform_round_reflection(self, agent, side: str, round_num: int, claim: str) -> Dict:
        """
        Perform self-critique for a specific agent after a round
        
        Returns:
            Dictionary with multi-dimensional scores and discovery needs
        """
        print(f"   > [{agent.name}] Performing self-reflection for Phase {round_num}...")
        
        # Extract context
        my_args = [a['text'] for a in self.debate_transcript if a.get('role') == side]
        opponent_side = "opponent" if side == "proponent" else "proponent"
        opponent_args = [a['text'] for a in self.debate_transcript if a.get('role') == opponent_side]
        
        display_side = "Plaintiff" if side == "proponent" else "Defense"
        opp_display = "Defense" if side == "proponent" else "Plaintiff"

        prompt = f"""You are the {agent.job_title} ({display_side} Counsel). 
        You have just completed Phase {round_num} of the proceedings.
        
        CLAIM: {claim}
        
        YOUR ARGUMENTS SO FAR:
        {" ".join(my_args[-2:])}
        
        {opp_display.upper()} COUNSEL'S CHALLENGES:
        {" ".join(opponent_args[-2:]) if opponent_args else "No challenges yet."}
        
        Perform a strictly professional self-audit:
        1. Logical Coherence: Evaluate the flow and structural integrity of your arguments.
        2. Evidence Novelty: Have you introduced truly new information or just repeated old points?
        3. Rebuttal Coverage: How effectively did you address the {opp_display.lower()} counsel's latest points?
        
        Identify:
        - Critical gaps in your current evidence base.
        - Premises you haven't sufficiently supported.
        
        Respond ONLY in valid JSON format:
        {{
            "scores": {{
                "logic": 0.0-1.0,
                "novelty": 0.0-1.0,
                "rebuttal": 0.0-1.0
            }},
            "flaws_identified": ["...", "..."],
            "discovery_need": "Specific evidence lookup query to fill a gap (1 sentence)",
            "refined_stance": "Summary of your improved position"
        }}"""
        
        response = agent.llm.generate(prompt)
        try:
            import json
            import re
            match = re.search(r'\{[\s\S]*\}', response)
            reflection_data = json.loads(match.group()) if match else {}
            
            # Weighted score calculation
            s = reflection_data.get("scores", {})
            logic = float(s.get("logic", 0.5))
            novelty = float(s.get("novelty", 0.5))
            rebuttal = float(s.get("rebuttal", 0.5))
            total_score = (0.4 * logic) + (0.3 * novelty) + (0.3 * rebuttal)
            
            reflection_data["total_score"] = round(total_score, 3)
            reflection_data["side"] = side
            reflection_data["round"] = round_num
            
            # Add legacy structure compatibility for FinalVerdict
            # (total_score 0.5 -> 0, 1.0 -> +0.3, 0.0 -> -0.3)
            reflection_data["self_reflection"] = {
                "confidence_adjustment": round((total_score - 0.5) * 0.6, 3)
            }
            
            self.reflection_history.append(reflection_data)
            
            # Print summary to console for visibility in logs
            print(f"     * Logic: {logic:.2f}, Novelty: {novelty:.2f}, Rebuttal: {rebuttal:.2f}")
            print(f"     * Total Score: {total_score:.3f}")
            if reflection_data.get("discovery_need"):
                print(f"     * Discovery Need: {reflection_data['discovery_need']}")
                
            return reflection_data
        except Exception as e:
            print(f"   > [Warning] Reflection parsing failed: {e}")
            return {
                "scores": {"logic": 0.5, "novelty": 0.5, "rebuttal": 0.5},
                "total_score": 0.5,
                "discovery_need": "",
                "side": side,
                "round": round_num
            }

    def save_reflection_history(self, claim_id: str = "unknown", filename: str = "self_reflection.json"):
        """Save history to disk"""
        import json
        try:
            from logging_extension import append_framework_json
            append_framework_json(filename.replace('.json', '.jsonl'), claim_id, self.reflection_history)
        except ImportError:
            with open(filename, "w") as f:
                json.dump(self.reflection_history, f, indent=2)
