"""
Self-Reflection Module

Winner performs self-critique and re-evaluates their arguments
"""

from typing import Dict, List
import json

class SelfReflection:
    """
    Enables winner to critically review their own arguments
    """
    
    def __init__(self, winner_side: str, winner_agent, debate_transcript: Dict):
        """
        Initialize self-reflection
        
        Args:
            winner_side: "proponent" or "opponent"
            winner_agent: The DebateAgent who won
            debate_transcript: Full debate transcript
        """
        self.winner_side = winner_side
        self.winner_agent = winner_agent
        self.debate_transcript = debate_transcript
    
    def perform_reflection(self) -> Dict:
        """
        Winner performs self-critique
        
        Returns:
            Reflection results with identified flaws and corrected stance
        """
        print("\n" + "="*60)
        print("SELF-REFLECTION ROUND")
        print("="*60)
        print(f"Winner: {self.winner_agent.name} ({self.winner_side})")
        print("Performing self-critique...\n")
        
        # Extract winner's arguments
        winner_args = self._extract_side_arguments(self.winner_side)
        
        # Extract opponent's critiques
        opponent_side = "opponent" if self.winner_side == "proponent" else "proponent"
        opponent_critiques = self._extract_critiques(opponent_side)
        
        # Generate self-reflection prompt
        prompt = f"""You previously argued {self._get_stance_text()} the following claim:

CLAIM: {self.debate_transcript['claim']}

YOUR ARGUMENTS:
{self._format_arguments(winner_args)}

OPPONENT'S CRITIQUES:
{self._format_arguments(opponent_critiques)}

Now, critically review your own arguments with complete honesty:

1. **Identify Logical Flaws**: Are there any weaknesses in your reasoning?
2. **Acknowledge Valid Opponent Points**: Which of the opponent's critiques are legitimate?
3. **Evidence Misinterpretations**: Did you misinterpret any evidence?
4. **Corrected Stance**: Based on this reflection, what is your refined position?
5. **Confidence Adjustment**: Should your confidence increase or decrease? By how much? (provide a number between -0.3 and +0.3)

Provide a thorough, honest self-critique:"""
        
        # Get reflection from winner's LLM
        reflection_text = self.winner_agent.llm.generate(prompt)
        
        # Parse confidence adjustment
        confidence_adjustment = self._extract_confidence_adjustment(reflection_text)
        
        result = {
            "winner": self.winner_side,
            "winner_agent": self.winner_agent.name,
            "winner_persona": self.winner_agent.persona_key,
            "original_arguments": winner_args[:3],  # First 3 for brevity
            "opponent_critiques": opponent_critiques[:2],
            "self_reflection": {
                "full_text": reflection_text,
                "confidence_adjustment": confidence_adjustment
            }
        }
        
        # Save results
        with open("self_reflection.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Self-reflection complete")
        print(f"Confidence adjustment: {confidence_adjustment:+.2f}")
        
        return result
    
    def _extract_side_arguments(self, side: str) -> List[str]:
        """Extract all arguments from one side"""
        arguments = []
        for round_data in self.debate_transcript['rounds']:
            for arg in round_data['arguments']:
                if arg['role'] == side:
                    arguments.append(arg['text'])
        return arguments
    
    def _extract_critiques(self, side: str) -> List[str]:
        """Extract critiques from opponent"""
        critiques = []
        for round_data in self.debate_transcript['rounds']:
            if 'critiques' in round_data:
                for critique in round_data['critiques']:
                    # Critiques are from the critic, but we want opponent's counter-arguments
                    pass
            # Use opponent's arguments as implicit critiques
            for arg in round_data['arguments']:
                if arg['role'] == side:
                    critiques.append(arg['text'])
        return critiques
    
    def _get_stance_text(self) -> str:
        """Get stance text (FOR or AGAINST)"""
        return "FOR" if self.winner_side == "proponent" else "AGAINST"
    
    def _format_arguments(self, args: List[str]) -> str:
        """Format arguments for prompt"""
        formatted = []
        for i, arg in enumerate(args[:3], 1):  # Limit to 3 to avoid token limits
            formatted.append(f"Argument {i}:\n{arg[:800]}...")  # Truncate long arguments
        return "\n\n".join(formatted)
    
    def _extract_confidence_adjustment(self, reflection_text: str) -> float:
        """
        Extract confidence adjustment from reflection text
        
        Returns:
            Float between -0.3 and +0.3
        """
        import re
        
        # Look for patterns like "decrease by 0.1", "+0.15", "-0.2", etc.
        patterns = [
            r'[-+]?\d*\.?\d+',  # Any number
            r'decrease.*?(\d*\.?\d+)',
            r'increase.*?(\d*\.?\d+)',
            r'adjust.*?by.*?([-+]?\d*\.?\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reflection_text.lower())
            if matches:
                try:
                    # Get first number found
                    num_str = matches[0] if isinstance(matches[0], str) else str(matches[0])
                    adjustment = float(num_str)
                    
                    # Check if text mentions "decrease" or "lower"
                    if any(word in reflection_text.lower() for word in ['decrease', 'lower', 'reduce', 'less']):
                        adjustment = -abs(adjustment)
                    
                    # Clamp to [-0.3, +0.3]
                    adjustment = max(-0.3, min(0.3, adjustment))
                    return adjustment
                except:
                    continue
        
        # Default: slight decrease due to acknowledging opponent's points
        return -0.05
