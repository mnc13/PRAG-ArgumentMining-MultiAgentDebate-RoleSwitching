"""
Judge Evaluation System

Multiple judges evaluate debate transcript and select provisional winner
"""

from typing import List, Dict
from groq_client import GroqLLMClient
from openai_client import OpenAILLMClient
import os
import json

class JudgeEvaluator:
    """
    Multi-judge evaluation system for debate transcripts
    """
    
    def __init__(self):
        """
        Initialize judges with different LLM providers for diversity
        """
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        self.judges = [
            {
                "name": "Logic & Reasoning Expert",
                "llm": GroqLLMClient(
                    api_key=groq_key,
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                    system_prompt="You are an expert in logical reasoning and argumentation. Evaluate arguments for logical consistency and soundness.",
                    temperature=0.3
                ),
                "focus": "logical_consistency"
            },
            {
                "name": "Evidence Quality Expert",
                "llm": GroqLLMClient(
                    api_key=groq_key,
                    model_name="llama-3.1-70b-versatile",
                    system_prompt="You are an expert in scientific evidence evaluation. Assess the quality and relevance of cited sources.",
                    temperature=0.3
                ),
                "focus": "evidence_quality"
            },
            {
                "name": "Scientific Accuracy Expert",
                "llm": OpenAILLMClient(
                    api_key=openai_key,
                    model_name="gpt-4o-mini",
                    system_prompt="You are a scientific accuracy expert. Evaluate claims for factual correctness and proper interpretation.",
                    temperature=0.3
                ),
                "focus": "scientific_accuracy"
            }
        ]
    
    def evaluate_debate(self, debate_transcript: Dict) -> Dict:
        """
        Each judge evaluates the debate and scores both sides
        
        Args:
            debate_transcript: Full debate transcript from MAD
            
        Returns:
            Judge evaluation results with provisional winner
        """
        print("\n" + "="*60)
        print("JUDGE EVALUATION")
        print("="*60 + "\n")
        
        claim = debate_transcript['claim']
        
        # Extract arguments from both sides
        proponent_args = self._extract_side_arguments(debate_transcript, 'proponent')
        opponent_args = self._extract_side_arguments(debate_transcript, 'opponent')
        
        judge_results = []
        
        for judge in self.judges:
            print(f"Judge: {judge['name']} evaluating...")
            
            # Score both sides
            proponent_scores = self._score_arguments(
                judge, 
                proponent_args, 
                opponent_args,
                "proponent"
            )
            
            opponent_scores = self._score_arguments(
                judge,
                opponent_args,
                proponent_args,
                "opponent"
            )
            
            # Determine winner for this judge
            winner = "proponent" if proponent_scores['total'] > opponent_scores['total'] else "opponent"
            
            judge_result = {
                "judge_name": judge['name'],
                "model": judge['llm'].model_name,
                "proponent_scores": proponent_scores,
                "opponent_scores": opponent_scores,
                "winner": winner,
                "reasoning": self._generate_reasoning(judge, proponent_args, opponent_args, winner)
            }
            
            judge_results.append(judge_result)
            print(f"  Winner: {winner} (Proponent: {proponent_scores['total']}, Opponent: {opponent_scores['total']})")
        
        # Aggregate scores
        total_proponent = sum(j['proponent_scores']['total'] for j in judge_results)
        total_opponent = sum(j['opponent_scores']['total'] for j in judge_results)
        
        provisional_winner = "proponent" if total_proponent > total_opponent else "opponent"
        
        # Calculate confidence based on score margin
        total_scores = total_proponent + total_opponent
        confidence = abs(total_proponent - total_opponent) / total_scores if total_scores > 0 else 0.5
        
        result = {
            "claim": claim,
            "judges": judge_results,
            "aggregate_scores": {
                "proponent": total_proponent,
                "opponent": total_opponent
            },
            "provisional_winner": provisional_winner,
            "confidence": round(confidence, 3)
        }
        
        # Save results
        with open("judge_evaluation.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\nProvisional Winner: {provisional_winner}")
        print(f"Aggregate Scores - Proponent: {total_proponent}, Opponent: {total_opponent}")
        print(f"Confidence: {confidence:.3f}")
        
        return result
    
    def _extract_side_arguments(self, transcript: Dict, role: str) -> List[str]:
        """Extract all arguments and expert testimonies for one side"""
        arguments = []
        for round_data in transcript['rounds']:
            # Regular arguments
            for arg in round_data['arguments']:
                if arg['role'] == role:
                    arguments.append(arg['text'])
            # Expert testimonies requested by this side
            if 'expert_testimonies' in round_data:
                for expert in round_data['expert_testimonies']:
                    if expert.get('requesting_side') == role:
                        arguments.append(f"[Expert Testimony Supporting {role.capitalize()}]: {expert['text']}")
        return arguments
    
    def _score_arguments(self, judge: Dict, side_args: List[str], 
                        opponent_args: List[str], side_name: str) -> Dict:
        """
        Judge scores arguments on 4 criteria
        
        Returns:
            Dict with scores for each criterion and total
        """
        # Combine arguments for evaluation
        args_text = "\n\n".join(side_args[:3])  # Use first 3 arguments to avoid token limits
        opp_text = "\n\n".join(opponent_args[:2])
        
        prompt = f"""Evaluate the following arguments on a scale of 0-10 for each criterion:

{side_name.upper()} ARGUMENTS:
{args_text}

OPPONENT ARGUMENTS (for context):
{opp_text}

Score the {side_name}'s arguments on:
1. Evidence Quality (0-10): Relevance and strength of cited sources
2. Logical Consistency (0-10): Internal coherence and sound reasoning
3. Persuasiveness (0-10): Clarity and convincingness
4. Scientific Accuracy (0-10): Factual correctness

Respond in JSON format:
{{
  "evidence_quality": <score>,
  "logical_consistency": <score>,
  "persuasiveness": <score>,
  "scientific_accuracy": <score>
}}"""
        
        response = judge['llm'].generate(prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                # Fallback scores if parsing fails
                scores = {
                    "evidence_quality": 7,
                    "logical_consistency": 7,
                    "persuasiveness": 7,
                    "scientific_accuracy": 7
                }
        except:
            scores = {
                "evidence_quality": 7,
                "logical_consistency": 7,
                "persuasiveness": 7,
                "scientific_accuracy": 7
            }
        
        scores['total'] = sum(scores.values())
        return scores
    
    def _generate_reasoning(self, judge: Dict, proponent_args: List[str], 
                           opponent_args: List[str], winner: str) -> str:
        """Generate judge's reasoning for their decision"""
        prompt = f"""Explain in 2-3 sentences why the {winner} won this debate based on your evaluation.

PROPONENT ARGUMENTS:
{proponent_args[0][:500]}...

OPPONENT ARGUMENTS:
{opponent_args[0][:500]}...

Provide concise reasoning:"""
        
        reasoning = judge['llm'].generate(prompt)
        return reasoning.strip()
