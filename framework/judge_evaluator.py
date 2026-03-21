"""
Judicial Panel Evaluation System

Three independent judges perform holistic evaluation of debate transcripts
using appellate-style deliberation with majority voting.
"""

from typing import List, Dict
from openrouter_client import OpenRouterLLMClient
import os
import json
from collections import Counter


class JudicialPanel:
    """
    3-Judge deliberative panel for independent holistic debate evaluation
    """
    
    def __init__(self):
        """
        Initialize three independent judges with different LLM models
        """
        # All judges use OpenRouter for consistency
        self.judges = [
            {
                "name": "Judge 1",
                "llm": OpenRouterLLMClient(
                    model_name="deepseek/deepseek-r1",
                    system_prompt="You are an independent appellate judge presiding over a legal proceeding. Your role is to perform a comprehensive holistic evaluation of the case, focusing on evidence admissibility, logical coherence of advocacy, and scientific accuracy of expert testimonies.",
                    temperature=0.3
                ),
                "model": "deepseek/deepseek-r1"
            },
            {
                "name": "Judge 2",
                "llm": OpenRouterLLMClient(
                    model_name="nousresearch/hermes-3-llama-3.1-405b",
                    system_prompt="You are an independent appellate judge presiding over a legal proceeding. Your role is to perform a comprehensive holistic evaluation of the case, focusing on evidence admissibility, logical coherence of advocacy, and scientific accuracy of expert testimonies.",
                    temperature=0.3
                ),
                "model": "nousresearch/hermes-3-llama-3.1-405b"
            },
            {
                "name": "Judge 3",
                "llm": OpenRouterLLMClient(
                    model_name="qwen/qwen3-235b-a22b-2507",
                    system_prompt="You are an independent appellate judge presiding over a legal proceeding. Your role is to perform a comprehensive holistic evaluation of the case, focusing on evidence admissibility, logical coherence of advocacy, and scientific accuracy of expert testimonies.",
                    temperature=0.3
                ),
                "model": "qwen/qwen3-235b-a22b-2507"
            }
        ]
    
    def evaluate_debate(self, debate_transcript: Dict, 
                       admitted_evidence: List = None,
                       role_switch_history: Dict = None,
                       prag_metrics: Dict = None,
                       critic_evaluations: List[Dict] = None,
                       reflection_history: List[Dict] = None) -> Dict:
        """
        Each judge independently evaluates the full debate transcript
        
        Args:
            debate_transcript: Full debate transcript from MAD
            admitted_evidence: List of evidence items admitted by negotiation judge
            role_switch_history: Role-switching consistency report (optional)
            prag_metrics: Summary of Progressive RAG evolution and novelty (optional)
            critic_evaluations: Round-by-round evaluations from the Critic Agent (optional)
            reflection_history: History of agent self-reflections (optional)
            
        Returns:
            Judicial panel results with majority verdict and opinions
        """
        print("\n" + "="*60)
        print("JUDICIAL PANEL EVALUATION")
        print("="*60 + "\n")
        
        claim = debate_transcript['claim']
        
        # Extract arguments from both sides
        proponent_args = self._extract_side_arguments(debate_transcript, 'proponent')
        opponent_args = self._extract_side_arguments(debate_transcript, 'opponent')
        
        # Extract evidence references
        evidence_summary = self._extract_evidence_summary(debate_transcript, admitted_evidence)
        
        # Role-switch summary
        role_switch_summary = self._format_role_switch(role_switch_history) if role_switch_history else "No role-switching performed."
        
        judge_verdicts = []
        
        for judge in self.judges:
            print(f"{judge['name']} ({judge['model']}) deliberating...")
            
            verdict = self._judge_evaluate(
                judge,
                claim,
                proponent_args,
                opponent_args,
                evidence_summary,
                role_switch_summary,
                debate_transcript,
                prag_metrics,
                critic_evaluations,
                reflection_history
            )
            
            judge_verdicts.append(verdict)
            print(f"  Verdict: {verdict['verdict']}")
            print(f"  Evidence Strength: {verdict['evidence_strength']}/10")
            print(f"  Argument Validity: {verdict['argument_validity']}/10")
            print(f"  Scientific Reliability: {verdict['scientific_reliability']}/10\n")
        
        # Aggregate verdicts using majority voting
        # NOTE: A 1-1-1 split defaults to Judge 1 (DeepSeek-R1) as presiding tie-breaker
        aggregation = self._aggregate_verdicts(judge_verdicts)
        
        result = {
            "claim": claim,
            "judge_verdicts": judge_verdicts,
            "final_verdict": aggregation['final_verdict'],
            "majority_opinion": aggregation['majority_opinion'],
            "dissenting_opinion": aggregation['dissenting_opinion'],
            "vote_breakdown": aggregation['vote_breakdown']
        }
        
        # Save results
        import json
        try:
            from logging_extension import append_framework_json
            append_framework_json("judge_evaluation.jsonl", claim, result)
        except ImportError:
            with open("judge_evaluation.json", "w") as f:
                json.dump(result, f, indent=2)
        
        print(f"\nFinal Verdict: {aggregation['final_verdict']}")
        print(f"Vote Breakdown: {aggregation['vote_breakdown']}")
        if aggregation['dissenting_opinion']:
            print(f"Dissent Present: Yes")
        
        return result
    
    def _judge_evaluate(self, judge: Dict, claim: str, 
                       proponent_args: List[str], opponent_args: List[str],
                       evidence_summary: str, role_switch_summary: str,
                       full_transcript: Dict, prag_metrics: Dict = None,
                       critic_evaluations: List[Dict] = None,
                       reflection_history: List[Dict] = None) -> Dict:
        """
        Single judge performs 5-stage holistic evaluation
        
        Returns:
            Dict with claim_summary, scores, verdict, and reasoning
        """
        # Prepare comprehensive prompt for 5-stage evaluation
        prompt = f"""You are an appellate judge evaluating the following proceedings for medical fact-checking.

PROCEEDINGS RECORD:
CLAIM: {claim}

PLAINTIFF COUNSEL'S ARGUMENTS:
{chr(10).join(proponent_args)}

DEFENSE COUNSEL'S ARGUMENTS:
{chr(10).join(opponent_args)}

ADMITTED EVIDENCE & EXPERT TESTIMONIES:
{evidence_summary}

ROLE-SWITCH HISTORY (ADVERSARY CONSISTENCY):
{role_switch_summary}

EVIDENCE DISCOVERY METRICS (PRAG EVOLUTION):
{json.dumps(prag_metrics, indent=2) if prag_metrics else "No P-RAG data available."}

INDEPENDENT CRITIC EVALUATIONS (PROCESS INTEGRITY):
{json.dumps(critic_evaluations, indent=2) if critic_evaluations else "No critic data available."}

AGENT SELF-REFLECTION TRENDS:
{json.dumps(reflection_history, indent=2) if reflection_history else "No reflection data available."}

Perform the following evaluation stages:

STAGE 1 - CASE RECONSTRUCTION
Identify:
- Core claim being adjudicated
- Main supporting arguments from Plaintiff Counsel
- Main counterarguments from Defense Counsel

STAGE 2 - EVIDENCE & TESTIMONY WEIGHTING
Evaluate the evidence and expert witness testimonies for:
- Relevance to the claim/premises
- Scientific credibility (peer-reviewed sources, professional credentials)
- Testimony strength (how well expert opinions back the advocacy)
- Consistency with admitted exhibits

Provide a score: Evidence Strength (0-10)
- 0-3: Weak, irrelevant, or unreliable evidence
- 4-6: Moderate evidence with some limitations
- 7-10: Strong, credible, highly relevant evidence

STAGE 3 - LOGICAL COHERENCE ANALYSIS
Detect:
- Logical contradictions
- Unsupported inferential leaps
- Fallacies (ad hominem, straw man, false dichotomy, etc.)
- Misuse or misrepresentation of evidence

Provide a score: Argument Validity (0-10)
- 0-3: Severely flawed logic, multiple fallacies
- 4-6: Some logical issues but generally coherent
- 7-10: Sound reasoning, minimal logical flaws

STAGE 4 - SCIENTIFIC/TECHNICAL CONSISTENCY
Check:
- Alignment with established biomedical/scientific consensus
- Correctness of interpretation of studies and data
- Risk of misrepresentation or cherry-picking

Provide a score: Scientific Reliability (0-10)
- 0-3: Contradicts consensus, major misinterpretations
- 4-6: Partially aligned, some interpretation issues
- 7-10: Well-aligned with consensus, accurate interpretations

STAGE 5 - DISCOVERY RIGOR & TRANSPARENCY
Analyze the PRAG metrics:
- Query Evolution: How well did counsels refine their discovery requests?
- Evidence Novelty: Did the proceedings reach a point of diminishing returns?
- Role of the Court: Assess the impact of judicial query refinement on evidence quality.
- Transparency: How does the retrieval history affect your confidence in the exhaustiveness of the case?

STAGE 6 - JUDICIAL VERDICT
Based on your evaluation, determine:
- SUPPORTED: The claim is well-supported by the evidence and arguments
- NOT SUPPORTED: The claim is not adequately supported or is refuted
- INCONCLUSIVE: Insufficient evidence or arguments are too balanced

Provide written justification including:
- Key decisive factors that led to your verdict
- Specific evidence references (cite by source ID if available)
- Reasoning summary (2-3 sentences)

Respond ONLY in valid JSON format:
{{
  "claim_summary": "Brief summary of the core claim and debate",
  "evidence_strength": <score 0-10>,
  "argument_validity": <score 0-10>,
  "scientific_reliability": <score 0-10>,
  "verdict": "SUPPORTED" or "NOT SUPPORTED" or "INCONCLUSIVE",
  "reasoning": "Detailed justification for your verdict"
}}"""

        response = judge['llm'].generate(prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                verdict_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            # Validate required fields
            required_fields = ['claim_summary', 'evidence_strength', 'argument_validity', 
                             'scientific_reliability', 'verdict', 'reasoning']
            for field in required_fields:
                if field not in verdict_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate verdict value
            if verdict_data['verdict'] not in ['SUPPORTED', 'NOT SUPPORTED', 'INCONCLUSIVE']:
                verdict_data['verdict'] = 'INCONCLUSIVE'
            
            # Ensure scores are integers 0-10
            for score_field in ['evidence_strength', 'argument_validity', 'scientific_reliability']:
                verdict_data[score_field] = max(0, min(10, int(verdict_data[score_field])))
            
        except Exception as e:
            print(f"  [WARNING] Failed to parse judge response: {e}")
            print(f"  [WARNING] Using fallback verdict")
            # Fallback verdict
            verdict_data = {
                "claim_summary": f"Evaluation of: {claim}",
                "evidence_strength": 5,
                "argument_validity": 5,
                "scientific_reliability": 5,
                "verdict": "INCONCLUSIVE",
                "reasoning": "Unable to parse structured evaluation. Defaulting to inconclusive."
            }
        
        # Add judge metadata
        verdict_data['judge_name'] = judge['name']
        verdict_data['model'] = judge['model']
        
        return verdict_data
    
    def _aggregate_verdicts(self, judge_verdicts: List[Dict]) -> Dict:
        """
        Aggregate judge verdicts using majority voting and construct opinions
        
        Returns:
            Dict with final_verdict, majority_opinion, dissenting_opinion, vote_breakdown
        """
        # Count votes
        vote_counts = Counter(v['verdict'] for v in judge_verdicts)
        final_verdict = vote_counts.most_common(1)[0][0]
        
        # Separate majority and dissenting judges
        majority_judges = [v for v in judge_verdicts if v['verdict'] == final_verdict]
        dissenting_judges = [v for v in judge_verdicts if v['verdict'] != final_verdict]
        
        # Construct majority opinion
        majority_opinion = self._synthesize_opinion(majority_judges, "majority")
        
        # Construct dissenting opinion if exists
        dissenting_opinion = None
        if dissenting_judges:
            dissenting_opinion = self._synthesize_opinion(dissenting_judges, "dissent")
        
        return {
            "final_verdict": final_verdict,
            "majority_opinion": majority_opinion,
            "dissenting_opinion": dissenting_opinion,
            "vote_breakdown": dict(vote_counts)
        }
    
    def _synthesize_opinion(self, judges: List[Dict], opinion_type: str) -> str:
        """
        Synthesize a coherent opinion from multiple judges' reasoning
        
        Args:
            judges: List of judge verdict dicts
            opinion_type: "majority" or "dissent"
            
        Returns:
            Synthesized opinion text
        """
        if not judges:
            return ""
        
        if len(judges) == 1:
            # Single judge opinion
            judge = judges[0]
            return f"{judge['judge_name']} ({judge['model']}) - {judge['verdict']}: {judge['reasoning']}"
        
        # Multiple judges with same verdict - synthesize
        verdict = judges[0]['verdict']
        judge_names = ", ".join([j['judge_name'] for j in judges])
        
        # Combine reasoning
        combined_reasoning = []
        for judge in judges:
            combined_reasoning.append(f"- {judge['judge_name']}: {judge['reasoning']}")
        
        opinion = f"{opinion_type.capitalize()} Opinion ({judge_names}) - {verdict}:\n\n"
        opinion += "\n".join(combined_reasoning)
        
        return opinion
    
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
                        arguments.append(f"[Expert Testimony]: {expert['text']}")
        return arguments
    
    def _extract_evidence_summary(self, transcript: Dict, admitted_evidence: List) -> str:
        """
        Create a summary of evidence used in the debate
        """
        if not admitted_evidence:
            # Fallback: extract from transcript
            evidence_ids = set()
            for round_data in transcript.get('rounds', []):
                for arg in round_data.get('arguments', []):
                    # Extract source IDs from argument text (simple regex)
                    import re
                    ids = re.findall(r'\b\d{8}\b', arg['text'])
                    evidence_ids.update(ids)
            
            if evidence_ids:
                return f"Evidence sources cited: {', '.join(sorted(evidence_ids))}"
            else:
                return "No specific evidence sources identified."
        
        # Format admitted evidence
        evidence_lines = []
        for i, ev in enumerate(admitted_evidence[:10], 1):  # Limit to 10 for brevity
            source_id = ev.source_id if hasattr(ev, 'source_id') else ev.get('source_id', 'unknown')
            text_preview = ev.text[:150] if hasattr(ev, 'text') else ev.get('text', '')[:150]
            evidence_lines.append(f"{i}. Source {source_id}: {text_preview}...")
        
        return "\n".join(evidence_lines)
    
    def _format_role_switch(self, role_switch_history: Dict) -> str:
        """Format role-switching history for judge review"""
        if not role_switch_history:
            return "No role-switching performed."
        
        analysis = role_switch_history.get('analysis', 'No analysis available.')
        return f"Role-Switching Consistency Analysis:\n{analysis}"
    
    def _format_arguments(self, args: List[str]) -> str:
        """Format arguments for prompt (limit to avoid token overflow)"""
        if not args:
            return "No arguments presented."
        
        formatted = []
        for i, arg in enumerate(args[:5], 1):  # Limit to 5 arguments
            # Truncate long arguments
            arg_text = arg[:800] if len(arg) > 800 else arg
            formatted.append(f"Argument {i}:\n{arg_text}\n")
        
        if len(args) > 5:
            formatted.append(f"... and {len(args) - 5} more arguments")
        
        return "\n".join(formatted)
