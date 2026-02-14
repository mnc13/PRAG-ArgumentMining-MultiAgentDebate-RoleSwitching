"""
Multi-Agent Debate (MAD) System

Implements debate agents with dynamic personas and multi-round debate orchestration.
"""

import json
from typing import List, Dict
from models import Claim, Evidence, DebateState
from llm_client import LLMClient, GeminiLLMClient
from prag_engine import ProgressiveRAG
from personas import validate_unique_models
import os

class DebateAgent:
    """
    Individual debate agent with persona and LLM, adapted for scientific debate simulation
    """
    def __init__(self, persona_config: dict, role: str, prag_engine: ProgressiveRAG):
        """
        Initialize debate agent
        
        Args:
            persona_config: Full persona configuration dictionary
            role: "proponent", "opponent", "judge", or "expert"
            prag_engine: ProgressiveRAG instance for evidence requests
        """
        from personas import create_llm_client
        
        self.persona_config = persona_config
        self.role = role
        self.prag = prag_engine
        
        # Create LLM client using factory
        self.llm = create_llm_client(self.persona_config)
        
        # Enforce job-title naming
        self.job_title = self.persona_config.get("role", role.capitalize())
        self.name = self.job_title # No person names allowed
        
        self.expertise = self.persona_config.get("expertise", [])
        self.provider = self.persona_config["llm_provider"]
        self.model = self.persona_config["llm_model"]
        self.persona_key = self.persona_config.get("persona_key", "debate_agent")
        
    def generate_argument(self, claim: Claim, evidence: List[Evidence], debate_history: List[Dict]) -> str:
        """
        Generate argument based on debate role and evidence
        """
        # Format evidence for context
        evidence_text = "\n\n".join([
            f"Evidence {i+1} (Source: {ev.source_id}):\n{ev.text[:500]}..."
            for i, ev in enumerate(evidence)
        ])
        
        # Format debate history (Join outside f-string)
        history_lines = [f"{arg['agent']}: {arg['text']}" for arg in debate_history[-5:]]
        history_text = "\n\n".join(history_lines) if history_lines else "Opening of the case."
        
        # Courtroom proceedings instructions
        debate_context = """
        You are participating in a structured legal proceeding. 
        - Maintain a professional, factual, and strictly evidence-based tone.
        - Focus on proving or refuting the claim using the provided medical evidence and expert witness testimony.
        - State your arguments clearly and concisely as you would in a courtroom.
        - DIRECT OUTPUT ONLY: Do not reveal your internal thought process, scratchpad, or "thinking" steps. Output only your final argument.
        """

        if self.role == "proponent":
            role_instruction = "As Plaintiff Counsel, present your case in SUPPORT of the claim. Use evidence and expert testimony to persuade the Court."
        elif self.role == "opponent":
            role_instruction = "As Defense Counsel, present your case AGAINST the claim. Identify flaws and challenge the plaintiff's evidence and witnesses."
        elif self.role == "judge":
            role_instruction = "As the Court, oversee the proceedings. Summarize the current state of arguments and ask probing questions to both counsels."
        else:  # expert
            role_instruction = f"As an Expert Witness ({self.job_title}), provide your unbiased professional testimony regarding: {', '.join(self.expertise)}."
        
        prompt = f"""
        {debate_context}
        
        Claim: {claim.text}

        Your Role: {self.job_title}
        Instruction: {role_instruction}

        Available Evidence:
        {evidence_text}

        Recent Debate History:
        {history_text}

        Provide your statement (2-3 paragraphs, cite evidence by source ID):"""
        
        argument = self.llm.generate(prompt, max_tokens=512)
        return argument

    def request_expert(self, debate_history: List[Dict]) -> Dict:
        """
        Propose summoning an expert to the judge
        
        Returns:
            Dictionary with 'expert_type' and 'reasoning' or None
        """
        if self.role not in ["proponent", "opponent"]:
            return None

        history_summary = "\n".join([f"{a['agent']}: {a['text'][:200]}..." for a in debate_history[-3:]])
        prompt = f"""
        Based on the current state of the proceedings, do you need to call an expert witness to clarify a specific point?
        
        Recent Proceedings:
        {history_summary}

        If yes, specify the type of expertise needed and why. If no, say 'None'.
        Format: {{"expert_type": "...", "reasoning": "..."}} or "None"
        """
        
        response = self.llm.generate(prompt)
        try:
            if "None" in response: return None
            import json
            import re
            match = re.search(r'\{[^}]+\}', response)
            return json.loads(match.group()) if match else None
        except:
            return None

    def evaluate_expert_request(self, requester: str, request: Dict) -> bool:
        """
        Judge-only: Decide whether to grant an expert request
        """
        if self.role != "judge": return False

        prompt = f"""
        {requester} has requested to call an expert witness: {request['expert_type']}
        Reasoning: {request['reasoning']}

        As the Court, is this expert witness necessary for the thorough resolution of this case? 
        Respond only with 'Granted' or 'Denied' followed by a brief reason.
        """
        response = self.llm.generate(prompt)
        return "Granted" in response

    def refine_query(self, original_query: str, debate_context: str) -> str:
        """
        Judge-only: Review and refine a counsel's proposed search query
        """
        if self.role != "judge": return original_query

        prompt = f"""
        As the Court, you must maintain the quality and focus of evidence discovery.
        A counsel has proposed the following search query to retrieve additional medical exhibits:
        
        Proposed Query: "{original_query}"
        
        Context of proceedings:
        {debate_context}
        
        Refine this query to be more precise, narrow the scope if necessary, and ensure it follows scientific rigor.
        Respond ONLY with the refined query string.
        """
        refined_query = self.llm.generate(prompt)
        return refined_query.strip().strip('"')

    def propose_query_gap(self, debate_context: str) -> str:
        """
        Counsel: Identify a gap in evidence and propose a specific need
        """
        if self.role not in ["proponent", "opponent"]:
            return ""

        prompt = f"""As {self.job_title}, analyze the current proceedings and identify a critical gap in the available medical exhibits. 
        What specific evidence do you need to request to strengthen your case or challenge the opposition?
        
        Context: {debate_context}
        
        Propose exactly one specific evidence need (1 sentence):"""
        specific_need = self.llm.generate(prompt)
        return specific_need.strip()

    def check_debate_completion(self, debate_history: List[Dict]) -> bool:
        """
        Judge-only: Decide if enough evidence has been presented to conclude the debate
        """
        if self.role != "judge": return False

        history_summary = "\n".join([f"{a['agent']}: {a['text'][:200]}..." for a in debate_history])
        prompt = f"""
        As the Court, review the proceedings. Have both counsels had sufficient opportunity to present their medical evidence and arguments?
        
        Record Summary:
        {history_summary}

        Should the proceedings continue or should we move to final deliberation?
        Respond 'Wait' to continue or 'Close' to finish.
        """
        
        response = self.llm.generate(prompt)
        if "Close" in response: return True
        return False
    
    def request_evidence(self, debate_context: str, specific_need: str = None) -> List[Evidence]:
        """
        Request additional evidence via P-RAG (maintain scientific rigor)
        """
        if specific_need is None:
            prompt = f"""As {self.job_title}, what specific scientific evidence do you need to request from the medical archives to support your case?
            
            Context: {debate_context}
            
            Your request (1 sentence):"""
            specific_need = self.llm.generate(prompt)
        
        query = self.prag.formulate_query(debate_context, specific_need)
        evidence = self.prag.retrieve_progressive(
            query, 
            top_k=3, 
            context=f"Phase {self.prag.round_counter} - {self.job_title}"
        )
        return evidence



class CriticAgent:
    """
    Independent Critic Agent to observe and evaluate the debate each round
    """
    def __init__(self, model_config: dict = None):
        from personas import create_llm_client
        if model_config is None:
            # Default to a strong model for critique with full persona config
            model_config = {
                "name": "Critic Agent",
                "role": "Independent Critic",
                "expertise": ["logical analysis", "scientific rigor", "legal argumentation"],
                "system_prompt": "You are the Independent Critic Agent. Your role is to evaluate the debate rounds for logical coherence, evidence coverage, and rebuttal quality. Provide objective, actionable feedback.",
                "llm_provider": "openrouter",
                "llm_model": "deepseek/deepseek-r1",
                "temperature": 0.3
            }
        self.llm = create_llm_client(model_config)
        self.name = "Critic Agent"

    def evaluate_round(self, round_num: int, claim: str, transcript: List[Dict]) -> Dict:
        """
        Evaluate the latest round of the debate
        """
        history_summary = "\n".join([f"{a['agent']}: {a['text'][:400]}..." for a in transcript[-4:]])
        
        prompt = f"""
        You are the Critic Agent observing a courtroom-style scientific debate.
        Claim: {claim}
        Round: {round_num}
        
        Recent Proceedings:
        {history_summary}
        
        Analyze both the Plaintiff and Defense Counsel's performance in this round.
        Score each side (0.0 to 1.0) on:
        1. Logical Coherence: Argument flow and structure.
        2. Evidence Coverage: How well they used admitted exhibits.
        3. Rebuttal Coverage: Did they address the opponent's strongest points?
        
        Identify any premises that remain "unresolved" or under-supported.
        Provide actionable recommendations for both sides to improve their discovery and arguments.
        
        Respond ONLY in valid JSON format:
        {{
            "plaintiff": {{ "logic": 0.0, "evidence": 0.0, "rebuttal": 0.0, "reasoning": "..." }},
            "defense": {{ "logic": 0.0, "evidence": 0.0, "rebuttal": 0.0, "reasoning": "..." }},
            "unresolved_premises": ["...", "..."],
            "recommendations": {{
                "plaintiff": ["...", "..."],
                "defense": ["...", "..."],
                "queries": ["suggested search query 1", "suggested search query 2"]
            }},
            "debate_resolved": true/false
        }}
        """
        
        response = self.llm.generate(prompt)
        try:
            import json
            import re
            match = re.search(r'\{[\s\S]*\}', response)
            return json.loads(match.group()) if match else {}
        except:
            return {}
