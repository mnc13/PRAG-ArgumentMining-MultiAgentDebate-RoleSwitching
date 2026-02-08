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
            role: "proponent", "opponent", "judge", "critic", or "expert"
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
        
        # Simulation instructions
        debate_context = """
        You are participating in a structured scientific debate. 
        - Maintain a clinical, factual, and strictly evidence-based tone.
        - Focus on proving or refuting the claim using the provided medical evidence and expert testimony.
        - State your arguments clearly and concisely.
        - DIRECT OUTPUT ONLY: Do not reveal your internal thought process, scratchpad, or "thinking" steps. Output only your final argument.
        """

        if self.role == "proponent":
            role_instruction = "Present your case in SUPPORT of the claim. Use evidence to persuade the Moderator."
        elif self.role == "opponent":
            role_instruction = "Present your case AGAINST the claim. Identify flaws and pose challenges to the proponent's evidence."
        elif self.role == "judge":
            role_instruction = "Oversee the discussion. Summarize the current state of arguments and ask probing questions to both sides."
        elif self.role == "critic":
            role_instruction = "Provide a neutral scientific analysis of the current debate. Identify logical gaps and evidentiary weaknesses."
        else:  # expert
            role_instruction = f"Provide your unbiased expert testimony as a {self.job_title} regarding: {', '.join(self.expertise)}."
        
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
        Based on the current state of the simulation, do you need to summon a scientific expert witness to clarify a specific point?
        
        Recent Debate Activity:
        {history_summary}

        If yes, specify the type of expertise needed and why. If no, say 'None'.
        Format: {{"expert_type": "...", "reasoning": "..."}} or "None"
        """
        
        response = self.llm.generate(prompt)
        try:
            if "None" in response: return None
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
        The {requester} has requested to summon an expert witness: {request['expert_type']}
        Reasoning: {request['reasoning']}

        As the Moderator, is this expert necessary for the thorough resolution of the debate? 
        Respond only with 'Granted' or 'Denied' followed by a brief reason.
        """
        response = self.llm.generate(prompt)
        return "Granted" in response

    def check_debate_completion(self, debate_history: List[Dict]) -> bool:
        """
        Judge-only: Decide if enough evidence has been presented to conclude the debate
        """
        if self.role != "judge": return False

        history_summary = "\n".join([f"{a['agent']}: {a['text'][:200]}..." for a in debate_history])
        prompt = f"""
        As the Moderator (Judge), review the debate record. Have both sides had sufficient opportunity to present their medical evidence and arguments?
        
        Record Summary:
        {history_summary}

        Should the debate continue or should we move to final evaluation?
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
            prompt = f"""As {self.job_title}, what specific scientific evidence do you need to request from the archives to build your case?
            
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

    def critique_argument(self, argument: Dict) -> Dict:
        """Maintain critic role as scientific analyst"""
        prompt = f"""Analyze this testimony/argument for logical consistency and evidentiary strength.
        
        Statement by {argument['agent']}:
        {argument['text']}
        
        Provide:
        1. Points of Strength
        2. Points of Contradiction/Weakness
        3. Suggested Cross-Examination questions
        
        Format as JSON."""
        
        critique_text = self.llm.generate(prompt)
        return {
            "critic": self.job_title,
            "target_agent": argument['agent'],
            "critique": critique_text
        }
