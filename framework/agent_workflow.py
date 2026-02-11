import re
import os
from typing import List
from models import Claim, Argument, Evidence, DebateState
from llm_client import LLMClient

class ArgumentMiner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def mine_arguments(self, claim: Claim) -> Argument:
        prompt = (
            f"As a Scientific Analyst, decompose the following clinical claim into atomic, testable premises "
            f"suitable for evidence retrieval.\n"
            f"Claim: {claim.text}\n"
            f"Format: Return only the list of premises, one per line. No numbering or prefixes."
        )
        response = self.llm_client.generate(prompt)
        # Handle various list formats (numbered, bulleted, etc) if LLM doesn't follow strict format
        premises = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line: continue
            # Remove common prefixes like 1. or - or *
            line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
            if line:
                premises.append(line)
        return Argument(claim_id=claim.id, premises=premises)

class EvidenceFirstDebateAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def negotiate_evidence(self, debate_state: DebateState) -> List[Evidence]:
        """
        Simulates agents discussing and agreeing on shared evidence.
        """
        evidence_texts = [f"{i}: {ev.text[:100]}..." for i, ev in enumerate(debate_state.evidence_pool)]
        prompt = (f"Review the evidence pool for claim '{debate_state.claim.text}':\n"
                  + "\n".join(evidence_texts) + 
                  "\nSelect the most relevant evidence items for the shared set.")
        
        # In a real implementation, this would involve multi-turn dialogue between agents.
        # Here we mock a single decision to populate the shared set.
        self.llm_client.generate(prompt) # Just to trigger the mock logic
        
        # For this test, simply select the top 2 if available
        return debate_state.evidence_pool[:2]
