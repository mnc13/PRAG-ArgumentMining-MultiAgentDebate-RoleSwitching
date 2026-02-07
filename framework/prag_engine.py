"""
Progressive RAG (P-RAG) Engine

Enables agents to request targeted evidence retrieval during debate rounds.
Tracks retrieval history and manages context-aware queries.
"""

import json
from typing import List, Dict
from models import Evidence
from llm_client import LLMClient

class ProgressiveRAG:
    def __init__(self, vector_retriever, llm_client: LLMClient):
        """
        Initialize P-RAG engine
        
        Args:
            vector_retriever: VectorRetriever instance for semantic search
            llm_client: LLM for query formulation
        """
        self.retriever = vector_retriever
        self.llm = llm_client
        self.retrieval_history = []
        self.round_counter = 0
    
    def formulate_query(self, debate_context: str, agent_request: str) -> str:
        """
        Use LLM to formulate a targeted query based on debate context
        
        Note: Uses the LLM passed during initialization (typically Groq GPT)
        
        Args:
            debate_context: Summary of debate so far
            agent_request: Specific information requested by agent
            
        Returns:
            Refined query string for retrieval
        """
        prompt = f"""Based on the following debate context and agent request, formulate a precise search query 
to retrieve relevant scientific evidence.

Debate Context:
{debate_context}

Agent Request:
{agent_request}

Generate a concise search query (1-2 sentences) that will retrieve the most relevant evidence:"""
        
        query = self.llm.generate(prompt)
        return query.strip()
    
    def retrieve_progressive(self, query: str, top_k: int = 3, context: str = "") -> List[Evidence]:
        """
        Perform targeted retrieval and log to history
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            context: Context for this retrieval (e.g., "Round 2 - Opponent request")
            
        Returns:
            List of Evidence objects
        """
        # Perform retrieval
        evidence = self.retriever.retrieve(query, top_k=top_k)
        
        # Log to history
        self.retrieval_history.append({
            "round": self.round_counter,
            "query": query,
            "context": context,
            "num_results": len(evidence),
            "evidence_ids": [ev.source_id for ev in evidence]
        })
        
        return evidence
    
    def start_new_round(self):
        """Increment round counter for tracking"""
        self.round_counter += 1
    
    def get_retrieval_summary(self) -> Dict:
        """
        Get summary of all P-RAG retrievals
        
        Returns:
            Dictionary with retrieval statistics and history
        """
        return {
            "total_retrievals": len(self.retrieval_history),
            "rounds_with_prag": len(set(r["round"] for r in self.retrieval_history)),
            "history": self.retrieval_history
        }
    
    def save_history(self, filepath: str = "prag_history.json"):
        """Save retrieval history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_retrieval_summary(), f, indent=2)
