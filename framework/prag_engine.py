"""
Progressive RAG (P-RAG) Engine

Enables counsels to request targeted evidence retrieval during proceedings (exhibit discovery).
Tracks retrieval history and manages context-aware queries.
"""

from models import Evidence
from llm_client import LLMClient
import numpy as np
from typing import List, Dict, Tuple

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
        self.total_evidence_pool = []  # Track all accepted evidence
        
        # Hyperparameters
        self.novelty_threshold = 0.2
        self.redundancy_sim_threshold = 0.85
        self.redundancy_ratio_threshold = 0.7
        self.relevance_gain_threshold = 0.05
        self.max_iterations = 10
    
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
        prompt = f"""Based on the following proceedings context and legal request, formulate a precise search query 
to retrieve relevant medical exhibits and evidence.

Debate Context:
{debate_context}

Agent Request:
{agent_request}

Generate a concise search query (1-2 sentences) that will retrieve the most relevant evidence:"""
        
        query = self.llm.generate(prompt)
        return query.strip()
    
    def retrieve_progressive(self, query: str, top_k: int = 3, context: str = "") -> List[Evidence]:
        """
        Perform targeted retrieval with novelty scoring and stopping criteria
        """
        # 1. Perform retrieval
        raw_evidence = self.retriever.retrieve(query, top_k=top_k)
        
        if not raw_evidence:
            return []

        # 2. Calculate Novelty Scores
        scored_evidence, avg_novelty = self._calculate_novelty(raw_evidence)
        
        # 3. Filter by Novelty (Reject < 0.2)
        accepted_evidence = [e for e in scored_evidence if e.novelty_score >= self.novelty_threshold]
        rejected_count = len(scored_evidence) - len(accepted_evidence)
        
        # 4. Check Redundancy Ratio
        redundant_count = sum(1 for e in scored_evidence if e.novelty_score < (1 - self.redundancy_sim_threshold))
        redundancy_ratio = redundant_count / len(scored_evidence) if scored_evidence else 0
        
        # 5. Check Relevance Gain (Compare to last round if exists)
        avg_relevance = np.mean([e.relevance_score for e in accepted_evidence]) if accepted_evidence else 0
        last_avg_relevance = self.retrieval_history[-1].get("avg_relevance", 0) if self.retrieval_history else 0
        relevance_gain = avg_relevance - last_avg_relevance
        
        # 6. Check Stopping Criteria
        stop_reason = None
        if self.round_counter >= self.max_iterations:
            stop_reason = "Maximum iterations reached"
        elif redundancy_ratio > self.redundancy_ratio_threshold:
            stop_reason = f"High redundancy detected ({redundancy_ratio:.2f} > {self.redundancy_ratio_threshold})"
        elif self.round_counter > 1 and relevance_gain < self.relevance_gain_threshold:
             stop_reason = f"Diminishing relevance gain ({relevance_gain:.4f} < {self.relevance_gain_threshold})"

        # Log to history
        retrieval_entry = {
            "round": self.round_counter,
            "query": query,
            "context": context,
            "num_retrieved": len(raw_evidence),
            "num_accepted": len(accepted_evidence),
            "num_rejected": rejected_count,
            "avg_novelty": float(avg_novelty),
            "avg_relevance": float(avg_relevance),
            "relevance_gain": float(relevance_gain),
            "redundancy_ratio": float(redundancy_ratio),
            "stop_reason": stop_reason,
            "evidence_ids": [ev.source_id for ev in accepted_evidence]
        }
        self.retrieval_history.append(retrieval_entry)
        
        if accepted_evidence:
            self.total_evidence_pool.extend(accepted_evidence)

        if stop_reason:
            print(f"   > [PRAG STOP] {stop_reason}")
        
        return accepted_evidence

    def _calculate_novelty(self, new_evidence: List[Evidence]) -> Tuple[List[Evidence], float]:
        """
        Calculate novelty_score = 1 - max_cosine_similarity(new_doc, existing_pool)
        """
        if not self.total_evidence_pool:
            for ev in new_evidence:
                ev.novelty_score = 1.0
            return new_evidence, 1.0

        # Get embeddings from retriever model
        model = getattr(self.retriever, 'model', None)
        if not model:
            # Fallback if no model available
            for ev in new_evidence:
                ev.novelty_score = 1.0
            return new_evidence, 1.0

        # Encode pool and new evidence
        pool_texts = [e.text for e in self.total_evidence_pool]
        new_texts = [e.text for e in new_evidence]
        
        pool_embs = model.encode(pool_texts, convert_to_numpy=True, normalize_embeddings=True)
        new_embs = model.encode(new_texts, convert_to_numpy=True, normalize_embeddings=True)
        
        novelty_scores = []
        for i, new_emb in enumerate(new_embs):
            # Cosine similarity is just dot product because normalized
            similarities = np.dot(pool_embs, new_emb)
            max_sim = np.max(similarities)
            novelty = 1.0 - float(max_sim)
            new_evidence[i].novelty_score = novelty
            novelty_scores.append(novelty)
            
        return new_evidence, np.mean(novelty_scores)
    
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
    
    def save_history(self, filepath: str = "prag_history.json", claim_id: str = "unknown"):
        """Save retrieval history to JSON file"""
        import json
        try:
            from logging_extension import append_framework_json
            append_framework_json(filepath.replace('.json', '.jsonl'), claim_id, self.get_retrieval_summary())
        except ImportError:
            with open(filepath, 'w') as f:
                json.dump(self.get_retrieval_summary(), f, indent=2)
