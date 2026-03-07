import time
import json
from deepseek_argument_miner import BaselineArgumentMiner
from deepseek_utils import BaselineOpenRouterClient, safe_parse_json

class BaselinePipeline:
    def __init__(self, retriever, model_name="deepseek/deepseek-v3.2"):
        self.retriever = retriever
        self.argument_miner = BaselineArgumentMiner(model_name=model_name)
        self.verdict_client = BaselineOpenRouterClient(model_name=model_name, temperature=0.1, max_tokens=512)
        
        self.system_prompt = """You are an expert biomedical fact-checker. You will be given a claim and a set of 
retrieved PubMed abstracts as evidence. Your task is to determine whether the claim 
is SUPPORTED or REFUTED by the evidence."""

    def process_claim(self, claim):
        start_time = time.time()
        
        print(f"Processing Claim ID: {claim.id}")
        print(f"Claim Text: {claim.text}")
        
        total_tokens = 0
        retrieval_calls = 0
        
        # Step 2: Premise decomposition
        premises, decomp_tokens = self.argument_miner.mine_arguments(claim.text)
        total_tokens += decomp_tokens
        
        print(f"\nExtracted Premises ({len(premises)}):")
        for i, p in enumerate(premises):
            print(f"  {i+1}. {p}")
            
        # Step 3: Retrieval
        evidence_pool = {}
        
        # a) Initial retrieval
        initial_ev = self.retriever.retrieve(claim.text, top_k=5)
        retrieval_calls += 1
        for ev in initial_ev:
            evidence_pool[ev.source_id] = ev
            
        # b) Premise-grounded retrieval
        for premise in premises:
            premise_ev = self.retriever.retrieve(premise, top_k=3)
            retrieval_calls += 1
            for ev in premise_ev:
                evidence_pool[ev.source_id] = ev
                
        evidence_count = len(evidence_pool)
        print(f"\nRetrieved {evidence_count} unique evidence documents. Retrieval calls: {retrieval_calls}")
        
        # Format evidence for prompt
        evidence_text = ""
        for i, (source_id, ev) in enumerate(list(evidence_pool.items())[:20]):
            # Truncate to 400 chars
            text_trunc = ev.text[:400] + ("..." if len(ev.text) > 400 else "")
            evidence_text += f"{i+1}. [PMID: {source_id}] {text_trunc}\n\n"
            print(f"  - Ev {i+1} [PMID: {source_id}]: {text_trunc[:100]}...")
            
        # Step 4: Verdict LLM
        user_prompt = f"""CLAIM: {claim.text}

EVIDENCE:
{evidence_text}

Based on the evidence above, determine whether the claim is SUPPORTED or REFUTED.

Respond in the following JSON format only, with no additional text:
{{
  "verdict": "SUPPORT" or "REFUTE",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<2–4 sentence explanation that clearly justifies the verdict, explicitly citing specific evidence IDs and explaining how they support or contradict the claim>"
}}

Rules:
- verdict must be exactly "SUPPORT" or "REFUTE" (no other values)
- confidence must reflect how strongly the evidence supports your verdict
- If evidence is mixed or weak, confidence should be lower (0.5–0.65)
- If evidence strongly and consistently points one way, confidence should be higher (0.80–0.95)
- reasoning should be a few sentences, not just a short phrase, and should briefly weigh any conflicting or weak evidence before giving the final justification."""

        print("\nSending to Verdict LLM...")
        result = self.verdict_client.generate(self.system_prompt, user_prompt, request_logprobs=True, enable_reasoning=True)
        total_tokens += result.get('tokens', 0)
        
        content = result.get('content', '')
        logprobs_data = result.get('logprobs')

        print("\nLLM Response:")
        print(content if content else "[EMPTY RESPONSE]")

        if result.get('error'):
            print(f"Verdict LLM returned an error: {result['error']}")
            parsed_data = None
        else:
            parsed_data = safe_parse_json(content, self.verdict_client, self.system_prompt, user_prompt)

        # Handle retry
        if isinstance(parsed_data, tuple):
            parsed_json, retry_result = parsed_data
            total_tokens += retry_result.get('tokens', 0)
            logprobs_data = retry_result.get('logprobs')
        else:
            parsed_json = parsed_data
            
        if not parsed_json:
            verdict = "PARSE_ERROR"
            confidence = 0.0
            reasoning = "Failed to parse JSON constraints."
        else:
            verdict = parsed_json.get("verdict", "UNKNOWN")
            try:
                confidence = float(parsed_json.get("confidence", 0.0))
            except (ValueError, TypeError):
                confidence = 0.0
            reasoning = parsed_json.get("reasoning", "")
            
        # Extract logprob
        logprob_verdict_token = None
        if logprobs_data:
            # find first token matching SUPPORT or REFUTE
            for lp in logprobs_data:
                token_str = lp.get('token', '').strip().upper()
                if "SUPPORT" in token_str or "REFUTE" in token_str:
                    logprob_verdict_token = lp.get('logprob')
                    break
                    
        ground_truth = claim.metadata.get('label', 'UNKNOWN')
        correct = (verdict == ground_truth) if verdict in ["SUPPORT", "REFUTE"] else False
        
        process_time = time.time() - start_time
        
        print(f"\nGROUND TRUTH: {ground_truth}")
        print(f"METRICS: tok={total_tokens} retr={retrieval_calls} ev={evidence_count} conf={confidence} time={process_time:.2f}s")
        
        return {
            "claim_id": claim.id,
            "claim_text": claim.text,
            "verdict": verdict,
            "ground_truth": ground_truth,
            "correct": correct,
            "confidence": confidence,
            "logprob_verdict_token": logprob_verdict_token,
            "reasoning": reasoning,
            "token_count": total_tokens,
            "retrieval_calls": retrieval_calls,
            "evidence_count": evidence_count,
            "processing_time_seconds": process_time
        }
