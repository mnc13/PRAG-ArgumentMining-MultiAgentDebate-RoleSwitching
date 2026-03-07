import time
from gpt_utils import GPTClient, parse_verdict_response
from gpt_argument_miner import GPTArgumentMiner


class GPTPipeline:
    """
    Fact-checking pipeline backed by GPT (default: gpt-5-mini).

    gpt-5-mini is a reasoning model: its max_completion_tokens budget covers
    BOTH internal chain-of-thought AND visible output. We use 16 000 tokens
    for the verdict call so there is headroom after reasoning completes.
    """

    SYSTEM_PROMPT = (
        "You are an expert biomedical fact-checker. "
        "You will receive a claim and a set of PubMed abstracts as evidence. "
        "Analyse the evidence carefully, then state your conclusion. "
        "At the very end of your response you MUST include the following three lines "
        "exactly as shown (replace the angle-bracket parts):\n\n"
        "FINAL VERDICT: SUPPORT\n"
        "CONFIDENCE: 0.85\n"
        "REASONING: <five to seven sentences citing specific evidence IDs>\n\n"
        "Use SUPPORT if the evidence supports the claim, REFUTE if it contradicts it. "
        "CONFIDENCE must be a decimal between 0.0 and 1.0."
    )

    def __init__(self, retriever, model_name="gpt-5-mini"):
        self.retriever      = retriever
        self.argument_miner = GPTArgumentMiner(model_name=model_name)
        # 16 000 tokens: reasoning model needs large budget so reasoning + output both fit
        self.verdict_client = GPTClient(model_name=model_name, temperature=0.1, max_tokens=16000)

    def process_claim(self, claim) -> dict:
        start_time   = time.time()
        total_tokens = 0

        print(f"Processing Claim ID  : {claim.id}")
        print(f"Claim Text           : {claim.text}")

        # ── 1. Premise decomposition ───────────────────────────────────
        premises, decomp_tokens = self.argument_miner.mine_arguments(claim.text)
        total_tokens += decomp_tokens

        print(f"\nExtracted Premises ({len(premises)}):")
        for i, p in enumerate(premises):
            print(f"  {i+1}. {p}")

        # ── 2. Evidence retrieval ──────────────────────────────────────
        evidence_pool   = {}
        retrieval_calls = 0

        for ev in self.retriever.retrieve(claim.text, top_k=5):
            evidence_pool[ev.source_id] = ev
        retrieval_calls += 1

        for premise in premises:
            for ev in self.retriever.retrieve(premise, top_k=3):
                evidence_pool[ev.source_id] = ev
            retrieval_calls += 1

        evidence_count = len(evidence_pool)
        print(
            f"\nRetrieved {evidence_count} unique evidence documents. "
            f"Retrieval calls: {retrieval_calls}"
        )

        # ── 3. Build evidence text (cap at 20 docs, 400 chars each) ───
        evidence_lines = []
        for i, (source_id, ev) in enumerate(list(evidence_pool.items())[:20]):
            trunc = ev.text[:400] + ("..." if len(ev.text) > 400 else "")
            evidence_lines.append(f"{i+1}. [PMID: {source_id}] {trunc}")
            print(f"  - Ev {i+1} [PMID: {source_id}]: {trunc[:100]}...")

        evidence_text = "\n\n".join(evidence_lines)

        # ── 4. Verdict prompt ──────────────────────────────────────────
        user_prompt = (
            f"CLAIM: {claim.text}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            "Analyse the evidence above and determine whether the claim is SUPPORTED "
            "or REFUTED. Walk through the key evidence briefly, then end your response "
            "with EXACTLY these three lines (no extra text after them):\n\n"
            "FINAL VERDICT: SUPPORT   ← or REFUTE\n"
            "CONFIDENCE: 0.00         ← decimal 0.0–1.0\n"
            "REASONING: <your 1–3 sentence justification citing PMID numbers>\n"
        )

        print("\nSending to Verdict LLM (GPT)...")
        result        = self.verdict_client.generate(self.SYSTEM_PROMPT, user_prompt)
        total_tokens += result.get("tokens", 0)

        content = result.get("content", "") or ""
        print("\nLLM Response:")
        print(content if content else "[EMPTY RESPONSE]")

        if result.get("error"):
            print(f"Verdict LLM error: {result['error']}")
            verdict    = "API_ERROR"
            confidence = 0.0
            reasoning  = result["error"]
        else:
            parsed     = parse_verdict_response(content)
            verdict    = parsed["verdict"]
            confidence = parsed["confidence"]
            reasoning  = parsed["reasoning"]

        # ── 5. Metrics ─────────────────────────────────────────────────
        ground_truth = claim.metadata.get("label", "UNKNOWN")
        correct      = (verdict == ground_truth) if verdict in ("SUPPORT", "REFUTE") else False
        process_time = time.time() - start_time

        print(f"\nGROUND TRUTH : {ground_truth}")
        print(f"VERDICT      : {verdict}  (correct={correct})")
        print(
            f"METRICS      : tok={total_tokens} retr={retrieval_calls} "
            f"ev={evidence_count} conf={confidence:.2f} time={process_time:.2f}s"
        )

        return {
            "claim_id":                claim.id,
            "claim_text":              claim.text,
            "verdict":                 verdict,
            "ground_truth":            ground_truth,
            "correct":                 correct,
            "confidence":              confidence,
            "reasoning":               reasoning,
            "token_count":             total_tokens,
            "retrieval_calls":         retrieval_calls,
            "evidence_count":          evidence_count,
            "processing_time_seconds": process_time,
        }