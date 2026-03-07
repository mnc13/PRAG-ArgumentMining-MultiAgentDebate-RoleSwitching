# PRAG Framework — Pipeline Overview
> **Last updated:** March 2026 | Entry point: `framework/run_eval_extended.py`

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Entry Point: `run_eval_extended.py`](#2-entry-point-run_eval_extendedpy)
3. [Stage 1: Data Loading](#3-stage-1-data-loading)
4. [Stage 2: Preprocessing & Extraction](#4-stage-2-preprocessing--extraction)
5. [Stage 3: Argument Mining](#5-stage-3-argument-mining)
6. [Stage 4: Initial RAG Retrieval](#6-stage-4-initial-rag-retrieval--pubmed-faiss)
7. [Stage 5: Evidence Negotiation & Arbitration](#7-stage-5-evidence-negotiation--arbitration)
8. [Stage 6–7: Multi-Agent Debate (MAD) Courtroom Proceedings](#8-stage-67-multi-agent-debate-mad-courtroom-proceedings)
9. [Stage 8: Role-Switching (Legal Consistency Check)](#9-stage-8-role-switching-legal-consistency-check)
10. [Stage 9: Judicial Panel Evaluation](#10-stage-9-judicial-panel-evaluation)
11. [Stage 10–11: Final Verdict Generation](#11-stage-1011-final-verdict-generation)
12. [Extension Layer: Metrics Logging](#12-extension-layer-metrics-logging)
13. [Outputs & Artifacts](#13-outputs--artifacts)
14. [Risk Analysis](#14-risk-analysis)
15. [Recommendations](#15-recommendations)

---

## 1. System Architecture Overview

The PRAG (Progressive RAG) framework is a **multi-agent medical fact-checking system** that models the COVID-19 claim verification process as a **courtroom legal proceeding**. It operates on the `Check-COVID` dataset and produces SUPPORT / REFUTE / INCONCLUSIVE verdicts for each claim.

### High-Level Flow

```
run_eval_extended.py  (entry point + metrics wrapper)
        │
        ├── monkey-patches LLM APIs (token counting)
        ├── patches RAG retriever (retrieval counting)
        └── calls main_pipeline.main()
                │
                ├─ [1] Data Load → claims from covidCheck_test_no_NEI.json
                ├─ [2] Preprocessing → ClaimExtractor
                ├─ [3] Argument Mining → DeepSeek-R1 decomposition
                ├─ [4] Initial RAG → PubMedRetriever (FAISS)
                ├─ [5] Evidence Negotiation → EvidenceNegotiator (6-point procedure)
                ├─ [6-7] MAD Proceedings → MADOrchestrator (up to 10 rounds)
                │          ├─ Per-round: Evidence Discovery (P-RAG)
                │          ├─ Per-round: Argument Generation
                │          ├─ Per-round: Expert Witness Testimony
                │          ├─ Per-round: Self-Reflection (SelfReflection)
                │          ├─ Per-round: Critic Evaluation (CriticAgent)
                │          └─ Adaptive Convergence Check → stops early
                ├─ [8] Role-Switching → RoleSwitcher
                ├─ [9] Judicial Panel → JudicialPanel (3 LLM judges)
                ├─ [11] Final Verdict → FinalVerdict (confidence + reasoning)
                └─ ExtensionState → claims_added.jsonl, runs_added.jsonl
```

### Models Used (as observed in logs)

| Role | Model | Provider |
|------|-------|----------|
| Argument Miner / Core LLM | `deepseek/deepseek-r1` | OpenRouter |
| Proponent / Opponent Agents | Fixed via `personas.py` AGENT_SLOTS | OpenRouter |
| Expert Witness (dynamic) | `expertise_extractor` config | OpenRouter |
| Critic Agent | `personas.py` AGENT_SLOTS critic | OpenRouter |
| Consistency Analyzer (Role-Switch) | `deepseek/deepseek-chat` | OpenRouter |
| Judge 1 | `deepseek/deepseek-r1` | OpenRouter |
| Judge 2 | `nousresearch/hermes-3-llama-3.1-405b` | OpenRouter |
| Judge 3 | `qwen/qwen3-235b-a22b-2507` | OpenRouter |
| Embedding (FAISS retrieval) | `sentence-transformers/all-MiniLM-L6-v2` | Local |

---

## 2. Entry Point: `run_eval_extended.py`

**File:** `framework/run_eval_extended.py`

This is the **primary runner script** that wraps `main_pipeline.py` non-destructively. It performs the following before launching the pipeline:

### 2.1 Argument Parsing

```bash
python run_eval_extended.py --limit 1 --offset 0 --runs 1 --force --inconclusive-policy A
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--limit` | 1 | Number of claims to process |
| `--offset` | 0 | Skip first N claims |
| `--runs` | 1 | Number of repeated runs (multi-run stability) |
| `--force` | False | Re-process already-processed claims |
| `--inconclusive-policy` | `A` | How to handle INCONCLUSIVE verdicts: A→SUPPORT, B→REFUTE, C→Exclude |

### 2.2 Monkey Patching (Non-Destructive Instrumentation)

Before calling `main_pipeline.main()`, `apply_monkey_patches()` intercepts:

- **`requests.post`** (OpenRouter calls) — captures `prompt_tokens`, `completion_tokens`, `total_tokens` from every HTTP response JSON and accumulates them in `ExtensionState`.
- **`openai.resources.chat.completions.Completions.create`** — captures `usage` from ChatCompletion responses.
- **`openai.resources.responses.Responses.create`** — for GPT-5/Responses API (uses `input_tokens` / `output_tokens` field names).
- **`rag_engine.PubMedRetriever.retrieve`** — increments `current_claim_retrievals` counter and `current_claim_evidence` per call.
- **`final_verdict.FinalVerdict.generate_verdict`** — after the verdict is generated, `extract_and_log_claim_metrics()` is called automatically.

All patches include a `_patched` attribute guard to prevent double-patching on repeated runs.

### 2.3 Multi-Run Support

For `--runs N > 1`, the loop runs the full pipeline N times per claim. On all runs except the last, `--no-mark-processed` is passed so the claim is not locked as done. This enables **repeated-run stability testing**.

---

## 3. Stage 1: Data Loading

**File:** `framework/main_pipeline.py` (lines 53–70)  
**Class:** `DataLoader` → `data_loader.py`

- Loads **`Check-COVID/test/covidCheck_test_no_NEI.json`** (test split, no NEI labels).
- Each claim object contains: `id`, `text`, `metadata` (including `label`: SUPPORT or REFUTE).
- The pipeline maintains **`framework/outcome/processed_claims.txt`** to track processed claim IDs. Skip logic prevents re-processing unless `--force` is set.
- Claims are sliced by `[offset : offset + limit]` immediately after loading.

**Processed Claims Format:**
```
5f889c84e95347460249fc52:0    ← run 0 of this claim done
5f889c84e95347460249fc52      ← all runs of this claim done (only if --no-mark-processed absent)
```

---

## 4. Stage 2: Preprocessing & Extraction

**File:** `framework/main_pipeline.py` (line 108–113)  
**Class:** `ClaimExtractor` → `preprocessing.py`

- Copies the raw claim text into an `extracted_claim` object.
- Preserves `id` and `metadata` from the original claim.
- Currently a pass-through (`preprocessing.py` is minimal, 384 bytes).

---

## 5. Stage 3: Argument Mining

**File:** `framework/main_pipeline.py` (lines 115–126)  
**Class:** `ArgumentMiner` → `agent_workflow.py`  
**LLM:** DeepSeek-R1 via OpenRouter

The claim text is decomposed into atomic, verifiable **premises/sub-claims** using an LLM.

**What happens:**
1. `ArgumentMiner(llm).mine_arguments(extracted_claim)` sends the claim to DeepSeek-R1.
2. Returns an `argument` object with a `premises` list (typically 5–10 sub-premises).
3. These premises are used as queries for the Negotiation Engine's **premise-grounded shared retrieval**.

**Example (from execution log):**
```
Claim: "Cloth masks are just as effective at preventing infection as medical masks."
→ Premises:
  1. Cloth masks reduce the risk of respiratory infection when used properly.
  2. Medical masks reduce the risk of respiratory infection when used properly.
  3. The filtration efficiency of cloth masks is equivalent to that of medical masks.
  ... (10 total premises)
```

---

## 6. Stage 4: Initial RAG Retrieval — PubMed FAISS

**File:** `framework/main_pipeline.py` (lines 128–148)  
**Class:** `PubMedRetriever` → `rag_engine.py`

### Infrastructure
| File | Size | Description |
|------|------|-------------|
| `pubmed_faiss.index` | ~1.45 GB | FAISS IndexFlatIP (normalized inner product) |
| `pubmed_meta.jsonl` | ~1.0 GB | JSON metadata per PubMed article |
| `pubmed_meta_offsets.npy` | ~7.5 MB | Byte offsets for fast disk-seek into meta file |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim sentence embeddings |

### Retrieval Process
1. Encode the claim text using `SentenceTransformer` with `normalize_embeddings=True`.
2. Run `faiss_index.search(query_embedding, top_k=5)` → gets cosine similarity scores and indices.
3. Seek into `pubmed_meta.jsonl` using pre-computed offsets.
4. Build `Evidence` objects: `text = "[Journal Year] Title\n\nAbstract"`, `source_id = PMID`.

**Output:** 5 initial evidence items with relevance scores.

---

## 7. Stage 5: Evidence Negotiation & Arbitration

**File:** `framework/main_pipeline.py` (lines 150–177)  
**Class:** `EvidenceNegotiator` → `negotiation_engine.py`

This is a **6-point pre-debate evidence preparation protocol** modelled on legal discovery:

| Step | Action | Details |
|------|--------|---------|
| **Step 1** | Premise-Grounded Shared Retrieval | Runs `retriever.retrieve(premise, top_k=3)` for each of the N mined premises. Deduplicates. Produces a **shared pool**. |
| **Step 2** | Stance / Perspective Retrieval | LLM generates a proponent-conditioned query and an opponent-conditioned query. Retrieves `top_k=3` for each → **proponent_pool** and **opponent_pool**. |
| **Step 3** | Multi-Agent Negotiation Injection | Both sides "review" each other's pools (LLM simulation). Currently simulated; LLM responses are generated but not deeply parsed for the pool update. |
| **Step 4** | Judicial Arbitration | For each candidate evidence item, LLM calculates `weight = relevance × credibility` (both 0–1). Items with `weight > 0.5` are admitted; `0.1–0.5` are disputed. |
| **(Saved)** | Negotiation State JSON | Saved to `outcome/negotiation_state_{claim_id}_{run_index}.json`. |
| **Extract** | Admissible Evidence Set | The set of evidence IDs with `weight > 0.5` is extracted as `final_evidence_set` to pass to MAD. |

**Admissibility Weight Formula:**
```
weight = round(relevance × credibility, 3)

where:
  relevance ∈ [0, 1]   — How directly does the evidence address the claim premises?
  credibility ∈ [0, 1] — Does the evidence come from reliable scientific sources?
  
Admitted if weight > 0.5
Disputed  if 0.1 < weight ≤ 0.5
Excluded  if weight ≤ 0.1
```

**Observed in logs:** Typically 20–30 candidates, with 15–26 admitted (often all receive weight = 0.75 in practice, suggesting a calibration issue).

---

## 8. Stage 6–7: Multi-Agent Debate (MAD) Courtroom Proceedings

**File:** `framework/main_pipeline.py` (lines 179–208)  
**Classes:** `MADOrchestrator` → `mad_orchestrator.py`, `ProgressiveRAG` → `prag_engine.py`

### 8.1 Initialization

```python
prag = ProgressiveRAG(retriever, llm)
mad = MADOrchestrator(claim, final_evidence_set, [], prag)
```

Agents initialized from `personas.py` `AGENT_SLOTS`:
- `proponent` → "Plaintiff Counsel"
- `opponent` → "Defense Counsel"
- `judge` → "The Court"
- `experts` → empty list (dynamic)

### 8.2 Per-Round Structure (up to 10 rounds)

Each round of the debate executes 5 steps in sequence:

#### Step 1: Evidence Discovery (Integrative Discovery)
For each side (proponent, then opponent):

1. Agent proposes a **gap query** (`propose_query_gap(debate_context)`) — identifies what evidence it needs.
2. Reflection engine provides a **reflection gap** from the previous round's self-reflection.
3. These are combined into a `discovery_prompt`.
4. P-RAG formulates a targeted search query (`formulate_query(debate_context, discovery_prompt)`).
5. The `judge` agent **refines the query** (`refine_query(original_query, debate_context)`).
6. `prag.retrieve_progressive(refined_query, top_k=3)` is called → new evidence fetched with novelty filtering.
7. Accepted evidence is added to `evidence_pool`.

#### Step 2: Argument Generation
Each side calls `agent.generate_argument(claim, evidence_pool, debate_transcript)` → LLM generates a legally-styled argument referencing specific Evidence IDs.

#### Step 3: Expert Witness Testimony
For each side:
- Agent proposes an expert type (`request_expert(debate_transcript)`).
- The judge evaluates if the request is granted (`evaluate_expert_request(side, expert_req)`).
- If granted, `extract_single_expert(expert_type, claim_text)` creates a dynamic expert config.
- Expert generates testimony using `generate_argument()`.

#### Step 4: Multi-Round Self-Reflection
For each side, `self_reflection.perform_round_reflection(agent, side, round_num, claim_text)` returns:
- `logic`: 0–1 — logical coherence of the agent's arguments
- `novelty`: 0–1 — novelty of evidence introduced
- `rebuttal`: 0–1 — quality of counter-argument
- `total_score` = weighted combination
- `discovery_need` — text describing what evidence is still needed (fed back to Step 1 next round)

#### Step 5: Critic Agent Evaluation
`CriticAgent.evaluate_round(round_num, claim_text, debate_transcript)` returns:
- `recommendations` — per-side list of improvements
- `debate_resolved` — boolean signal that all premises are addressed

### 8.3 Adaptive Convergence (Early Stopping)

After each round (starting from round 2), the MAD checks 4 stopping conditions in order:

| Condition | Threshold | Stop Reason |
|-----------|-----------|-------------|
| Reflection score delta | `|delta| < 0.05` | "Reflection plateau" |
| Critic resolution signal | `debate_resolved == True` | "Critic resolution" |
| Evidence novelty | `avg_novelty < 0.1` (2 consecutive rounds) | "Novelty stabilization" |
| Judicial signal | `judge.check_debate_completion()` | "Judicial signal" |

**Observed in logs:** Most debates stop at **rounds 2–4** via judicial signal or reflection plateau.

### 8.4 Progressive RAG (P-RAG) Engine — `prag_engine.py`

On each `retrieve_progressive()` call:

1. Retrieve `top_k=3` raw evidence via FAISS.
2. Compute **novelty scores**: `novelty_i = 1 - max_cosine_sim(doc_i, existing_pool)` using SentenceTransformer embeddings.
3. Filter: keep evidence where `novelty ≥ 0.2` (novelty threshold).
4. Compute redundancy ratio: `redundant_count / total_retrieved`.
5. Compute relevance gain vs previous round.
6. Check stopping criteria (max iterations, high redundancy `>0.7`, diminishing gain `<0.05`).
7. Log to `retrieval_history`.

---

## 9. Stage 8: Role-Switching (Legal Consistency Check)

**File:** `framework/main_pipeline.py` (lines 189–193)  
**Class:** `RoleSwitcher` → `role_switcher.py`

1. The proponent and opponent agents are **swapped in-place** (Agent A becomes Defense Counsel, Agent B becomes Plaintiff Counsel).
2. `mad.reset_state()` clears transcript, resets P-RAG, and resets reflection history.
3. A **new full debate** runs with `max_rounds=10` (also with adaptive convergence), saved with suffix `_switched`.
4. `check_consistency(original_result, switched_result)` calls `deepseek/deepseek-chat` to:
   - Analyze if Agent A maintains logical consistency when switching from Plaintiff → Defense.
   - Analyze if Agent B maintains logical consistency when switching from Defense → Plaintiff.
   - Report any contradictions.
   - Return an overall consistency score (0–10).
5. The consistency report is saved to `artifacts/outcome/all_output_jsons/role_switch_report.jsonl`.

**Use in Final Verdict:** Role-switch consistency (`consistent` keyword detection) adds `+0.10` to the final confidence score; inconsistency subtracts `-0.05`.

---

## 10. Stage 9: Judicial Panel Evaluation

**File:** `framework/main_pipeline.py` (lines 195–215)  
**Class:** `JudicialPanel` → `judge_evaluator.py`

### Three Independent Judges

| Judge | Model | Temperature |
|-------|-------|-------------|
| Judge 1 | `deepseek/deepseek-r1` | 0.3 |
| Judge 2 | `nousresearch/hermes-3-llama-3.1-405b` | 0.3 |
| Judge 3 | `qwen/qwen3-235b-a22b-2507` | 0.3 |

### 5-Stage Holistic Evaluation (per judge)

Each judge performs a 5-stage structured evaluation using a single detailed prompt:

| Stage | Focus | Score Type |
|-------|-------|------------|
| Stage 1 - Case Reconstruction | Identify core claim, plaintiff/defense arguments | Narrative |
| Stage 2 - Evidence & Testimony Weighting | Relevance, scientific credibility, testimony strength | `evidence_strength` 0–10 |
| Stage 3 - Logical Coherence | Detect fallacies, contradictions, unsupported leaps | `argument_validity` 0–10 |
| Stage 4 - Scientific/Technical Consistency | Alignment with biomedical consensus | `scientific_reliability` 0–10 |
| Stage 5 - Discovery Rigor & Transparency | P-RAG query evolution, novelty, court's query refinement | Narrative |
| Stage 6 - Judicial Verdict | SUPPORTED / NOT SUPPORTED / INCONCLUSIVE | Binary verdict |

Each judge's response is parsed as JSON: `{ claim_summary, evidence_strength, argument_validity, scientific_reliability, verdict, reasoning }`.

### Verdict Aggregation (Majority Voting)

```python
vote_counts = Counter([v['verdict'] for v in judge_verdicts])
final_verdict = vote_counts.most_common(1)[0][0]
```

Majority and dissenting opinions are synthesized from individual judge reasoning text.

**Observed in logs (claim 5f889c84...):**
```
Judge 1 (deepseek-r1):           NOT SUPPORTED  | Ev:6, Arg:7, Sci:6
Judge 2 (llama-405b-instruct):   NOT SUPPORTED  | Ev:7, Arg:8, Sci:8
Judge 3 (qwen3-235b):            INCONCLUSIVE   | Ev:7, Arg:6, Sci:7
Final: NOT SUPPORTED (2-1 majority)
```

Results saved to `artifacts/outcome/all_output_jsons/judge_evaluation.jsonl`.

---

## 11. Stage 10–11: Final Verdict Generation

**File:** `framework/main_pipeline.py` (lines 211–222)  
**Class:** `FinalVerdict` → `final_verdict.py`

### Verdict Mapping

```
Judicial Panel:  SUPPORTED     → Pipeline Verdict: SUPPORT
                 NOT SUPPORTED → Pipeline Verdict: REFUTE
                 INCONCLUSIVE  → Pipeline Verdict: SUPPORT (default – plaintiff counsel wins)
```

### Confidence Calculation

The final confidence score is computed from three components:

#### Component 1: Consensus Strength (max 0.80)
```python
consensus_strength = winning_votes / total_votes
# e.g., 2-1 majority: 2/3 = 0.667
margin_score = consensus_strength * 0.8
# = 0.667 × 0.8 = 0.533
```

#### Component 2: Quality Score from Judge Rubric (max 0.30)
```python
avg_evidence_strength     = mean(v['evidence_strength'] for v in judge_verdicts)   # 0-10
avg_argument_validity     = mean(v['argument_validity'] for v in judge_verdicts)    # 0-10
avg_scientific_reliability= mean(v['scientific_reliability'] for v in judge_verdicts)# 0-10

quality_score = ((avg_ev + avg_arg + avg_sci) / 30) * 0.3
# e.g., (6+7.5+7)/30 × 0.3 = 0.205
```

#### Component 3: Adjustments
```python
# Role-switch consistency
if consistent: adjustments += 0.10
else:          adjustments -= 0.05

# Self-reflection adjustment (capped at -0.15 on downside)
reflection_adj = self_reflection_result['self_reflection']['confidence_adjustment']
if reflection_adj < 0:
    reflection_adj = max(-0.15, reflection_adj)
adjustments += reflection_adj
```

#### Final Confidence
```python
final_confidence = margin_score + quality_score + adjustments
# Floor to 0.1 if consensus > 0.5 but computed value < 0.1
final_confidence = clamp(final_confidence, 0.0, 1.0)
```

**Observed in logs:** Claim `5f889c84` confidence = **0.876** (2-1 majority × 0.8 + quality scores + role-switch consistent).

### Outputs

- `outcome/all_verdicts.jsonl` — appended with `{claim_id, verdict, confidence, ground_truth, correct}`
- `artifacts/outcome/all_output_jsons/final_verdict.jsonl` — full detailed verdict
- `outcome/processed_claims.txt` — claim ID and run key marked as done

---

## 12. Extension Layer: Metrics Logging

**Files:** `logging_extension.py`, `metrics_extension.py`

After `generate_verdict()` completes, the monkey-patched wrapper calls `extract_and_log_claim_metrics()` which:

1. Reads `final_verdict.jsonl`, `judge_evaluation.jsonl`, `debate_transcript.jsonl`, `debate_transcript_switched.jsonl` from disk.
2. Extracts ground truth, predicted label, confidence, normal rounds, switched rounds.
3. Records judge votes dict.
4. Appends a full claim record to `artifacts/metrics/claims_added.jsonl`.
5. Prints the **Extra Metrics Block**:
   ```
   === EXTRA METRICS (ADDED) ===
   [CLAIM {id}] rounds_norm=4 rounds_switch=2 tok=223690 retr=25 ev=77 conf=0.876
   [CLAIM {id}] judges: NOT SUPPORTED, NOT SUPPORTED, INCONCLUSIVE | kappa_mean=0.333
   =============================
   ```
6. Resets `ExtensionState` for the next claim.

After all claims are processed, `compile_and_log_run_summary()` computes dataset-level metrics and appends to `artifacts/metrics/runs_added.jsonl` and `artifacts/metrics/run_reports_added.md`.

---

## 13. Outputs & Artifacts

### Per-Claim Outputs (in `framework/outcome/`)

| File | Description |
|------|-------------|
| `logs/execution_log_{claim_id}_{run_id}.txt` | Full dual-logged execution transcript |
| `negotiation_state_{claim_id}_{run_index}.json` | Evidence negotiation state (pools, weights) |
| `all_verdicts.jsonl` | Compact verdict record per claim |
| `processed_claims.txt` | Progress tracker |

### Intermediate JSON Artifacts (in `artifacts/outcome/all_output_jsons/`)

| File | Description |
|------|-------------|
| `final_verdict.jsonl` | Full verdict dict with reasoning |
| `judge_evaluation.jsonl` | Per-judge scores and aggregated result |
| `debate_transcript.jsonl` | Full MAD transcript (normal) |
| `debate_transcript_switched.jsonl` | Full MAD transcript (role-switched) |
| `role_switch_report.jsonl` | Consistency analysis report |
| `judge_visibility.jsonl` | PRAG query evolution and totals |
| `prag_history.jsonl` | P-RAG retrieval history per claim |
| `self_reflection.json` | Self-reflection history |

### Metrics Artifacts (in `artifacts/metrics/`)

| File | Description |
|------|-------------|
| `claims_added.jsonl` | Per-claim: tokens, rounds, retrievals, votes |
| `runs_added.jsonl` | Per-run: full classification + efficiency metrics |
| `run_reports_added.md` | Human-readable run summary |

---

## 14. Risk Analysis

### 🔴 Critical Risks

| Risk | Location | Impact |
|------|----------|--------|
| **KS Stability is hardcoded** | `run_eval_extended.py` line 406 | The `ks` dict used in the run summary is a static placeholder `{D_t: {1:0.8, 2:0.4, ...}}`. Real KS statistics are NOT computed from actual per-round data. |
| **Single claim per run** | `main_pipeline.py` default `--limit 1` | The pipeline processes one claim at a time; large-scale evaluation requires orchestration or batching. |
| **Negotiation weight calibration** | `negotiation_engine.py` `_calculate_weight()` | In practice, logs show all evidence gets `weight=0.75` (likely LLM fallback). The filtering threshold of 0.5 admits everything, defeating the purpose of selective admission. |
| **INCONCLUSIVE defaults to SUPPORT** | `final_verdict.py` line 52 | Both the Judicial Panel fallback and the main verdict mapping route INCONCLUSIVE to SUPPORT/proponent, introducing a systematic bias. |

### 🟡 Moderate Risks

| Risk | Location | Impact |
|------|----------|--------|
| **Expert testimony repetition** | `judge_evaluator.py`, `mad_orchestrator.py` | Logs show expert witnesses often copy prior expert testimony nearly verbatim (same model, similar system prompt). Diversity is low. |
| **Groq token counter not populated** | `logging_extension.py` `current_claim_groq_tokens` | Groq is tracked in `ExtensionState` but `groq_client.py` has no corresponding patch in `apply_monkey_patches()`. |
| **Token overflow risk in judge prompts** | `judge_evaluator.py` | The full debate transcript, PRAG metrics, critic evaluations, and reflection history are all injected into a single judge prompt. Multi-round debates may exceed context limits. |
| **Argument truncation in verdict reasoning** | `final_verdict.py` line 236 | Arguments are truncated to 300 chars; may lose critical nuance in reasoning chain. |
| **Race condition on verdict file** | `main_pipeline.py` lines 238–240 | Multiple runs appending to `all_verdicts.jsonl` concurrently could corrupt the file if parallelism were introduced. |

### 🟢 Low Risks

| Risk | Location | Impact |
|------|----------|--------|
| **FAISS cold start** | `rag_engine.py` | First claim in a session loads 1.45 GB FAISS index into memory — significant latency. |
| **OpenRouter rate limits** | All LLM calls | No retry/backoff logic visible; rate limit errors will propagate as exceptions. |
| **Missing `run_index` suffix** | `main_pipeline.py` line 162 | `negotiation_state_{claim_id}.json` overwrites on each run (no `run_index` suffix). Multi-run tracking may be incorrect for negotiation state. |

---

## 15. Recommendations

### Immediate Fixes

1. **Fix KS Stability** — Replace the hardcoded `ks` dictionary with real `analyze_stability()` calls using per-claim confidence traces (accumulated in `ExtensionState.stability_traces`).

2. **Fix negotiation weight calibration** — Log raw LLM responses for `_calculate_weight()`. Consider a stricter threshold (e.g., 0.6) or per-evidence-type calibration to ensure the filtering step is non-trivial.

3. **Patch Groq client** — Add the same requests/session-level token interception to `groq_client.py` as done for OpenRouter in `apply_monkey_patches()`.

4. **Add `run_index` to negotiation state filename** — `negotiation_state_{claim_id}_{run_index}.json` to avoid overwrites in multi-run mode.

### Improvements

5. **INCONCLUSIVE handling** — Rather than defaulting to SUPPORT, consider a separate "abstain" class or score it as 0.5 confidence and flag separately in evaluation metrics.

6. **Expert witness deduplification** — Track previously generated expert testimony summaries and add a diversity constraint or use different model configurations per expert type.

7. **Token-safe judge prompt** — Truncate debate transcript / reflection history to a configurable max token budget before constructing the judge evaluation prompt.

8. **Argument truncation** — Increase or remove the 300-char truncation in `final_verdict._generate_reasoning()` to preserve argument quality in the output.

9. **Retry/backoff for LLM calls** — Wrap all `llm.generate()` calls in a retry decorator with exponential backoff for rate limit resilience.

10. **Batch evaluation mode** — Add a `--batch-size N` flag to `run_eval_extended.py` and run N claims sequentially within a single FAISS-loaded session to amortize index loading cost.
