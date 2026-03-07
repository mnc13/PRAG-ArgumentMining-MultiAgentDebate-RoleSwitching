# PRAG Framework — Metrics Audit
> **Last updated:** March 2026 | Sources: `metrics_extension.py`, `logging_extension.py`, `run_eval_extended.py`, `final_verdict.py`, `judge_evaluator.py`, `prag_engine.py`, `negotiation_engine.py`, `mad_orchestrator.py`, `self_reflection.py`

---

## Table of Contents
1. [Metrics Overview](#1-metrics-overview)
2. [Per-Claim Operational Metrics](#2-per-claim-operational-metrics)
3. [Evidence Quality Metrics (Negotiation)](#3-evidence-quality-metrics-negotiation)
4. [P-RAG Progressive Retrieval Metrics](#4-p-rag-progressive-retrieval-metrics)
5. [Self-Reflection Scoring](#5-self-reflection-scoring)
6. [Judicial Panel Scores](#6-judicial-panel-scores)
7. [Final Verdict Confidence Score](#7-final-verdict-confidence-score)
8. [Dataset-Level Classification Metrics](#8-dataset-level-classification-metrics)
9. [Judge Inter-Rater Reliability (Cohen's Kappa)](#9-judge-inter-rater-reliability-cohens-kappa)
10. [AUC-ROC & Threshold Sweep](#10-auc-roc--threshold-sweep)
11. [Debating Stability (KS Statistic)](#11-debating-stability-ks-statistic)
12. [Efficiency Metrics](#12-efficiency-metrics)
13. [Inconclusive Sensitivity Analysis](#13-inconclusive-sensitivity-analysis)
14. [Metrics Data Flow Summary](#14-metrics-data-flow-summary)
15. [Risk Analysis](#15-risk-analysis)
16. [Recommendations](#16-recommendations)

---

## 1. Metrics Overview

The PRAG framework produces metrics at **four levels of granularity**:

| Level | Scope | Where Computed | Stored |
|-------|-------|----------------|--------|
| **Evidence-level** | Per candidate evidence item | `negotiation_engine.py` | `negotiation_state_{id}.json` |
| **Round-level** | Per MAD debate round | `mad_orchestrator.py`, `prag_engine.py`, `self_reflection.py` | `debate_transcript.jsonl` |
| **Claim-level** | Per claim processed | `run_eval_extended.py` → `extract_and_log_claim_metrics()` | `claims_added.jsonl` |
| **Dataset-level** | Across all claims in a run | `run_eval_extended.py` → `compile_and_log_run_summary()` | `runs_added.jsonl`, `run_reports_added.md` |

---

## 2. Per-Claim Operational Metrics

**Source:** `run_eval_extended.py` → `ExtensionState`, `extract_and_log_claim_metrics()`  
**Output file:** `artifacts/metrics/claims_added.jsonl`

| Metric | Type | Description | How Computed |
|--------|------|-------------|--------------|
| `token_total` | `int` | Total tokens consumed for this claim | Accumulated by monkey-patching all LLM API responses; incremented on every `requests.post` or `Completions.create` call during the claim's lifecycle. |
| `token_input` | `int` | Input (prompt) tokens | Summed from `prompt_tokens` field of every API response during claim processing. |
| `token_output` | `int` | Output (completion) tokens | Summed from `completion_tokens` / `output_tokens` across all API calls. |
| `token_openai` | `int` | Tokens via OpenAI API | Tracked separately via `Completions.create` and `Responses.create` patches. |
| `token_openrouter` | `int` | Tokens via OpenRouter API | Tracked separately via `requests.post` intercept on OpenRouter calls. |
| `token_groq` | `int` | Tokens via Groq API | Class variable exists (reset in `reset_claim_state()`) but **no Groq patch currently implemented** — always 0. |
| `token_models` | `dict` | Per-model token breakdown | `{model_name: {in, out, tot}}` — built from model name extracted from each API response. |
| `retrieval_calls` | `int` | Number of `PubMedRetriever.retrieve()` calls | Patched in `apply_monkey_patches()`: incremented once per `retrieve()` invocation. |
| `evidence_count` | `int` | Total evidence items returned across all retrievals | Patched: `+= len(res)` after each `retrieve()` call. |
| `rounds_normal` | `int` | Number of debate rounds in the normal MAD run | `len(debate_transcript.jsonl['rounds'])` read from disk. |
| `rounds_switched` | `int` | Number of debate rounds in the role-switched run | `len(debate_transcript_switched.jsonl['rounds'])` read from disk. |
| `total_rounds` | `int` | Sum of normal + switched rounds | `rounds_normal + rounds_switched` |
| `pred_label` | `str` | Predicted verdict (SUPPORT / REFUTE / INCONCLUSIVE) | From `final_verdict.jsonl['verdict']`. |
| `gt_label` | `str` | Ground truth label | From claim `metadata['label']`, cross-checked with `final_verdict.jsonl['ground_truth_label']`. |
| `correct` | `bool` / `None` | Whether prediction matches ground truth | `pred == gt` if `gt != UNKNOWN else None`. |
| `confidence` | `float` | Final confidence score | From `final_verdict.jsonl['confidence']`. |
| `judge_votes` | `dict` | Per-judge verdicts | `{judge_name: verdict}` extracted from `judge_evaluation.jsonl['judge_verdicts']`. |

**Console output per claim:**
```
=== EXTRA METRICS (ADDED) ===
[CLAIM {id}] rounds_norm=4 rounds_switch=2 tok=223690 retr=25 ev=77 conf=0.876
[CLAIM {id}] judges: NOT SUPPORTED, NOT SUPPORTED, INCONCLUSIVE | kappa_mean=0.333
=============================
```

> **Note on `kappa_mean` in console:** The per-claim `kappa_pair_mean` printed here is a simplified metric: `pairs_match / 3.0` (number of agreeing pairs among 3 judges, divided by 3). This is **not** Cohen's Kappa — it is a raw agreement ratio. Full Cohen's Kappa is computed at dataset level.

---

## 3. Evidence Quality Metrics (Negotiation)

**Source:** `negotiation_engine.py` → `_calculate_weight()`, `judge_arbitration()`  
**Output file:** `outcome/negotiation_state_{claim_id}_{run_index}.json`

### 3.1 Admissibility Weight

| Metric | Type | Range | Formula |
|--------|------|-------|---------|
| `relevance` | `float` | [0, 1] | LLM-rated: "How directly does this evidence address the premises of the claim?" |
| `credibility` | `float` | [0, 1] | LLM-rated: "Does the evidence come from a reliable scientific context?" |
| `weight` | `float` | [0, 1] | `round(relevance × credibility, 3)` |

**LLM Prompt for Evidence Weighting:**
```
Evaluate the scientific relevance and credibility of the following medical evidence for the claim.
CLAIM: {claim}
EVIDENCE: {evidence_text[:1000]}

Provide evaluation with:
1. Relevance: How directly does this evidence address the premises? (0.0-1.0)
2. Credibility: Does it come from a reliable scientific context? (0.0-1.0)

Respond ONLY in JSON: {"relevance": 0.0-1.0, "credibility": 0.0-1.0, "reason": "..."}
```

**Admissibility Decision Logic:**
```python
if weight > 0.5:       → ADMITTED to MAD evidence pool
elif weight > 0.1:     → DISPUTED (flagged, not yet admitted)
else:                  → EXCLUDED
```

**Why used:** Implements a pre-trial discovery filter analogous to legal admissibility rules. Prevents irrelevant or low-credibility evidence from entering the debate, theoretically improving debate quality.

**Observed Issue:** In practice, most/all evidence items receive `weight = 0.75` (the LLM fallback or insufficient calibration), meaning the filter is effectively disabled.

---

## 4. P-RAG Progressive Retrieval Metrics

**Source:** `prag_engine.py` → `retrieve_progressive()`, `_calculate_novelty()`  
**Output file:** `artifacts/outcome/all_output_jsons/prag_history.jsonl`

### 4.1 Novelty Score

| Metric | Type | Range | Formula |
|--------|------|-------|---------|
| `novelty_score` | `float` per doc | [0, 1] | `1.0 - max_cosine_similarity(doc, existing_pool)` |
| `avg_novelty` | `float` per retrieval | [0, 1] | `mean(novelty_scores)` across retrieved batch |

**Calculation Detail:**
```python
# Encode all documents in the existing evidence pool
pool_embs = model.encode(pool_texts, normalize_embeddings=True)

# For each new document:
new_emb = model.encode([new_text], normalize_embeddings=True)
similarities = np.dot(pool_embs, new_emb)   # cosine sim (normalized = dot product)
max_sim = np.max(similarities)
novelty = 1.0 - float(max_sim)
```

**Filter:** Documents with `novelty < 0.2` (novelty threshold) are **rejected**.  
**Why used:** Ensures each retrieval round produces genuinely new information rather than duplicating existing evidence. This drives the evidence diversification in the legal proceedings.

### 4.2 Redundancy Ratio

| Metric | Type | Formula |
|--------|------|---------|
| `redundancy_ratio` | `float` [0, 1] | `sum(1 for e if novelty < (1 - 0.85)) / total_retrieved` |

Where `redundancy_sim_threshold = 0.85`, so novelty threshold for "redundant" = `1 - 0.85 = 0.15`.

**Why used:** Stops retrieval if more than 70% of retrieved evidence is redundant (similarity > 0.85 with existing pool). Computational efficiency gate.

### 4.3 Relevance Gain

| Metric | Type | Formula |
|--------|------|---------|
| `avg_relevance` | `float` | `mean(e.relevance_score for e in accepted_evidence)` — FAISS inner-product score (cosine sim) |
| `relevance_gain` | `float` | `avg_relevance - prev_round_avg_relevance` |

**Why used:** Measures whether each new retrieval round improves the average quality of accepted evidence. Below threshold (`0.05`) triggers adaptive stopping.

### 4.4 P-RAG Stopping Criteria

Applied in order starting from round 2:

| Condition | Threshold | Description |
|-----------|-----------|-------------|
| Max iterations | `round_counter >= 10` | Hard cap on retrieval rounds per MAD round |
| High redundancy | `redundancy_ratio > 0.7` | Evidence pool is saturated |
| Diminishing relevance | `relevance_gain < 0.05` | New round below relevance improvement threshold |

**Log output when triggered:**
```
> [PRAG STOP] Diminishing relevance gain (-0.0240 < 0.05)
```

### 4.5 Retrieval History Record

Each `retrieve_progressive()` call logs to `retrieval_history`:
```json
{
  "round": 3,
  "query": "...",
  "context": "Round 3 - Plaintiff Counsel",
  "num_retrieved": 3,
  "num_accepted": 2,
  "num_rejected": 1,
  "avg_novelty": 0.31,
  "avg_relevance": 0.72,
  "relevance_gain": -0.024,
  "redundancy_ratio": 0.33,
  "stop_reason": "Diminishing relevance gain (-0.024 < 0.05)",
  "evidence_ids": ["32965398", "34957644"]
}
```

---

## 5. Self-Reflection Scoring

**Source:** `self_reflection.py` → `perform_round_reflection()`  
**Used in:** `mad_orchestrator.py` Step 4 of each MAD round  
**Output file:** `artifacts/outcome/all_output_jsons/self_reflection.json`

### 5.1 Core Reflection Scores (per side, per round)

| Score | Type | Range | Description |
|-------|------|-------|-------------|
| `logic` | `float` | [0, 1] | Logical coherence of the agent's arguments in this round |
| `novelty` | `float` | [0, 1] | How novel/different are the arguments compared to prior rounds |
| `rebuttal` | `float` | [0, 1] | Quality and effectiveness of counter-arguments to the opponent |
| `total_score` | `float` | [0, 1] | Weighted combination of above three scores |
| `confidence_adjustment` | `float` | [-1, 1] | Adjustment to final verdict confidence |
| `discovery_need` | `str` | — | Text description of what evidence the agent still needs |

**Why used:** Self-reflection serves two purposes:
1. **Convergence signal** — The delta of `total_score` between rounds triggers adaptive stopping.
2. **Confidence adjustment** — The winning side's final reflection's `confidence_adjustment` is added to the verdict confidence (see Section 7).

**Convergence Rule:**
```python
delta_score = total_ref_score_current - last_total_reflection_score
if abs(delta_score) < 0.05:  # plateau
    → ADAPTIVE STOP: "Reflection plateau"
```

**Log output:**
```
[Plaintiff Counsel] Performing self-reflection for Phase 3...
  * Logic: 0.80, Novelty: 0.40, Rebuttal: 0.60
  * Total Score: 0.620
  * Discovery Need: Find head‑to‑head clinical studies...
```

---

## 6. Judicial Panel Scores

**Source:** `judge_evaluator.py` → `_judge_evaluate()`  
**Output file:** `artifacts/outcome/all_output_jsons/judge_evaluation.jsonl`

### Per-Judge Scores (per claim)

| Score | Type | Range | Description | Rubric |
|-------|------|-------|-------------|--------|
| `evidence_strength` | `int` | 0–10 | Quality and relevance of admitted evidence and expert testimony | 0-3: Weak/unreliable; 4-6: Moderate; 7-10: Strong/credible |
| `argument_validity` | `int` | 0–10 | Logical coherence of advocacy; detection of fallacies | 0-3: Multiple fallacies; 4-6: Some issues; 7-10: Sound reasoning |
| `scientific_reliability` | `int` | 0–10 | Alignment with biomedical consensus; correct citation | 0-3: Contradicts consensus; 4-6: Partially aligned; 7-10: Well-aligned |
| `verdict` | `str` | SUPPORTED / NOT SUPPORTED / INCONCLUSIVE | Judge's binary verdict on the claim | N/A |
| `reasoning` | `str` | — | Textual justification with evidence citations | N/A |

**Evaluation stages:**
1. Case Reconstruction  
2. Evidence & Testimony Weighting → `evidence_strength`  
3. Logical Coherence Analysis → `argument_validity`  
4. Scientific/Technical Consistency → `scientific_reliability`  
5. Discovery Rigor & Transparency (narrative only)  
6. Judicial Verdict → `verdict`

**JSON response structure required from each judge:**
```json
{
  "claim_summary": "Brief summary...",
  "evidence_strength": 7,
  "argument_validity": 8,
  "scientific_reliability": 6,
  "verdict": "NOT SUPPORTED",
  "reasoning": "Detailed justification..."
}
```

**Why three judges at different temperatures (0.3)?** Low temperature reduces LLM stochasticity and promotes consistent, analytical reasoning appropriate for judicial evaluation. Three independent judges enable majority voting to reduce single-model bias.

---

## 7. Final Verdict Confidence Score

**Source:** `final_verdict.py` → `_calculate_confidence()`  
**Used for:** Global classification metrics (as the continuous score for AUC)

### Formula Breakdown

```
final_confidence = (Component1: consensus_strength)
                 + (Component2: quality_score)
                 + (Component3: adjustments)
```

#### Component 1 — Judicial Consensus Strength (weight 0.80)

```python
consensus_strength = winning_votes / total_votes
# 3-0: 1.0, 2-1: 0.667, 1-2: 0.333 (minority), 1-1-1: 0.333

margin_score = consensus_strength * 0.8   # Maximum: 0.8
```

**Why:** The strength of unanimity is the primary driver of confidence. A 3-0 verdict is more credible than 2-1.

#### Component 2 — Judge Quality Scores (weight 0.30)

```python
avg_evidence_strength      = mean([v['evidence_strength'] for v in judge_verdicts])      # 0-10
avg_argument_validity      = mean([v['argument_validity'] for v in judge_verdicts])       # 0-10
avg_scientific_reliability = mean([v['scientific_reliability'] for v in judge_verdicts])  # 0-10

quality_score = ((avg_ev + avg_arg + avg_sci) / 30) * 0.3
# Normalises 0-30 range to [0, 0.30]
```

**Why:** The actual quality of evidence and argumentation should modulate confidence beyond just vote counting.

#### Component 3 — Adjustments

```python
# a) Role-switching consistency bonus/penalty
if role_switch_consistent:     adjustments += 0.10
else:                          adjustments -= 0.05

# Role switch consistency check (heuristic keyword detection):
consistency_keywords    = ['consistent', 'maintained', 'coherent', 'logical']
inconsistency_keywords  = ['inconsistent', 'contradicted', 'conflicting', 'incoherent']
→ consistent_count > inconsistent_count → True

# b) Self-reflection confidence adjustment (capped)
reflection_adj = winner_reflection['self_reflection']['confidence_adjustment']
if reflection_adj < 0:
    reflection_adj = max(-0.15, reflection_adj)   # downside capped at -0.15
adjustments += reflection_adj
```

**Why:** Role-switching consistency is a proxy for argument robustness — agents that argue well from either side demonstrate genuine understanding. Self-reflection allows the winning side to signal its own confidence in its arguments.

#### Final Clamp

```python
# Ensure minimum confidence if there is consensus
if final_confidence < 0.1 and consensus_strength > 0.5:
    final_confidence = 0.1

final_confidence = max(0.0, min(1.0, final_confidence))
```

---

## 8. Dataset-Level Classification Metrics

**Source:** `metrics_extension.py` → `compute_classification_metrics()`  
**Computed by:** `run_eval_extended.py` → `compile_and_log_run_summary()`  
**Output:** `artifacts/metrics/runs_added.jsonl`

### 8.1 Input

- `y_true`: list of ground truth labels (`SUPPORT`, `REFUTE`)
- `y_pred`: list of predicted labels (from `final_verdict.jsonl` per claim)

Labels with `gt_label ∈ {UNKNOWN, None, ""}` are filtered out before computing metrics.

### 8.2 Accuracy

```
Accuracy = (number of correct predictions) / (total predictions)
         = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / n
```

### 8.3 Per-Class Precision, Recall, F1

For each class `c ∈ {SUPPORT, REFUTE, INCONCLUSIVE}`:

```
TP_c = confusion[c][c]
FP_c = sum(confusion[other][c] for other ≠ c)
FN_c = sum(confusion[c][other] for other ≠ c)
TN_c = n - TP_c - FP_c - FN_c

Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)
```

### 8.4 Macro Averages

```
Macro Precision = mean(Precision_c for c in classes)
Macro Recall    = mean(Recall_c for c in classes)
Macro F1        = mean(F1_c for c in classes)
Balanced Accuracy = Macro Recall   ← (by definition)
```

> **Why Balanced Accuracy = Macro Recall?** For multi-class classification, balanced accuracy is defined as the average of per-class recall (sensitivity). This is equivalent to macro-averaged recall.

### 8.5 Micro Averages (Implementation Note)

```python
micro_precision = accuracy
micro_recall    = accuracy
micro_f1        = accuracy
```

In the current implementation, micro metrics are simply set equal to accuracy. This is **mathematically correct for balanced datasets** (equal class priors, no INCONCLUSIVE), but can be misleading when there is class imbalance or when INCONCLUSIVE predictions exist.

### 8.6 Confusion Matrix

```python
confusion = {c1: {c2: count for c2 in classes} for c1 in classes}
# confusion[true_label][pred_label] += 1
```

---

## 9. Judge Inter-Rater Reliability (Cohen's Kappa)

**Source:** `metrics_extension.py` → `compute_cohens_kappa()`, `compute_judge_reliability()`  
**Computed at:** Dataset level (across all claims)

### 9.1 Cohen's Kappa (Pairwise)

```
κ = (p_o - p_e) / (1 - p_e)

where:
  p_o = observed agreement = sum(r1_i == r2_i) / n
  p_e = expected agreement by chance = sum(P(class c | rater1) × P(class c | rater2))

Special case: if p_e = 1.0, κ = 1.0
```

**Interpretation guide:**
| κ Range | Interpretation |
|---------|----------------|
| < 0 | Less than chance agreement |
| 0.0–0.20 | Slight agreement |
| 0.21–0.40 | Fair agreement |
| 0.41–0.60 | Moderate agreement |
| 0.61–0.80 | Substantial agreement |
| 0.81–1.00 | Almost perfect agreement |

### 9.2 All Kappa Metrics Computed

| Metric | Description |
|--------|-------------|
| `k_12` | Kappa between Judge 1 and Judge 2 |
| `k_13` | Kappa between Judge 1 and Judge 3 |
| `k_23` | Kappa between Judge 2 and Judge 3 |
| `mean_kappa` | `mean(k_12, k_13, k_23)` — primary reliability metric |
| `k_gt1` | Kappa of Judge 1 vs ground truth (after label mapping) |
| `k_gt2` | Kappa of Judge 2 vs ground truth |
| `k_gt3` | Kappa of Judge 3 vs ground truth |
| `avg_raw_agreement` | `mean(p_o_12, p_o_13, p_o_23)` — raw pairwise agreement |
| `unanimity_rate` | `sum(j1==j2==j3) / n` — fraction of claims with unanimous verdict |
| `split_rate` | `1 - unanimity_rate` |

**Label Mapping (for GT comparison):**
```python
SUPPORTED     → SUPPORT
NOT SUPPORTED → REFUTE
INCONCLUSIVE  → INCONCLUSIVE
```

**Why used:** Inter-rater reliability of the 3-judge panel is crucial for establishing the validity of the judicial evaluation process. `mean_kappa` is the key headline metric for the panel's consistency.

**Console output per claim (simplified):**
```
kappa_mean=0.333   ← note: this is raw agreement ratio, not Cohen's Kappa
```

**Full Kappa computed at run-summary level:**
```
Kappa: κ12=0.720 κ13=0.650 κ23=0.700 mean=0.690
```

---

## 10. AUC-ROC & Threshold Sweep

**Source:** `metrics_extension.py` → `compute_auc_and_sweep()`  
**Computed at:** Dataset level  
**Positive class:** `SUPPORT`

### 10.1 AUC (Area Under the ROC Curve)

```python
# Binarize ground truth
y_binary = [1 if yt == "SUPPORT" else 0 for yt in y_true]

# Sort by descending confidence (predicted probability of SUPPORT)
sorted_indices = argsort([-c for c in confidences])

# Compute trapezoidal AUC
auc += (fp_rate - prev_fp_rate) × (tp_rate + prev_tp_rate) / 2.0
```

**Implementation note:** The `confidence` score is used as the probability estimate for the positive class. Since the confidence is computed from vote consensus and quality scores (not a probabilistic model), AUC interpretation requires care — a high `confidence` for a REFUTE verdict does not map to the SUPPORT positive class probability.

**Only computed when:** `total_p > 0 and total_n > 0` (both SUPPORT and REFUTE examples present).

**Why used:** AUC is a threshold-independent measure of ranking quality. It quantifies how well the system's confidence score separates SUPPORT from REFUTE claims.

### 10.2 Threshold Sweep

For thresholds `τ ∈ {0.30, 0.35, 0.40, ..., 0.70}`:

```python
preds = ["SUPPORT" if confidence >= τ else "REFUTE" for confidence in confidences]
metrics = compute_classification_metrics(y_true, preds)
sweep_results.append({"threshold": τ, "accuracy": acc, "macro_f1": mf1})
```

**Why used:** Identifies the optimal confidence threshold for binarizing the system's output, useful for calibrating the SUPPORT / REFUTE decision boundary.

---

## 11. Debating Stability (KS Statistic)

**Source:** `metrics_extension.py` → `compute_ks_statistic()`, `analyze_stability()`  
**Current status in pipeline:** ⚠️ **HARDCODED PLACEHOLDER** — not computed from real data

### 11.1 Intended Purpose

The KS (Kolmogorov-Smirnov) statistic measures how much the **distribution of confidence scores shifts** between consecutive debate rounds. A small KS value means the debate has "stabilized."

### 11.2 Mathematical Definition

```
D_t = sup_x |F_t(x) - F_{t-1}(x)|

where:
  F_t(x) = empirical CDF of confidence scores at round t
  F_{t-1}(x) = empirical CDF at round t-1
  D_t → 0 means distributions between rounds are converging
```

**Computation (if called correctly):**
```python
# For each x in the union of all observed values:
f1 = searchsorted(sorted_dist1, x, side='right') / n1   # CDF value
f2 = searchsorted(sorted_dist2, x, side='right') / n2
d  = abs(f1 - f2)
D_t = max(d over all x)
```

### 11.3 Stabilization Check

```python
# For epsilon in [0.03, 0.05, 0.07]:
stab_round = first round r where D_r < epsilon
```

### 11.4 Current Issue (Critical)

In `run_eval_extended.py` line 406:
```python
ks = {"D_t": {1: 0.8, 2: 0.4, 3: 0.1, 4: 0.04}, "stabilization_rounds": {"eps_0.05": 4}}
```

This is a **static placeholder** that is always output regardless of actual debate dynamics. The `analyze_stability()` function exists and is correct but is **never called** in the current pipeline. The `ExtensionState.stability_traces` list is populated in design but the per-round confidence distribution is not actually written to it in the current implementation.

**Log output reflects the placeholder:**
```
Stability: D_1=0.800, D_2=0.400, D_3=0.100..., stop_round(0.05)=4
```

---

## 12. Efficiency Metrics

**Source:** `run_eval_extended.py` → `compile_and_log_run_summary()` → `eff` dict  
**Computed at:** Dataset (run) level

| Metric | Formula | Description |
|--------|---------|-------------|
| `avg_tokens` | `sum(h['token_total']) / len(history)` | Average tokens consumed per claim (all LLM calls combined) |
| `avg_rounds` | `sum(h['total_rounds']) / len(history)` | Average number of debate rounds per claim (normal + switched) |
| `avg_evidence` | `sum(h['evidence_count']) / len(history)` | Average number of evidence items retrieved (total, all `retrieve()` calls) |
| `avg_retrieval_calls` | `sum(h['retrieval_calls']) / len(history)` | Average number of `retrieve()` calls per claim |

**Console output:**
```
Cost: avg_tok=223690.0 avg_round=6.0 avg_evidence=77.0
```

**Why used:** Efficiency metrics quantify the computational cost of the pipeline. They are essential for understanding the token budget per claim, which directly maps to API cost.

---

## 13. Inconclusive Sensitivity Analysis

**Source:** `run_eval_extended.py` → `compile_and_log_run_summary()` → `inconc` dict  
**Computed at:** Dataset level (only if any predictions are INCONCLUSIVE)

Three policies are tested for how INCONCLUSIVE verdicts are handled:

| Policy | Mapping | Description |
|--------|---------|-------------|
| **Policy A** | INCONCLUSIVE → SUPPORT | Plaintiff wins by default |
| **Policy B** | INCONCLUSIVE → REFUTE | Defense wins by default |
| **Policy C** | Exclude INCONCLUSIVE | Compute metrics only on non-INCONCLUSIVE claims |

For Policies A and B, full `compute_classification_metrics()` is run on the remapped predictions.  
For Policy C:
```python
y_true_c = [yt for yt, yp in zip(y_true, y_pred) if yp != "INCONCLUSIVE"]
y_pred_c = [yp for yp in y_pred if yp != "INCONCLUSIVE"]
coverage  = sum(yp != "INCONCLUSIVE" for yp in y_pred) / len(y_pred) * 100
```

**Console output:**
```
--- INCONCLUSIVE SENSITIVITY ---
Policy A (→SUP): Acc=0.750
Policy B (→REF): Acc=0.650
Policy C (Excl): Acc=0.800 Coverage=85.0%
```

**Why used:** INCONCLUSIVE verdicts occur when the judicial panel cannot reach a majority. The sensitivity analysis measures how robust the accuracy metric is to the treatment of these ambiguous cases.

---

## 14. Metrics Data Flow Summary

```
Per-Claim:
  API Calls ──────────────→ ExtensionState.token_* (monkey patches)
  retrieve() calls ────────→ ExtensionState.retrieval_calls, evidence_count
  
  After FinalVerdict.generate_verdict():
    extract_and_log_claim_metrics()
      → reads: final_verdict.jsonl, judge_evaluation.jsonl,
               debate_transcript.jsonl, debate_transcript_switched.jsonl
      → writes claims_added.jsonl
      → prints EXTRA METRICS block
      → resets ExtensionState

After all claims:
  compile_and_log_run_summary()
    → compute_classification_metrics(y_true, y_pred) → accuracy, F1, etc.
    → compute_auc_and_sweep(y_true, confidences) → AUC, threshold sweep
    → compute_judge_reliability(j_list, y_true) → kappas
    → eff metrics (avg_tokens, avg_rounds, etc.)
    → ks = HARDCODED PLACEHOLDER
    → writes runs_added.jsonl
    → appends run_reports_added.md
```

---

## 15. Risk Analysis

### 🔴 Critical Issues

| Metric | Risk | Impact |
|--------|------|--------|
| **KS Stability D_t** | Hardcoded placeholder values | Published stability results are fabricated — this is a data integrity issue if used in thesis/paper. |
| **Per-claim kappa_mean in logs** | Mislabeled — it's raw agreement ratio (`pairs_match/3`), not Cohen's Kappa | Confusion between a simplified local metric and the full dataset-level Cohen's Kappa. |
| **Evidence weight calibration** | Almost all weights = 0.75 (likely LLM fallback or uncalibrated prompt) | The admissibility filter is effectively disabled; all evidence is admitted regardless of quality. |
| **Confidence as AUC score** | `confidence` is not a proper probability for SUPPORT | AUC computed treating REFUTE-confidence as 1-P(SUPPORT) is incorrect if both verdicts can have high confidence. |

### 🟡 Moderate Issues

| Metric | Risk | Impact |
|--------|------|--------|
| **Groq token tracking** | `token_groq` is always 0 | If Groq is used for any calls, those tokens are invisible, underreporting actual cost. |
| **Micro metrics = accuracy** | Misleading for imbalanced classes | In binary (SUPPORT/REFUTE) or when INCONCLUSIVE verdicts exist, micro-F1 ≠ accuracy. |
| **Self-reflection `confidence_adjustment`** | Source and range unclear from current code | Risk of large negative adjustments (capped at -0.15) disproportionately tanking confidence. |
| **Consistency check (keyword-based)** | `_check_role_switch_consistency()` uses simple keyword in/not-in check | Highly dependent on exact LLM wording; inconsistent behavior across runs. |
| **AUC with very small datasets** | AUC is unreliable with `n < 20` | When processing only a few claims (default `--limit 1`), AUC is meaningless. |

### 🟢 Low Issues

| Metric | Risk | Impact |
|--------|------|--------|
| **`balanced_accuracy = macro_recall`** | Accurate but may confuse readers | This is mathematically correct but should be documented. |
| **GT label sourcing** | Multiple fallback sources in `extract_and_log_claim_metrics()` | Minor edge case if metadata is missing, defaults to UNKNOWN and claim is excluded from metrics. |

---

## 16. Recommendations

### Fix Immediately

1. **Implement real KS Stability** — Replace the hardcoded `ks` dict in `run_eval_extended.py` with real calls to `analyze_stability()`. Requires populating `ExtensionState.stability_traces` with per-claim, per-round confidence lists. Consider storing `final_confidence` after each round of the debate as a proxy.

2. **Fix AUC confidence interpretation** — The `confidence` score should be interpreted per-verdict: for AUC against "SUPPORT" as positive, use `conf if pred == SUPPORT else (1 - conf)` to get a proper probability proxy.

3. **Rename `kappa_mean` in console output** — The per-claim `kappa_pair_mean` printed in the EXTRA METRICS block is not Cohen's Kappa. Rename to `judge_agreement_ratio` or document clearly.

4. **Fix evidence weight calibration** — Either (a) tune the admissibility prompt to produce more granular scores, or (b) log all `_calculate_weight()` raw LLM responses for analysis.

### Improve

5. **Patch Groq client** — Add `requests.Session` or direct response interception to `groq_client.py` matching the OpenRouter pattern.

6. **Proper micro-F1 implementation** — Calculate micro-F1 from the pooled TP/FP/FN across all classes rather than setting it equal to accuracy.

7. **Keyword-less consistency check** — Replace the keyword-counting approach in `_check_role_switch_consistency()` with an LLM rating of the consistency report, or at minimum use a binary regex pattern for a `score` field.

8. **Calibrate confidence → support probability** — Document clearly that `confidence` is a composite signal, not a calibrated probability. Consider adding a Platt scaling or isotonic regression calibration step if ground truth labels are available.

9. **Per-claim AUC reporting** — Currently AUC is a single run-level value. With small test sets, report confidence intervals (bootstrap or binomial) alongside AUC.

10. **Add INCONCLUSIVE as a real metric class** — Rather than treating it as a policy choice, consider tracking "abstention rate" and "abstention precision" (what fraction of abstentions were for genuinely ambiguous cases).
