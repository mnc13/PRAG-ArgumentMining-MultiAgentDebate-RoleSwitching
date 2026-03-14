# METRICS OVERVIEW

## Comprehensive Metrics Documentation for the PRAG Multi-Agent Debate Framework

> **Document Purpose:** Formal reference for all metrics computed across the pipeline, including their mathematical definitions, units, code locations, value ranges, inter-dependencies, and logging destinations. Suitable as supplementary material for a research paper.

---

## Table of Contents

1. [Metrics System Overview](#1-metrics-system-overview)
2. [Category 1 — Argument Mining Metrics](#2-category-1--argument-mining-metrics)
3. [Category 2 — Evidence Retrieval Metrics](#3-category-2--evidence-retrieval-metrics)
4. [Category 3 — Evidence Negotiation & Arbitration Metrics](#4-category-3--evidence-negotiation--arbitration-metrics)
5. [Category 4 — Progressive RAG (P-RAG) Metrics](#5-category-4--progressive-rag-p-rag-metrics)
6. [Category 5 — Per-Round Debate Metrics](#6-category-5--per-round-debate-metrics)
7. [Category 6 — Self-Reflection Metrics](#7-category-6--self-reflection-metrics)
8. [Category 7 — Critic Agent Metrics](#8-category-7--critic-agent-metrics)
9. [Category 8 — Role-Switching Consistency Metrics](#9-category-8--role-switching-consistency-metrics)
10. [Category 9 — Judicial Panel Metrics](#10-category-9--judicial-panel-metrics)
11. [Category 10 — Final Verdict & Confidence Metrics](#11-category-10--final-verdict--confidence-metrics)
12. [Category 11 — Classification & Aggregate Metrics](#12-category-11--classification--aggregate-metrics)
13. [Category 12 — Efficiency Metrics](#13-category-12--efficiency-metrics)
14. [Category 13 — Token Usage Metrics](#14-category-13--token-usage-metrics)
15. [Metrics Dependency Flow](#15-metrics-dependency-flow)
16. [Logging & Storage](#16-logging--storage)
17. [Special Cases & Assumptions](#17-special-cases--assumptions)

---

## 1. Metrics System Overview

The pipeline computes metrics at **13 distinct levels**, spanning from fine-grained per-round agent behavior to coarse-grained final dataset-level classification performance. The primary purposes of the metrics system are:

1. **Adaptive control** — Real-time metrics (novelty score, reflection delta, critic signal) directly influence pipeline control flow: stopping early, triggering expert witnesses, or adjusting queries.
2. **Quality evaluation** — Intermediate metrics (admissibility weight, judge scores) characterize the quality of evidence and argumentation.
3. **Research evaluation** — Final metrics (accuracy, macro-F1, AUC, Cohen's Kappa, KS stability) provide dataset-level performance indicators for comparison with baselines.
4. **Transparency & explainability** — All intermediate metrics are saved alongside outputs, enabling full traceability of verdicts.

### Agents Producing Metrics

| Agent | Role | Metrics Produced |
|---|---|---|
| `ArgumentMiner` | Premise decomposition | Premise count |
| `PubMedRetriever` | Initial retrieval | Retrieval score, top-K count |
| `EvidenceNegotiator` | Discovery & arbitration | Admissibility weight, relevance, credibility |
| `ProgressiveRAG` | Per-round retrieval | Novelty score, relevance gain, redundancy ratio |
| `DebateAgent` (Proponent/Opponent) | Argument generation | Self-reflection scores, discovery needs |
| `SelfReflection` | Per-round self-audit | Logic, novelty, rebuttal, total score |
| `CriticAgent` | Round quality evaluation | Argument quality scores, resolution signal |
| `RoleSwitcher` | Consistency testing | Consistency score |
| `JudicialPanel` (3 judges) | Holistic evaluation | Evidence strength, argument validity, scientific reliability, verdict |
| `FinalVerdict` | Verdict synthesis | Confidence score |
| `run_eval_extended` | Experiment wrapper | All aggregate classification metrics |

---

## 2. Category 1 — Argument Mining Metrics

### 2.1 Premise Count

| Field | Value |
|---|---|
| **Name** | `premise_count` |
| **Meaning** | Number of atomic, testable premises decomposed from the original claim |
| **Unit / Range** | Integer ≥ 1 |
| **Computed in** | `agent_workflow.py` → `ArgumentMiner.mine_arguments()` |
| **Logged to** | Execution log (`execution_log_{id}.txt`) via `print()` |
| **Example** | Claim: *"COVID-19 vaccines prevent severe disease."* → Premises: 2 |
| **Relationships** | Drives `Step 1` of `EvidenceNegotiator.prepare_pools()`: one retrieval call per premise |

**Notes:** Premises are extracted by prompting DeepSeek-R1 (`openrouter_client.py`) with a structured decomposition prompt. The count is not stored as a structured field but is observable from the log output `[DECOMPOSED PREMISES/ARGUMENTS]`.

---

## 3. Category 2 — Evidence Retrieval Metrics

These metrics characterize the quality of the initial PubMed FAISS retrieval performed once per claim at the start of the pipeline.

### 3.1 FAISS Similarity Score (Initial Retrieval)

| Field | Value |
|---|---|
| **Name** | `relevance_score` (initial) |
| **Meaning** | Inner product similarity between normalized query embedding and normalized document embedding. Equivalent to cosine similarity. Higher = more relevant. |
| **Unit / Range** | Float ∈ [0.0, 1.0] (for normalized `IndexFlatIP`) |
| **Computed in** | `rag_engine.py` → `PubMedRetriever.retrieve()` |
| **Stored in** | `Evidence.relevance_score` dataclass field (`models.py`) |
| **Logged to** | `execution_log_{id}.txt` — initial evidence block |
| **Example** | 0.87 (high relevance), 0.41 (moderate relevance) |
| **Relationships** | This score is later overwritten in the negotiation phase by the `admissibility_weight` |

### 3.2 Top-K Retrieved Count

| Field | Value |
|---|---|
| **Name** | `initial_top_k` |
| **Meaning** | Number of evidence items retrieved in the initial PubMed search |
| **Unit / Range** | Fixed integer = 5 (hardcoded in `main_pipeline.py`) |
| **Computed in** | `main_pipeline.py` → `retriever.retrieve(extracted_claim.text, top_k=5)` |
| **Logged to** | Execution log |

---

## 4. Category 3 — Evidence Negotiation & Arbitration Metrics

### 4.1 Stance-Conditioned Retrieval Pool Size

| Field | Value |
|---|---|
| **Name** | `pool_size` (shared / proponent / opponent) |
| **Meaning** | Number of unique evidence items in each discovery pool after deduplication |
| **Unit / Range** | Integer ≥ 0 |
| **Computed in** | `negotiation_engine.py` → `EvidenceNegotiator.prepare_pools()` |
| **Logged to** | Execution log: `"> Aggregated N shared evidence items."` |
| **Example** | shared_pool: 6, proponent_pool: 3, opponent_pool: 3 |
| **Relationships** | Pool items are candidates for admissibility scoring in `judge_arbitration()` |

### 4.2 Relevance Score (Arbitration)

| Field | Value |
|---|---|
| **Name** | `relevance` |
| **Meaning** | LLM-assigned score of how directly the evidence addresses the claim premises |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `negotiation_engine.py` → `EvidenceNegotiator._calculate_weight()` |
| **Stored in** | `negotiation_state["judge_state"]["admissible_evidence"][*]["relevance"]` |
| **Logged to** | Execution log: `Evidence Arbitration [{source_id}]` block; `negotiation_state_{id}.json` |
| **Example** | 0.85 (directly addresses a key premise) |
| **Relationships** | Multiplied with `credibility` to produce `admissibility_weight` |

### 4.3 Credibility Score (Arbitration)

| Field | Value |
|---|---|
| **Name** | `credibility` |
| **Meaning** | LLM-assigned score of the scientific reliability of the evidence source |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `negotiation_engine.py` → `EvidenceNegotiator._calculate_weight()` |
| **Stored in** | Same as relevance |
| **Example** | 0.90 (peer-reviewed, high-impact journal) |
| **Relationships** | Multiplied with `relevance` to produce `admissibility_weight` |

### 4.4 Admissibility Weight

| Field | Value |
|---|---|
| **Name** | `admissibility_weight` (also `weight`) |
| **Meaning** | Joint score representing the overall fitness of an evidence item for admission into the debate proceedings |
| **Formula** | `weight = relevance × credibility` |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `negotiation_engine.py` → `EvidenceNegotiator._calculate_weight()` |
| **Stored in** | `Evidence.relevance_score` (overwritten); `negotiation_state_{id}.json` |
| **Logged to** | `negotiation_state_{id}.json`; execution log arbitration block |
| **Example** | 0.765 = 0.85 × 0.90 |
| **Thresholds** | `> 0.5` → **Admitted**; `0.1–0.5` → **Disputed**; `≤ 0.1` → **Rejected** |
| **Relationships** | All admitted items (sorted descending by weight) form the `final_evidence_set` passed to the MAD phase |

### 4.5 Admitted Evidence Count

| Field | Value |
|---|---|
| **Name** | `admitted_count` |
| **Meaning** | Number of evidence items passing the admissibility threshold (weight > 0.5) |
| **Unit / Range** | Integer ≥ 0 |
| **Computed in** | `negotiation_engine.py` → `EvidenceNegotiator.judge_arbitration()` |
| **Logged to** | Execution log: `"> Admitted N high-weight items. Flagged M for dispute."` |
| **Relationships** | Feeds `MADOrchestrator.evidence_pool` as the starting evidence set |

### 4.6 Disputed Evidence Count

| Field | Value |
|---|---|
| **Name** | `disputed_count` |
| **Meaning** | Number of evidence items scoring between 0.10 and 0.50 (uncertain admissibility) |
| **Unit / Range** | Integer ≥ 0 |
| **Computed in** | `negotiation_engine.py` → `EvidenceNegotiator.judge_arbitration()` |
| **Stored in** | `negotiation_state["judge_state"]["disputed_items"]` |

---

## 5. Category 4 — Progressive RAG (P-RAG) Metrics

These metrics are computed on every call to `ProgressiveRAG.retrieve_progressive()`, which may be triggered multiple times per round per side.

### 5.1 Novelty Score (Per Evidence Item)

| Field | Value |
|---|---|
| **Name** | `novelty_score` |
| **Meaning** | Fraction of new information contributed by a retrieved document relative to the existing evidence pool. A score of 1.0 means the document is completely new; 0.0 means it is a duplicate. |
| **Formula** | `novelty_score = 1 − max_cosine_similarity(new_doc_emb, pool_emb_i for all pool_emb_i)` |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `prag_engine.py` → `ProgressiveRAG._calculate_novelty()` |
| **Stored in** | `Evidence.novelty_score` (`models.py`) |
| **Logged to** | `prag_history.jsonl`; debate transcript `new_evidence[*]["novelty"]` |
| **Example** | 0.76 (somewhat novel), 0.04 (near-duplicate of existing pool item) |
| **Filter threshold** | Items with `novelty_score < 0.2` are **rejected** |
| **Relationships** | Aggregated into `avg_novelty` per round; used in adaptive convergence check |

### 5.2 Average Novelty (Per Retrieval Call)

| Field | Value |
|---|---|
| **Name** | `avg_novelty` |
| **Meaning** | Mean novelty score across all retrieved items in a single retrieval call |
| **Formula** | `avg_novelty = mean(novelty_score_i for i in retrieved)` |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `prag_engine.py` → `ProgressiveRAG._calculate_novelty()` return value |
| **Stored in** | `prag_engine.retrieval_history[*]["avg_novelty"]` |
| **Logged to** | `prag_history.jsonl`; `mad_orchestrator.py` round_data["prag_metrics"] |
| **Adaptive stop** | If `avg_novelty < 0.10` in two consecutive rounds → **terminate debate** |

### 5.3 Average Relevance (Per Retrieval Call)

| Field | Value |
|---|---|
| **Name** | `avg_relevance` |
| **Meaning** | Mean FAISS similarity score of accepted (novelty-filtered) evidence items for a given retrieval call |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `prag_engine.py` → `ProgressiveRAG.retrieve_progressive()` |
| **Stored in** | `prag_engine.retrieval_history[*]["avg_relevance"]` |
| **Relationships** | Used to compute `relevance_gain` |

### 5.4 Relevance Gain

| Field | Value |
|---|---|
| **Name** | `relevance_gain` |
| **Meaning** | Improvement in average relevance between the current and previous retrieval call. Measures whether new queries are finding more relevant evidence. |
| **Formula** | `relevance_gain = avg_relevance_t − avg_relevance_{t−1}` |
| **Unit / Range** | Float ∈ [−1.0, 1.0] (typically small positive or near-zero) |
| **Computed in** | `prag_engine.py` → `ProgressiveRAG.retrieve_progressive()` |
| **Stored in** | `prag_engine.retrieval_history[*]["relevance_gain"]` |
| **Adaptive stop** | If `relevance_gain < 0.05` after round 1 → **stop P-RAG** |

### 5.5 Redundancy Ratio

| Field | Value |
|---|---|
| **Name** | `redundancy_ratio` |
| **Meaning** | Fraction of retrieved items that are highly similar to existing pool items (novelty < 0.15) |
| **Formula** | `redundancy_ratio = count(novelty_score < 0.15) / total_retrieved` |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `prag_engine.py` → `ProgressiveRAG.retrieve_progressive()` |
| **Stored in** | `prag_engine.retrieval_history[*]["redundancy_ratio"]` |
| **Adaptive stop** | If `redundancy_ratio > 0.70` → **stop P-RAG** |

### 5.6 P-RAG Retrieval Counts (Per Call)

| Field | Value |
|---|---|
| **Name** | `num_retrieved`, `num_accepted`, `num_rejected` |
| **Meaning** | Raw retrieved count; count passing novelty filter; count failing novelty filter |
| **Unit / Range** | Integers ≥ 0 |
| **Computed in** | `prag_engine.py` → `retrieve_progressive()` |
| **Stored in** | `prag_engine.retrieval_history[*]` |

### 5.7 P-RAG Stop Reason

| Field | Value |
|---|---|
| **Name** | `stop_reason` |
| **Meaning** | Text label explaining why P-RAG halted for a given retrieval call |
| **Possible values** | `"Maximum iterations reached"` / `"High redundancy detected"` / `"Diminishing relevance gain"` / `null` (no stop) |
| **Computed in** | `prag_engine.py` → `retrieve_progressive()` |
| **Stored in** | `prag_engine.retrieval_history[*]["stop_reason"]` |

---

## 6. Category 5 — Per-Round Debate Metrics

These metrics are produced and aggregated by the `MADOrchestrator` at the conclusion of each debate round.

### 6.1 New Evidence Count (Per Round)

| Field | Value |
|---|---|
| **Name** | `new_evidence_count` |
| **Meaning** | Number of novel evidence items admitted to the pool in a given round, summed over both sides |
| **Unit / Range** | Integer ≥ 0 |
| **Computed in** | `mad_orchestrator.py` → `run_debate_round()` |
| **Stored in** | `debate_transcript.jsonl` → `rounds[*]["new_evidence"]` (list length) |
| **Relationships** | Used to compute `avg_novelty` for adaptive convergence |

### 6.2 Expert Witness Count (Per Round)

| Field | Value |
|---|---|
| **Name** | `expert_count` |
| **Meaning** | Number of expert witness testimonies admitted during a round |
| **Unit / Range** | Integer ≥ 0; typically 0–2 per round |
| **Computed in** | `mad_orchestrator.py` → `run_debate_round()` |
| **Stored in** | `debate_transcript.jsonl` → `rounds[*]["expert_testimonies"]` (list length) |

### 6.3 Reflection Delta (Round Convergence Signal)

| Field | Value |
|---|---|
| **Name** | `delta_score` |
| **Meaning** | Change in total self-reflection score between consecutive rounds. Measures whether argument quality is still improving. |
| **Formula** | `delta_score = total_reflection_score_t − total_reflection_score_{t−1}` |
| **Unit / Range** | Float (typically ∈ [−0.5, 0.5]); positive = improvement |
| **Computed in** | `mad_orchestrator.py` → `run_full_debate()` |
| **Logged to** | Execution log: `"[Convergence] Score Delta: X.XXXX"` |
| **Adaptive stop** | If `|delta_score| < 0.05` for rounds > 1 → **terminate debate** ("Reflection plateau") |
| **Relationships** | Derived from `SelfReflection.total_score` aggregated over both agents |

### 6.4 Debate Convergence Reason

| Field | Value |
|---|---|
| **Name** | `stop_reason` (debate-level) |
| **Meaning** | Categorical label explaining why the debate terminated |
| **Possible values** | `"Reflection plateau"` / `"Critic resolution"` / `"Novelty stabilization"` / `"Judicial signal"` |
| **Computed in** | `mad_orchestrator.py` → `run_full_debate()` |
| **Stored in** | `debate_transcript.jsonl` → `convergence_metrics["stop_reason"]` |

### 6.5 Total Debate Rounds (Normal)

| Field | Value |
|---|---|
| **Name** | `rounds_normal` |
| **Meaning** | Number of rounds completed in the primary (non-switched) debate |
| **Unit / Range** | Integer ∈ [1, 10] |
| **Computed in** | `run_eval_extended.py` → `extract_and_log_claim_metrics()` |
| **Stored in** | `claims_added.jsonl` → `rounds_normal` |

### 6.6 Total Debate Rounds (Switched)

| Field | Value |
|---|---|
| **Name** | `rounds_switched` |
| **Meaning** | Number of rounds completed in the role-switched debate |
| **Unit / Range** | Integer ∈ [1, 10] |
| **Computed in** | `run_eval_extended.py` → `extract_and_log_claim_metrics()` |
| **Stored in** | `claims_added.jsonl` → `rounds_switched` |

### 6.7 Total Rounds

| Field | Value |
|---|---|
| **Name** | `total_rounds` |
| **Meaning** | Sum of normal and switched debate rounds for a single claim |
| **Formula** | `total_rounds = rounds_normal + rounds_switched` |
| **Unit / Range** | Integer ∈ [2, 20] |
| **Stored in** | `claims_added.jsonl` → `total_rounds` |

---

## 7. Category 6 — Self-Reflection Metrics

Self-reflection is performed after every debate round by each side, using the `SelfReflection` class (`self_reflection.py`).

### 7.1 Logic Score

| Field | Value |
|---|---|
| **Name** | `logic` |
| **Meaning** | Agent's self-assessed score of its argument's logical coherence and structural integrity |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `self_reflection.py` → `SelfReflection.perform_round_reflection()` (LLM response parsed) |
| **Stored in** | `self_reflection.jsonl` → `reflection_history[*]["scores"]["logic"]` |
| **Weight in formula** | 0.40 (highest weight) |

### 7.2 Novelty Score (Reflection)

| Field | Value |
|---|---|
| **Name** | `novelty` (reflection context) |
| **Meaning** | Agent's self-assessment of whether it introduced genuinely new information vs. repeating earlier points |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `self_reflection.py` → `perform_round_reflection()` |
| **Stored in** | `self_reflection.jsonl` → `reflection_history[*]["scores"]["novelty"]` |
| **Weight in formula** | 0.30 |

### 7.3 Rebuttal Score

| Field | Value |
|---|---|
| **Name** | `rebuttal` |
| **Meaning** | Agent's self-assessment of how effectively it addressed the opponent's key challenges in the last round |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `self_reflection.py` → `perform_round_reflection()` |
| **Stored in** | `self_reflection.jsonl` → `reflection_history[*]["scores"]["rebuttal"]` |
| **Weight in formula** | 0.30 |

### 7.4 Total Reflection Score

| Field | Value |
|---|---|
| **Name** | `total_score` |
| **Meaning** | Weighted composite self-evaluation score for a single agent in a single round |
| **Formula** | `total_score = 0.4 × logic + 0.3 × novelty + 0.3 × rebuttal` |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `self_reflection.py` → `perform_round_reflection()` |
| **Stored in** | `self_reflection.jsonl` → `reflection_history[*]["total_score"]` |
| **Logged to** | Execution log: `"* Total Score: X.XXX"` |
| **Relationships** | Aggregated across both sides to produce `total_reflection_score` used in `delta_score` computation; also produces `confidence_adjustment` |

### 7.5 Confidence Adjustment (Reflection)

| Field | Value |
|---|---|
| **Name** | `confidence_adjustment` |
| **Meaning** | Additive modifier to final confidence score derived from the winner's last self-reflection total score |
| **Formula** | `confidence_adjustment = (total_score − 0.5) × 0.6` |
| **Unit / Range** | Float ∈ [−0.30, +0.30]; positive if `total_score > 0.5`, negative otherwise |
| **Computed in** | `self_reflection.py` → `perform_round_reflection()` (appended to `reflection_data["self_reflection"]`) |
| **Stored in** | `self_reflection.jsonl` → `"self_reflection": {"confidence_adjustment": ...}` |
| **Relationships** | Consumed by `final_verdict.py` → `FinalVerdict._calculate_confidence()` |
| **Cap** | Negative adjustments are capped at `−0.15` in the final verdict to prevent verdict collapse |

### 7.6 Discovery Need (Reflection)

| Field | Value |
|---|---|
| **Name** | `discovery_need` |
| **Meaning** | A natural language query identifying a gap in the agent's current evidence base, produced by LLM self-audit |
| **Type** | String (1-sentence query) |
| **Computed in** | `self_reflection.py` → `perform_round_reflection()` |
| **Usage** | Injected into the next round's P-RAG query formulation as `reflection_gap` in `mad_orchestrator.py` |
| **Logged to** | Execution log: `"* Discovery Need: ..."` |

---

## 8. Category 7 — Critic Agent Metrics

The `CriticAgent` produces round-level quality evaluations, independent of the debating agents.

### 8.1 Critic Logic Score (Per Side)

| Field | Value |
|---|---|
| **Name** | `plaintiff["logic"]`, `defense["logic"]` |
| **Meaning** | Critic's assessment of argument flow and structural quality for each side |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `mad_system.py` → `CriticAgent.evaluate_round()` |
| **Stored in** | `debate_transcript.jsonl` → `rounds[*]["critic_evaluation"]["plaintiff"]["logic"]` |

### 8.2 Critic Evidence Coverage Score (Per Side)

| Field | Value |
|---|---|
| **Name** | `plaintiff["evidence"]`, `defense["evidence"]` |
| **Meaning** | Critic's assessment of how well each side grounded arguments in admitted exhibits |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `mad_system.py` → `CriticAgent.evaluate_round()` |

### 8.3 Critic Rebuttal Coverage Score (Per Side)

| Field | Value |
|---|---|
| **Name** | `plaintiff["rebuttal"]`, `defense["rebuttal"]` |
| **Meaning** | Critic's evaluation of how thoroughly each side addressed the opponent's arguments |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `mad_system.py` → `CriticAgent.evaluate_round()` |

### 8.4 Unresolved Premises

| Field | Value |
|---|---|
| **Name** | `unresolved_premises` |
| **Meaning** | List of claim sub-premises that the Critic identifies as insufficiently addressed by either side |
| **Type** | List of strings |
| **Computed in** | `mad_system.py` → `CriticAgent.evaluate_round()` |
| **Stored in** | `debate_transcript.jsonl` → `rounds[*]["critic_evaluation"]["unresolved_premises"]` |

### 8.5 Debate Resolved Flag

| Field | Value |
|---|---|
| **Name** | `debate_resolved` |
| **Meaning** | Boolean signal from the Critic indicating whether the core premises of the claim have been sufficiently argued to proceed to verdict |
| **Type** | Boolean |
| **Computed in** | `mad_system.py` → `CriticAgent.evaluate_round()` |
| **Stored in** | `debate_transcript.jsonl` → `rounds[*]["critic_evaluation"]["debate_resolved"]` |
| **Adaptive stop** | `True` → MAD terminates with `stop_reason = "Critic resolution"` |

---

## 9. Category 8 — Role-Switching Consistency Metrics

### 9.1 Consistency Score

| Field | Value |
|---|---|
| **Name** | `consistency_score` |
| **Meaning** | An LLM-assessed integer score (0–10) measuring whether agents maintained logically consistent argumentation positions after switching sides. A high score indicates robust, position-independent reasoning; a low score reveals positional bias. |
| **Unit / Range** | Integer ∈ [0, 10] |
| **Computed in** | `role_switcher.py` → `RoleSwitcher.check_consistency()` (LLM response parsed) |
| **Stored in** | `role_switch_report.jsonl` → `"consistency_score"` |
| **Logged to** | Execution log: consistency analysis JSON block |
| **Threshold** | `≥ 6` → `is_consistent = True`; `< 6` → `is_consistent = False` |
| **Relationships** | Consumed by `FinalVerdict._check_role_switch_consistency()` to compute `rs_adj` |

### 9.2 Is Consistent (Boolean)

| Field | Value |
|---|---|
| **Name** | `is_consistent` |
| **Meaning** | Binary flag derived from `consistency_score`. True if the agents' argumentation is robust to role-reversal. |
| **Formula** | `is_consistent = (consistency_score >= 6)` |
| **Type** | Boolean |
| **Stored in** | `role_switch_report.jsonl` → `"is_consistent"` |

### 9.3 Role-Switch Confidence Adjustment

| Field | Value |
|---|---|
| **Name** | `rs_adj` |
| **Meaning** | Additive modifier to final confidence based on role-switching consistency quality |
| **Formula** | `rs_adj = +0.10 if score ≥ 7; 0.0 if 5 ≤ score < 7; −0.05 if score < 5` |
| **Unit / Range** | Float ∈ {−0.05, 0.0, +0.10} |
| **Computed in** | `final_verdict.py` → `FinalVerdict._calculate_confidence()` |
| **Relationships** | Added to `base_confidence` in final confidence calculation |

### 9.4 Agent Consistency Analysis Text

| Field | Value |
|---|---|
| **Name** | `agent_a_analysis`, `agent_b_analysis`, `contradictions_found`, `reasoning` |
| **Meaning** | LLM-generated natural language descriptions of each agent's consistency performance and identified contradictions |
| **Type** | Strings |
| **Stored in** | `role_switch_report.jsonl` → `"analysis"` nested object |

---

## 10. Category 9 — Judicial Panel Metrics

Three independent appellate judges each produce a structured evaluation. All judges operate at `temperature=0.3`.

### 10.1 Evidence Strength Score (Per Judge)

| Field | Value |
|---|---|
| **Name** | `evidence_strength` |
| **Meaning** | Judge's assessment of the scientific quality, relevance, and credibility of all admitted exhibits and expert testimonies |
| **Unit / Range** | Integer ∈ [0, 10]; rubric: 0–3 = weak, 4–6 = moderate, 7–10 = strong |
| **Computed in** | `judge_evaluator.py` → `JudicialPanel._judge_evaluate()` |
| **Stored in** | `judge_evaluation.jsonl` → `judge_verdicts[*]["evidence_strength"]` |

### 10.2 Argument Validity Score (Per Judge)

| Field | Value |
|---|---|
| **Name** | `argument_validity` |
| **Meaning** | Judge's assessment of the logical coherence of arguments from both sides, checking for fallacies, contradictions, and inferential leaps |
| **Unit / Range** | Integer ∈ [0, 10]; rubric: 0–3 = severely flawed, 4–6 = moderate, 7–10 = sound |
| **Computed in** | `judge_evaluator.py` → `JudicialPanel._judge_evaluate()` |
| **Stored in** | `judge_evaluation.jsonl` → `judge_verdicts[*]["argument_validity"]` |

### 10.3 Scientific Reliability Score (Per Judge)

| Field | Value |
|---|---|
| **Name** | `scientific_reliability` |
| **Meaning** | Judge's evaluation of alignment with established biomedical/scientific consensus and correctness of study interpretations |
| **Unit / Range** | Integer ∈ [0, 10]; rubric: 0–3 = contradicts consensus, 4–6 = partial, 7–10 = well-aligned |
| **Computed in** | `judge_evaluator.py` → `JudicialPanel._judge_evaluate()` |
| **Stored in** | `judge_evaluation.jsonl` → `judge_verdicts[*]["scientific_reliability"]` |

### 10.4 Individual Judge Verdict

| Field | Value |
|---|---|
| **Name** | `verdict` (per judge) |
| **Meaning** | Binary classification decision by a single appellate judge |
| **Possible values** | `"SUPPORTED"` / `"NOT SUPPORTED"` / `"INCONCLUSIVE"` |
| **Computed in** | `judge_evaluator.py` → `JudicialPanel._judge_evaluate()` |
| **Stored in** | `judge_evaluation.jsonl` → `judge_verdicts[*]["verdict"]` |

### 10.5 Panel Final Verdict (Majority Vote)

| Field | Value |
|---|---|
| **Name** | `final_verdict` (panel) |
| **Meaning** | Majority-voted verdict across the three judges |
| **Formula** | `final_verdict = argmax(Counter([v1, v2, v3]))` |
| **Possible values** | `"SUPPORTED"` / `"NOT SUPPORTED"` / `"INCONCLUSIVE"` |
| **Computed in** | `judge_evaluator.py` → `JudicialPanel._aggregate_verdicts()` |
| **Stored in** | `judge_evaluation.jsonl` → `"final_verdict"` |
| **Relationships** | Directly determines `FinalVerdict.verdict` output label |

### 10.6 Vote Breakdown

| Field | Value |
|---|---|
| **Name** | `vote_breakdown` |
| **Meaning** | Dictionary count of how many judges voted for each verdict class |
| **Type** | `dict`, e.g., `{"SUPPORTED": 2, "NOT SUPPORTED": 1}` |
| **Computed in** | `judge_evaluator.py` → `JudicialPanel._aggregate_verdicts()` |
| **Stored in** | `judge_evaluation.jsonl` → `"vote_breakdown"` |
| **Relationships** | `winning_votes / total_votes` = `consensus_strength` used in confidence calculation |

### 10.7 Average Judge Quality Scores

| Field | Value |
|---|---|
| **Name** | `avg_evidence_strength`, `avg_argument_validity`, `avg_scientific_reliability` |
| **Meaning** | Mean of each quality score across all three judges |
| **Formula** | `avg_X = sum(judge_i["X"] for i in [1,2,3]) / 3` |
| **Unit / Range** | Float ∈ [0.0, 10.0] |
| **Computed in** | `final_verdict.py` → `FinalVerdict._calculate_confidence()` |
| **Relationships** | Combined into `quality_score` component of final confidence |

---

## 11. Category 10 — Final Verdict & Confidence Metrics

### 11.1 Consensus Strength

| Field | Value |
|---|---|
| **Name** | `consensus_strength` |
| **Meaning** | Fraction of judges who agreed on the winning verdict. Reflects unanimity of the panel. |
| **Formula** | `consensus_strength = winning_votes / total_votes` |
| **Unit / Range** | Float ∈ {0.33, 0.67, 1.00} (for a 3-judge panel) |
| **Computed in** | `final_verdict.py` → `FinalVerdict._calculate_confidence()` |

### 11.2 Quality Score (Verdict Confidence Component)

| Field | Value |
|---|---|
| **Name** | `quality_score` |
| **Meaning** | Normalized composite of the three average judge dimension scores |
| **Formula** | `quality_score = (avg_evidence + avg_validity + avg_reliability) / 30 × 0.3` |
| **Unit / Range** | Float ∈ [0.0, 0.30] |
| **Computed in** | `final_verdict.py` → `FinalVerdict._calculate_confidence()` |

### 11.3 Base Confidence

| Field | Value |
|---|---|
| **Name** | `base_confidence` |
| **Meaning** | Initial confidence estimate before role-switch and reflection adjustments |
| **Formula** | `base_confidence = consensus_strength × 0.8 + quality_score` |
| **Unit / Range** | Float ∈ [0.0, 1.10] before clamping |
| **Computed in** | `final_verdict.py` → `FinalVerdict._calculate_confidence()` |

### 11.4 Final Confidence Score

| Field | Value |
|---|---|
| **Name** | `confidence` |
| **Meaning** | Final calibrated probability-like score reflecting the system's certainty in its verdict |
| **Formula** | `confidence = clamp(base_confidence + rs_adj + reflection_adj, 0.0, 1.0)` |
| **Unit / Range** | Float ∈ [0.0, 1.0]; minimum enforced at 0.10 if consensus > 50% |
| **Computed in** | `final_verdict.py` → `FinalVerdict._calculate_confidence()` |
| **Stored in** | `final_verdict.jsonl`, `all_verdicts.jsonl`, `claims_added.jsonl` |
| **Example** | 0.847 (strong unanimous panel + high quality scores + consistent role-switching) |

### 11.5 Final Verdict Label

| Field | Value |
|---|---|
| **Name** | `verdict` |
| **Meaning** | Final system prediction mapped to the dataset label space |
| **Mapping** | `SUPPORTED → "SUPPORT"` / `NOT SUPPORTED → "REFUTE"` / `INCONCLUSIVE → "INCONCLUSIVE"` |
| **Computed in** | `final_verdict.py` → `FinalVerdict.generate_verdict()` |
| **Stored in** | `final_verdict.jsonl`, `all_verdicts.jsonl`, `claims_added.jsonl` |

### 11.6 Correctness Flag

| Field | Value |
|---|---|
| **Name** | `correct` |
| **Meaning** | Boolean indicating whether the predicted verdict matches the ground truth label |
| **Formula** | `correct = (verdict == ground_truth)` (None if ground truth is UNKNOWN) |
| **Type** | Boolean or null |
| **Computed in** | `final_verdict.py` → `FinalVerdict.generate_verdict()` |
| **Stored in** | `final_verdict.jsonl`, `all_verdicts.jsonl`, `claims_added.jsonl` |

---

## 12. Category 11 — Classification & Aggregate Metrics

Computed once per run over all processed claims by `compile_and_log_run_summary()` in `run_eval_extended.py`, using functions from `metrics_extension.py`.

### 12.1 Accuracy

| Field | Value |
|---|---|
| **Name** | `accuracy` |
| **Meaning** | Proportion of claims where the predicted verdict matches the ground truth |
| **Formula** | `accuracy = Σ(correct_i) / N` |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `metrics_extension.py` → `compute_classification_metrics()` |
| **Stored in** | `runs_summary.jsonl` → metrics block |

### 12.2 Macro Precision / Recall / F1

| Field | Value |
|---|---|
| **Name** | `macro_precision`, `macro_recall`, `macro_f1` |
| **Meaning** | Unweighted per-class averages of precision, recall, and F1. Treats all classes equally regardless of frequency. |
| **Formula** | `macro_F1 = mean(F1_c for c in classes)` |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `metrics_extension.py` → `compute_classification_metrics()` |

### 12.3 Micro Precision / Recall / F1

| Field | Value |
|---|---|
| **Name** | `micro_precision`, `micro_recall`, `micro_f1` |
| **Meaning** | For multi-class problems, micro-averaging gives each sample equal weight. Equivalent to accuracy in this setting. |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `metrics_extension.py` → `compute_classification_metrics()` |

### 12.4 Balanced Accuracy

| Field | Value |
|---|---|
| **Name** | `balanced_accuracy` |
| **Meaning** | Macro recall — gives equal weight to each class, adjusting for class imbalance |
| **Formula** | `balanced_accuracy = macro_recall` |
| **Unit / Range** | Float ∈ [0.0, 1.0] |
| **Computed in** | `metrics_extension.py` → `compute_classification_metrics()` |

### 12.5 Per-Class Precision / Recall / F1

| Field | Value |
|---|---|
| **Name** | `per_class["SUPPORT"]["precision"]`, `per_class["REFUTE"]["f1"]`, etc. |
| **Meaning** | Classification quality metrics computed independently for each outcome class |
| **Computed in** | `metrics_extension.py` → `compute_classification_metrics()` |
| **Stored in** | `runs_summary.jsonl` → `metrics["per_class"]` |

### 12.6 Confusion Matrix

| Field | Value |
|---|---|
| **Name** | `confusion_matrix` |
| **Meaning** | Square matrix of true vs. predicted label counts across all classes |
| **Type** | Nested dict, e.g., `{"SUPPORT": {"SUPPORT": 18, "REFUTE": 2, "INCONCLUSIVE": 0}, ...}` |
| **Computed in** | `metrics_extension.py` → `compute_classification_metrics()` |
| **Stored in** | `runs_summary.jsonl` → `metrics["confusion_matrix"]` |

### 12.7 AUC-ROC

| Field | Value |
|---|---|
| **Name** | `auc` |
| **Meaning** | Area under the Receiver Operating Characteristic curve for the binary SUPPORT / non-SUPPORT classification, computed from per-claim confidence scores |
| **Formula** | Trapezoidal rule on empirical ROC curve |
| **Unit / Range** | Float ∈ [0.0, 1.0]; 0.5 = random, 1.0 = perfect |
| **Computed in** | `metrics_extension.py` → `compute_auc_and_sweep()` |
| **Stored in** | `runs_summary.jsonl` → `metrics["auc"]` |
| **Note** | Returns `null` if all claims share the same class (degenerate case) |

### 12.8 Threshold Sweep (AUC Extension)

| Field | Value |
|---|---|
| **Name** | `threshold_sweep` |
| **Meaning** | Accuracy and macro-F1 evaluated at 9 confidence thresholds from 0.30 to 0.70 (step 0.05), using the per-claim confidence score as a binary classifier decision boundary |
| **Type** | List of `{"threshold": t, "accuracy": a, "macro_f1": f}` |
| **Computed in** | `metrics_extension.py` → `compute_auc_and_sweep()` |
| **Stored in** | `runs_summary.jsonl` → `metrics["threshold_sweep"]` |

### 12.9 INCONCLUSIVE Policy Metrics

| Field | Value |
|---|---|
| **Name** | `inconclusive["A"]`, `inconclusive["B"]`, `inconclusive["C"]` |
| **Meaning** | Full classification metric sets under three post-hoc INCONCLUSIVE handling policies |
| **Policy A** | Re-label all INCONCLUSIVE as SUPPORT |
| **Policy B** | Re-label all INCONCLUSIVE as REFUTE |
| **Policy C** | Exclude all INCONCLUSIVE predictions; report coverage separately |
| **Computed in** | `run_eval_extended.py` → `compile_and_log_run_summary()` |
| **Stored in** | `runs_summary.jsonl` → `inconclusive` block |

---

## 13. Category 12 — Efficiency Metrics

Tracked per run across all processed claims.

### 13.1 Average Tokens Per Claim

| Field | Value |
|---|---|
| **Name** | `avg_tokens` |
| **Meaning** | Mean total token count (input + output combined) consumed per claim |
| **Formula** | `avg_tokens = Σ(token_total_i) / N` |
| **Unit / Range** | Float; typically 10,000–200,000+ tokens per claim |
| **Computed in** | `run_eval_extended.py` → `compile_and_log_run_summary()` |
| **Stored in** | `runs_summary.jsonl` → efficiency block |

### 13.2 Average Rounds Per Claim

| Field | Value |
|---|---|
| **Name** | `avg_rounds` |
| **Meaning** | Mean total debate rounds (normal + switched) per claim |
| **Formula** | `avg_rounds = Σ(total_rounds_i) / N` |
| **Unit / Range** | Float ∈ [2, 20] |
| **Computed in** | `run_eval_extended.py` → `compile_and_log_run_summary()` |

### 13.3 Average Evidence Per Claim

| Field | Value |
|---|---|
| **Name** | `avg_evidence` |
| **Meaning** | Mean number of evidence items retrieved and considered per claim (across all retrieval stages) |
| **Computed in** | `run_eval_extended.py` → `compile_and_log_run_summary()` |

### 13.4 Average Retrieval Calls Per Claim

| Field | Value |
|---|---|
| **Name** | `avg_retrieval_calls` |
| **Meaning** | Mean number of times the PubMed FAISS retriever was invoked per claim (includes initial + all P-RAG calls) |
| **Computed in** | `run_eval_extended.py` → `compile_and_log_run_summary()` |

---

## 14. Category 13 — Token Usage Metrics

Tracked per claim via monkey-patching of OpenAI and OpenRouter API clients in `run_eval_extended.py`.

### 14.1 Token Usage Breakdown (Per Claim)

| Metric | Description |
|---|---|
| `token_total` | Total tokens (input + output) across all LLM calls for the claim |
| `token_input` | Total input tokens only |
| `token_output` | Total output tokens only |
| `token_openai` | Total tokens through OpenAI API (GPT models) |
| `token_openai_input` | Input tokens via OpenAI |
| `token_openai_output` | Output tokens via OpenAI |
| `token_openrouter` | Total tokens through OpenRouter API |
| `token_openrouter_input` | Input tokens via OpenRouter |
| `token_openrouter_output` | Output tokens via OpenRouter |
| `token_groq` | Total tokens through Groq API (if used) |
| `token_models` | Per-model breakdown: `{"model_name": {"in": N, "out": N, "tot": N}, ...}` |

**Computed in:** `run_eval_extended.py` → `apply_monkey_patches()` (intercepting `requests.post`, `openai.completions.create`, and `openai.responses.create`)

**Stored in:** `claims_added.jsonl` per-claim record

**Logged to:** Execution log: `"[Token Usage] Model: ..., Input: ..., Output: ..., Total: ..."`

---

## 15. Metrics Dependency Flow

The following diagram illustrates how metrics feed into each other across pipeline stages:

```
[Argument Mining]
  └─ premise_count
       └─→ drives retrieval call count in prepare_pools()

[Initial Retrieval]
  └─ relevance_score (FAISS similarity)
       └─→ overwritten by admissibility_weight in negotiation

[Evidence Negotiation]
  ├─ relevance (LLM)
  ├─ credibility (LLM)
  └─ admissibility_weight = relevance × credibility
       ├─→ admitted / disputed / rejected classification
       └─→ final_evidence_set → MADOrchestrator.evidence_pool

[Progressive RAG — Per Round Per Side]
  ├─ novelty_score (per item) → avg_novelty → CONVERGENCE CHECK
  ├─ avg_relevance → relevance_gain → P-RAG STOP
  ├─ redundancy_ratio → P-RAG STOP
  ├─ num_accepted → new_evidence_count (per round)
  └─ stop_reason → prag_history.jsonl

[Self-Reflection — Per Round Per Agent]
  ├─ logic, novelty, rebuttal
  ├─ total_score = 0.4×logic + 0.3×novelty + 0.3×rebuttal
  ├─ delta_score = total_score_t − total_score_{t−1} → DEBATE CONVERGENCE CHECK
  ├─ confidence_adjustment = (total_score − 0.5) × 0.6 → FINAL CONFIDENCE
  └─ discovery_need → next round P-RAG query

[Critic Agent — Per Round]
  ├─ logic, evidence, rebuttal (per side)
  ├─ unresolved_premises
  └─ debate_resolved → DEBATE CONVERGENCE CHECK

[Role-Switching]
  ├─ consistency_score ∈ [0, 10]
  ├─ is_consistent = (consistency_score ≥ 6)
  └─ rs_adj = f(consistency_score) → FINAL CONFIDENCE

[Judicial Panel — 3 Judges]
  ├─ evidence_strength, argument_validity, scientific_reliability (per judge)
  ├─ verdict (per judge) → vote_breakdown
  ├─ final_verdict = majority(v1, v2, v3)
  └─ avg scores → quality_score → FINAL CONFIDENCE

[Final Verdict]
  ├─ consensus_strength = winning_votes / 3
  ├─ quality_score = (avg_ev + avg_av + avg_sr) / 30 × 0.3
  ├─ base_confidence = consensus_strength × 0.8 + quality_score
  ├─ confidence = clamp(base_confidence + rs_adj + reflection_adj, 0, 1)
  └─ correct = (verdict == ground_truth)

[Aggregate — Run Level]
  ├─ accuracy, macro_F1, micro_F1, balanced_accuracy
  ├─ AUC, threshold_sweep
  ├─ Cohen's Kappa (pairwise + vs. GT)
  ├─ KS Stability (D_t per round)
  └─ efficiency: avg_tokens, avg_rounds, avg_evidence
```

---

## 16. Logging & Storage

### 16.1 Per-Claim Structured Output Files

All files are stored under `framework/outcome/all_output_jsons/` as **append-only JSONL**, one line per claim:

| File | Content | Written by |
|---|---|---|
| `claims_added.jsonl` | All per-claim metrics (tokens, rounds, scores, verdict) | `logging_extension.py` → `append_jsonl()` |
| `final_verdict.jsonl` | Full verdict with confidence, reasoning, key evidence | `final_verdict.py` → `logging_extension.append_framework_json()` |
| `judge_evaluation.jsonl` | 3-judge scores, verdicts, majority opinion | `judge_evaluator.py` → `append_framework_json()` |
| `debate_transcript.jsonl` | Per-round arguments, expert testimonies, P-RAG metrics | `mad_orchestrator.py` → `append_framework_json()` |
| `debate_transcript_switched.jsonl` | Same as above for role-switched debate | `mad_orchestrator.py` → `append_framework_json()` |
| `self_reflection.jsonl` | Reflection history per round per agent | `self_reflection.py` → `append_framework_json()` |
| `prag_history.jsonl` | Full P-RAG retrieval history | `prag_engine.py` → `append_framework_json()` |
| `role_switch_report.jsonl` | Consistency analysis and score | `role_switcher.py` → `append_framework_json()` |
| `judge_visibility.jsonl` | Judge-visible PRAG query evolution | `mad_orchestrator.py` → `append_framework_json()` |
| `runs_summary.jsonl` | Aggregate classification metrics per run | `logging_extension.py` → `log_run_summary()` |

### 16.2 Per-Claim Execution Logs

Text logs at `framework/outcome/logs/execution_log_{claim_id}_{run_id}.txt` capture a complete, human-readable dual-stream trace of all pipeline steps, LLM inputs/outputs, metric computations, and intermediate values.

### 16.3 Simple Verdict File

`framework/outcome/all_verdicts.jsonl` contains a compact summary per claim:
```json
{"claim_id": "42", "verdict": "SUPPORT", "confidence": 0.847, "ground_truth": "SUPPORT", "correct": true}
```

### 16.4 Progress Tracker

`framework/outcome/processed_claims.txt` records completed claim IDs (format: `{claim_id}:{run_index}`) to enable resumable runs.

### 16.5 JSONL Format Convention

All structured JSONL files follow the schema:
```json
{"timestamp": "...", "claim_id": "...", "run_id": "...", "data": { ... }}
```
The `data` field contains the actual metric payload. Written via `logging_extension.append_framework_json()`.

---

## 17. Special Cases & Assumptions

### 17.1 Novelty Score — Empty Pool Edge Case
When `ProgressiveRAG.total_evidence_pool` is empty (first retrieval of a debate), all candidate items receive `novelty_score = 1.0` by default. This ensures no evidence is rejected on the first call. Implemented in `prag_engine.py` → `_calculate_novelty()`.

### 17.2 Confidence Adjustment Clamping
The reflection-derived `confidence_adjustment` is capped at `−0.15` even if the raw formula produces a more negative value (e.g., when `total_score ≈ 0`). This prevents a poor self-evaluation from completely nullifying a unanimous judicial verdict. Implemented in `final_verdict.py` → `_calculate_confidence()`.

### 17.3 INCONCLUSIVE Handling
The pipeline outputs `INCONCLUSIVE` when the judicial panel cannot reach a clear majority. This is a valid third class. Three post-hoc policy remappings (A/B/C) are evaluated at aggregate level but only one is selected via `--inconclusive-policy` CLI flag for primary reporting.

### 17.4 Minimum Confidence Floor
If `consensus_strength > 0.5` (at least 2 of 3 judges agree), a minimum confidence of `0.10` is enforced even after subtracting adjustments. This prevents a near-zero confidence when there is clear panel consensus.

### 17.5 KS Stability Approximation
The KS statistic `D_t` is computed over per-claim final confidence scores rather than true per-round intra-claim confidence distributions. Extracting genuine per-round confidence values would require architectural changes to expose intermediate state. This is acknowledged as an approximation.

### 17.6 Judge Score Normalization
Raw judge scores are integers 0–10. They are normalized to [0, 1] (dividing by 10) when computing `quality_score` inside `_calculate_confidence()`. The `judge_evaluation.jsonl` file always stores the raw integer range.

### 17.7 Cohen's Kappa — Single Claim Invalidity
Pairwise Cohen's Kappa for a single claim (one vote per judge) is statistically undefined. Kappa is therefore computed **only at the run level** (dataset level), over all claims processed in a run. The per-claim `judge_votes` dict is stored for later aggregation.

### 17.8 Role-Switch JSON Parse Failures
If the LLM consistency analysis response cannot be parsed as valid JSON, `consistency_score` defaults to `5` and `is_consistent` to `False`. This results in a neutral `rs_adj = 0.0`. Fallback implemented in `final_verdict.py` → `_check_role_switch_consistency()`.
