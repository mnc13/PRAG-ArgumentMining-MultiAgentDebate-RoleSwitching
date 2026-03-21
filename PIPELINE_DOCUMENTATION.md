# PRAG-ArgumentMining-MultiAgentDebate-RoleSwitching: Pipeline Documentation

> **Document Purpose:** Complete technical documentation of the fact-checking pipeline, suitable as a foundation for a research paper. All sections include references to the relevant source files and functions.

---

## Table of Contents

1. [Overview of the Pipeline](#1-overview-of-the-pipeline)
2. [Datasets and Preprocessing](#2-datasets-and-preprocessing)
3. [Methodology](#3-methodology)
   - [3.1 Argument Mining](#31-argument-mining)
   - [3.2 Initial RAG Retrieval (PubMed)](#32-initial-rag-retrieval-pubmed)
   - [3.3 Evidence Negotiation & Arbitration](#33-evidence-negotiation--arbitration)
   - [3.4 Progressive RAG (P-RAG)](#34-progressive-rag-p-rag)
   - [3.5 Multi-Agent Debate (MAD) Orchestration](#35-multi-agent-debate-mad-orchestration)
   - [3.6 Self-Reflection & Critic Agent](#36-self-reflection--critic-agent)
   - [3.7 Role-Switching Consistency Test](#37-role-switching-consistency-test)
   - [3.8 Judicial Panel Evaluation](#38-judicial-panel-evaluation)
   - [3.9 Final Verdict Generation](#39-final-verdict-generation)
4. [Experimental Setup](#4-experimental-setup)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Code Structure and File Descriptions](#6-code-structure-and-file-descriptions)
7. [Additional Notes](#7-additional-notes)

---

## 1. Overview of the Pipeline

This system implements a **courtroom-style multi-agent debate pipeline** for automated fact-checking of medical and COVID-19–related claims. The design draws on legal metaphors to structure the reasoning process: claims are treated as cases, agents take on roles of Plaintiff Counsel (proponent) and Defense Counsel (opponent), and a judicial panel issues a verdict.

### High-Level Inputs and Outputs

| Item | Description |
|---|---|
| **Input** | A medical claim (natural language sentence), e.g., from the Check-COVID dataset |
| **External Knowledge** | 1.4 GB+ PubMed FAISS index of COVID-19 literature (2020–2024) |
| **Output** | A structured verdict: `SUPPORT`, `REFUTE`, or `INCONCLUSIVE`, plus confidence score, reasoning chain, and evidence traces |

### Pipeline Execution Flow

The pipeline is triggered from `framework/run_eval_extended.py`, which wraps `framework/main_pipeline.py`. The complete sequence for **each claim** is as follows:

```
Claim Input
     │
     ▼
[1] Preprocessing          → ClaimExtractor (preprocessing.py)
     │
     ▼
[2] Argument Mining        → ArgumentMiner (agent_workflow.py) via DeepSeek-R1
     │
     ▼
[3] Initial Retrieval      → PubMedRetriever (rag_engine.py) — top-5 FAISS search
     │
     ▼
[4] Evidence Negotiation   → EvidenceNegotiator (negotiation_engine.py)
    │    ┌─ Stance-Conditioned Retrieval (Proponent & Opponent)
    │    └─ LLM-scored Admissibility Weighting
     │
     ▼
[5] Multi-Agent Debate     → MADOrchestrator (mad_orchestrator.py)
    │    ┌─ Progressive RAG (prag_engine.py) per round
    │    ├─ DebateAgent: Proponent / Opponent
    │    ├─ Expert Witnesses (expertise_extractor.py) on demand
    │    ├─ Self-Reflection (self_reflection.py) per round per side
    │    └─ Critic Evaluation (mad_system.py::CriticAgent) per round
     │
     ▼
[6] Role-Switching         → RoleSwitcher (role_switcher.py) — re-runs debate
     │
     ▼
[7] Judicial Panel         → JudicialPanel (judge_evaluator.py)
    │    └─ 3 independent appellate judges, majority voting
     │
     ▼
[8] Final Verdict          → FinalVerdict (final_verdict.py)
    │    └─ Confidence weighting from consensus + judge scores + role-switch + reflection
     │
     ▼
Output: SUPPORT / REFUTE / INCONCLUSIVE + confidence score
```

**Token counting**, **retrieval tracking**, and **run logging** are handled transparently by `logging_extension.py` and `metrics_extension.py` via monkey-patching (injected in `run_eval_extended.py`).

---

## 2. Datasets and Preprocessing

### 2.1 Primary Fact-Checking Dataset: Check-COVID

- **Source file loaded:** `Check-COVID/test/covidCheck_test_data.json`
- **Format:** JSON array containing claim objects
- **Fields per claim:**
  - `id` — unique integer identifier
  - `claim` — the text of the medical claim (natural language)
  - `label` — ground truth: `SUPPORT` or `REFUTE` (NEI claims excluded)
  - `evidence` — list of reference evidence items (used for evaluation only, not for retrieval)
- **Loaded by:** `DataLoader.load_specific_file()` in `framework/data_loader.py`
- **Loading modes:** Handles both JSON list and JSONL formats automatically via fallback parsing

The test set is intentionally limited to binary-labeled claims (Supports / Refutes) by excluding NEI (Not Enough Information) labels. This design choice simplifies evaluation to a binary classification task with an optional `INCONCLUSIVE` outcome from the pipeline.

### 2.2 Retrieval Corpus: PubMed COVID-19 Literature

The retrieval corpus consists of COVID-19–related biomedical abstracts collected from PubMed (2020–2024). The corpus is preprocessed offline into a FAISS binary index for fast semantic search.

| File | Description |
|---|---|
| `framework/pubmed_faiss.index` | FAISS `IndexFlatIP` with ~1.4 GB of normalized embeddings |
| `framework/pubmed_meta.jsonl` | JSONL metadata (PMID, title, abstract chunk, journal, year) |
| `framework/pubmed_meta_offsets.npy` | Numpy byte offset array for O(1) random access into the meta file |
| `framework/pubmed_covid_2020_2024_edat_upto_2024.jsonl.gz` | Raw source corpus (~272 MB compressed) |

**Build scripts (run offline, not part of the inference pipeline):**
- `build_pubmed_corpus.py` — Downloads abstracts from PubMed API
- `build_pubmed_corpus_sliced.py` — Sliced version for large corpora
- `build_faiss.py` — Encodes texts with `sentence-transformers/all-MiniLM-L6-v2` and builds the FAISS index

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors, normalized for cosine similarity via inner product `IndexFlatIP`).

### 2.3 Preprocessing of Claims

Preprocessing is minimal by design. The raw claim text is passed directly through `ClaimExtractor.extract_claim()` (`framework/preprocessing.py`), which creates a `Claim` dataclass instance. No tokenization, stemming, or normalization is applied at this stage to preserve the full semantics of the claim for LLM consumption.

The claim ID and metadata label are preserved and carried forward through all pipeline stages for final accuracy evaluation.

### 2.4 Train / Test Splits

All experiments are run on the **test split only** (`covidCheck_test_data.json`). The pipeline does not perform any training (it is a zero-shot, inference-only system). There is no train/dev/test split in the traditional ML sense — the pipeline processes each test claim independently using pre-trained LLMs and the pre-built PubMed index.

---

## 3. Methodology

### 3.1 Argument Mining

**File:** `framework/agent_workflow.py` — class `ArgumentMiner`  
**LLM used:** DeepSeek-R1 via OpenRouter (`framework/openrouter_client.py`)

The claim text is deconstructed into a set of **atomic, testable premises** using an LLM with the following prompt strategy:

> *"As a Scientific Analyst, decompose the following clinical claim into atomic, testable premises suitable for evidence retrieval..."*

The output is a structured list of premises (one per line). Parsing handles common formatting artifacts (numbering, bullets, asterisks) via regex cleaning. These premises serve two downstream purposes:
1. **Premise-grounded retrieval** in the negotiation phase
2. **Scoring reference** in self-reflection and critic evaluation

**Output type:** `Argument(claim_id, premises: List[str])`

---

### 3.2 Initial RAG Retrieval (PubMed)

**File:** `framework/rag_engine.py` — class `PubMedRetriever`

The system performs **semantic retrieval** over the PubMed FAISS index:

1. The claim text is encoded using `SentenceTransformer('all-MiniLM-L6-v2')` with `normalize_embeddings=True`
2. A FAISS `search()` call returns top-K nearest neighbors by inner product (cosine similarity due to normalization)
3. Metadata is read from `pubmed_meta.jsonl` using byte offsets stored in `pubmed_meta_offsets.npy` for O(1) disk access
4. Each retrieved item is returned as an `Evidence(text, source_id, relevance_score)` object

**Hyperparameters:**
- `top_k = 5` for initial retrieval

The `source_id` is the PMID or DOI of the article. Context enrichment includes year and journal prefix (e.g., `[Nature Medicine 2021] ...`) if available.

---

### 3.3 Evidence Negotiation & Arbitration

**File:** `framework/negotiation_engine.py` — class `EvidenceNegotiator`

This module implements a **4-step legal discovery procedure** to filter and weight evidence before the debate begins:

#### Step 1 — Premise-Grounded Shared Retrieval
For each decomposed premise from Step 3.1, the PubMed retriever is queried (`top_k=3`). All results are merged and deduplicated by `source_id` into a **shared pool**.

#### Step 2 — Stance-Conditioned Retrieval
Using the LLM, stance-specific search queries are generated for each side:
- **Proponent (Plaintiff Counsel):** Generates a query searching for evidence *supporting* the claim
- **Opponent (Defense Counsel):** Generates a query searching for evidence *challenging* the claim

Each side performs retrieval (`top_k=3`) over the PubMed index, building individual **proponent_pool** and **opponent_pool**.

#### Step 3 — Negotiation Injection
Both sides "review" each other's discovery pools via LLM prompts, simulating the disclosure and challenge process. This step influences the conceptual framing of the debate but does not directly modify the evidence pools.

#### Step 4 — Judicial Arbitration (Admissibility Scoring)
All candidate evidence items from all three pools are merged and deduplicated. For each evidence item, the LLM (`miner_llm`, same DeepSeek-R1 instance) evaluates **admissibility**:

```
Admissibility Weight = Relevance × Credibility
```

where `Relevance ∈ [0, 1]` and `Credibility ∈ [0, 1]` are LLM-assigned scores. Items with `weight > 0.5` are admitted; items with `0.1 < weight ≤ 0.5` are flagged as disputed. Admitted evidence is sorted by weight descending.

The output is a `negotiation_state` JSON saved to `outcome/negotiation_state_{claim_id}.json`, and the final **admitted evidence set** is passed to the MAD phase.

---

### 3.4 Progressive RAG (P-RAG)

**File:** `framework/prag_engine.py` — class `ProgressiveRAG`

P-RAG enables **targeted, round-by-round evidence discovery** during the debate. Unlike the initial retrieval which is static, P-RAG is designed to adaptively retrieve novel evidence in response to the evolving debate context.

#### Query Formulation
An LLM query is formulated based on:
- Rolling debate context (last 4 messages)
- Agent-specific discovery need (identified by the agent in the same round)
- Reflection-driven gaps from the prior round's self-reflection

The judge (Court agent) then **refines** the query to improve precision before executing retrieval.

#### Novelty Scoring
After retrieval, each new evidence item is scored against the existing evidence pool:
```
novelty_score = 1 - max_cosine_similarity(new_doc_embedding, pool_embeddings)
```
Cosine similarity is computed as a dot product over `normalize_embeddings=True` vectors from the SentenceTransformer model.

#### Stopping Criteria (applied per retrieval call)
| Criterion | Threshold |
|---|---|
| Novelty filter | Accept only items with `novelty_score ≥ 0.2` |
| Redundancy ratio | Stop if `> 70%` of retrieved items are redundant |
| Relevance gain | Stop if relevance gain `< 0.05` vs. previous round |
| Max iterations | Hard cap at 10 retrieval calls |

**Hyperparameters (defined in `ProgressiveRAG.__init__()`):**

| Parameter | Value |
|---|---|
| `novelty_threshold` | 0.20 |
| `redundancy_sim_threshold` | 0.85 |
| `redundancy_ratio_threshold` | 0.70 |
| `relevance_gain_threshold` | 0.05 |
| `max_iterations` | 10 |

---

### 3.5 Multi-Agent Debate (MAD) Orchestration

**File:** `framework/mad_orchestrator.py` — class `MADOrchestrator`  
**File:** `framework/mad_system.py` — class `DebateAgent`, `CriticAgent`  
**File:** `framework/personas.py` — `AGENT_SLOTS` registry

#### Agent Configuration

The debate involves four **fixed agent slots** defined in `personas.py::AGENT_SLOTS`:

| Role | Name | LLM Model | Provider | Temp |
|---|---|---|---|---|
| Proponent | Plaintiff Counsel | `gpt-5-mini` | OpenAI | 0.5 |
| Opponent | Defense Counsel | `deepseek/deepseek-v3.2` | OpenRouter | 0.5 |
| Judge | The Court | `qwen/qwen3-235b-a22b-2507` | OpenRouter | 0.2 |
| Expert Witness | Dynamic | `nousresearch/hermes-3-llama-3.1-405b` | OpenRouter | 0.5 |
| Critic | Critic Agent | `deepseek/deepseek-r1` | OpenRouter | 0.3 |

Each agent uses a **different LLM model** enforced by `validate_unique_models()` in `personas.py`, ensuring that no two agents share the same model to promote genuine reasoning diversity.

#### Per-Round Debate Procedure (5 Steps)

Each round (`MADOrchestrator.run_debate_round()`) executes:

1. **Evidence Discovery (Integrative Loop):** Proponent and Opponent each identify a gap in the evidence base, propose a search query, which is refined by the Judge (Court), and executed via P-RAG. Novel evidence is added to the shared pool.

2. **Argument Generation:** Each side generates a structured legal argument grounded in the current evidence pool and debate history (`DebateAgent.generate_argument()`).

3. **Expert Witness Testimony:** Each side may request a domain expert. The Judge evaluates the request and may grant it; if granted, a new expert agent is dynamically generated via `expertise_extractor.extract_single_expert()` (using Hermes 405B) and provides testimony.

4. **Self-Reflection:** Each side's agent performs a structured self-critique (`SelfReflection.perform_round_reflection()`), scoring themselves on logic, novelty, and rebuttal coverage. The resulting `discovery_need` is injected into the next round's query formulation.

5. **Critic Evaluation:** The Critic Agent (`CriticAgent.evaluate_round()`, backed by DeepSeek-R1) scores both sides independently on logic, evidence coverage, and rebuttal, and identifies unresolved premises. The critic can signal `debate_resolved=True` to trigger early stopping.

#### Adaptive Convergence / Stopping

The debate runs for up to `max_rounds=10` but terminates early under any of:
- **Reflection plateau:** |Δ total_reflection_score| < 0.05 for two consecutive rounds
- **Critic resolution:** `debate_resolved=True` from the Critic Agent
- **Evidence novelty exhaustion:** Average novelty < 0.10 across two consecutive rounds
- **Judicial signal:** The Court agent responds "Close" to a prompt asking if deliberation should begin

---

### 3.6 Self-Reflection & Critic Agent

**Self-Reflection — `framework/self_reflection.py` — class `SelfReflection`**

After each round, each counsel performs an introspective evaluation of their own performance:

- **Logic score** (`0.0–1.0`): Structural coherence of arguments
- **Novelty score** (`0.0–1.0`): Whether new information was introduced
- **Rebuttal score** (`0.0–1.0`): How thoroughly the opposing counsel's points were addressed

**Weighted total score:**
```
total_score = 0.4 × logic + 0.3 × novelty + 0.3 × rebuttal
```

**Confidence adjustment** (used in final verdict computation):
```
confidence_adjustment = (total_score − 0.5) × 0.6
```
This maps the total score from `[0, 1]` to `[−0.30, +0.30]`, creating a symmetric positive/negative contribution to the final confidence score.

---

**Critic Agent — `framework/mad_system.py` — class `CriticAgent`**

An independent observer backed by DeepSeek-R1 evaluates each round:
- Scores both sides (0.0–1.0) on logic, evidence coverage, and rebuttal
- Identifies unresolved premises
- Provides actionable recommendations for both sides
- Signals whether the debate has been sufficiently resolved (`debate_resolved` flag)

---

### 3.7 Role-Switching Consistency Test

**File:** `framework/role_switcher.py` — class `RoleSwitcher`

To test **argument robustness**, the pipeline enforces a role-switch after the primary debate. Plaintiff Counsel and Defense Counsel physically swap agents: the model that argued for the claim now argues against it, and vice versa.

The orchestrator resets all state (`MADOrchestrator.reset_state()`) and re-runs the full debate with `max_rounds=10` using the suffix `_switched` for artifact files.

**Consistency Analysis:**

A separate LLM (DeepSeek Chat via OpenRouter, `temperature=0.3`) receives both debate transcripts and analyzes:
- Does Agent A maintain logical consistency when forced to switch sides?
- Are contradictions present in the arguments?
- A `consistency_score ∈ [0, 10]` and a Boolean `is_consistent` flag (threshold: score ≥ 6)

The consistency score is used as an input to the confidence weighting in the final verdict.

---

### 3.8 Judicial Panel Evaluation

**File:** `framework/judge_evaluator.py` — class `JudicialPanel`

Three independent appellate judges evaluate the complete case record (both debate transcripts, admitted evidence, P-RAG metrics, critic evaluations, and self-reflection history).

**Judge LLM assignments:**

| Judge | Model | Provider |
|---|---|---|
| Judge 1 | `deepseek/deepseek-r1` | OpenRouter |
| Judge 2 | `nousresearch/hermes-3-llama-3.1-405b` | OpenRouter |
| Judge 3 | `qwen/qwen3-235b-a22b-2507` | OpenRouter |

All judges operate at `temperature=0.3` for low variance.

#### 6-Stage Evaluation per Judge

| Stage | Task |
|---|---|
| Stage 1 | Case Reconstruction — identify claim, main arguments from each side |
| Stage 2 | Evidence & Testimony Weighting — score Evidence Strength (0–10) |
| Stage 3 | Logical Coherence Analysis — score Argument Validity (0–10) |
| Stage 4 | Scientific/Technical Consistency — score Scientific Reliability (0–10) |
| Stage 5 | Discovery Rigor & Transparency — analyze P-RAG query evolution and novelty |
| Stage 6 | Judicial Verdict — SUPPORTED / NOT SUPPORTED / INCONCLUSIVE |

#### Majority Voting Aggregation

The three individual verdicts are aggregated via **majority voting** (`collections.Counter`). If ≥ 2 judges agree, that verdict is the final panel verdict. The system handles split decisions (1-1-1) by taking the most common vote. Majority and dissenting opinions are synthesized textually.

---

### 3.9 Final Verdict Generation

**File:** `framework/final_verdict.py` — class `FinalVerdict`

The final verdict maps the judicial panel outcome to the dataset label space:
- `SUPPORTED` → `"SUPPORT"`
- `NOT SUPPORTED` → `"REFUTE"`
- `INCONCLUSIVE` → `"INCONCLUSIVE"` (subject to policy remapping downstream)

#### Confidence Score Calculation

```
base_confidence = consensus_strength × 0.8 + quality_score × 0.3

where:
  consensus_strength = winning_votes / total_votes
  quality_score = (avg_evidence + avg_validity + avg_reliability) / 30

final_confidence = base_confidence + rs_adj + reflection_adj
```

| Adjustment | Source | Range |
|---|---|---|
| `rs_adj` | Role-switch consistency score | `−0.05` to `+0.10` |
| `reflection_adj` | Winner's last self-reflection | `−0.15` to `+0.30` |

The final confidence is clamped to `[0.0, 1.0]`. A minimum of 0.10 is enforced if consensus > 50%.

**Saved output (`all_verdicts.jsonl`):**
```json
{
  "claim_id": "...",
  "verdict": "SUPPORT | REFUTE | INCONCLUSIVE",
  "confidence": 0.73,
  "ground_truth": "SUPPORT",
  "correct": true
}
```

---

## 4. Experimental Setup

### 4.1 Software Environment

| Component | Version / Details |
|---|---|
| Python | 3.x (virtual environment at `framework/venv/`) |
| Sentence Transformers | Used for embedding, version from requirements |
| FAISS | `faiss-cpu` for vector index |
| PyTorch | `1.7.0` |
| NumPy | `1.19.1` |
| Transformers | `3.4.0` |
| `python-dotenv` | Used for loading API keys from `.env` |

### 4.2 LLM API Dependencies

The pipeline relies on **three LLM API providers**:

| Provider | Use Case | API Key Env Var |
|---|---|---|
| OpenAI | Plaintiff Counsel (GPT-5-mini) | `OPENAI_API_KEY` |
| OpenRouter | Defense Counsel, Judge, Expert, Critic, Consistency Analyzer, Mining | `OPENROUTER_API_KEY` |
| Groq | Optional (client available) | `GROQ_API_KEY` |

API keys are loaded via `python-dotenv` from a `.env` file in the framework directory.

LLM clients are implemented in:
- `framework/openai_client.py` — `OpenAILLMClient` wrapping the OpenAI Responses/Chat API
- `framework/openrouter_client.py` — `OpenRouterLLMClient` using raw `requests.post`
- `framework/groq_client.py` — `GroqLLMClient` using Groq's OpenAI-compatible API

### 4.3 Execution

The pipeline entry point is:
```bash
cd framework
python run_eval_extended.py --limit N --offset M [--force] [--runs R] [--inconclusive-policy {A,B,C}]
```

| Argument | Description |
|---|---|
| `--limit` | Number of claims to process |
| `--offset` | Skip first N claims (for batching) |
| `--force` | Re-process already completed claims |
| `--runs` | Number of repeated runs (same claims) |
| `--inconclusive-policy` | **A** = treat INCONCLUSIVE as SUPPORT; **B** = as REFUTE; **C** = exclude |

### 4.4 Reproducibility Notes

- **Progress tracking:** Processed claims are logged in `framework/outcome/processed_claims.txt` using `{claim_id}:{run_index}` keys, preventing duplicate processing on restart.
- **Execution logs:** Dual-stream logging per claim to `framework/outcome/logs/execution_log_{claim_id}_{run_id}.txt` captures both terminal and file output simultaneously.
- **Artifact files:** All structured outputs (debate transcripts, judge evaluations, PRAG history, verdicts) are saved as JSONL to `framework/outcome/all_output_jsons/`, one record appended per claim.
- **Token tracking:** Per-claim token usage (input/output/total, broken down by OpenAI and OpenRouter separately) is logged via monkey-patching `requests.post` and `openai.resources.chat.completions.Completions.create`.
- **Run IDs:** Each run is assigned a deterministic run ID (`uuid5` hash of the argument string), stored in `ExtensionState.run_id`.

---

## 5. Evaluation Metrics

All metrics are computed in `framework/metrics_extension.py` and logged by the `log_run_summary()` function in `framework/logging_extension.py`.

### 5.1 Classification Metrics

**Function:** `compute_classification_metrics(y_true, y_pred)` — `metrics_extension.py`

| Metric | Formula |
|---|---|
| Accuracy | `correct / total` |
| Macro Precision | Mean per-class precision |
| Macro Recall | Mean per-class recall |
| Macro F1 | Mean per-class F1 |
| Micro Precision / Recall / F1 | Equal to accuracy for multi-class |
| Balanced Accuracy | Equal to macro recall |
| Per-Class P/R/F1 | TP, FP, FN computed from confusion matrix per class |

The confusion matrix is computed fully across all label classes observed in `y_true ∪ y_pred`.

### 5.2 INCONCLUSIVE Policy Remapping

When the model outputs `INCONCLUSIVE`, three post-hoc policies are evaluated:

| Policy | Effect |
|---|---|
| **A** | Map INCONCLUSIVE → SUPPORT |
| **B** | Map INCONCLUSIVE → REFUTE |
| **C** | Exclude INCONCLUSIVEs from metric computation (coverage reported separately) |

All three variants of classification metrics are computed and logged when at least one INCONCLUSIVE prediction exists.

### 5.3 AUC-ROC and Threshold Sweep

**Function:** `compute_auc_and_sweep(y_true, confidences, pos_class="SUPPORT")` — `metrics_extension.py`

- **AUC:** Computed via trapezoidal rule on the empirical ROC curve, treating `SUPPORT` as the positive class
- **Threshold sweep:** Accuracy and macro-F1 are computed at thresholds from 0.30 to 0.70 in steps of 0.05, using the per-claim confidence score as the decision boundary

### 5.4 Judge Reliability (Cohen's Kappa)

**Function:** `compute_judge_reliability(judge_votes_list, y_true)` — `metrics_extension.py`

Pairwise Cohen's Kappa is computed for all three judge pairs (J1 vs J2, J1 vs J3, J2 vs J3):

```
κ = (P_o − P_e) / (1 − P_e)
```

where `P_o` is observed agreement and `P_e` is chance agreement.

Additional statistics:
- **Mean Kappa** (across all three pairs)
- **Judge vs. Ground Truth** Kappa (each judge individually)
- **Unanimity Rate** and **Split Rate** (3-0 vs 2-1 decisions)
- **Average Raw Agreement** (P_o) averaged over all pairs

**Kappa interpretation:**

| Range | Interpretation |
|---|---|
| κ < 0.20 | Slight agreement |
| 0.21–0.40 | Fair agreement |
| 0.41–0.60 | Moderate agreement |
| 0.61–0.80 | Substantial agreement |
| > 0.80 | Almost perfect agreement |

### 5.5 KS Stability Analysis

**Function:** `compute_ks_statistic(dist1, dist2)`, `analyze_stability(traces)` — `metrics_extension.py`

The **Kolmogorov-Smirnov (KS) statistic** is used to measure how much the distribution of model confidence scores changes across debate rounds:

```
D_t = sup_x |F_t(x) − F_{t−1}(x)|
```

where `F_t` is the empirical CDF of confidences at round `t`.

Stabilization is reported at epsilon thresholds of `{0.03, 0.05, 0.07}`, identifying the earliest round where `D_t < ε`.

> **Note:** In the current implementation, the per-round confidence distributions are approximated (the proxy uses final confidence scores across claims) since extracting per-round confidence from the multi-LLM pipeline would require deep architectural changes. This is noted as a limitation.

### 5.6 Efficiency Metrics

Logged per run:

| Metric | Description |
|---|---|
| `avg_tokens` | Average total tokens consumed per claim |
| `avg_rounds` | Average number of debate rounds per claim |
| `avg_evidence` | Average number of evidence items retrieved per claim |
| `avg_retrieval_calls` | Average P-RAG retrieval function call count per claim |

---

## 6. Code Structure and File Descriptions

```
PRAG-ArgumentMining-MultiAgentDebate-RoleSwitching/
├── Check-COVID/                         # Dataset directory
│   └── test/
│       └── covidCheck_test_data.json   # Primary evaluation dataset
│
├── framework/                           # All pipeline source code
│   ├── run_eval_extended.py             # [ENTRY POINT] Extended eval wrapper + monkey patches
│   ├── main_pipeline.py                 # Core orchestration: load claim → run pipeline → save verdict
│   │
│   ├── models.py                        # Dataclasses: Claim, Evidence, Argument, DebateState
│   ├── data_loader.py                   # DataLoader: load claims from JSON/JSONL
│   ├── preprocessing.py                 # ClaimExtractor: claim text normalization (passthrough)
│   │
│   ├── rag_engine.py                    # PubMedRetriever, VectorRetriever, SimpleRetriever
│   ├── build_faiss.py                   # [OFFLINE] Encodes corpus, builds FAISS index
│   ├── build_pubmed_corpus.py           # [OFFLINE] Downloads PubMed abstracts via API
│   ├── build_pubmed_corpus_sliced.py    # [OFFLINE] Sliced variant for large downloads
│   │
│   ├── agent_workflow.py                # ArgumentMiner, EvidenceFirstDebateAgent
│   ├── mad_system.py                    # DebateAgent, CriticAgent
│   ├── mad_orchestrator.py              # MADOrchestrator: manages multi-round debate
│   ├── self_reflection.py              # SelfReflection: per-round introspective scoring
│   ├── prag_engine.py                  # ProgressiveRAG: novelty-filtered dynamic retrieval
│   ├── negotiation_engine.py           # EvidenceNegotiator: discovery + arbitration
│   ├── expertise_extractor.py          # Dynamic expert persona generation via LLM
│   ├── personas.py                     # AGENT_SLOTS registry, create_llm_client() factory
│   ├── role_switcher.py                # RoleSwitcher: role reversal + consistency check
│   ├── judge_evaluator.py              # JudicialPanel: 3-judge evaluation + majority voting
│   ├── final_verdict.py                # FinalVerdict: confidence scoring + output generation
│   │
│   ├── logging_extension.py            # ExtensionState, append_jsonl, log_run_summary
│   ├── metrics_extension.py            # compute_classification_metrics, AUC, Kappa, KS
│   │
│   ├── llm_client.py                   # LLMClient base class, MockLLMClient, GeminiLLMClient
│   ├── openai_client.py                # OpenAILLMClient (Chat + Responses API)
│   ├── openrouter_client.py            # OpenRouterLLMClient (requests-based)
│   ├── groq_client.py                  # GroqLLMClient (OpenAI-compatible Groq API)
│   │
│   ├── pubmed_faiss.index              # FAISS binary index (~1.4 GB)
│   ├── pubmed_meta.jsonl               # Article metadata for retrieval (~1 GB)
│   ├── pubmed_meta_offsets.npy         # Byte offsets for O(1) JSONL access
│   │
│   ├── outcome/                        # All pipeline outputs (verdicts, logs, JSONL)
│   │   ├── logs/                       # Per-claim dual execution logs
│   │   ├── all_output_jsons/           # JSONL files: debate transcripts, judges, verdicts
│   │   ├── all_verdicts.jsonl          # Summary verdicts per claim
│   │   └── processed_claims.txt        # Progress tracker for resumable runs
│   │
│   └── venv/                           # Python virtual environment
│
├── requirements.txt                     # Python package list
└── PIPELINE_DOCUMENTATION.md           # This file
```

### Files Not in the Main Pipeline (Utility / Offline)

| File | Purpose |
|---|---|
| `evaluate_results.py` | Post-hoc evaluation from saved verdicts |
| `combine_all_metrics.py` | Combines per-claim metrics into aggregate tables |
| `summarize_added_metrics.py` | Summarizes `claims_added.jsonl` content |
| `sync_logs_to_outcomes.py` | Syncs execution logs to outcome directories |
| `rescan_and_fix_metrics.py` | Corrects and rescans saved metric files |
| `merge_pubmed_slices.py` | Merges sliced PubMed corpus files |
| `query_faiss.py` | Standalone FAISS query test tool |
| `filter_years.py` | Filters corpus by publication year |
| `run_eval_healthver.py` | Variant runner for HealthVer dataset |
| `logging_for_healthver.py` | Logging variant for HealthVer pipeline |
| `baseline/` | Baseline pipelines (DeepSeek-only, GPT-4-only) |

---

## 7. Additional Notes

### 7.1 Assumptions

- The pipeline assumes **binary-labeled claims** (SUPPORT / REFUTE). The NEI class is excluded from evaluation, and the dataset is filtered accordingly (`covidCheck_test_data.json`).
- The retrieval system assumes the PubMed FAISS index has been pre-built and all three index files (`pubmed_faiss.index`, `pubmed_meta.jsonl`, `pubmed_meta_offsets.npy`) are present in the `framework/` directory.
- All LLMs are accessed as **external API services**. Output variability is inherent to these models and temperature settings.

### 7.2 LLM Heterogeneity by Design

The system uses **intentionally different LLM models** for each agent role. This design decision (enforced by `validate_unique_models()` in `personas.py`) serves to:
1. Prevent agent collusion from shared internal representations
2. Introduce genuine disagreement reflecting different reasoning styles
3. Improve diversity of arguments and evidence perspectives

### 7.3 Limitations

- **KS Stability approximation:** The current implementation uses final-round confidence distributions as a proxy for per-round stability. True KS stability would require tracking the confidence distribution after every individual debate round, which would require significant architectural changes.
- **Expense:** Each claim requires numerous LLM API calls (ArgumentMining, Evidence Negotiation, up to 10 debate rounds × 2 sides × multiple prompts, role-switching, plus 3 judge evaluations). API costs scale directly with claim count and debate length.
- **Determinism:** Due to temperature > 0 for most agents, results are not fully deterministic across runs. Multi-run averaging is supported via `--runs` argument.
- **Context window:** Evidence text is truncated to 500 characters for debate prompts and 1000 characters for arbitration to fit within LLM context windows. This may cause loss of crucial details in longer abstracts.
- **PubMed retrieval scope:** The retrieval corpus is bounded by the `pubmed_covid_2020_2024` collection. Claims that require more recent literature or non-COVID biomedical knowledge may suffer reduced retrieval quality.

### 7.4 Novelty of the Approach

The key novelties of this system over prior fact-checking approaches are:

1. **Evidence Negotiation Phase:** A structured pre-debate discovery procedure where evidence is scored for admissibility using both relevance and credibility, modeled on legal discovery proceedings.
2. **Role-Switching Consistency Test:** A novel evaluation of argument robustness by forcing agents to swap roles, probing whether the logic is position-independent.
3. **Adaptive P-RAG:** Evidence retrieval is not a one-time step but a dynamic, round-by-round process driven by the current state of argumentation and adaptive stopping criteria.
4. **Heterogeneous Multi-LLM Panel:** The use of distinct LLM families for each agent prevents shared reasoning artifacts, and the three-judge majority voting framework draws on deliberative democracy principles.
