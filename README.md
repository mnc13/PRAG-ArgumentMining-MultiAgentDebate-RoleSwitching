# PROClaim : Courtroom-Style Multi-Agent Debate with Role-Switching and Progressive RAG for Controversial Claim Verification

> **A courtroom-style multi-agent debate framework for automated fact-checking, combining Progressive Retrieval-Augmented Generation (P-RAG), adversarial argument mining, and judicial panel evaluation.**

<div align="center">

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![PubMed Corpus](https://img.shields.io/badge/corpus-PubMed%20COVID--19-red.svg)]()

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation & Requirements](#installation--requirements)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples & Expected Output](#examples--expected-output)
- [Code Structure](#code-structure)
- [Evaluation & Metrics](#evaluation--metrics)
- [Extending the Framework](#extending-the-framework)
- [References & Acknowledgments](#references--acknowledgments)

---

## Overview

**PROClaim : Courtroom-Style Multi-Agent Debate with Role-Switching and Progressive RAG for Controversial Claim Verification** is a fact-checking pipeline for claim verification. It models the fact-checking process as a **structured legal proceeding**, where:

- A **Plaintiff Counsel** (proponent LLM agent) argues *for* the claim
- A **Defense Counsel** (opponent LLM agent) argues *against* the claim
- A presiding **Court Agent** (judge LLM) oversees proceedings, moderates evidence requests, and determines when to close arguments
- A **3-judge judicial panel** independently evaluates the full record and issues a majority verdict

The system grounds all argumentation in a large-scale **PubMed FAISS index** (~1.4 GB, 2020–2024 COVID literature), retrieved progressively through an adaptive retrieval mechanism (**P-RAG**) that filters for novelty and terminates when the evidence well runs dry.

**Primary application:** Verification of controversial claims, evaluated on the **Check-COVID** dataset.

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Argument Mining** | Decomposes claims into atomic, testable premises via DeepSeek-R1 |
| ⚖️ **Evidence Negotiation** | Stance-conditioned discovery + LLM-scored admissibility (Relevance × Credibility) |
| 🔄 **Progressive RAG (P-RAG)** | Round-by-round adaptive retrieval with novelty scoring and automatic stopping criteria |
| 🤖 **Multi-Agent Debate** | Heterogeneous LLM agents (OpenAI, OpenRouter) debate in structured proceedings |
| 🧑‍⚖️ **Expert Witness System** | Dynamically generated expert personas called on demand during proceedings |
| 🪞 **Role-Switching** | Agents swap sides to test argument robustness and logical consistency |
| 👩‍⚖️ **Judicial Panel** | 3 independent appellate judges vote via majority ruling |
| 📊 **Extended Metrics** | Accuracy, Macro-F1, AUC-ROC, Cohen's Kappa (inter-judge reliability), KS Stability |
| 💾 **Resumable Runs** | Progress tracked via `processed_claims.txt`; supports offset/limit batching |
| 🔌 **LLM-Agnostic Design** | Pluggable LLM clients for OpenAI, OpenRouter |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_eval_extended.py                        │
│   (Entry point — wraps main_pipeline with metrics + logging)    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                 ┌───────────▼───────────┐
                 │    main_pipeline.py   │
                 └───────────┬───────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │ 1. Load Claim  (data_loader.py)      │
          │ 2. Preprocess  (preprocessing.py)    │
          │ 3. Mine Args   (agent_workflow.py)   │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │ 4. Initial PubMed FAISS Retrieval    │
          │    (rag_engine.PubMedRetriever)      │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │ 5. Evidence Negotiation & Arbitration│
          │    (negotiation_engine.py)           │
          │    ▸ Premise-grounded retrieval      │
          │    ▸ Stance-conditioned retrieval    │
          │    ▸ LLM admissibility scoring       │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │ 6. Multi-Agent Debate (MAD)           │
          │    (mad_orchestrator.py)             │
          │    Per round:                        │
          │    ▸ P-RAG discovery (prag_engine.py)│
          │    ▸ Argument generation             │
          │    ▸ Expert witness testimony        │
          │    ▸ Self-reflection scoring         │
          │    ▸ Critic evaluation               │
          │    ▸ Adaptive convergence check      │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │ 7. Role-Switching (role_switcher.py) │
          │    ▸ Agents swap sides               │
          │    ▸ Full debate re-run              │
          │    ▸ Consistency analysis            │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │ 8. Judicial Panel (judge_evaluator)  │
          │    ▸ 3 independent judges            │
          │    ▸ 6-stage holistic evaluation     │
          │    ▸ Majority voting                 │
          └──────────────────┬──────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │ 9. Final Verdict (final_verdict.py)  │
          │    ▸ SUPPORT / REFUTE / INCONCLUSIVE │
          │    ▸ Confidence score                │
          │    ▸ Reasoning chain                 │
          └─────────────────────────────────────┘
```

---

## Installation & Requirements

### Prerequisites

- Python 3.8+
- Access to **OpenAI**, **OpenRouter**
- ~2 GB disk space for the PubMed FAISS index files (pre-built; not included in repo due to size)

### 1. Clone the Repository

```bash
git clone https://github.com/mnc13/PRAG-ArgumentMining-MultiAgentDebate-RoleSwitching.git
cd PRAG-ArgumentMining-MultiAgentDebate-RoleSwitching
```

### 2. Create a Virtual Environment

```bash
cd framework
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The key runtime dependencies are:

| Package | Purpose |
|---|---|
| `faiss-cpu` | Vector similarity search over PubMed index |
| `sentence-transformers` | Sentence embedding (`all-MiniLM-L6-v2`) |
| `openai` | OpenAI API client (GPT models) |
| `requests` | HTTP client for OpenRouter API |
| `numpy` | Numerical operations, offset arrays |
| `python-dotenv` | API key management via `.env` |
| `torch` | PyTorch backend for transformer models |

> **Note:** The `requirements.txt` in the root directory lists all transitive dependencies for full reproducibility. For a leaner install, install only core packages listed above.

### 4. Set Up PubMed FAISS Index

The retrieval corpus index is **not included** in the repository due to size. Follow:

```bash
# Download PubMed abstracts (COVID-19, 2020-2024)
python framework/build_pubmed_corpus.py

# Build FAISS index
python framework/build_faiss.py
```

---

## Configuration

### API Keys

Create a `.env` file in the `framework/` directory:

```env
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```

### Agent LLM Configuration

LLM model assignments for each agent role are centrally defined in [`framework/personas.py`](framework/personas.py):

```python
AGENT_SLOTS = {
    "proponent":   { "llm_provider": "openai",      "llm_model": "gpt-5-mini",                    "temperature": 0.5 },
    "opponent":    { "llm_provider": "openrouter",   "llm_model": "deepseek/deepseek-v3.2",        "temperature": 0.5 },
    "judge":       { "llm_provider": "openrouter",   "llm_model": "qwen/qwen3-235b-a22b-2507",     "temperature": 0.2 },
    "expert_slot": { "llm_provider": "openrouter",   "llm_model": "nousresearch/hermes-3-llama-3.1-405b", "temperature": 0.5 },
    "critic":      { "llm_provider": "openrouter",   "llm_model": "deepseek/deepseek-r1",          "temperature": 0.3 },
}
```

To substitute a different model, update the `llm_model` and `llm_provider` fields. The framework enforces that **each agent must use a distinct model** via `validate_unique_models()`.

---

## Usage

All commands should be run from the `framework/` directory.

### Basic Run

```bash
cd framework
python run_eval_extended.py --limit 1
```

Processes one claim from the Check-COVID test set and saves all outputs to `framework/outcome/`.

### Full Command Reference

```bash
python run_eval_extended.py [OPTIONS]

Options:
  --limit N                  Number of claims to process (default: 1)
  --offset M                 Skip first M claims in the dataset (default: 0)
  --force                    Re-process claims already in processed_claims.txt
  --runs R                   Number of repeated runs per claim set (default: 1)
```

### Batch Processing

```bash
# Process claims 0-9
python run_eval_extended.py --limit 10 --offset 0

# Process claims 10-19 (resumable)
python run_eval_extended.py --limit 10 --offset 10

# Re-run all claims, forcing re-evaluation
python run_eval_extended.py --limit 50 --force

# Multi-run (for variance estimation)
python run_eval_extended.py --limit 10 --runs 3
```

### Resume Interrupted Runs

Runs are automatically resumable. Already-processed claims are tracked in `framework/outcome/processed_claims.txt`. Simply re-run the same command — completed claims will be skipped.

---

## Examples & Expected Output

### Input

A claim from the Check-COVID test set:
```json
{
  "id": "42",
  "claim": "COVID-19 vaccines are effective at preventing severe disease and hospitalization.",
  "label": "SUPPORT"
}
```

### Pipeline Output (console summary)

```text
=== PRAG EVALUATION EXTENSION LAYER ===
Executing 1 runs. Output safely appending to ../artifacts/metrics

1. Loading Data...
2. Preprocessing & Extraction...
   Extracted: Increase health care capacity would have no effect on the duration of the COVID-19 pandemic.

3. Argument Mining...
   [Token Usage] Model: deepseek-r1 | Total: 876
   [DECOMPOSED PREMISES/ARGUMENTS]:
   - 1. Healthcare capacity increase does not reduce COVID-19 transmission rates.
   - 2. Case fatality ratio of COVID-19 is independent of healthcare capacity.
   - 3. Pandemic duration is primarily determined by factor-unrelated to healthcare capacity (e.g., NPIs).

4. Initial RAG Retrieval...
   [INITIAL RETRIEVED EVIDENCE]:
   - Evidence 1 (ID: 33937565): Hospital capacity during the COVID-19 pandemic.
   - Evidence 2 (ID: 33781225): Effects of medical resource capacities on outcomes.

5. Evidence Negotiation & Arbitration...
   --- [Negotiator] Perspectives: Plaintiff & Defense gathered 6 specific items.
   [JUDICIAL ADMISSION] Admitted 2 exhibits for global discovery (IDs: 33793611, 33781225).

6. Courtroom Proceedings (MAD + Role-Switching)...
   [Phase 1-5] Multi-round debate with expert witnesses.
   [EXPERT TESTIMONY]: "...healthcare capacity directly impacts key pandemic outcomes... shortening the acute crisis phase."
   [CONSISTENCY ANALYSIS]: Agent A: 9/10 | Agent B: 7/10

8. Judicial Panel Evaluation...
   Judge 1 (deepseek-r1):         NOT SUPPORTED — Evidence Strength: 6/10
   Judge 2 (hermes-3-llama-405b): NOT SUPPORTED — Evidence Strength: 8/10
   Judge 3 (qwen3-235b):          NOT SUPPORTED — Evidence Strength: 8/10

=============================
[CLAIM 5f8a0530e95347460249fc61_0] rounds_norm=3 rounds_switch=2 tok=271,359 retr=25 ev=77 conf=1.000
[CLAIM 5f8a0530e95347460249fc61_0] judges: NOT SUPPORTED, NOT SUPPORTED, NOT SUPPORTED
=============================

11. Generating Final Verdict...
    Verdict: REFUTE
    Confidence: 1.000
    Correct: True (GT: REFUTE)
```

### Saved Artifacts

All structured outputs are written to `framework/outcome/`:

| File | Description |
|---|---|
| `all_verdicts.jsonl` | Per-claim verdicts with confidence and correctness |
| `all_output_jsons/final_verdict.jsonl` | Detailed verdict with reasoning chain |
| `all_output_jsons/debate_transcript.jsonl` | Full multi-round debate record |
| `all_output_jsons/judge_evaluation.jsonl` | JSON scores from the 3-judge panel |
| `all_output_jsons/debate_transcript_switched.jsonl` | Role-switched debate record |
| `all_output_jsons/negotiation_state_{id}.json` | Evidence pools and admissibility scores |
| `logs/execution_log_{id}_{run_id}.txt` | Full dual-stream execution log |
| `../artifacts/metrics/claims_added.jsonl` | Extended metrics for the evaluation layer |

---

## Code Structure

```
PRAG-ArgumentMining-MultiAgentDebate-RoleSwitching/
│
├── Check-COVID/                        # Fact-checking dataset
│   └── test/
│       └── covidCheck_test_data.json   # Test claims (SUPPORT/REFUTE, no NEI)
│
├── framework/                          # All pipeline source code
│   │
│   ├── run_eval_extended.py            # ★ ENTRY POINT — runs evaluation + extended metrics
│   ├── main_pipeline.py                # Core claim processing loop
│   │
│   │   ── Core Data Layer ──
│   ├── models.py                       # Dataclasses: Claim, Evidence, Argument, DebateState
│   ├── data_loader.py                  # DataLoader (JSON/JSONL auto-detection)
│   ├── preprocessing.py                # ClaimExtractor (claim normalization)
│   │
│   │   ── Retrieval Layer ──
│   ├── rag_engine.py                   # PubMedRetriever (FAISS), VectorRetriever, SimpleRetriever
│   ├── prag_engine.py                  # ProgressiveRAG — novelty-filtered adaptive retrieval
│   │
│   │   ── Debate Engine ──
│   ├── agent_workflow.py               # ArgumentMiner — premise decomposition
│   ├── negotiation_engine.py           # EvidenceNegotiator — discovery + arbitration
│   ├── mad_system.py                   # DebateAgent, CriticAgent
│   ├── mad_orchestrator.py             # MADOrchestrator — per-round debate management
│   ├── self_reflection.py              # SelfReflection — per-round introspective scoring
│   ├── personas.py                     # AGENT_SLOTS registry + create_llm_client() factory
│   ├── expertise_extractor.py          # Dynamic expert witness persona generation
│   ├── role_switcher.py                # RoleSwitcher — side swap + consistency analysis
│   │
│   │   ── Evaluation Layer ──
│   ├── judge_evaluator.py              # JudicialPanel — 3-judge evaluation + majority voting
│   ├── final_verdict.py                # FinalVerdict — confidence weighting + output
│   ├── logging_extension.py            # ExtensionState, JSONL logging, run summaries
│   ├── metrics_extension.py            # Accuracy, F1, AUC, Cohen's Kappa, KS Stability
│   │
│   │   ── LLM Client Layer ──
│   ├── llm_client.py                   # LLMClient base class, 
│   ├── openai_client.py                # OpenAILLMClient (Chat + Responses API)
│   ├── openrouter_client.py            # OpenRouterLLMClient (requests-based)
│   │
│   │   ── Corpus Files (not in repo) ──
│   ├── pubmed_faiss.index              # FAISS binary index (~1.4 GB)
│   ├── pubmed_meta.jsonl               # Article metadata (~1.0 GB)
│   ├── pubmed_meta_offsets.npy         # Byte offsets for O(1) metadata access (~7.5 MB)
│   │
│   │   ── Offline Build Scripts ──
│   ├── build_faiss.py                  # Encodes PubMed corpus, builds FAISS index
│   ├── build_pubmed_corpus.py          # Downloads PubMed abstracts from NCBI API
│   ├── build_pubmed_corpus_sliced.py   # Sliced variant for large datasets
│   ├── merge_pubmed_slices.py          # Merges sliced corpus files
│   │
│   │   ── Offline Analysis Scripts ──
│   ├── evaluate_results.py             # Post-hoc evaluation from saved verdicts
│   ├── combine_all_metrics.py          # Aggregate metrics from multiple runs
│   ├── summarize_added_metrics.py      # Summarizes claims_added.jsonl
│   ├── sync_logs_to_outcomes.py        # Syncs execution logs to outcome directories
│   ├── rescan_and_fix_metrics.py       # Corrects corrupted metric files
│   │
│   ├── baseline/                       # Baseline pipelines (non-debate, single LLM)
│   │   ├── deepseek_pipeline.py        # DeepSeek-only baseline
│   │   ├── gpt_pipeline.py             # GPT-only baseline
│   │   └── ...
│   │
│   ├── outcome/                        # Output directory (auto-created at runtime)
│   │   ├── all_verdicts.jsonl
│   │   ├── processed_claims.txt
│   │   ├── logs/
│   │   └── all_output_jsons/
│   │
│   ├── requirements.txt                # Python dependencies (framework-specific)
│   └── venv/                           # Virtual environment (not committed)
│
├── requirements.txt                    # Full pinned dependency list (root)
├── PIPELINE_DOCUMENTATION.md          # In-depth technical documentation
└── README.md                          # This file
```

---

## Evaluation & Metrics

Metrics are computed per run in [`framework/metrics_extension.py`](framework/metrics_extension.py) and logged by [`framework/logging_extension.py`](framework/logging_extension.py).

### Classification Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall fraction of correct predictions |
| Macro Precision / Recall / F1 | Unweighted average across classes |
| Balanced Accuracy | Macro recall (accounts for class imbalance) |
| Per-Class P/R/F1 | Computed per label from confusion matrix |

### Judge Reliability (Cohen's Kappa)

Pairwise Cohen's Kappa is reported for all three judge pairs and for each judge vs. ground truth. Unanimity rate and split rate quantify panel cohesion.

```
κ = (P_o − P_e) / (1 − P_e)
```

### KS Stability

The Kolmogorov-Smirnov statistic `D_t` measures the shift in confidence score distributions across rounds, with stabilization points reported at `ε ∈ {0.03, 0.05, 0.07}`.

---

## Extending the Framework

### Adding a New LLM Provider

Create a new client in `framework/` and register the provider in `create_llm_client()` inside `personas.py`

### Swapping Agent Models

Edit the `AGENT_SLOTS` dictionary in [`framework/personas.py`](framework/personas.py). Each slot requires:
- `llm_provider`: `"openai"`, `"openrouter"`
- `llm_model`: Model identifier string
- `temperature`: Float
- `system_prompt`: Role-specific instruction string

### Using a Different Dataset

Implement a new loader or extend `DataLoader` in [`framework/data_loader.py`](framework/data_loader.py). The only requirement is that it returns a list of `Claim(id, text, metadata={'label': ...})` objects. Then point `main_pipeline.py`'s `test_file_path` to your new dataset file.

### Adding New Evaluation Metrics

Add a function to [`framework/metrics_extension.py`](framework/metrics_extension.py) and call it in `compile_and_log_run_summary()` inside [`framework/run_eval_extended.py`](framework/run_eval_extended.py).

---

## References & Acknowledgments

### Dataset

- **Check-COVID:** A COVID-19 fact-checking benchmark derived from scientific literature. All claims in this project are evaluated on the test split.

### Retrieval Corpus

- **PubMed COVID-19 Literature (2020–2024):** Biomedical abstracts retrieved via the NCBI Entrez API. Indexed using Facebook AI Similarity Search (FAISS).

### Models & Tools

| Tool | Use |
|---|---|
| [DeepSeek-R1](https://api-docs.deepseek.com/) | Argument mining, critic agent, Judge 1 |
| [DeepSeek-V3.2](https://api-docs.deepseek.com/) | Defense Counsel |
| [GPT-5-mini](https://platform.openai.com/docs) | Plaintiff Counsel |
| [Qwen3-235B](https://openrouter.ai/) | Presiding Judge (Court), Judge 3 |
| [Hermes-3-Llama-3.1-405B](https://openrouter.ai/) | Expert Witnesses, Judge 2 |
| [DeepSeek-Chat](https://openrouter.ai/) | Consistency analysis |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Sentence embedding for FAISS retrieval |
| [FAISS](https://github.com/facebookresearch/faiss) | Billion-scale vector similarity search |

---

## Contact

For questions about this framework, please open a GitHub issue or contact the repository maintainer.

---

<div align="center">
<sub>Built for research in automated fact-checking · Not intended for clinical decision-making</sub>
</div>
