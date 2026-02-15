# Framework Execution Flow Analysis

This document traces the execution path from `main_pipeline.py` to the runtime events recorded in `execution_log_5f2d9300e95347460249fbb3_1.txt`.

## 1. Argument Mining
**Log Section:**
```text
3. Argument Mining...
   [DECOMPOSED PREMISES/ARGUMENTS]:
   - 1. Severe COVID-19 cases can be consistently defined...
```
**Code Mapping:**
- **File:** `framework/main_pipeline.py` calls `miner.mine_arguments(extracted_claim)`.
- **Class:** `ArgumentMiner` in `framework/agent_workflow.py`.
- **Method:** `mine_arguments(self, claim: Claim) -> Argument`.
- **Logic:** Calls the LLM to decompose the claim into atomic, testable premises.

## 2. Initial RAG Retrieval (FAISS)
**Log Section:**
```text
4. Initial RAG Retrieval...
   [DEBUG] Checking paths:
   Index: .../pubmed_faiss.index (Exists: True)
   Loading FAISS index from ...
   [INITIAL RETRIEVED EVIDENCE]:
```
**Code Mapping:**
- **File:** `framework/main_pipeline.py` initializes `PubMedRetriever`.
- **Class:** `PubMedRetriever` in `framework/rag_engine.py`.
- **Method:** `__init__` loads `pubmed_faiss.index`, `pubmed_meta.jsonl`, and `pubmed_meta_offsets.npy`.
- **Method:** `retrieve(self, query: str, top_k: int = 5)` executes the search using `self.index.search` and retrieves metadata via file offsets.

## 3. Evidence Negotiation
**Log Section:**
```text
5. Evidence Negotiation & Arbitration...
--- [Negotiator] Step 1: Premise-Grounded Shared Retrieval ---
...
--- [The Court] Step 4: Evidence Arbitration & Admissibility ---
   > Admitted 4 high-weight items.
```
**Code Mapping:**
- **File:** `framework/main_pipeline.py` initializes `EvidenceNegotiator` and calls `negotiator.judge_arbitration()`.
- **Class:** `EvidenceNegotiator` in `framework/negotiation_engine.py`.
- **Method:** `negotiate_phase` and `judge_arbitration` handle the evidence pool filtering and initial admissibility check before the debate starts.

## 4. Multi-Agent Debate (Proceedings)
**Log Section:**
```text
6. Initializing Multi-Agent Legal Proceedings ( Courtroom MAD)...
7. Presiding Over Courtroom Proceedings...
============================================================
PROCEEDINGS PHASE 1
============================================================
--- [Plaintiff Counsel] Step 1: Evidence Discovery (Integrative Discovery) ---
...
--- [Defense Counsel] Step 2: Generating Legal Argument ---
```
**Code Mapping:**
- **File:** `framework/main_pipeline.py` calls `mad.run_full_debate(max_rounds=10)`.
- **Class:** `MADOrchestrator` in `framework/mad_orchestrator.py`.
- **Method:** `run_full_debate` iterates through rounds, calling `run_debate_round`.
- **Inner Loop (`run_debate_round`):**
    - **Discovery:** `self.agents[side].propose_query_gap` -> `self.prag.formulate_query` -> `self.agents['judge'].refine_query`.
    - **Retrieval:** `self.prag.retrieve_progressive`.
    - **Argument:** `self.agents[side].generate_argument`.

## 5. Metric Calculation (Reflection & Convergence)
**Log Section:**
```text
--- [Audit] Step 4: Multi-Round Self-Reflection ---
   > [Plaintiff Counsel] Performing self-reflection for Phase 1...
     * Logic: 0.60, Novelty: 0.30, Rebuttal: 0.40
...
--- [Convergence] Score Delta: 1.1600 ---
```
**Code Mapping:**
- **File:** `framework/mad_orchestrator.py` inside `run_debate_round`.
- **Class:** `SelfReflection` (imported from `self_reflection`).
- **Method:** `perform_round_reflection` generates the scores (Logic, Novelty, Rebuttal).
- **Convergence:** Inside `run_full_debate`, the delta is calculated: `delta_score = total_ref_score - self.last_total_reflection_score`.

## 6. Expert Witness Testimony
**Log Section:**
```text
--- [The Court] Step 3: Evaluating Expert Witness Requirements ---
   > [Plaintiff Counsel] Proposed Expert Witness Type: ...
   > [The Court] REQUEST GRANTED. Calling expert witness...
[EXPERT TESTIMONY]: ...
```
**Code Mapping:**
- **File:** `framework/mad_orchestrator.py` inside `run_debate_round`.
- **Method:** Checks `self.agents[side].request_expert`. If granted by `self.agents['judge'].evaluate_expert_request`, it dynamically instantiates a new `DebateAgent` as an expert and calls `generate_argument`.

## 7. Probabilistic Fusion (Final Verdict)
**Log Section:**
```text
11. Generating Final Verdict...
============================================================
FINAL VERDICT GENERATION (PROBABILISTIC FUSION)
============================================================
Negotiation P_neg: 0.1824 (L_neg: -1.5000)
Panel L_panel: -2.3877
Verdict: REFUTE (Confidence: 0.9598)
```
**Code Mapping:**
- **File:** `framework/main_pipeline.py` calls `verdict_generator.generate_verdict()`.
- **Class:** `FinalVerdict` in `framework/final_verdict.py`.
- **Method:** `generate_verdict`.
    - **Logic:**
        1.  **Negotiation Log-Odds ($L_{neg}$):** `math.log(p_neg / (1 - p_neg))` (using `claim_metrics` passed from `main_pipeline.py`).
        2.  **Panel Log-Odds ($L_{panel}$):** Sum of individual judge log-odds `math.log(q_j / (1 - q_j))`.
        3.  **Fusion:** `L_total = L_neg + L_panel`.
        4.  **Probability:** `p_final = 1 / (1 + math.exp(-L_total))`.





# Multi-Agent Debate (MAD) Pipeline Documentation

## 1. System Overview

This pipeline implements a **Multi-Agent Debate (MAD)** framework designed to verify scientific claims through simulated legal proceedings. Unlike standard RAG systems that simply retrieve and summarize, this system instantiates adversarial agents (**"Plaintiff" vs. "Defense"**) who litigate the validity of a claim using medical evidence, supervised by a **"Judicial"** component.

**Target Claim:**  
> "In the most severe COVID-19 cases, women often had a higher level of antibodies than men."

**Final Verdict:**  
**NOT SUPPORTED** (Probability: **2.01%**)

---

## 2. Pipeline Execution Flow

---

### Phase 1: Claim Ingestion & Decomposition

The process begins by ingesting the natural language claim and breaking it down into atomic, testable propositions.

**Component:** `ArgumentMiner`  
**Action:**  
- Decomposes the main claim into 10 distinct premises  
  - Example premises:
    - "Antibody levels are quantitatively measurable."
    - "Comparisons account for confounding variables."
- Each premise is independently debatable and evidence-grounded.

**Purpose:**  
Ensures the debate addresses specific logical facets rather than just the general statement.

---

### Phase 2: Knowledge Retrieval (RAG & FAISS)

Before the debate, the system retrieves the raw scientific corpus.

**Component:** `PubMedRetriever`  
**Action:**
- Loads the FAISS index (`pubmed_faiss.index`)
- Retrieves candidate evidence based on vector similarity search
- Ranks documents using embedding similarity

**Output:**  
A pool of **Initial Retrieved Evidence** (Source IDs).

---

### Phase 3: Evidence Negotiation & Arbitration

Agents do not get access to all retrieved data immediately. They must negotiate for admissibility.

**Component:** `EvidenceNegotiator`

#### Step 1 — Premise-Grounded Retrieval
- Agents propose evidence relevant to specific premises.
- Each proposal must explicitly justify relevance.

#### Step 2 — Arbitration
- The "Court" reviews the proposed evidence.
- Assigns an admissibility score.

**Metric — Admissibility Weight (0.0 – 1.0):**
- Measures how relevant a retrieved paper is to the decomposed premises.
- Higher weight = more likely to be admitted into debate.

**Example:**
- Source `34766646` → Weight: `0.80` → Admitted

**Outcome:**  
A filtered list of **Admissible Evidence** available for the courtroom phase.

---

### Phase 4: Courtroom Proceedings (The Debate)

Core execution loop managed by `MADOrchestrator`.

---

#### A. Integrative Discovery

Agents formulate SQL-like structured queries to "extract" information from admitted evidence.

- **Plaintiff Action:**  
  Requests data showing stability of antibodies in women.
  
- **Defense Action:**  
  Requests data showing peak tiers in men to introduce confounding by severity.

---

#### B. Argument Generation

- **Plaintiff Counsel:**  
  Argues women have more durable antibody responses (citing Source `35458488`).

- **Defense Counsel:**  
  Argues men often have higher initial levels due to disease severity (confounding variable), citing Source `34766646`.

---

#### C. Expert Witness Testimony

**Trigger:**  
When agents reach a technical impasse.

**Process:**
- Judge evaluates request.
- If granted, a temporary `DebateAgent` (Expert persona) is instantiated.

**Log Event Insight:**
The Expert clarifies:
> "Stability" ≠ "Magnitude of antibody level"

This distinction weakens the Plaintiff’s alignment with the original claim.

---

### Phase 5: Self-Reflection & Role Switching

Designed to prevent hallucination and improve robustness.

**Component:** `SelfReflection`

#### Self-Scoring Metrics (0.0 – 1.0)
- Logic Score
- Novelty Score
- Rebuttal Score

#### Role Switching
- Plaintiff ↔ Defense
- Ensures arguments remain consistent under adversarial inversion.

---

### Phase 6: Final Verdict (Probabilistic Fusion)

The debate concludes with a final probabilistic judgment.

**Component:** `FinalVerdict`

#### Method: Probabilistic Argument Fusion

**Inputs:**
1. Negotiation Probability ($P_{neg}$)  
   - Initial likelihood based on raw evidence strength
2. Panel Evaluation  
   - 3 distinct LLMs (DeepSeek, Llama-3, Qwen) independently vote

---

### Mathematical Aggregation

1. Convert probability to log-odds:

$$
L = \ln\left(\frac{p}{1 - p}\right)
$$

2. Aggregate:

$$
L_{final} = L_{neg} + L_{panel}
$$

3. Convert back to probability:

$$
P_{final} = \frac{e^{L_{final}}}{1 + e^{L_{final}}}
$$

---

### Final Result

The system computed:

$$
P_{final} = 0.0201
$$

→ **2.01% probability that the claim is true**  
→ Final Verdict: **REFUTE**

---

## 3. Key Metrics & Definitions

| Metric | Range | Description | Used In |
|---------|--------|-------------|----------|
| **Admissibility Weight** | 0.0 – 1.0 | Relevance of retrieved paper to decomposed premises | Evidence Negotiation |
| **Logic Score** | 0.0 – 1.0 | Coherence and logical consistency of argument | Self-Reflection |
| **Novelty Score** | 0.0 – 1.0 | Measures new information vs repetition | Self-Reflection |
| **Rebuttal Score** | 0.0 – 1.0 | Effectiveness of countering opponent’s points | Self-Reflection |
| **Convergence Delta** | ≥ 0.0 | Difference in reflection scores between rounds. If Δ < 0.05 → early stop | MAD Orchestrator |
| **Log-Odds ($L$)** | $-\infty$ to $+\infty$ | Logarithmic representation of probability odds | Final Verdict |
| **Evidence Strength** | 0 – 10 | Judicial panel rating of raw data quality | Judicial Evaluation |
| **Scientific Reliability** | 0 – 10 | Alignment with scientific consensus | Judicial Evaluation |

---

## Design Philosophy Summary

- Premise-level reasoning instead of claim-level summarization  
- Evidence negotiation before argumentation  
- Adversarial robustness via role switching  
- Probabilistic fusion instead of majority voting  
- Quantitative stopping criteria (Convergence Delta)  
- Judicial supervision and expert escalation  

---

**End of Documentation**
