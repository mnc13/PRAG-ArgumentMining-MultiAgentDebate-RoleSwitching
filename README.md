# Ablation Framework Analysis

## 1. Overview

This ablation analysis implements **six distinct ablation experiments**, each of which removes or modifies exactly one functional subsystem of the main pipeline while keeping all other components intact. The experiments are evaluated on the **Check-COVID** benchmark dataset (`covidCheck_test_data.json`), using the same evaluation infrastructure as the main pipeline.

Each ablation is designed to answer a precise research question:

| Ablation | Research Question |
|---|---|
| Ablation 1 | What is the performance of a stripped-down MAD baseline with no advanced components? |
| Ablation 2 | How much does Role-Switching contribute to system accuracy and robustness? |
| Ablation 3 | What is the contribution of the multi-judge panel over a single judge? |
| Ablation 4 | How much does Progressive RAG (P-RAG) contribute to evidence quality and classification accuracy? |
| Ablation 5 | Does adaptive convergence improve over a fixed-round debate? |
| Ablation 6 | How much does Self-Reflection contribute to argument quality and verdict confidence? |

---

## 2. Relationship to the Main Pipeline

The **main pipeline** (`run_eval_extended.py`, `main_pipeline.py`) implements the complete PRAG-ArgumentMining-MAD-RoleSwitching architecture with eleven sequential processing stages:

```
Stage 1: Input Processing & Claim Normalization
Stage 2: Preprocessing & Claim Extraction
Stage 3: Argument Mining (DeepSeek-R1)
Stage 4: Initial RAG Retrieval (FAISS / PubMed)
Stage 5: Evidence Negotiation & Judicial Arbitration
Stage 6: MAD Initialization (P-RAG + Agents)
Stage 7: Courtroom Proceedings (MAD with Adaptive Convergence)
    - Step 1: P-RAG Evidence Discovery (Integrative: Gap + Reflection)
    - Step 2: Argument Generation (Plaintiff / Defense)
    - Step 3: Expert Witness Testimony
    - Step 4: Multi-Round Self-Reflection
    - Step 5: Critic Agent Evaluation
Stage 8: Role-Switching Consistency Check
Stage 9: Judicial Panel Evaluation (3 Judges: GPT-4.1, Qwen3-235B, DeepSeek-V3.2)
Stage 10: [Internal deliberation]
Stage 11: Final Verdict Generation (Confidence-Weighted)
```

Each ablation experiment selectively disables or replaces one or more of these stages. The key components removed across all ablations are:

| Component | Module | Ablations Removing It |
|---|---|---|
| Evidence Negotiation | `negotiation_engine.py` | Ablation 1 |
| P-RAG (Progressive RAG) | `prag_engine.py` | Ablation 1, 4 |
| Expert Witnesses | `expertise_extractor.py` | Ablation 1 |
| Critic Agent | `mad_system.py::CriticAgent` | Ablation 1 |
| Self-Reflection | `self_reflection.py` | Ablation 1, 6 |
| Adaptive Convergence | `mad_orchestrator.py::run_full_debate` | Ablation 1, 4, 5 |
| Role-Switching | `role_switcher.py` | Ablation 1, 2 |
| 3-Judge Panel | `judge_evaluator.py::JudicialPanel` | Ablation 1, 3 |
| Reflection→P-RAG Feedback | `mad_orchestrator.py` | Ablation 6 |
| Reflection confidence adjustment | `final_verdict.py` | Ablation 6 |

---

## 3. Ablation Experiment Inventory

### Ablation 1: Standard MAD Baseline

**Script:** `run_ablation1_standard_mad.py`  
**Output directory:** `framework/ablation/ablation1/`  
**Run ID prefix:** `ablation1`

#### What Component Is Removed / Modified

This is the most radical ablation: it removes **all** novel architectural contributions simultaneously and reduces the system to a minimal MAD baseline. Specifically removed:

- Evidence Negotiation (Stage 5)
- P-RAG progressive evidence retrieval (Stage 6/7 Step 1)
- Expert Witnesses (Stage 7 Step 3)
- Self-Reflection (Stage 7 Step 4)
- Critic Agent (Stage 7 Step 5)
- Adaptive Convergence (any stopping logic)
- Role-Switching (Stage 8)
- Multi-Judge Panel → replaced by **single judge** (Qwen3-235B-A22B)

The MAD phase is simplified to a **hard-coded 3-round loop** with only two adversarial agents (Plaintiff Counsel: GPT-5-mini; Defense Counsel: DeepSeek-V3.2), with no internal feedback mechanisms.

#### Hypothesis

The hypothesis is that the main pipeline's performance advantage over this baseline directly reflects the aggregate value of all novel components. Ablation 1 serves as the **lower bound** for the system's architecture. Any accuracy gain from the main pipeline above this baseline is attributable to the combined effect of P-RAG, Evidence Negotiation, Self-Reflection, Role-Switching, Expert Witnesses, and the multi-judge panel.

#### Architecture Diagram

```
Claim Text
    │
    ▼
[Stage 2] Claim Extraction (ClaimExtractor)
    │
    ▼
[Stage 3] Argument Mining (DeepSeek-R1)
    │
    ▼
[Stage 4] Initial RAG Retrieval (PubMedRetriever, top_k=5)
    │
    ▼ evidence_pool (5 documents)
[MAD Loop: Fixed 3 Rounds]
┌─────────────────────────────────────────┐
│  Round r (r ∈ {1,2,3})                 │
│  - Plaintiff Counsel → generate_argument│
│  - Defense Counsel   → generate_argument│
│  No P-RAG, no expert, no reflection    │
└─────────────────────────────────────────┘
    │
    ▼
[Single Judge Evaluation] (Qwen3-235B)
  - Evidence Strength: 0–10
  - Argument Validity: 0–10
  - Scientific Reliability: 0–10
  - Verdict: SUPPORTED / NOT SUPPORTED / INCONCLUSIVE
    │
    ▼
[Confidence Calculation]
  - margin_score = 0.8 (fixed single-judge weight)
  - quality_score = ((ev + arg + sci) / 30) * 0.3
  - final_conf = margin_score + quality_score
    │
    ▼
Final Verdict (SUPPORT / REFUTE / INCONCLUSIVE)
```

#### Module-Level Explanation

- **`ProgressiveRAG`** is instantiated as a `dummy_prag` and passed to `DebateAgent` constructors but is **never called** for retrieval. Evidence remains fixed at the initial 5 documents throughout all 3 rounds.
- The single-judge response is parsed from raw JSON; on parse failure, the system defaults to `INCONCLUSIVE` with scores of 5/10.
- Confidence is computed directly in `run_ablation` (not via `FinalVerdict`) using a simplified formula.
- `kappa_pair_mean` is reported as `"N/A"` since there is only one judge.

#### Data Flow Differences

```
Main Pipeline: Claim → Negotiation → P-RAG Evidence Pool → MAD(10 rounds adaptive) → 3-Judge Panel → FinalVerdict
Ablation 1:   Claim → Initial RAG(5 docs) → MAD(3 rounds fixed) → Single Judge → Inline Confidence
```

#### Expected Impact on Performance

Since all enrichment mechanisms are disabled, Ablation 1 is expected to yield **significantly lower accuracy** than the main pipeline. The fixed 3-round structure prevents the debate from naturally converging, and the single judge introduces greater decision variance. The small initial evidence pool (5 documents) limits the epistemic basis of each argument.

---

### Ablation 2: Without Role-Switching

**Script:** `run_ablation2_no_role_switch.py`  
**Output directory:** `framework/ablation/ablation2/`  
**Run ID prefix:** `ablation2`

#### What Component Is Removed / Modified

This ablation retains the **complete system** (Evidence Negotiation, P-RAG, Expert Witnesses, Self-Reflection, Critic, Adaptive Convergence, 3-Judge Panel) but **removes Stage 8 (Role-Switching)** entirely.

Implementation detail: After `MADOrchestrator.run_full_debate()` completes, the code inserts a **synthetic consistency report**:
```python
consistency_report = {
    "is_consistent": True,
    "consistency_score": 5,
    "analysis": "Ablation 2: Role-Switching Disabled"
}
```
This neutral stub prevents downstream `FinalVerdict._calculate_confidence()` from applying any role-switch penalty or bonus. The `FinalVerdict` module detects the special `"analysis"` string and forces `rs_adj = 0.0`.

In the full pipeline, `role_switch_adjustment` can range from `−0.05` (inconsistency penalty) to `+0.10` (consistency bonus).

#### Hypothesis

Role-switching tests **argument robustness**: the system forces each agent to argue the opposite position, and checks whether the final verdict is consistent across both debate configurations. The hypothesis is that removing this check will reduce system **calibration** and **confidence reliability**, even if classification accuracy is not dramatically affected. We also hypothesize that the debate without role-switching may converge prematurely due to unchallenged argumentation patterns.

#### Architecture Diagram

```
... [Stages 2–7: identical to main pipeline] ...
    │
    │ (NO STAGE 8)
    │
    ▼
[Judicial Panel Evaluation] (3 Judges)
    │
    ▼
[FinalVerdict]
  consistency_check = BYPASSED (fixed stub, score=5, adj=0.0)
  reflection_adj = live from self-reflection
```

#### Data Flow Differences

The switched-debate branch exists in the main pipeline as follows:
```
Main Pipeline: MAD→ RoleSwitcher → SwitchedDebate → consistency_report → FinalVerdict
Ablation 2:   MAD→ (stub)       → skip           → fixed_report      → FinalVerdict
```
The `rounds_switched` field in metrics records is `0` for all claims.

#### Expected Impact on Performance

Without role-switching, the system loses its internal cross-validation mechanism. We expect:
- No change in classification accuracy if arguments are already well-formed.
- Potential increase in overconfidence (no negative `rs_adj` from inconsistency detection).
- Loss of `consistency_score` signal in output metadata.

---

### Ablation 3: Single Judge

**Script:** `run_ablation3_single_judge.py`  
**Output directory:** `framework/ablation/ablation3/`  
**Run ID prefix:** `ablation3`

#### What Component Is Removed / Modified

This ablation retains everything from the main pipeline (Evidence Negotiation, P-RAG, Expert Witnesses, Self-Reflection, Critic, Adaptive Convergence, Role-Switching) but **replaces the 3-judge panel** (Stage 9) with a **single judge** (Qwen3-235B-A22B).

The `JudicialPanel` class (which orchestrates three heterogeneous LLM judges) is replaced by a direct call to a single `OpenRouterLLMClient`. The single judge responds to the same 5-stage evaluation prompt used by the panel. The response is wrapped into a `mock_judge_result` dictionary that mimics the multi-judge panel output schema, with `majority_opinion` and `dissenting_opinion` fields populated accordingly.

```python
mock_judge_result = {
    "final_verdict": verdict_data['verdict'],
    "judge_verdicts": [verdict_data],
    "vote_breakdown": {verdict_data['verdict']: 1},
    "majority_opinion": f"Single Judge ({judge_llm.model_name}) - {verdict_data['verdict']}: {reasoning}",
    "dissenting_opinion": None
}
```

This mock is passed directly to `FinalVerdict`, which computes confidence as if the single-judge were a unanimous panel.

#### Hypothesis

The 3-judge panel in the main pipeline aggregates verdicts from three models with different architectures and training characteristics (GPT-4.1, Qwen3-235B, DeepSeek-V3.2). The ensemble is expected to reduce individual judge variance and improve calibration. Ablation 3 tests whether model diversity adds real value. The hypothesis is that the single-judge variant will show **higher variance** in predictions and **lower inter-judge agreement** statistics (trivially: all kappa metrics are undefined for n=1 judge).

#### Architecture Diagram

```
... [Stages 2–8: identical to main pipeline] ...
    │
    ▼
[Single Judge Evaluation] (Qwen3-235B-A22B)
  5-stage prompt:
    Stage 1: Case Reconstruction
    Stage 2: Evidence & Testimony Weighting (0–10)
    Stage 3: Logical Coherence Analysis (0–10)
    Stage 4: Scientific/Technical Consistency (0–10)
    Stage 5: Judicial Verdict (SUPPORTED / NOT SUPPORTED / INCONCLUSIVE)
    │
    ▼
[Mock Judge Result] → same schema as JudicialPanel output
    │
    ▼
[FinalVerdict] (standard, with reflection_adj included)
```

#### Expected Impact on Performance

We expect lower **inter-judge kappa** statistics (which collapse to undefined since κ requires multiple raters). Accuracy may remain competitive if the single judge is high-quality, but calibration will be noisier. The absence of deliberation between judges means edge-case claims may be mislabeled.

---

### Ablation 4: Without P-RAG

**Script:** `run_ablation4_no_prag.py`  
**Output directory:** `framework/ablation/ablation4/`  
**Run ID prefix:** `ablation4`

#### What Component Is Removed / Modified

This ablation retains Evidence Negotiation, Self-Reflection, Critic, Role-Switching, Expert Witnesses, and the 3-Judge Panel, but **removes P-RAG** from the MAD loop. The debate is constrained to **exactly 3 rounds** (no adaptive convergence).

Two custom subclasses are introduced:

1. **`DummyPRAG(ProgressiveRAG)`**: A stub class with `disabled = True`. Its `retrieve_progressive()` always returns `[]`, and `formulate_query()` is a no-op passthrough. This allows the orchestrator to detect `prag.disabled` and skip the discovery block cleanly.

2. **`FixedRoundsOrchestrator(MADOrchestrator)`**: Overrides `run_debate_round()` to skip the evidence discovery block when PRAG is disabled (prints `"[P-RAG DISABLED] Evidence Discovery skipped"`). Overrides `run_full_debate()` to execute exactly `max_rounds=3` iterations with **no adaptive stopping logic**.

3. **`FixedRoundsRoleSwitcher(RoleSwitcher)`**: A compatible subclass that ensures the switched debate also runs for exactly 3 rounds.

#### Hypothesis

Progressive RAG is the engine that dynamically expands the evidence pool during debate. Each round, each agent proposes a targeted discovery query (gap-driven + reflection-driven), which is refined by the judicial agent and executed against the FAISS/PubMed index. Removing P-RAG means the evidence pool is **frozen** at the initial negotiated set throughout all rounds.

The hypothesis is that P-RAG significantly improves **evidence recall** and **factual grounding**, and its removal will result in lower classification accuracy, particularly for ambiguous claims where initial evidence is insufficient.

#### Architecture Diagram

```
... [Stages 2–5: identical to main pipeline] ...
    │
    ▼ final_evidence_set (from Negotiation)
[FixedRoundsOrchestrator]
┌──────────────────────────────────────────────────────────────┐
│  Rounds 1–3 (fixed)                                          │
│  Step 1: [P-RAG DISABLED] — Evidence Discovery SKIPPED       │
│  Step 2: Argument Generation (Plaintiff / Defense)           │
│  Step 3: Expert Witness Testimony                            │
│  Step 4: Self-Reflection (runs but no discovery_need output) │
│  Step 5: Critic Evaluation                                   │
│  NO adaptive stopping check                                  │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
[FixedRoundsRoleSwitcher] → 3 switched rounds
    │
    ▼
[3-Judge Panel] → FinalVerdict
```

#### Expected Impact on Performance

Without P-RAG, each agent must argue solely from the pre-negotiated evidence set, which is assembled before the debate begins. This may penalise complex claims requiring diverse evidence perspectives across multiple argument turns. Expected: lower accuracy and evidence_count metrics vs. the main pipeline.

---

### Ablation 5: Fixed Rounds

**Script:** `run_ablation5_fixed_rounds.py`  
**Output directory:** `framework/ablation/ablation5/`  
**Run ID prefix:** `ablation5`

#### What Component Is Removed / Modified

This ablation retains the **full system** including P-RAG, Evidence Negotiation, Expert Witnesses, Self-Reflection, Critic, Role-Switching, and the 3-Judge Panel. The **only change** is that **adaptive convergence stopping logic is disabled**.

Two custom subclasses are introduced:

1. **`FixedRoundsOrchestrator(MADOrchestrator)`**: Overrides only `run_full_debate()`. The inner `run_debate_round()` method is **not** overridden, so all processing (P-RAG, argument generation, expert witnesses, self-reflection, critic evaluation) runs identically to the main pipeline **per round**. Only the convergence check is removed: the `delta_score < 0.05` reflection plateau test, critic resolution test, novelty stabilization test, and judicial signal test are all bypassed. The debate always runs exactly `max_rounds=3` rounds.

2. **`FixedRoundsRoleSwitcher(RoleSwitcher)`**: Ensures the switched debate also runs exactly 3 rounds.

#### Hypothesis

In the main pipeline, `MADOrchestrator.run_full_debate()` implements four adaptive stopping conditions:
- Reflection plateau (delta_score < 0.05)
- Critic resolution signal
- Evidence novelty stabilization (avg_novelty < 0.1 for two consecutive rounds)
- Judicial completion signal

These conditions allow the debate to terminate early when further argumentation adds no value. The hypothesis is that forcing exactly 3 rounds is **suboptimal**: some claims converge after round 1 or 2 and are harmed by unnecessary additional computation, while other claims that need more rounds are cut off at 3.

#### Architecture Diagram

```
... [Stages 2–5: identical to main pipeline] ...
    │
    ▼
[FixedRoundsOrchestrator]
┌────────────────────────────────────────────────────────────────┐
│  Rounds 1–3 (fixed, regardless of convergence signals)         │
│  Step 1: P-RAG Evidence Discovery (active, gap+reflection)     │
│  Step 2: Argument Generation                                   │
│  Step 3: Expert Witness                                        │
│  Step 4: Self-Reflection (active, scores tracked)              │
│  Step 5: Critic (active, signals evaluated but NOT acted upon) │
│  NO break conditions evaluated                                 │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
[FixedRoundsRoleSwitcher] → 3 fixed switched rounds
    │
    ▼
[3-Judge Panel] → FinalVerdict
```

#### Expected Impact on Performance

Without adaptive stopping, the debate may:
1. Over-run for claims that converge early → wasted compute, potentially degraded accuracy due to argument dilution.
2. Under-run for complex claims that need more than 3 rounds.

The efficiency metrics (avg_tok, avg_round) will be deterministically 3.0 rounds per claim, contrasting with the main pipeline's variable round count.

---

### Ablation 6: Without Self-Reflection

**Script:** `run_ablation6_no_self_reflection.py`  
**Output directory:** `framework/ablation/ablation6/`  
**Run ID prefix:** `ablation6`

#### What Component Is Removed / Modified

This ablation retains Evidence Negotiation, P-RAG, Expert Witnesses, Critic, Role-Switching, Adaptive Convergence (partially), and the 3-Judge Panel. It **removes multi-round self-reflection** (Stage 7 Step 4) from the debate loop and its downstream effects:

1. **Reflection step skipped**: `NoReflectionOrchestrator.run_debate_round()` prints `"[Ablation] Step 4: Multi-Round Self-Reflection SKIPPED"` and does not call `SelfReflection.perform_round_reflection()`.

2. **Reflection→P-RAG feedback removed**: In the main pipeline, `reflection_discovery_needs` (extracted from reflection output) is concatenated with the gap proposal to form the discovery prompt. In this ablation, the discovery prompt is the **gap proposal only** — the reflection component of the integrative query is absent.

3. **Reflection plateau stopping removed**: The `delta_score < 0.05` convergence check is removed from `run_full_debate()`. The remaining convergence checks (critic resolution, novelty stabilization, judicial signal) are retained.

4. **Reflection confidence adjustment neutralized**: A custom `AblationFinalVerdict(FinalVerdict)` subclass overrides `_calculate_confidence()` to **explicitly set `reflection_adj = 0.0`** (with no `adjustments += reflection_adj` line). It also overrides `generate_verdict()` to stamp `self_reflection_adjustment: 0.0` in the result metadata.

5. **Empty reflection history**: `reflection_history=[]` is passed to `JudicialPanel.evaluate_debate()`, and `AblationFinalVerdict` is instantiated with `{}` as the reflection result.

#### Hypothesis

Self-reflection in the main pipeline serves three roles: (a) it feeds `discovery_need` back into P-RAG to steer evidence retrieval toward argument weaknesses; (b) its `total_score` delta drives adaptive convergence; (c) its `confidence_adjustment` modifies the final verdict confidence. Removing self-reflection tests all three simultaneously.

The hypothesis is that self-reflection primarily benefits **evidence quality** via its feedback to P-RAG. Without this feedback, P-RAG queries are less targeted, yielding lower-novelty evidence and potentially degrading classification for edge-case claims.

#### Architecture Diagram

```
... [Stages 2–5: identical to main pipeline] ...
    │
    ▼
[NoReflectionOrchestrator]
┌───────────────────────────────────────────────────────────────────────┐
│  Rounds 1–N (adaptive, max=10)                                        │
│  Step 1: P-RAG Evidence Discovery (gap_proposal ONLY, no reflection)  │
│  Step 2: Argument Generation                                           │
│  Step 3: Expert Witness Testimony                                      │
│  Step 4: [SKIPPED] Self-Reflection                                     │
│  Step 5: Critic Evaluation (still active)                              │
│  Convergence: Critic signal, Novelty stabilization, Judicial signal    │
│  (Reflection plateau check: REMOVED)                                   │
└───────────────────────────────────────────────────────────────────────┘
    │
    ▼
[NoReflectionRoleSwitcher] → switched debate
    │
    ▼
[3-Judge Panel]
    │
    ▼
[AblationFinalVerdict]
  reflection_adj = 0.0 (hard-coded)
```

#### Expected Impact on Performance

Removing reflection while keeping all other components may have a counter-intuitive effect: P-RAG may run more aggressively (more rounds before convergence) since the reflection plateau stopping condition is removed, potentially retrieving more diverse evidence. This could explain the paradoxical finding in the experimental results (see Section 3 of `ABLATION_METRICS_AUDIT.md`).

---

## 4. Comparative Framework Summary

### Structural Differences

| Feature | Main Pipeline | Abl. 1 | Abl. 2 | Abl. 3 | Abl. 4 | Abl. 5 | Abl. 6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Evidence Negotiation | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| P-RAG (progressive retrieval) | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| Expert Witnesses | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Self-Reflection | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Reflection → P-RAG Feedback | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ |
| Critic Agent | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Adaptive Convergence | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | Partial |
| Role-Switching | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| 3-Judge Panel | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Reflection Confidence Adj. | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Max Debate Rounds | 10 | 3 | 10 | 10 | 3 | 3 | 10 |

### Algorithmic Differences

| Component | Main Pipeline Behavior | Ablation Modification |
|---|---|---|
| Evidence Discovery | `gap_proposal + reflection_gap` concatenated | Abl. 6: `gap_proposal` only |
| Stopping Logic | 4 criteria: plateau, critic, novelty, judicial | Abl. 4/5: none; Abl. 6: 3 criteria |
| Confidence Formula | `margin + quality + rs_adj + reflection_adj` | Abl. 1: simplified inline; Abl. 2: `rs_adj=0`; Abl. 6: `reflection_adj=0` |
| Judge Aggregation | Majority vote over 3 heterogeneous LLMs | Abl. 3: single LLM decision |
| Role-Switch Score | Live computation via `RoleSwitcher.check_consistency()` | Abl. 2: stub `{score: 5, adj: 0.0}` |

---

## 5. Experimental Philosophy

The six ablations are designed around the principle of **controlled single-factor removal**: each experiment modifies exactly one independent variable relative to the system configuration of its predecessor or the full pipeline. This ensures that observed metric differences are causally attributable to the removed component.

The choice of ablations reflects the three core theoretical claims of the paper:

1. **Evidence Quality Hypothesis** (tested by Ablations 1, 4): P-RAG and Evidence Negotiation improve evidence grounding beyond static retrieval. Ablation 1 tests the aggregate effect; Ablation 4 isolates P-RAG specifically.

2. **Deliberation Robustness Hypothesis** (tested by Ablations 2, 3, 5, 6): The multi-agent deliberation mechanism — including role-switching, multi-judge consensus, adaptive convergence, and self-reflection — produces more robust verdicts. Each ablation isolates one aspect:
   - Ablation 2: role-switching as argument robustness check
   - Ablation 3: judge diversity as decision robustness
   - Ablation 5: adaptive convergence as computational efficiency mechanism
   - Ablation 6: self-reflection as argument quality feedback loop

3. **Component Interaction Hypothesis**: Some components are hypothesized to interact (e.g., P-RAG and self-reflection share a feedback channel). The ablations reveal these interactions through unexpected performance patterns.


