# Technical Pipeline Architecture Report: Check-COVID

This document provides a technical decomposition of the 12-step Check-COVID pipeline. It details the inputs, processes, actors, and outputs for each stage of the simulation.

## 1. Data Loading
- **File**: `data_loader.py`
- **Who**: `DataLoader` class.
- **Input**: Raw directory path containing JSONL files (e.g., `climate-fever.jsonl`).
- **Process**: Reads the file line-by-line, parses JSON objects, and maps them to a structured `Claim` dataclass.
- **Output**: A list of `Claim` objects containing `id`, `text`, and `metadata`.

## 2. Preprocessing & Extraction
- **File**: `preprocessing.py`
- **Who**: `ClaimExtractor` powered by `Gemini-1.5-Flash`.
- **Input**: Raw claim text.
- **Process**: Normalizes text, removes noise, and extracts core semantic meaning. It also preserves original metadata.
- **Output**: A cleaned `Claim` object ready for downstream LLM processing.

## 3. Argument Mining
- **File**: `agent_workflow.py`
- **Who**: `ArgumentMiner` powered by `meta-llama/llama-4-maverick-17b`.
- **Input**: `Claim` object.
- **Process**: Performs semantic decomposition to identify underlying premises, required evidence types, and implicit assumptions. This breaks a complex claim into verifiable "atomic" components.
- **Output**: An `Argument` object containing a list of `premises`.

## 4. Initial RAG Retrieval
- **File**: `rag_engine.py` (inherits from `PubMedRetriever`)
- **Who**: `VectorRetriever` using FAISS and `sentence-transformers/all-MiniLM-L6-v2`.
- **Input**: Cleaned claim text.
- **Process**: Converts claim text into a vector embedding and performs a similarity search against a local index of 33 million PubMed abstracts.
- **Output**: Top 5 `Evidence` objects, each containing a snippet of the abstract and a `source_id`.

## 5. Evidence-First Debate (Negotiation)
- **File**: `agent_workflow.py`
- **Who**: `EvidenceFirstDebateAgent`.
- **Input**: List of 5 `Evidence` objects.
- **Process**: Two virtual personas analyze the evidence pool. They negotiate which sources are most relevant/strong and discard weak or irrelevant ones. Why? This ensures the MAD system starts with high-quality, agreed-upon data.
- **Output**: A `shared_evidence` list (usually 2-3 high-relevance sources).

## 6. Scientific Debate Simulation Initialization
- **File**: `mad_orchestrator.py` & `personas.py`
- **Who**: `MADOrchestrator`.
- **Input**: `Claim`, `shared_evidence`, and `AGENT_SLOTS` configuration.
- **Process**: Initializes 4 distinct agents:
  - **Proponent**: Argues in favor of the claim using technical reasoning.
  - **Opponent**: Identifies methodological flaws and counter-evidence.
  - **Judge (Moderator)**: Oversees flow and evaluates expert requests.
  - **Critic (Analyst)**: Performs per-round consistency checks.
- **Output**: A live MAD environment with initialized LLM clients.

## 7. Multi-Agent Debate Rounds & Technical Testimony
- **File**: `mad_orchestrator.py` & `mad_system.py`
- **Who**: `DebateAgent` class and `expertise_extractor.py`.
- **Input**: `debate_transcript` and `evidence_pool`.
- **Process**: 
  - Agents generate arguments citing scientific IDs.
  - **Expert Summoning**: If an agent feels a specific field (e.g., "Clinical Virologist") is missing, they request an expert. The **Judge** evaluates the request. If granted, a dynamic persona is built in `expertise_extractor.py`, and a new testimony is generated.
- **Output**: Multi-turn transcript including expert reports and evidence-backed claims.

## 8. Analyzing Proceedings & Consistency
- **File**: `role_switcher.py`
- **Who**: `RoleSwitcher`.
- **Input**: Full debate transcript.
- **Process**: The system **swaps** the Proponent and Opponent agents. The same agents must now argue for the opposite side. This is a "robustness test" to see if they contradict their previous logic.
- **Output**: Two transcripts (Original and Switched).

## 9. Consistency Analysis
- **File**: `role_switcher.py`
- **Who**: Consistency Analyst (Maverick-17b).
- **Process**: Compares Agent A's logic as a Proponent vs. Agent A's logic as an Opponent.
- **Output**: A consistency report with a score (0.0-1.0) and a qualitative analysis.

## 10. Judge Evaluation & Scoring
- **File**: `judge_evaluator.py`
- **Who**: 3 specialized judges (Logic, Evidence, Accuracy).
- **Input**: Final transcript.
- **Calculation Logic**:
  - Each judge scores 4 categories (0-10): Evidence Quality, Logic, Persuasiveness, Scientific Accuracy.
  - **Aggregate Score**: Sum of all scores from all 3 judges (Max 120 per side).
  - **Confidence**: `abs(Total_Proponent - Total_Opponent) / Total_Scores`. High margins indicate high confidence; close scores indicate low confidence.
- **Output**: `judge_result` JSON object.

## 11. Self-Reflection Round
- **File**: `self_reflection.py`
- **Who**: The "Winning" side agent.
- **Input**: Their own arguments + Critic's feedback.
- **Process**: The winning agent must perform a "self-critique," identifying weaknesses in their own case. This is an LLM-debiasing step.
- **Output**: A `confidence_adjustment` (e.g., -0.20 if they find significant flaws in their reasoning).

## 12. Final Verdict Generation
- **File**: `final_verdict.py`
- **Who**: `FinalVerdict` aggregator.
- **Calculation**: 
  - **Base Confidence**: Margin from Judge evaluation.
  - **Consistency Penalty/Bonus**: +/- 0.10 based on the consistency report.
  - **Reflection Adjustment**: Added/Subtracted based on the winner's self-critique.
- **Output**: Final `SUPPORT/REFUTE` verdict with a normalized confidence score (0.0-1.0).

## Persona LLM Diversity & Configurations

To ensure a robust and unbiased debate, each role in the Check-COVID pipeline is assigned a distinct LLM persona with specific behavioral parameters (Temperature, System Prompts).

| Role | Agent Title | LLM Model | Provider | Temp | Primary Focus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proponent** | Scientific Proponent | `gpt-4o-mini` | OpenAI | 0.5 | Verification & Supporting Logic |
| **Opponent** | Scientific Opponent | `llama-4-maverick-17b` | Groq | 0.5 | Falsification & Counter-Evidence |
| **Moderator** | Scientific Moderator | `llama-3.1-70b-versatile`| Groq | 0.2 | Rule Enforcement & Completion Check|
| **Analyst** | Scientific Analyst | `mixtral-8x7b-32768` | Groq | 0.4 | Intra-round Consistency Analysis |
| **Expert** | Clinical Specialist | `llama-4-maverick-17b` | Groq | 0.7 | Deep Technical Domain Knowledge |

### Judicial Model Matrix
Final evaluation is performed by an independent panel to prevent bias from the simulation agents.

1.  **Logic & Reasoning Judge**: Powered by `meta-llama/llama-4-maverick-17b` (Temp: 0.3). Focuses on the structural integrity of the arguments.
2.  **Evidence Quality Judge**: Powered by `llama-3.1-70b-versatile` (Temp: 0.3). Focuses on the validity and relevance of the 33M PubMed citations.
3.  **Scientific Accuracy Judge**: Powered by `gpt-4o-mini` (Temp: 0.3). Cross-references results against known clinical benchmarks.

### Essential Parameters
- **System Prompts**: Each agent receives a unique persona instruction that defines their "job title" and "behavioral constraints" (e.g., strictly clinical language).
- **Temperature Control**: Low temperatures (0.2-0.3) are used for "Rational" roles (Judge, Analyst) to minimize hallucinations. Higher temperatures (0.5-0.7) are used for "Synthesizing" roles (Proponent, Expert) to allow for more complex semantic connections.
- **Job-Title Enforcement**: Agents are forbidden from using personal names and must identify only by their technical role (e.g., "Clinical Epidemiologist").

