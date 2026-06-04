"""
Ablation 1: Standard MAD Baseline — HealthVer Dataset
- Removes Evidence Negotiation
- Removes P-RAG (Initial RAG only)
- Removes Expert Witnesses, Critic, Self-Reflection, Adaptive Convergence, Role-Switching
- 2 Agents: openai/gpt-5-mini (Proponent) vs deepseek/deepseek-v3.2 (Opponent)
  ** ALL 3 models served via OpenRouter (no direct OpenAI API) **
- Fixed 3 rounds always
- Single Judge Evaluation (qwen/qwen3-235b-a22b-2507 via OpenRouter)
- Dataset: HealthVer (healthver_sample_100.jsonl)
- Output:  framework/ablation/HealthVerStd Mad/
"""

import os

# Load .env keys FIRST — before any API client touches os.getenv()
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import sys
import json
import time
import re
import argparse
from copy import deepcopy

# ---------------------------------------------------------------------------
# Path setup — must happen before importing framework modules that read
# module-level globals (logging_extension paths etc.)
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

ABLATION_NAME   = "HealthVerStd Mad"
ABLATION_BASE_DIR    = os.path.join(script_dir, "ablation", ABLATION_NAME)
ABLATION_LOGS_DIR    = os.path.join(ABLATION_BASE_DIR, "logs")
ABLATION_OUTCOMES_DIR = os.path.join(ABLATION_BASE_DIR, "outcomes")
os.makedirs(ABLATION_LOGS_DIR,    exist_ok=True)
os.makedirs(ABLATION_OUTCOMES_DIR, exist_ok=True)

# Patch logging_extension paths BEFORE importing from it
import logging_extension
logging_extension.ARTIFACTS_DIR  = os.path.join(ABLATION_OUTCOMES_DIR, "metrics")
logging_extension.CLAIMS_FILE    = os.path.join(logging_extension.ARTIFACTS_DIR, "claims_added.jsonl")
logging_extension.RUNS_FILE      = os.path.join(logging_extension.ARTIFACTS_DIR, "runs_added.jsonl")
logging_extension.STABILITY_FILE = os.path.join(logging_extension.ARTIFACTS_DIR, "stability_added.jsonl")
logging_extension.REPORT_FILE    = os.path.join(logging_extension.ARTIFACTS_DIR, "run_reports_added.md")
logging_extension.ALL_OUTPUT_JSONS_DIR = ABLATION_OUTCOMES_DIR
os.makedirs(logging_extension.ARTIFACTS_DIR, exist_ok=True)

from filelock import FileLock
from logging_extension import ExtensionState, print_extra_claim_metrics

# Safe concurrent writes
orig_append_jsonl = logging_extension.append_jsonl
def locked_append_jsonl(filepath: str, data: dict):
    with FileLock(filepath + ".lock"):
        orig_append_jsonl(filepath, data)
logging_extension.append_jsonl = locked_append_jsonl

# Framework imports
from preprocessing import ClaimExtractor
from rag_engine import PubMedRetriever
from agent_workflow import ArgumentMiner
from mad_system import DebateAgent
from prag_engine import ProgressiveRAG
from openrouter_client import OpenRouterLLMClient
from models import Claim

# Monkey-patch token / retrieval counters
from run_eval_extended import apply_monkey_patches

# ---------------------------------------------------------------------------
# HealthVer data loader  (simple — data already has id, claim, label)
# ---------------------------------------------------------------------------

def load_healthver(jsonl_path: str):
    """Load HealthVer JSONL and return list of Claim objects."""
    claims = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            c = Claim(
                id=str(entry["id"]),
                text=entry["claim"],
                metadata={"label": entry.get("label", "UNKNOWN")}
            )
            claims.append(c)
    return claims

# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(args):
    outcome_dir = ABLATION_OUTCOMES_DIR
    logs_dir    = ABLATION_LOGS_DIR

    # Resume support — skip already-processed claim IDs
    processed_claims_path = os.path.join(outcome_dir, "processed_claims.txt")
    processed_ids = set()
    if os.path.exists(processed_claims_path):
        with open(processed_claims_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    processed_ids.add(line.strip())

    # Load dataset
    healthver_path = os.path.join(
        script_dir, "..", "Other-datasets", "healthver_sample_100.jsonl"
    )
    all_claims = load_healthver(healthver_path)
    if not all_claims:
        print("[ERROR] No claims loaded from HealthVer dataset. Aborting.")
        return

    # Slice with --offset / --limit
    start_idx = args.offset
    end_idx   = args.offset + args.limit if args.limit is not None else len(all_claims)
    all_claims = all_claims[start_idx:end_idx]
    print(f"[INFO] Processing {len(all_claims)} claims (offset={args.offset}, limit={args.limit})")

    # ---------------------------------------------------------------------------
    # Shared infrastructure (initialised once, reused across all claims)
    # ---------------------------------------------------------------------------

    # PubMed RAG (initial retrieval only — no P-RAG rounds in Ablation 1)
    index_path   = os.path.join(script_dir, "pubmed_faiss.index")
    meta_path    = os.path.join(script_dir, "pubmed_meta.jsonl")
    offsets_path = os.path.join(script_dir, "pubmed_meta_offsets.npy")
    retriever    = PubMedRetriever(index_path, meta_path, offsets_path)

    # Argument miner LLM (deepseek-r1 via OpenRouter)
    miner_llm = OpenRouterLLMClient(model_name="deepseek/deepseek-r1")
    miner     = ArgumentMiner(miner_llm)

    # Dummy P-RAG (constructor requirement for DebateAgent; not used in Ablation 1)
    dummy_prag = ProgressiveRAG(retriever, miner_llm)

    # ---------------------------------------------------------------------------
    # Agent configs — ALL via OpenRouter
    # ---------------------------------------------------------------------------
    proponent_config = {
        "name":        "Plaintiff Counsel",
        "role":        "Plaintiff Counsel",
        "llm_provider": "openrouter",
        "llm_model":   "openai/gpt-5-mini",        # OpenRouter-hosted GPT-5-mini
        "temperature": 0.5,
        "expertise":   ["legal advocacy", "evidence presentation", "clinical analysis"],
        "system_prompt": (
            "You are the Plaintiff Counsel in a legal proceeding. "
            "Present arguments supporting the claim using evidence. "
            "Maintain a professional legal advocacy tone. "
            "DIRECT OUTPUT ONLY: Do not reveal internal thoughts or scratchpad."
        )
    }

    opponent_config = {
        "name":        "Defense Counsel",
        "role":        "Defense Counsel",
        "llm_provider": "openrouter",
        "llm_model":   "deepseek/deepseek-v3.2",
        "temperature": 0.5,
        "expertise":   ["legal defense", "critical analysis"],
        "system_prompt": (
            "You are the Defense Counsel in a legal proceeding. "
            "Challenge the claim and the plaintiff's interpretation of evidence. "
            "Maintain a professional legal defense tone. "
            "DIRECT OUTPUT ONLY: Do not reveal internal thoughts or scratchpad."
        )
    }

    judge_llm = OpenRouterLLMClient(
        model_name="qwen/qwen3-235b-a22b-2507",
        system_prompt="You are an independent appellate judge presiding over a legal proceeding.",
        temperature=0.3
    )

    proponent = DebateAgent(proponent_config, "proponent", dummy_prag)
    opponent  = DebateAgent(opponent_config,  "opponent",  dummy_prag)

    # Initialise run state and monkey-patch token/retrieval counters
    ExtensionState.generate_run_id("healthver_ablation1_std_mad")
    apply_monkey_patches()

    print(f"\n=== HealthVerStd Mad — Ablation 1: Standard MAD ===")
    print(f"    Proponent : {proponent_config['llm_model']} (OpenRouter)")
    print(f"    Opponent  : {opponent_config['llm_model']} (OpenRouter)")
    print(f"    Judge     : {judge_llm.model_name} (OpenRouter)")
    print(f"    Output    : {ABLATION_BASE_DIR}\n")

    # ---------------------------------------------------------------------------
    # Claim loop
    # ---------------------------------------------------------------------------
    for input_claim in all_claims:
        claim_id_str = str(input_claim.id)

        if not args.force and claim_id_str in processed_ids:
            print(f"[SKIP] Claim {claim_id_str} already processed.")
            continue

        ExtensionState.reset_claim_state()

        # Per-claim dual logger (console + file)
        log_filename = os.path.join(logs_dir, f"execution_log_{claim_id_str}_0.txt")

        class DualLogger:
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, "w", encoding="utf-8")
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
            def flush(self):
                self.terminal.flush()
                self.log.flush()
            def close(self):
                self.log.close()

        dual_logger = DualLogger(log_filename)
        sys.stdout = dual_logger

        try:
            print(f"=== ABLATION 1 (HealthVer): STANDARD MAD — Claim {claim_id_str} ===")
            print(f"Claim text: {input_claim.text}\n")

            # ------------------------------------------------------------------
            # Step 2: Preprocessing & Extraction
            # ------------------------------------------------------------------
            print("2. Preprocessing & Extraction...")
            extractor = ClaimExtractor()
            extracted_claim = extractor.extract_claim(input_claim.text)
            extracted_claim.id       = input_claim.id
            extracted_claim.metadata = input_claim.metadata
            print(f"   Extracted: {extracted_claim.text}\n")

            # ------------------------------------------------------------------
            # Step 3: Argument Mining
            # ------------------------------------------------------------------
            print("3. Argument Mining...")
            argument = miner.mine_arguments(extracted_claim)
            print("   [DECOMPOSED PREMISES/ARGUMENTS]:")
            for i, p in enumerate(argument.premises):
                print(f"   - {i+1}. {p}")
            print("")

            # ------------------------------------------------------------------
            # Step 4: Initial RAG Retrieval
            # ------------------------------------------------------------------
            print("4. Initial RAG Retrieval...")
            print("   [DEBUG] Checking paths:")
            print(f"   Index:   {index_path}   (Exists: {os.path.exists(index_path)})")
            print(f"   Meta:    {meta_path}   (Exists: {os.path.exists(meta_path)})")
            print(f"   Offsets: {offsets_path} (Exists: {os.path.exists(offsets_path)})")

            retrieved_evidence = retriever.retrieve(extracted_claim.text, top_k=5)
            evidence_pool = retrieved_evidence
            print("   [INITIAL RETRIEVED EVIDENCE]:")
            for i, e in enumerate(evidence_pool):
                print(f"   - Evidence {i+1} (ID: {e.source_id}): {e.text}")
            print("")

            # ------------------------------------------------------------------
            # Step 7: MAD Loop (Fixed 3 rounds)
            # ------------------------------------------------------------------
            debate_transcript = {
                "claim":    extracted_claim.text,
                "claim_id": getattr(extracted_claim, "id", "Unknown"),
                "dataset":  "HealthVer",
                "agents":   {
                    "proponent": proponent.job_title,
                    "opponent":  opponent.job_title
                },
                "rounds": []
            }

            flat_transcript = []

            print("7. Presiding Over Courtroom Proceedings...\n")

            for round_num in range(1, 4):
                print("=" * 60)
                print(f"PROCEEDINGS PHASE {round_num}")
                print("=" * 60 + "\n")

                round_data = {"round_number": round_num, "arguments": []}

                # Proponent
                print(f"--- [Plaintiff Counsel] Step 2: Generating Legal Argument ---")
                p_arg   = proponent.generate_argument(extracted_claim, evidence_pool, flat_transcript)
                entry_p = {"agent": proponent.name, "role": "proponent", "text": p_arg}
                round_data["arguments"].append(entry_p)
                flat_transcript.append(entry_p)
                print(f"\n{p_arg}\n")

                # Opponent
                print(f"--- [Defense Counsel] Step 2: Generating Legal Argument ---")
                o_arg   = opponent.generate_argument(extracted_claim, evidence_pool, flat_transcript)
                entry_o = {"agent": opponent.name, "role": "opponent", "text": o_arg}
                round_data["arguments"].append(entry_o)
                flat_transcript.append(entry_o)
                print(f"\n{o_arg}\n")

                debate_transcript["rounds"].append(round_data)

            # ------------------------------------------------------------------
            # Step 9: Judicial Panel Evaluation
            # ------------------------------------------------------------------
            print("9. Judicial Panel Evaluation...\n")
            print("=" * 60)
            print("JUDICIAL PANEL EVALUATION")
            print("=" * 60 + "\n")

            print(f"Judge 1 ({judge_llm.model_name}) deliberating...")

            p_args_text = [
                a["text"]
                for r in debate_transcript["rounds"]
                for a in r["arguments"]
                if a["role"] == "proponent"
            ]
            o_args_text = [
                a["text"]
                for r in debate_transcript["rounds"]
                for a in r["arguments"]
                if a["role"] == "opponent"
            ]
            ev_summary = "\n".join([
                f"{i+1}. Source {e.source_id}: "
                f"{getattr(e, 'metadata', {}).get('title', 'N/A')}"
                for i, e in enumerate(evidence_pool)
            ])

            judge_prompt = f"""You are an appellate judge evaluating the following proceedings for medical fact-checking.

PROCEEDINGS RECORD:
CLAIM: {extracted_claim.text}

PLAINTIFF COUNSEL'S ARGUMENTS:
{chr(10).join(p_args_text)}

DEFENSE COUNSEL'S ARGUMENTS:
{chr(10).join(o_args_text)}

ADMITTED EVIDENCE:
{ev_summary}

Perform the following evaluation stages:
STAGE 1 - CASE RECONSTRUCTION
STAGE 2 - EVIDENCE & TESTIMONY WEIGHTING (Score 0-10)
STAGE 3 - LOGICAL COHERENCE ANALYSIS (Score 0-10)
STAGE 4 - SCIENTIFIC/TECHNICAL CONSISTENCY (Score 0-10)
STAGE 5 - JUDICIAL VERDICT
Determine: SUPPORTED, NOT SUPPORTED, or INCONCLUSIVE

Respond ONLY in valid JSON format:
{{
  "claim_summary": "Brief summary",
  "evidence_strength": <score 0-10>,
  "argument_validity": <score 0-10>,
  "scientific_reliability": <score 0-10>,
  "verdict": "SUPPORTED" or "NOT SUPPORTED" or "INCONCLUSIVE",
  "reasoning": "Detailed justification"
}}"""

            response = judge_llm.generate(judge_prompt)

            try:
                verdict_data = json.loads(re.search(r'\{[\s\S]*\}', response).group())
                for s in ["evidence_strength", "argument_validity", "scientific_reliability"]:
                    verdict_data[s] = max(0, min(10, int(verdict_data.get(s, 5))))
                if verdict_data.get("verdict") not in ["SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE"]:
                    verdict_data["verdict"] = "INCONCLUSIVE"
            except Exception:
                verdict_data = {
                    "claim_summary":        f"Evaluation of: {extracted_claim.text}",
                    "evidence_strength":    5,
                    "argument_validity":    5,
                    "scientific_reliability": 5,
                    "verdict":   "INCONCLUSIVE",
                    "reasoning": "JSON parse failure — defaulting to INCONCLUSIVE."
                }

            print(f"  Verdict              : {verdict_data['verdict']}")
            print(f"  Evidence Strength    : {verdict_data['evidence_strength']}/10")
            print(f"  Argument Validity    : {verdict_data['argument_validity']}/10")
            print(f"  Scientific Reliability: {verdict_data['scientific_reliability']}/10\n")

            verdict_data["judge_name"] = "Single Judge (Qwen3)"
            verdict_data["model"]      = judge_llm.model_name

            judge_eval_result = {
                "claim":         extracted_claim.text,
                "judge_verdicts": [verdict_data]
            }

            # ------------------------------------------------------------------
            # Persist intermediate artefacts (append-only JSONL)
            # ------------------------------------------------------------------
            logging_extension.append_framework_json(
                "debate_transcript.jsonl", extracted_claim.id, debate_transcript
            )
            logging_extension.append_framework_json(
                "judge_evaluation.jsonl", extracted_claim.id, judge_eval_result
            )

            # ------------------------------------------------------------------
            # Step 11: Final Verdict
            # ------------------------------------------------------------------
            print("11. Generating Final Verdict...\n")
            print("=" * 60)
            print("FINAL VERDICT GENERATION")
            print("=" * 60 + "\n")

            # Confidence from judge scores (same formula as original ablation1)
            margin_score  = 0.8
            quality_score = (
                (
                    verdict_data["evidence_strength"]
                    + verdict_data["argument_validity"]
                    + verdict_data["scientific_reliability"]
                ) / 30.0
            ) * 0.3
            final_conf = max(0.0, min(1.0, margin_score + quality_score))

            # Map judge verdict → canonical labels
            raw_verdict = verdict_data["verdict"]
            if raw_verdict == "NOT SUPPORTED":
                pred = "REFUTE"
            elif raw_verdict == "SUPPORTED":
                pred = "SUPPORT"
            else:
                pred = "INCONCLUSIVE"

            final_result = {
                "verdict":    pred,
                "confidence": final_conf,
                "reasoning":  verdict_data["reasoning"]
            }
            logging_extension.append_framework_json(
                "final_verdict.jsonl", extracted_claim.id, final_result
            )

            print(f"Verdict   : {pred}")
            print(f"Confidence: {final_conf:.3f}")

            # ------------------------------------------------------------------
            # Metrics record
            # ------------------------------------------------------------------
            gt      = extracted_claim.metadata.get("label", "UNKNOWN")
            correct = (pred == gt) if gt != "UNKNOWN" else None

            record = {
                "run_id":                   ExtensionState.run_id,
                "claim_id":                 claim_id_str,
                "dataset":                  "HealthVer",
                "gt_label":                 gt,
                "pred_label":               pred,
                "correct":                  correct,
                "confidence":               final_conf,
                "rounds_normal":            3,
                "rounds_switched":          0,
                "total_rounds":             3,
                "judge_votes":              {"judge_1": pred},
                "token_total":              ExtensionState.current_claim_tokens,
                "token_input":              ExtensionState.current_claim_input_tokens,
                "token_output":             ExtensionState.current_claim_output_tokens,
                "token_openai":             ExtensionState.current_claim_openai_tokens,
                "token_openai_input":       ExtensionState.current_claim_openai_input_tokens,
                "token_openai_output":      ExtensionState.current_claim_openai_output_tokens,
                "token_openrouter":         ExtensionState.current_claim_openrouter_tokens,
                "token_openrouter_input":   ExtensionState.current_claim_openrouter_input_tokens,
                "token_openrouter_output":  ExtensionState.current_claim_openrouter_output_tokens,
                "token_groq":               ExtensionState.current_claim_groq_tokens,
                "token_models":             ExtensionState.current_claim_model_tokens,
                "retrieval_calls":          ExtensionState.current_claim_retrievals,
                "evidence_count":           ExtensionState.current_claim_evidence
            }

            with FileLock(logging_extension.CLAIMS_FILE + ".lock"):
                with open(logging_extension.CLAIMS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

            # Console summary
            print_extra_claim_metrics(
                claim_id      = claim_id_str,
                normal_rounds = 3,
                switched_rounds = 0,
                tokens        = record["token_total"],
                retrievals    = record["retrieval_calls"],
                evidence      = record["evidence_count"],
                confidence    = final_conf,
                judge_summary = verdict_data["verdict"],
                kappa_pair_mean = "N/A",
                ground_truth  = gt,
                verdict       = pred
            )

            # Master verdict ledger
            verdicts_path = os.path.join(outcome_dir, "all_verdicts.jsonl")
            with FileLock(verdicts_path + ".lock"):
                with open(verdicts_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "claim_id":    claim_id_str,
                        "claim_text":  input_claim.text,
                        "verdict":     pred,
                        "confidence":  final_conf,
                        "ground_truth": gt,
                        "correct":     correct,
                        "dataset":     "HealthVer"
                    }) + "\n")

            # Mark as processed
            with FileLock(processed_claims_path + ".lock"):
                with open(processed_claims_path, "a", encoding="utf-8") as f:
                    f.write(claim_id_str + "\n")

            processed_ids.add(claim_id_str)

        except Exception as e:
            print(f"\n[ERROR] Claim {claim_id_str} failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = dual_logger.terminal
            dual_logger.close()

    # -------------------------------------------------------------------------
    # End-of-run summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALL CLAIMS PROCESSED — HealthVerStd Mad Ablation 1")
    print("=" * 60)
    print(f"Outputs saved to: {ABLATION_BASE_DIR}")

    # Quick accuracy summary from all_verdicts
    verdicts_path = os.path.join(outcome_dir, "all_verdicts.jsonl")
    if os.path.exists(verdicts_path):
        total, correct_count, incl = 0, 0, 0
        with open(verdicts_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                v = json.loads(line)
                if v.get("correct") is not None:
                    total += 1
                    if v["correct"]:
                        correct_count += 1
                if v.get("verdict") == "INCONCLUSIVE":
                    incl += 1
        if total > 0:
            print(f"Accuracy    : {correct_count}/{total} = {correct_count/total:.4f}")
            print(f"Inconclusive: {incl}/{total} = {incl/total:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation 1 Standard MAD — HealthVer dataset (all models via OpenRouter)"
    )
    parser.add_argument(
        "--limit",  type=int, default=None,
        help="Maximum number of claims to process (default: all)"
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Start index into the claim list (default: 0)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process claims that are already marked as done"
    )
    args = parser.parse_args()

    run_ablation(args)
