import os
import json
import re
from datetime import datetime

# Paths
BASE_DIR = r"d:\thesis\PRAG--ArgumentMining-MultiAgentDebate-RoleSwitching-CheckCOVID"
LOGS_DIR = os.path.join(BASE_DIR, "framework", "outcome", "logs")
PROCESSED_CLAIMS_FILE = os.path.join(BASE_DIR, "framework", "outcome", "processed_claims.txt")
CLAIMS_ADDED_FILE = os.path.join(BASE_DIR, "artifacts", "metrics", "claims_added.jsonl")
ALL_VERDICTS_FILE = os.path.join(BASE_DIR, "framework", "outcome", "all_verdicts.jsonl")

def get_run_index(log_filename):
    # Log files are named: execution_log_<claim_id>_run_<timestamp>_<hex>.txt
    # We need to sort logs for the same claim_id by timestamp to assign run_index (0, 1, 2...)
    match = re.search(r"execution_log_(.*)_run_(\d+)_", log_filename)
    if not match:
        return None, None, None
    claim_id = match.group(1)
    timestamp = int(match.group(2))
    return claim_id, timestamp, log_filename

def parse_log_details(log_path):
    """Extracts verdict, confidence, ground truth, and other metrics from the log file."""
    details = {
        "verdict": "INCONCLUSIVE",
        "confidence": 0.0,
        "ground_truth": "N/A",
        "correct": False,
        "rounds": 0,
        "rounds_normal": 0,
        "rounds_switched": 0,
        "token_total": 0,
        "retrieval_calls": 0,
        "evidence_count": 0,
        "p_final": 0.0,
        "run_id": ""
    }
    
    # Extract run_id from filename
    filename = os.path.basename(log_path)
    run_id_match = re.search(r"run_(\d+_[a-f0-9]+)", filename)
    if run_id_match:
        details["run_id"] = "run_" + run_id_match.group(1)

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Ground Truth
            gt_match = re.search(r"Ground Truth: (.*)", content)
            if gt_match:
                details["ground_truth"] = gt_match.group(1).strip()
            
            # Final Verdict
            v_match = re.search(r"Final Verdict: (.*)", content)
            if v_match:
                details["verdict"] = v_match.group(1).strip()
            
            # Correctness
            c_match = re.search(r"Correct: (True|False)", content)
            if c_match:
                details["correct"] = c_match.group(1) == "True"
            
            # Extra Metrics
            # [CLAIM <id>] rounds=7 tok=83631 retr=17 ev=53 p_final=1.000 conf=1.000
            m_match = re.search(r"\[CLAIM .*\] rounds=(\d+) tok=(\d+) retr=(\d+) ev=(\d+) p_final=([0-9.]+) conf=([0-9.]+)", content)
            if m_match:
                details["rounds"] = int(m_match.group(1))
                details["token_total"] = int(m_match.group(2))
                details["retrieval_calls"] = int(m_match.group(3))
                details["evidence_count"] = int(m_match.group(4))
                details["confidence"] = float(m_match.group(6))
                
            # Role Switching Rounds: Count PROCEEDINGS PHASE blocks
            # Before "ROLE-SWITCHING ROUND", they are normal. After, they are switched.
            parts = content.split("ROLE-SWITCHING ROUND")
            normal_part = parts[0]
            switched_part = parts[1] if len(parts) > 1 else ""
            
            details["rounds_normal"] = len(re.findall(r"PROCEEDINGS PHASE \d+", normal_part))
            details["rounds_switched"] = len(re.findall(r"PROCEEDINGS PHASE \d+", switched_part))
            
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        
    return details

def main():
    print("Starting synchronization...")
    
    # 1. Collect all logs and sort them to determine run_index
    all_log_files = [f for f in os.listdir(LOGS_DIR) if f.startswith("execution_log_") and f.endswith(".txt")]
    log_data = []
    for f in all_log_files:
        cid, ts, fname = get_run_index(f)
        if cid:
            log_data.append({"claim_id": cid, "timestamp": ts, "filename": fname})
    
    # Sort by claim_id then timestamp
    log_data.sort(key=lambda x: (x["claim_id"], x["timestamp"]))
    
    # Assign run_index and limit to 3 runs max per claim
    claim_run_counts = {}
    logs_by_claim_run = {} # (claim_id, run_index) -> details
    
    for entry in log_data:
        cid = entry["claim_id"]
        run_idx = claim_run_counts.get(cid, 0)
        
        # Only process if we have fewer than 3 runs for this claim AND it's a valid run
        if run_idx < 3:
            full_path = os.path.join(LOGS_DIR, entry["filename"])
            details = parse_log_details(full_path)
            
            # Skip runs that failed to complete (no tokens processed)
            if details["token_total"] == 0 and details["rounds"] == 0:
                print(f"Skipping failed/incomplete run: {entry['filename']}")
                continue
                
            claim_run_counts[cid] = run_idx + 1
            details["claim_id"] = cid
            details["run_index"] = run_idx
            
            logs_by_claim_run[(cid, run_idx)] = details
        else:
            print(f"Skipping extra run for {cid}: {entry['filename']} (already have 3 successful runs)")

    print(f"Kept {len(logs_by_claim_run)} runs from logs (max 3 per claim).")

    # 2. Load existing state
    def load_lines(path):
        if not os.path.exists(path): return []
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    processed_lines = load_lines(PROCESSED_CLAIMS_FILE)
    processed_set = set(processed_lines)

    verdict_lines = load_lines(ALL_VERDICTS_FILE)
    verdict_claims = set()
    for line in verdict_lines:
        try:
            v_data = json.loads(line)
            # Some old verdicts might not have run_index, so we'd need to match carefully.
            # But the user wants consistency.
            verdict_claims.add(v_data.get("claim_id")) 
        except: pass

    claims_added_lines = load_lines(CLAIMS_ADDED_FILE)
    claims_added_map = {} # (claim_id, run_id) -> data
    for line in claims_added_lines:
        try:
            c_data = json.loads(line)
            claims_added_map[(c_data["claim_id"], c_data["run_id"])] = c_data
        except: pass

    # 3. Synchronize
    new_processed = []
    new_verdicts = []
    new_claims_added = []
    
    # We rebuild or append? User wants them matched and consistent.
    # I'll build the complete list based on logs.
    
    for (cid, ridx), details in logs_by_claim_run.items():
        # processed_claims.txt
        entry_pc = f"{cid}:{ridx}"
        new_processed.append(entry_pc)
        
        # all_verdicts.jsonl
        # Note: all_verdicts usually doesn't have run_index in the key, but we should probably keep it consistent.
        # However, the user's file shows {"claim_id": "...", "verdict": "..."}
        # If there are multiple runs for the same claim, they should probably have unique entries if possible.
        # But looking at the user's all_verdicts.jsonl, it seems to just use claim_id.
        # If I have 60ac6953f9b9e03ea4d8e692_0 run 0 and run 1, they both show as 60ac6953f9b9e03ea4d8e692_0?
        # That would be a problem for consistency. I'll check the existing file again.
        
        # Re-check: the user's file shows claim_id: 60ac6953f9b9e03ea4d8e692_0 and 60ac6953f9b9e03ea4d8e692_1.
        # Those are different claims (ending in _0, _1).
        # My run restoration before used :1 for the SAME claim_id.
        
        v_entry = {
            "claim_id": cid if ridx == 0 else f"{cid}:{ridx}", # Match the :1 format if it's not run 0
            "verdict": details["verdict"],
            "confidence": details["confidence"],
            "ground_truth": details["ground_truth"],
            "correct": details["correct"]
        }
        new_verdicts.append(json.dumps(v_entry))
        
        # claims_added.jsonl
        ca_entry = {
            "run_id": details["run_id"],
            "claim_id": cid,
            "gt_label": details["ground_truth"],
            "pred_label": details["verdict"],
            "correct": details["correct"],
            "confidence": details["confidence"],
            "rounds_normal": details["rounds_normal"],
            "rounds_switched": details["rounds_switched"],
            "token_total": details["token_total"],
            "retrieval_calls": details["retrieval_calls"],
            "evidence_count": details["evidence_count"]
        }
        new_claims_added.append(json.dumps(ca_entry))

    # Write back
    print(f"Writing {len(new_processed)} entries to processed_claims.txt")
    with open(PROCESSED_CLAIMS_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_processed) + "\n")

    print(f"Writing {len(new_verdicts)} entries to all_verdicts.jsonl")
    with open(ALL_VERDICTS_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_verdicts) + "\n")

    print(f"Writing {len(new_claims_added)} entries to claims_added.jsonl")
    with open(CLAIMS_ADDED_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_claims_added) + "\n")

    print("Sync complete.")

if __name__ == "__main__":
    main()
