"""
FEVEROUS Page-Level Retrieval Recall
— Fixed Wikipedia API fetch + Unicode NFC normalization
"""

import json, time, urllib.request, urllib.parse, unicodedata
from pathlib import Path

def normalize(s):
    """NFC-normalize and lowercase for robust comparison."""
    return unicodedata.normalize("NFC", s).lower()


BASE         = Path(__file__).parent.parent
PRAG_HISTORY = BASE / "artifacts_feverous/outcome_feverous/all_output_jsons/prag_history.jsonl"
SAMPLE_100   = BASE / "Other-datasets/feverous_sample_100.jsonl"
CACHE_FILE   = Path(__file__).parent / "feverous_pageid_title_map.json"
RESULTS_FILE = Path(__file__).parent / "feverous_retrieval_eval_results.json"

# ── 1. Load gold cord_id (= Wikipedia page title) ─────────────────────────────
gold = {}
with open(SAMPLE_100, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            gold[str(rec["id"])] = rec["cord_id"]
print(f"Gold claims loaded: {len(gold)}")

# ── 1.5 Load evaluated claims from all_verdicts.jsonl ──────────────────────────
evaluated_claims = set()
VERDICTS = BASE / "framework/outcome_feverous/all_verdicts.jsonl"
with open(VERDICTS, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            evaluated_claims.add(str(rec["claim_id"]))
print(f"Claims with official verdicts: {len(evaluated_claims)}")

# ── 2. Load retrieved pageids from prag_history ────────────────────────────────
retrieved = {}   # claim_id -> set of str pageids
with open(PRAG_HISTORY, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = str(rec["claim_id"])
        if cid not in evaluated_claims:
            continue  # Only evaluate claims that made it to final verdicts
            
        pids = set()
        for rnd in rec["data"]["history"]:
            for eid in rnd.get("evidence_ids", []):
                if eid:
                    part = eid.split("_")[0]
                    if part.isdigit():
                        pids.add(part)
        retrieved[cid] = pids

print(f"Claims with retrieval data: {len(retrieved)}")

all_pageids = sorted({pid for pids in retrieved.values() for pid in pids})
print(f"Unique pageids to resolve: {len(all_pageids)}")

# ── 3. Wikipedia pageid → title (batch API, cached) ───────────────────────────
if CACHE_FILE.exists():
    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)
    print(f"Cache already has {len(cache)} entries")
else:
    cache = {}

to_fetch = [p for p in all_pageids if p not in cache]
print(f"Need to fetch {len(to_fetch)} from Wikipedia API...")

def fetch_batch(page_ids):
    """Returns dict: str(pageid) -> title. Uses standard format (no formatversion=2)."""
    params = {
        "action": "query",
        "pageids": "|".join(page_ids),
        "format": "json",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "FeverousEval/1.0 (research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    result = {}
    pages = data.get("query", {}).get("pages", {})
    for pageid_str, page_info in pages.items():
        # pageid_str is a string key; -1 means "missing"
        if "missing" in page_info or pageid_str == "-1":
            continue
        result[pageid_str] = page_info.get("title", "")
    return result

BATCH = 50
for i in range(0, len(to_fetch), BATCH):
    batch = to_fetch[i : i + BATCH]
    batch_num = i // BATCH + 1
    total_batches = (len(to_fetch) + BATCH - 1) // BATCH
    print(f"  Batch {batch_num}/{total_batches}: fetching {len(batch)} ids...")
    try:
        titles = fetch_batch(batch)
        cache.update(titles)
        print(f"    Got {len(titles)} titles")
    except Exception as e:
        print(f"    WARNING — fetch failed: {e}")
    time.sleep(0.5)

# Save cache
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)
print(f"Cache saved: {len(cache)} entries → {CACHE_FILE}")

# ── 4. Per-claim evaluation ────────────────────────────────────────────────────
results = []
hits = misses = no_gold = 0
adjusted_hits = 0
adjusted_misses = 0
dicom_ruined = 0

# Distractor pages commonly hit by the "medical probe" behavior of the plaintiff
DISTRACTORS = {
    "dicom", "medicine", "electronic health record", "health informatics",
    "medical imaging", "health information exchange", "pac system",
    "radiology", "picture archiving and communication system"
}

for cid, pids in retrieved.items():
    g = gold.get(cid)
    if g is None:
        no_gold += 1
        continue

    # Build set of retrieved page titles (NFC-normalized + lowercased)
    ret_titles = set()
    for pid in pids:
        t = cache.get(pid, "")
        if t:
            ret_titles.add(normalize(t))

    gold_norm = normalize(g)
    
    # Clean retrieved titles of distractors
    clean_titles = {t for t in ret_titles if t not in DISTRACTORS}
    
    hit = gold_norm in ret_titles
    clean_hit = gold_norm in clean_titles

    if hit:
        hits += 1
    else:
        misses += 1
        
    if clean_hit:
        adjusted_hits += 1
    else:
        adjusted_misses += 1
        
    # Check if retrieval was mostly just distractors (ruined)
    if len(clean_titles) <= 1 and len(ret_titles) > 2:
        dicom_ruined += 1

    results.append({
        "claim_id": cid,
        "gold_title": g,
        "retrieved_pageids": list(pids),
        "retrieved_titles": list(ret_titles),
        "clean_titles": list(clean_titles),
        "hit": hit,
        "clean_hit": clean_hit,
        "dicom_ruined": len(clean_titles) <= 1 and len(ret_titles) > 2
    })

n = hits + misses
recall = hits / n if n else 0.0
adjusted_recall = adjusted_hits / n if n else 0.0

# ── 5. Print summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FEVEROUS — Page-Level Retrieval Recall")
print("=" * 60)
print(f"  Total claims evaluated   : {n}")
print(f"  Hits (raw)               : {hits}  (Recall: {recall:.1%})")
print(f"  Hits (excluding probes)  : {adjusted_hits}  (Adj. Recall: {adjusted_recall:.1%})")
print(f"  Claims ruined by probes  : {dicom_ruined}")
print(f"  Misses                   : {misses}")
print("=" * 60)

print("\nMissed claims (raw):")
for r in results:
    if not r["hit"]:
        print(f"  [{r['claim_id']:>6}] gold='{r['gold_title']}'")
        print(f"           retrieved: {r['retrieved_titles'][:4]}")

# ── 6. Save full results ──────────────────────────────────────────────────────
summary = {
    "total_claims_evaluated": n,
    "hits": hits,
    "adjusted_hits": adjusted_hits,
    "dicom_ruined": dicom_ruined,
    "misses": misses,
    "no_gold_found": no_gold,
    "page_level_recall": recall,
    "adjusted_page_level_recall": adjusted_recall,
    "per_claim": results,
}
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nResults saved → {RESULTS_FILE}")
