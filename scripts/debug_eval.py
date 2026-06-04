import json

with open('scripts/feverous_retrieval_eval_results.json', encoding='utf-8') as f:
    r = json.load(f)

print(f"Total claims evaluated: {r['total_claims_evaluated']}")
print(f"Hits (raw): {r['hits']} -> Recall: {r['page_level_recall']:.1%}")
print(f"Hits (adjusted): {r['adjusted_hits']} -> Recall: {r['adjusted_page_level_recall']:.1%}")
print(f"Misses: {r['misses']}")
print(f"Claims where all retrieval was ruined by distractor: {r['dicom_ruined']}")
print("\nMissed claims (raw):")
for c in r['per_claim']:
    if not c['hit']:
        is_ruined = c['dicom_ruined']
        ruined_mark = " [RUINED BY DICOM PROBE]" if is_ruined else ""
        print(f"  [{c['claim_id']}] '{c['gold_title']}'{ruined_mark}")
        
