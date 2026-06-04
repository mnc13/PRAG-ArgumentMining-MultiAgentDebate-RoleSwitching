import json

with open('scripts/feverous_retrieval_eval_results.json', encoding='utf-8') as f:
    r = json.load(f)

print(f'Total evaluated: {r["total_claims_evaluated"]}')
print(f'Hits: {r["hits"]}')
print(f'Misses: {r["misses"]}')
print(f'No gold found: {r["no_gold_found"]}')
print(f'No retrieved titles: {r["no_retrieved_titles"]}')
print(f'Page-level Recall: {r["page_level_recall"]:.1%}  ({r["hits"]}/{r["total_claims_evaluated"]})')
print()
print('MISSED claims:')
for c in r['per_claim']:
    if not c['hit']:
        print(f'  claim_id={c["claim_id"]:>6}  gold="{c["gold_title"]}"')
        print(f'           retrieved_titles (first 3): {c["retrieved_titles"][:3]}')
