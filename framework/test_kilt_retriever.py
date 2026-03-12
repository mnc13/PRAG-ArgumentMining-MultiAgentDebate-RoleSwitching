from kilt_retriever import KILTWikipediaRetriever

retriever = KILTWikipediaRetriever()

test_queries = [
    "Ronaldo plays for Manchester United",
    "COVID-19 vaccines cause infertility",
    "The Eiffel Tower is located in Berlin"
]

for q in test_queries:
    print(f"\nQuery: {q}")
    results = retriever.retrieve(q, top_k=3)
    for r in results:
        print(f"  [{r.source_id}] score={r.relevance_score:.3f} | {r.text[:120]}...")

print("\nP-RAG model encode test:")
emb = retriever.model.encode(["test sentence"], normalize_embeddings=True)
print(f" Embedding shape: {emb.shape} ✓")
