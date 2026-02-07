import argparse
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def read_meta_at(meta_path: str, offset: int):
    with open(meta_path, "rb") as f:
        f.seek(int(offset))
        line = f.readline().decode("utf-8")
        return json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="pubmed_faiss.index")
    ap.add_argument("--meta", default="pubmed_meta.jsonl")
    ap.add_argument("--offsets", default="pubmed_meta_offsets.npy")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--q", required=True, help="your query text")
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    index = faiss.read_index(args.index)
    offsets = np.load(args.offsets)

    q_emb = model.encode([args.q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, args.k)

    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if idx == -1:
            continue

        meta = read_meta_at(args.meta, offsets[idx])

        print(f"\n#{rank}  score={float(score):.4f}")
        print(meta.get("title", ""))

        # Extra helpful fields from your new indexing script
        if "section" in meta or "chunk_id" in meta:
            print(f"Section={meta.get('section','')}  ChunkID={meta.get('chunk_id','')}")

        print(f"PMID={meta.get('pmid','')}  Year={meta.get('year','')}  DOI={meta.get('doi','')}")
        print(meta.get("url", ""))

        print("\n--- chunk ---")
        print((meta.get("text", "") or "")[:1200])


if __name__ == "__main__":
    main()
