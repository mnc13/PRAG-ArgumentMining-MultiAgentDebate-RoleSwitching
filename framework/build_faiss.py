import argparse, gzip, json, os, re
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# Evidence-heavy sections to prefer splitting into their own chunks
SECTION_PRIORITY = [
    "RESULTS",
    "FINDINGS",
    "INTERPRETATION",
    "CONCLUSION",
    "CONCLUSIONS",
]

# Section headers we try to detect in abstracts
SECTION_HEADERS = [
    "BACKGROUND",
    "INTRODUCTION",
    "OBJECTIVE",
    "OBJECTIVES",
    "AIM",
    "AIMS",
    "METHOD",
    "METHODS",
    "MATERIALS",
    "MATERIALS AND METHODS",
    "DESIGN",
    "SETTING",
    "PARTICIPANTS",
    "INTERVENTION",
    "MEASUREMENTS",
    "RESULT",
    "RESULTS",
    "FINDINGS",
    "DISCUSSION",
    "CONCLUSION",
    "CONCLUSIONS",
    "INTERPRETATION",
]

HEADER_RE = re.compile(
    r"(?:(?:^)|(?:\n)|(?:\s))(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:\s*",
    flags=re.IGNORECASE,
)

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_ws(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple spaces, but keep newlines
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def sentenceish_chunks(text: str, chunk_size: int = 1200, overlap_chars: int = 150) -> list[str]:
    """
    Split into sentence-ish chunks up to chunk_size characters, with small overlap.
    """
    text = normalize_ws(text)
    if not text:
        return []

    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    if not sents:
        return [text] if text else []

    chunks = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) + 1 <= chunk_size:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    if overlap_chars > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap_chars:]
            out.append((prev_tail + " " + chunks[i]).strip())
        return out

    return chunks


def split_by_sections(abstract: str) -> list[tuple[str, str]]:
    """
    Returns list of (SECTION, TEXT) segments. If no headers found, returns [].
    """
    abstract = normalize_ws(abstract)
    if not abstract:
        return []

    matches = list(HEADER_RE.finditer(abstract))
    if not matches:
        return []

    segments = []
    for i, m in enumerate(matches):
        sec = m.group(1).upper().strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(abstract)
        body = abstract[start:end].strip()
        if body:
            segments.append((sec, body))
    return segments


def chunk_abstract(abstract: str, chunk_size: int = 1200, overlap_chars: int = 150) -> list[tuple[str, str]]:
    """
    Returns list of (section_name, chunk_text).
    - If labeled sections exist, chunk each section separately (better evidence retrieval).
    - Else fallback to sentence-ish chunking on the whole abstract.
    """
    abstract = normalize_ws(abstract)
    if not abstract:
        return []

    segs = split_by_sections(abstract)
    if not segs:
        # No labels -> just sentence-ish chunking
        return [("ABSTRACT", c) for c in sentenceish_chunks(abstract, chunk_size, overlap_chars)]

    out = []
    for sec, body in segs:
        # Keep header in chunk text so the model sees "RESULTS:" etc.
        labeled = f"{sec}: {body}".strip()
        # Chunk within section if too long
        for c in sentenceish_chunks(labeled, chunk_size, overlap_chars):
            out.append((sec, c))
    return out


def section_rank(sec: str) -> int:
    sec = (sec or "").upper()
    # Lower is better (more evidence-y)
    for i, k in enumerate(SECTION_PRIORITY):
        if sec == k:
            return i
    return len(SECTION_PRIORITY) + 5


def iter_records(path_gz: str):
    with gzip.open(path_gz, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="pubmed_covid_2020_2024_edat_upto_2024.jsonl.gz")
    ap.add_argument("--out_index", default="pubmed_faiss.index")
    ap.add_argument("--out_meta", default="pubmed_meta.jsonl")
    ap.add_argument("--out_offsets", default="pubmed_meta_offsets.npy")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=150)
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()

    # Cosine similarity via inner product on normalized vectors
    index = faiss.IndexFlatIP(dim)

    offsets = []
    total_vectors = 0

    os.makedirs(os.path.dirname(args.out_index) or ".", exist_ok=True)

    with open(args.out_meta, "wb") as meta_f:
        buf_texts = []
        buf_metas = []

        for rec in tqdm(iter_records(args.infile), desc="Reading & chunking"):
            title = (rec.get("title") or "").strip()
            abstract = (rec.get("abstract") or "").strip()

            # For PubMed-based RAG, abstract is the evidence.
            # If abstract is missing, fall back to text_for_rag.
            if abstract:
                chunks = chunk_abstract(abstract, chunk_size=args.chunk_size, overlap_chars=args.overlap)
            else:
                base_text = (rec.get("text_for_rag") or "").strip()
                if not base_text:
                    continue
                # Chunk whatever we have
                chunks = [("TEXT", c) for c in sentenceish_chunks(base_text, args.chunk_size, args.overlap)]

            if not chunks:
                continue

            base_meta = {
                "source": rec.get("source", "pubmed"),
                "pmid": rec.get("pmid", ""),
                "doi": rec.get("doi", ""),
                "title": title,
                "year": rec.get("year", ""),
                "journal": rec.get("journal", ""),
                "url": rec.get("url", ""),
            }

            for ci, (sec, chunk) in enumerate(chunks):
                m = dict(base_meta)
                m["chunk_id"] = ci
                m["section"] = sec
                m["section_rank"] = section_rank(sec)

                # Store what you’ll show after retrieval
                # (include title for context without re-indexing it separately)
                if title:
                    m["text"] = f"{title}\n\n{chunk}".strip()
                    embed_text = f"{title}\n\n{chunk}".strip()
                else:
                    m["text"] = chunk
                    embed_text = chunk

                buf_texts.append(embed_text)
                buf_metas.append(m)

                if len(buf_texts) >= args.batch:
                    emb = model.encode(
                        buf_texts,
                        batch_size=args.batch,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).astype("float32")

                    index.add(emb)

                    for meta in buf_metas:
                        offsets.append(meta_f.tell())
                        meta_f.write((json.dumps(meta, ensure_ascii=False) + "\n").encode("utf-8"))

                    total_vectors += len(buf_texts)
                    buf_texts.clear()
                    buf_metas.clear()

        # Flush leftovers
        if buf_texts:
            emb = model.encode(
                buf_texts,
                batch_size=args.batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            index.add(emb)

            for meta in buf_metas:
                offsets.append(meta_f.tell())
                meta_f.write((json.dumps(meta, ensure_ascii=False) + "\n").encode("utf-8"))

            total_vectors += len(buf_texts)

    faiss.write_index(index, args.out_index)
    np.save(args.out_offsets, np.array(offsets, dtype=np.int64))

    print("Done.")
    print(f"Vectors indexed: {total_vectors}")
    print(f"FAISS index: {args.out_index}")
    print(f"Metadata:   {args.out_meta}")
    print(f"Offsets:    {args.out_offsets}")


if __name__ == "__main__":
    main()
