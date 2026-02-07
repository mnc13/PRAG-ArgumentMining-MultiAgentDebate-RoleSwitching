from typing import List, Dict
from models import Evidence
import re
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleRetriever:
    def __init__(self, corpus: Dict[str, Dict]):
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 3) -> List[Evidence]:
        """
        Retrieves relevant evidence from the corpus based on keyword overlap.
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        results = []

        for cord_id, doc in self.corpus.items():
            abstract_text = " ".join(doc['abstract'])
            doc_words = set(re.findall(r'\w+', abstract_text.lower()))
            overlap = len(query_words.intersection(doc_words))
            
            if overlap > 0:
                results.append(Evidence(
                    text=abstract_text,
                    source_id=cord_id,
                    relevance_score=float(overlap)
                ))

        # Sort by relevance score descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]

import json

# ... (Previous code)

class VectorRetriever:
    def __init__(self, corpus: Dict[str, Dict], model_name: str = 'all-MiniLM-L6-v2', index_path: str = 'vector_store.index', metadata_path: str = 'vector_metadata.json'):
        self.corpus = corpus
        self.index_path = index_path
        self.metadata_path = metadata_path
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self.doc_ids = []
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"Loading existing FAISS index from {index_path}...")
            self.index = faiss.read_index(index_path)
            print(f"Loading metadata from {metadata_path}...")
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.doc_ids = [m['cord_id'] for m in self.metadata]
        else:
            print("Creating new FAISS index (this may take a while)...")
            self.index = self._build_index()

    def _build_index(self):
        self.doc_ids = list(self.corpus.keys())
        texts = [" ".join(self.corpus[cid]['abstract']) for cid in self.doc_ids]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        # Save index
        faiss.write_index(index, self.index_path)
        
        # Save metadata
        metadata = []
        for cid in self.doc_ids:
            metadata.append({
                'cord_id': cid,
                'title': self.corpus[cid]['title']
            })
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return index

    def retrieve(self, query: str, top_k: int = 3) -> List[Evidence]:
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue # No match found
            cord_id = self.doc_ids[idx]
            doc = self.corpus[cord_id]
            abstract_text = " ".join(doc['abstract'])
            # FAISS returns L2 distance (smaller is better), we might want to convert to a similarity score or just keep it.
            # For consistency with "relevance", we can invert it or just pass it through.
            # Here we just pass the distance as "score" (Logic: Lower is better, but Evidence expects float)
            results.append(Evidence(
                text=abstract_text,
                source_id=cord_id,
                relevance_score=float(distances[0][i]) # Note: This is a distance, not similarity
            ))
        
        return results

class PubMedRetriever:
    def __init__(self, index_path: str = 'pubmed_faiss.index', meta_path: str = 'pubmed_meta.jsonl', offsets_path: str = 'pubmed_meta_offsets.npy', model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Retriever for the large-scale PubMed Corpus using FAISS and disk-based metadata.
        """
        self.index_path = index_path
        self.meta_path = meta_path
        self.offsets_path = offsets_path
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        if os.path.exists(index_path) and os.path.exists(offsets_path) and os.path.exists(meta_path):
            print(f"Loading FAISS index from {index_path}...")
            self.index = faiss.read_index(index_path)
            print(f"Loading offsets from {offsets_path}...")
            self.offsets = np.load(offsets_path)
        else:
            raise FileNotFoundError(f"Missing one or more required files: {index_path}, {offsets_path}, {meta_path}. Please run build_faiss.py first.")

    def _read_meta_at(self, offset: int) -> Dict:
        with open(self.meta_path, "rb") as f:
            f.seek(int(offset))
            line = f.readline().decode("utf-8")
            return json.loads(line)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        """
        Retrieves evidence using the PubMed FAISS index.
        """
        # Encode query
        # normalize_embeddings=True because index was built with normalized vectors (IndexFlatIP)
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        
        scores, ids = self.index.search(q_emb, top_k)
        
        results = []
        for i, idx in enumerate(ids[0]):
            if idx == -1: 
                continue 
            
            # Read metadata using offset
            try:
                offset = self.offsets[idx]
                meta = self._read_meta_at(offset)
                
                # Construct text and source_id
                # Use the 'text' field which contains title + chunk (populated in build_faiss.py)
                # or fallback to direct reconstruction
                text_content = meta.get("text", "")
                if not text_content:
                    title = meta.get("title", "")
                    chunk = meta.get("text_for_rag", "") or "" 
                    # Note: metadata schema might vary, build_faiss.py puts "text" field explicitly
                    text_content = f"{title}\n\n{chunk}".strip()

                # Use PMID as source_id, fallback to DOI, then 'unknown'
                source_id = meta.get("pmid") or meta.get("doi") or f"chunk_{idx}"
                
                # Add context like Year/Journal if available
                year = meta.get("year", "")
                journal = meta.get("journal", "")
                if year or journal:
                    text_content = f"[{journal} {year}] {text_content}"

                results.append(Evidence(
                    text=text_content,
                    source_id=str(source_id),
                    relevance_score=float(scores[0][i])
                ))
            except Exception as e:
                print(f"[ERROR] Failed to retrieve metadata for index {idx}: {e}")
                continue
                
        return results
