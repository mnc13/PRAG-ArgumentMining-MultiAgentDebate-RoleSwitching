import os
import requests
import logging
import re
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from models import Evidence
import wikipediaapi

logger = logging.getLogger(__name__)

HEADERS  = {"User-Agent": "PRAG_Argument_Debate (research_script)"}
WIKI_API = "https://en.wikipedia.org/w/api.php"

# Section titles that are almost always relevant for FEVEROUS fact-checking
# These are forced into the candidate pool regardless of semantic score
PRIORITY_SECTION_KEYWORDS = {
    "schedule", "results", "season results", "game results",
    "season", "record", "standings", "roster", "history",
    "career", "biography", "early life", "personal life",
    "discography", "filmography", "political positions",
    "elections", "awards", "honors", "statistics", "stats",
}


class KILTWikipediaRetriever:
    """
    Wikipedia retriever for FEVEROUS / encyclopaedic fact-checking.

    v4 key changes
    ──────────────
    1. HYBRID SCORING  :  final_score = α * semantic_score + (1-α) * keyword_score
       keyword_score = fraction of query tokens found in the chunk text.
       This ensures a chunk mentioning "September 19" + "Wake Forest" beats
       a generic intro paragraph even when semantic similarity is lower.

    2. PRIORITY SECTION PINNING  :  chunks from schedule/results/history
       sections are always included in the top-k candidates alongside the
       semantic top-k, so the scorer always sees the relevant section.

    3. SMART DEDUP  :  same page_id chunks are capped so one page can't
       crowd out all top-k slots with near-identical intro paragraphs.

    4. Everything from v3 retained:
       - Direct page probing (year + entity title synthesis)
       - Prefix search
       - Condensed query search
       - Section-level chunking (correct list iteration)
    """

    ALPHA = 0.6   # weight for semantic score; (1-ALPHA) for keyword score

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info("Initializing KILT Wikipedia Retriever (v4)...")
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=HEADERS["User-Agent"],
            language="en",
        )
        self._query_cache:   Dict[str, List[Evidence]] = {}
        self._passage_cache: Dict[str, str]            = {}
        self._total_queries  = 0
        self._cache_hits     = 0

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        self._total_queries += 1
        cache_key = f"{query.strip().lower()}||{top_k}"
        if cache_key in self._query_cache:
            self._cache_hits += 1
            return self._query_cache[cache_key]
        results = self._fetch_and_score(query, top_k)
        self._query_cache[cache_key] = results
        return results

    # ─────────────────────────────────────────────────────────────────
    # Chunking
    # ─────────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, page_id, title: str,
                    is_priority: bool = False) -> List[dict]:
        words = text.split()
        chunks = []
        step, size, min_w = 50, 100, 15
        for i, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start: start + size]
            if len(chunk_words) < min_w:
                break
            chunks.append({
                "id":          f"{page_id}_{i}",
                "title":       title,
                "text":        " ".join(chunk_words),
                "is_priority": is_priority,
            })
        return chunks

    # ─────────────────────────────────────────────────────────────────
    # Scoring helpers
    # ─────────────────────────────────────────────────────────────────

    def _keyword_score(self, chunk_text: str, query_tokens: List[str]) -> float:
        """
        Fraction of meaningful query tokens present in the chunk (case-insensitive).
        Tokens shorter than 3 chars or pure stop-words are skipped.
        """
        if not query_tokens:
            return 0.0
        chunk_lower = chunk_text.lower()
        hits = sum(1 for t in query_tokens if t in chunk_lower)
        return hits / len(query_tokens)

    def _tokenise_query(self, query: str) -> List[str]:
        """
        Break query into meaningful tokens for keyword matching.
        Keeps years, proper-noun fragments, and content words ≥ 3 chars.
        """
        stop = {
            "the","a","an","is","are","was","were","in","on","at","of","for",
            "with","by","to","that","this","it","its","does","did","do","have",
            "has","had","be","been","against","during","from","or","and","not",
            "no","both","game","played","held","beat","defeated","result",
            "which","when","where","who","what","their","they","them",
        }
        raw = re.findall(r'\b\w+\b', query.lower())
        return [t for t in raw if t not in stop and len(t) >= 3]

    # ─────────────────────────────────────────────────────────────────
    # Title discovery
    # ─────────────────────────────────────────────────────────────────

    def _extract_years(self, text: str) -> List[str]:
        return re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text)

    def _extract_proper_nouns(self, text: str) -> List[str]:
        tokens = re.findall(r'[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*', text)
        seen, out = set(), []
        for t in sorted(tokens, key=len, reverse=True):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _build_direct_probe_titles(self, query: str) -> List[str]:
        years  = self._extract_years(query)
        nouns  = self._extract_proper_nouns(query)
        ql     = query.lower()

        sport_suffixes: List[str] = []
        for sport in ("football","basketball","baseball",
                      "soccer","hockey","cricket","tennis","rugby"):
            if sport in ql:
                sport_suffixes += [sport, f"{sport} season", f"{sport} team"]
        if not sport_suffixes:
            sport_suffixes = ["season", ""]

        titles: List[str] = []
        for year in years:
            for noun in nouns[:6]:
                for suffix in sport_suffixes:
                    titles.append(f"{year} {noun} {suffix}".strip())
                    titles.append(f"{noun} {year} {suffix}".strip())
            if nouns:
                titles.append(f"{year} {nouns[0]}")

        for noun in nouns[:5]:
            titles.append(noun)

        seen, unique = set(), []
        for t in titles:
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def _search_api(self, query: str, srlimit: int = 10) -> List[str]:
        try:
            r = requests.get(WIKI_API, params={
                "action":"query","list":"search",
                "srsearch":query,"srlimit":srlimit,
                "format":"json","utf8":1,
            }, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return [x["title"] for x in r.json().get("query",{}).get("search",[])]
        except Exception as e:
            logger.warning(f"Search API failed for '{query}': {e}")
            return []

    def _prefix_search(self, prefix: str, limit: int = 5) -> List[str]:
        try:
            r = requests.get(WIKI_API, params={
                "action":"query","list":"prefixsearch",
                "pssearch":prefix,"pslimit":limit,
                "format":"json",
            }, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return [x["title"] for x in r.json().get("query",{}).get("prefixsearch",[])]
        except Exception:
            return []

    def _condense_query(self, query: str) -> str:
        stop = {
            "the","a","an","is","are","was","were","in","on","at","of","for",
            "with","by","to","that","this","it","its","does","did","do","have",
            "has","had","be","been","against","during","from","or","and","not",
            "no","both","game","played","held","won","lost","beat","defeated",
            "result","season","which","when","where","who","what",
        }
        words = [w for w in query.split() if w.lower() not in stop]
        return " ".join(words[:8])

    def _get_all_candidate_titles(self, query: str) -> List[str]:
        seen: set       = set()
        all_titles: List[str] = []

        def add(t: str):
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                all_titles.append(t)

        for t in self._build_direct_probe_titles(query):
            add(t)

        years = self._extract_years(query)
        nouns = self._extract_proper_nouns(query)
        for year in years:
            if nouns:
                for t in self._prefix_search(f"{year} {nouns[0]}", limit=5):
                    add(t)
                if len(nouns) > 1:
                    for t in self._prefix_search(f"{year} {nouns[1]}", limit=3):
                        add(t)

        condensed = self._condense_query(query)
        for t in self._search_api(condensed, srlimit=10):
            add(t)

        for t in self._search_api(query, srlimit=5):
            add(t)

        return all_titles

    # ─────────────────────────────────────────────────────────────────
    # Page fetching
    # ─────────────────────────────────────────────────────────────────

    def _is_priority_section(self, section_title: str) -> bool:
        return any(kw in section_title.lower() for kw in PRIORITY_SECTION_KEYWORDS)

    def _fetch_page_passages(self, title: str,
                              max_passages: int,
                              current_count: int) -> List[dict]:
        passages: List[dict] = []
        try:
            page = self.wiki.page(title)
            if not page.exists():
                return []
            page_id = page.pageid

            # Full-page text (not priority — let scoring decide)
            if page.text:
                for chunk in self._chunk_text(page.text, page_id, title,
                                               is_priority=False):
                    passages.append(chunk)
                    self._passage_cache[chunk["id"]] = chunk["text"]
                    if current_count + len(passages) >= max_passages:
                        return passages

            # Section-level chunks — mark priority sections
            sections = getattr(page, "sections", []) or []
            for section in sections:
                if current_count + len(passages) >= max_passages:
                    break
                sec_title = getattr(section, "title", "")
                sec_text  = getattr(section, "text",  "")
                if sec_text and len(sec_text.split()) >= 15:
                    is_pri = self._is_priority_section(sec_title)
                    safe   = re.sub(r"[^a-z0-9]", "_", sec_title.lower())
                    sec_id = f"{page_id}_sec_{safe}"
                    for chunk in self._chunk_text(sec_text, sec_id,
                                                   f"{title} § {sec_title}",
                                                   is_priority=is_pri):
                        passages.append(chunk)
                        self._passage_cache[chunk["id"]] = chunk["text"]
                        if current_count + len(passages) >= max_passages:
                            break
        except Exception as e:
            logger.warning(f"Failed to fetch/process page '{title}': {e}")
        return passages

    # ─────────────────────────────────────────────────────────────────
    # Fetch & Score  (hybrid)
    # ─────────────────────────────────────────────────────────────────

    def _fetch_and_score(self, query: str, top_k: int) -> List[Evidence]:
        candidate_titles = self._get_all_candidate_titles(query)
        if not candidate_titles:
            return []

        all_passages: List[dict] = []
        MAX_PASSAGES = 1000

        for title in candidate_titles:
            if len(all_passages) >= MAX_PASSAGES:
                break
            all_passages.extend(
                self._fetch_page_passages(title, MAX_PASSAGES, len(all_passages))
            )

        if not all_passages:
            logger.warning(f"No passages for: '{query}'")
            return []

        logger.info(f"Scoring {len(all_passages)} passages for: '{query[:70]}'")

        # ── Semantic scores ──────────────────────────────────────────
        query_emb    = self.model.encode([query], normalize_embeddings=True)[0]
        passage_embs = self.model.encode(
            [p["text"] for p in all_passages],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        sem_scores = np.dot(passage_embs, query_emb)           # shape (N,)

        # ── Keyword scores ───────────────────────────────────────────
        query_tokens = self._tokenise_query(query)
        kw_scores = np.array([
            self._keyword_score(p["text"], query_tokens)
            for p in all_passages
        ])

        # ── Hybrid scores ────────────────────────────────────────────
        hybrid = self.ALPHA * sem_scores + (1 - self.ALPHA) * kw_scores

        # ── Select top-k with per-page diversity cap ─────────────────
        # Sort by hybrid score descending
        sorted_indices = np.argsort(hybrid)[::-1]

        selected: List[int]    = []
        page_counts: Dict[str, int] = {}
        MAX_PER_PAGE = max(2, top_k // 2)   # e.g. top_k=5 → max 2-3 per page

        # First pass: priority sections always get a slot
        priority_slots = min(top_k // 2, 3)   # up to half of top_k for priority
        for idx in sorted_indices:
            if len([s for s in selected
                    if all_passages[s].get("is_priority")]) >= priority_slots:
                break
            if all_passages[idx].get("is_priority"):
                selected.append(idx)

        # Second pass: fill remaining slots with best hybrid scores (diversity)
        for idx in sorted_indices:
            if len(selected) >= top_k:
                break
            if idx in selected:
                continue
            page_root = str(all_passages[idx]["id"]).split("_sec_")[0].split("_")[0]
            if page_counts.get(page_root, 0) >= MAX_PER_PAGE:
                continue
            selected.append(idx)
            page_counts[page_root] = page_counts.get(page_root, 0) + 1

        # If still short (e.g. all pages hit cap), fill from remaining
        for idx in sorted_indices:
            if len(selected) >= top_k:
                break
            if idx not in selected:
                selected.append(idx)

        return [
            Evidence(
                text=(
                    f"[Wikipedia: {all_passages[i]['title']}] "
                    f"{all_passages[i]['text']}"
                ),
                source_id=str(all_passages[i]["id"]),
                relevance_score=float(hybrid[i]),
            )
            for i in selected[:top_k]
        ]

    # ─────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────

    def get_cache_stats(self) -> dict:
        return {
            "query_cache_hits": self._cache_hits,
            "total_queries":    self._total_queries,
            "unique_passages":  len(self._passage_cache),
        }