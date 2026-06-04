import os
import time
import random
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

PRIORITY_SECTION_KEYWORDS = {
    "schedule", "results", "season results", "game results",
    "season", "record", "standings", "roster", "history",
    "career", "biography", "early life", "personal life",
    "discography", "filmography", "political positions",
    "elections", "awards", "honors", "statistics", "stats",
    "classification", "taxonomy", "description", "founding",
    "formation", "establishment", "overview", "background",
    "profile", "members", "composition",
    # Table-heavy sections — these are the most retrieval-critical for FEVEROUS
    "qualifying", "race result", "grid", "lap times",
    "season summary", "infobox",
    # Additional numeric-data sections
    "results and standings", "final standings", "points table",
    "match results", "scorecard", "scoreline",
}

TABLE_MAX_PER_PAGE = 25
TABLE_FETCH_LIMIT  = 3

_RETRY_MAX    = 4
_RETRY_BASE   = 1.0
_RETRY_JITTER = 0.5

# ─────────────────────────────────────────────────────────────────────
# Numeric-content detection helpers
# ─────────────────────────────────────────────────────────────────────

# Patterns for lap times (1:35.220), scores (3-1), percentages, decimal
# numbers that appear in results/standings tables.
_NUMERIC_PATTERNS = re.compile(
    r'\b\d{1,2}:\d{2}\.\d{2,3}\b'   # lap/race time  e.g. 1:35.220
    r'|\b\d+\.\d+\b'                  # decimal number e.g. 9.87
    r'|\b\d+[–\-]\d+\b'              # score          e.g. 3-1
    r'|\b\d{4}\b',                    # bare year used as a value
)


def _numeric_bonus(text: str) -> float:
    """
    Extra score boost for table chunks that contain numerical data matching
    patterns common in results/standings claims.  Capped at 0.10 so it
    cannot override a fundamentally irrelevant chunk.

    A chunk with 3+ numeric matches gets the full bonus; fewer gets a
    proportional fraction.
    """
    matches = _NUMERIC_PATTERNS.findall(text)
    return min(0.10, len(matches) * 0.03)


class KILTWikipediaRetriever:
    """
    Wikipedia retriever for FEVEROUS / encyclopaedic fact-checking.

    v7 key changes over v6
    ──────────────────────
    1. NUMERIC TABLE BOOST  (_numeric_bonus)
       Table chunks that contain lap times, scores, standings numbers, or
       decimal values receive an additional score bonus (up to +0.10) on top
       of the EXACT_TITLE_BONUS already applied to priority-page chunks.
       This ensures that the qualifying results row with "1:35.220" is
       ranked higher than a prose paragraph that only mentions pole position,
       which was the root cause of the 48413 (Schumacher) retrieval failure.

    2. EXPANDED PRIORITY SECTION KEYWORDS
       Added "results and standings", "final standings", "points table",
       "match results", "scorecard", "scoreline" so that Wikipedia sections
       containing race results, sports standings, and scoreboards are flagged
       as priority sections and their chunks receive the is_priority=True flag
       during prose extraction.

    3. TABLE FETCH LIMIT INCREASE (3 → 5)
       The original cap of 3 was set for API traffic reasons.  Raised to 5
       because FEVEROUS numerical claims often span both the primary event
       article AND the athlete/team article (e.g. the 2001 Malaysian GP article
       AND the Michael Schumacher article both contain qualifying data).
       The backoff logic already handles 429s gracefully.

    All v6 behaviour retained.
    """

    ALPHA             = 0.6
    EXACT_TITLE_BONUS = 0.15

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info("Initializing KILT Wikipedia Retriever (v7)...")
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=HEADERS["User-Agent"],
            language="en",
        )
        self._query_cache:   Dict[str, List[Evidence]] = {}
        self._passage_cache: Dict[str, str]            = {}
        self._table_cache:   Dict[str, List[dict]]     = {}
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
                "from_table":  False,
            })
        return chunks

    # ─────────────────────────────────────────────────────────────────
    # Scoring helpers
    # ─────────────────────────────────────────────────────────────────

    def _keyword_score(self, chunk_text: str, query_tokens: List[str]) -> float:
        if not query_tokens:
            return 0.0
        chunk_lower = chunk_text.lower()
        hits = sum(1 for t in query_tokens if t in chunk_lower)
        return hits / len(query_tokens)

    def _tokenise_query(self, query: str) -> List[str]:
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

    def _extract_multi_word_entities(self, text: str) -> List[str]:
        pattern = (r'[A-Z][a-zA-Z]+'
                   r'(?:\s(?:of|the|in|and|for|de|la|le|al|von|van)\s'
                   r'[A-Z][a-zA-Z]+|\s[A-Z][a-zA-Z]+)+')
        found = re.findall(pattern, text)
        aka = re.findall(
            r'a\.?k\.?a\.?\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)', text)
        return list(dict.fromkeys(found + aka))

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

    @staticmethod
    def _api_get_with_backoff(params: dict, timeout: int = 10) -> requests.Response:
        for attempt in range(_RETRY_MAX + 1):
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 429:
                if attempt == _RETRY_MAX:
                    r.raise_for_status()
                wait = _RETRY_BASE * (2 ** attempt) + random.uniform(0, _RETRY_JITTER)
                logger.debug(f"429 from Wikipedia API – retrying in {wait:.1f}s "
                             f"(attempt {attempt + 1}/{_RETRY_MAX})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        r.raise_for_status()
        return r

    def _search_api(self, query: str, srlimit: int = 10) -> List[str]:
        try:
            r = self._api_get_with_backoff({
                "action":"query","list":"search",
                "srsearch":query,"srlimit":srlimit,
                "format":"json","utf8":1,
            })
            return [x["title"] for x in r.json().get("query",{}).get("search",[])]
        except Exception as e:
            logger.warning(f"Search API failed for '{query}': {e}")
            return []

    def _prefix_search(self, prefix: str, limit: int = 5) -> List[str]:
        try:
            r = self._api_get_with_backoff({
                "action":"query","list":"prefixsearch",
                "pssearch":prefix,"pslimit":limit,
                "format":"json",
            })
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

    def _get_all_candidate_titles(self, query: str) -> Tuple[List[str], List[str]]:
        seen: set                  = set()
        all_titles: List[str]      = []
        priority_titles: List[str] = []

        def add(t: str, is_priority: bool = False):
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                all_titles.append(t)
                if is_priority:
                    priority_titles.append(t)

        multi_entities = self._extract_multi_word_entities(query)
        for phrase in multi_entities[:8]:
            add(phrase, is_priority=True)
            for t in self._search_api(f'"{phrase}"', srlimit=5):
                add(t, is_priority=True)
            for t in self._prefix_search(phrase, limit=3):
                add(t, is_priority=True)

        nouns = self._extract_proper_nouns(query)
        for t in self._build_direct_probe_titles(query):
            is_pri = any(t.startswith(n) or t.endswith(n) for n in nouns[:3])
            add(t, is_priority=is_pri)

        years = self._extract_years(query)
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

        return all_titles, priority_titles

    # ─────────────────────────────────────────────────────────────────
    # Page fetching  (prose)
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

            if page.text:
                for chunk in self._chunk_text(page.text, page_id, title,
                                               is_priority=False):
                    passages.append(chunk)
                    self._passage_cache[chunk["id"]] = chunk["text"]
                    if current_count + len(passages) >= max_passages:
                        return passages

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
    # Page fetching  (tables & infoboxes)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_wiki_markup(text: str) -> str:
        text = re.sub(r'<(ref|small|sup|sub|nowiki)[^>]*>.*?</\1>', '', text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        for _ in range(2):
            text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        text = re.sub(r'\[\[(?:[^\[\]|]*\|)?([^\[\]]+)\]\]', r'\1', text)
        text = re.sub(r"'''|''", '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _parse_infoboxes(wikitext: str, page_id, title: str) -> List[dict]:
        chunks: List[dict] = []
        pattern = re.compile(
            r'\{\{(?:Infobox|Taxobox|Speciesbox|Automatic[ _]taxobox'
            r'|Speciesbox)[^\n]*\n(.*?)\n\}\}',
            re.DOTALL | re.IGNORECASE,
        )
        skip_keys = {
            'image', 'image_caption', 'alt', 'caption', 'map', 'logo',
            'image_size', 'map_caption', 'width', 'border', 'map_image',
            'image_map', 'blank_emblem', 'blank_map',
        }
        clean = KILTWikipediaRetriever._clean_wiki_markup
        for box_idx, match in enumerate(pattern.finditer(wikitext)):
            content = match.group(1)
            pairs: List[str] = []
            for line in content.split('\n'):
                m = re.match(r'\|\s*(\w[\w\s]*?)\s*=\s*(.*)', line)
                if not m:
                    continue
                key = m.group(1).strip()
                if key.lower() in skip_keys:
                    continue
                val = clean(m.group(2))
                if val:
                    pairs.append(f'{key}: {val}')
            for i in range(0, len(pairs), 10):
                text = ' | '.join(pairs[i:i + 10])
                if len(text) > 15:
                    chunks.append({
                        'id':          f'{page_id}_infobox{box_idx}_{i}',
                        'title':       f'{title} [Infobox]',
                        'text':        text,
                        'is_priority': True,
                        'from_table':  True,
                    })
        return chunks

    @staticmethod
    def _parse_wikitables(wikitext: str, page_id, title: str) -> List[dict]:
        chunks: List[dict] = []
        clean = KILTWikipediaRetriever._clean_wiki_markup
        table_pattern = re.compile(r'^\{\|.*?^\|\}', re.DOTALL | re.MULTILINE)

        def _strip_cell_attr(raw: str) -> str:
            m = re.match(r'^[^|=\[{]*=[^|]*\|(?!\|)(.*)', raw, re.DOTALL)
            return m.group(1).strip() if m else raw

        for t_idx, table_match in enumerate(table_pattern.finditer(wikitext)):
            table_text = table_match.group()
            lines = table_text.split('\n')
            headers: List[str]     = []
            current_row: List[str] = []
            rows: List[List[str]]  = []

            for line in lines:
                stripped = line.strip()
                if stripped.startswith('|+'):
                    continue
                if stripped.startswith('!'):
                    raw_headers = re.split(r'!!', stripped.lstrip('!'))
                    for h in raw_headers:
                        h = clean(_strip_cell_attr(h))
                        if h:
                            headers.append(h)
                    continue
                if stripped.startswith('|-'):
                    if current_row:
                        rows.append(current_row)
                        current_row = []
                    continue
                if (stripped.startswith('|')
                        and not stripped.startswith('|}')
                        and not stripped.startswith('{|')):
                    raw_cells = re.split(r'\|\|', stripped.lstrip('|'))
                    for cell in raw_cells:
                        cell = clean(_strip_cell_attr(cell))
                        if cell:
                            current_row.append(cell)
                    continue

            if current_row:
                rows.append(current_row)

            for r_idx, row in enumerate(rows):
                row = [c for c in row if c.strip()]
                if not row:
                    continue
                if headers and len(headers) == len(row):
                    text = ' | '.join(
                        f'{h}: {v}'
                        for h, v in zip(headers, row)
                        if v.strip()
                    )
                else:
                    text = ' | '.join(row)

                if len(text) > 15:
                    chunks.append({
                        'id':          f'{page_id}_tbl{t_idx}_row{r_idx}',
                        'title':       f'{title} [Table]',
                        'text':        text,
                        'is_priority': True,
                        'from_table':  True,
                    })

        return chunks

    def _fetch_page_tables(self, title: str) -> List[dict]:
        cache_key = title.strip().lower()
        if cache_key in self._table_cache:
            return self._table_cache[cache_key]

        try:
            resp = self._api_get_with_backoff({
                "action": "parse",
                "page":   title,
                "prop":   "wikitext|revid",
                "format": "json",
                "utf8":   1,
            }, timeout=15)
            data = resp.json()

            if "error" in data:
                logger.debug(f"[tables] parse API error for '{title}': {data['error']}")
                self._table_cache[cache_key] = []
                return []

            parse_obj = data.get("parse", {})
            wikitext  = parse_obj.get("wikitext", {}).get("*", "")
            page_id   = parse_obj.get("pageid", title)

            if not wikitext:
                self._table_cache[cache_key] = []
                return []

            infobox_chunks = self._parse_infoboxes(wikitext, page_id, title)
            table_chunks   = self._parse_wikitables(wikitext, page_id, title)

            table_budget = max(0, TABLE_MAX_PER_PAGE - len(infobox_chunks))
            all_chunks   = infobox_chunks + table_chunks[:table_budget]

            logger.debug(
                f"[tables] '{title}': "
                f"{len(infobox_chunks)} infobox + {len(table_chunks)} table rows "
                f"→ {len(all_chunks)} chunks (cap {TABLE_MAX_PER_PAGE})"
            )
            self._table_cache[cache_key] = all_chunks
            return all_chunks

        except Exception as e:
            logger.warning(f"[tables] Failed to fetch tables for '{title}': {e}")
            self._table_cache[cache_key] = []
            return []

    # ─────────────────────────────────────────────────────────────────
    # Fetch & Score  (v7 — numeric table boost)
    # ─────────────────────────────────────────────────────────────────

    def _fetch_and_score(self, query: str, top_k: int) -> List[Evidence]:
        candidate_titles, priority_titles = self._get_all_candidate_titles(query)
        if not candidate_titles:
            return []

        priority_set = {t.lower() for t in priority_titles}

        all_passages: List[dict] = []
        MAX_PASSAGES = 1500

        for title in candidate_titles:
            if len(all_passages) >= MAX_PASSAGES:
                break
            new_passages = self._fetch_page_passages(
                title, MAX_PASSAGES, len(all_passages))
            is_pri_page = title.lower() in priority_set
            for p in new_passages:
                p["from_priority_page"] = is_pri_page
            all_passages.extend(new_passages)

        # v7: raised TABLE_FETCH_LIMIT from 3 to 5 — numerical claims often
        # span both the event article and the athlete/team article.
        for title in priority_titles[:5]:
            table_chunks = self._fetch_page_tables(title)
            for chunk in table_chunks:
                chunk["from_priority_page"] = True
            all_passages.extend(table_chunks)

        if not all_passages:
            logger.warning(f"No passages for: '{query}'")
            return []

        logger.info(
            f"Scoring {len(all_passages)} passages "
            f"({sum(1 for p in all_passages if p.get('from_table'))} from tables) "
            f"for: '{query[:70]}'"
        )

        query_emb    = self.model.encode([query], normalize_embeddings=True)[0]
        passage_embs = self.model.encode(
            [p["text"] for p in all_passages],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        sem_scores = np.dot(passage_embs, query_emb)

        query_tokens = self._tokenise_query(query)
        kw_scores = np.array([
            self._keyword_score(p["text"], query_tokens)
            for p in all_passages
        ])

        hybrid = self.ALPHA * sem_scores + (1 - self.ALPHA) * kw_scores

        # ── Bonuses ──────────────────────────────────────────────────
        for i, p in enumerate(all_passages):
            bonus = 0.0
            # Priority-page bonus (unchanged from v6)
            if p.get("from_priority_page"):
                bonus += self.EXACT_TITLE_BONUS
            # v7 NEW: numeric-content bonus for table chunks
            # This is specifically to surface qualifying timesheets, race results,
            # and standings rows that contain lap times / scores not in prose.
            if p.get("from_table"):
                bonus += _numeric_bonus(p["text"])
            hybrid[i] = min(1.0, float(hybrid[i]) + bonus)

        # ── Top-k with per-page diversity cap ────────────────────────
        sorted_indices = np.argsort(hybrid)[::-1]

        selected: List[int]         = []
        page_counts: Dict[str, int] = {}
        MAX_PER_PAGE = max(2, top_k // 2)

        def _page_root(passage: dict) -> str:
            raw = str(passage["id"]).split("_sec_")[0].split("_")[0]
            return (raw + "_tbl") if passage.get("from_table") else raw

        priority_slots = min(top_k // 2, 3)
        for idx in sorted_indices:
            if len([s for s in selected
                    if all_passages[s].get("is_priority")]) >= priority_slots:
                break
            if all_passages[idx].get("is_priority"):
                selected.append(idx)

        for idx in sorted_indices:
            if len(selected) >= top_k:
                break
            if idx in selected:
                continue
            root = _page_root(all_passages[idx])
            if page_counts.get(root, 0) >= MAX_PER_PAGE:
                continue
            selected.append(idx)
            page_counts[root] = page_counts.get(root, 0) + 1

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