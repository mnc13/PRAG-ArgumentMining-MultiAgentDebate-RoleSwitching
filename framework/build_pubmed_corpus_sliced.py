import argparse
import gzip
import json
import os
import time
from datetime import date, datetime, timedelta

import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

from requests.exceptions import (
    ChunkedEncodingError,
    ReadTimeout,
    ConnectionError,
    HTTPError,
    RequestException,
)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# PubMed ESearch effectively limits access to the first 10,000 results per query.
# Keep a safety margin below 10k.
MAX_PER_QUERY = 9500


def parse_ymd(s: str) -> date:
    # Accept "YYYY/MM/DD" or "YYYY-MM-DD"
    s = s.strip().replace("-", "/")
    return datetime.strptime(s, "%Y/%m/%d").date()


def fmt_ymd(d: date) -> str:
    return d.strftime("%Y/%m/%d")


def get_with_retries(url, params, timeout, max_retries=8, backoff=2.0):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)

            # Retry on 429/5xx
            if r.status_code == 429 or (500 <= r.status_code <= 599):
                raise HTTPError(f"HTTP {r.status_code}", response=r)

            r.raise_for_status()
            return r

        except (ChunkedEncodingError, ReadTimeout, ConnectionError, HTTPError) as e:
            last_exc = e
            if attempt == max_retries:
                break

            sleep_s = backoff ** (attempt - 1)  # 1,2,4,8...
            code = ""
            if isinstance(e, HTTPError) and getattr(e, "response", None) is not None and e.response is not None:
                code = f" status={e.response.status_code}"

            # Respect Retry-After for 429 if present
            if isinstance(e, HTTPError) and getattr(e, "response", None) is not None and e.response is not None:
                if e.response.status_code == 429:
                    ra = e.response.headers.get("Retry-After")
                    if ra:
                        try:
                            sleep_s = max(sleep_s, float(ra))
                        except ValueError:
                            pass

            print(f"[warn] request failed ({type(e).__name__}{code}): retry {attempt}/{max_retries} in {sleep_s:.1f}s")
            time.sleep(sleep_s)

        except RequestException as e:
            last_exc = e
            if attempt == max_retries:
                break
            sleep_s = backoff ** (attempt - 1)
            print(f"[warn] request failed ({type(e).__name__}): retry {attempt}/{max_retries} in {sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise last_exc if last_exc else RuntimeError("Unknown request failure")


def esearch(term, mindate, maxdate, tool, email, api_key=""):
    url = f"{EUTILS}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": 0,
        "usehistory": "y",
        "datetype": "edat",
        "mindate": mindate,
        "maxdate": maxdate,
        "term": term,
        "tool": tool,
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    r = get_with_retries(url, params=params, timeout=60)
    data = r.json()["esearchresult"]
    return int(data["count"]), data["webenv"], data["querykey"]


def efetch_xml(webenv, query_key, retstart, retmax, tool, email, api_key=""):
    url = f"{EUTILS}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "query_key": query_key,
        "WebEnv": webenv,
        "retstart": retstart,
        "retmax": retmax,
        "tool": tool,
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key

    r = get_with_retries(url, params=params, timeout=180)
    return r.text


def get_text(node, path):
    if node is None:
        return ""
    x = node.find(path)
    return (x.text or "").strip() if x is not None and x.text else ""


def parse_article(pubmed_article):
    pmid = get_text(pubmed_article, ".//MedlineCitation/PMID")

    article = pubmed_article.find(".//Article")
    title = ""
    if article is not None:
        at = article.find("ArticleTitle")
        if at is not None:
            title = "".join(at.itertext()).strip()

    abstract = ""
    abstract_nodes = pubmed_article.findall(".//Abstract/AbstractText")
    if abstract_nodes:
        parts = []
        for n in abstract_nodes:
            label = (n.attrib.get("Label") or "").strip()
            txt = "".join(n.itertext()).strip()
            if not txt:
                continue
            parts.append(f"{label}: {txt}" if label else txt)
        abstract = "\n".join(parts).strip()

    journal = get_text(pubmed_article, ".//Journal/Title")

    year = get_text(pubmed_article, ".//PubDate/Year")
    medline_date = get_text(pubmed_article, ".//PubDate/MedlineDate")
    pub_year = year or (medline_date.split(" ")[0] if medline_date else "")

    doi = ""
    for aid in pubmed_article.findall(".//ArticleIdList/ArticleId"):
        if (aid.attrib.get("IdType") == "doi") and (aid.text or "").strip():
            doi = aid.text.strip()
            break

    authors = []
    for a in pubmed_article.findall(".//AuthorList/Author"):
        last = get_text(a, "LastName")
        initials = get_text(a, "Initials")
        if last:
            authors.append(f"{last} {initials}".strip())

    mesh_terms = []
    for mh in pubmed_article.findall(".//MeshHeading/DescriptorName"):
        if (mh.text or "").strip():
            mesh_terms.append(mh.text.strip())

    return {
        "source": "pubmed",
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": pub_year,
        "authors": authors,
        "mesh_terms": mesh_terms,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        "text_for_rag": (title + "\n\n" + abstract).strip(),
    }


def split_range_if_needed(start: date, end: date, term: str, tool: str, email: str, api_key: str):
    """
    Recursively split [start, end] until the PubMed hit count fits under MAX_PER_QUERY.
    Yields (start, end, count).
    """
    count, _, _ = esearch(term, fmt_ymd(start), fmt_ymd(end), tool, email, api_key)

    if count == 0:
        return

    if count <= MAX_PER_QUERY:
        yield (start, end, count)
        return

    if start >= end:
        # Extremely rare: >10k in one day. We'll still yield it, but fetch will only get first ~10k.
        print(f"[warn] {fmt_ymd(start)} has {count} hits (> {MAX_PER_QUERY}). You may not be able to fetch all of them via E-utilities.")
        yield (start, end, count)
        return

    mid = start + timedelta(days=(end - start).days // 2)
    yield from split_range_if_needed(start, mid, term, tool, email, api_key)
    yield from split_range_if_needed(mid + timedelta(days=1), end, term, tool, email, api_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="pubmed_slices", help="Folder to write per-slice .jsonl.gz files")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--mindate", default="2020/01/01")
    ap.add_argument("--maxdate", default="2024/12/31")
    ap.add_argument("--term", default='((COVID-19[Title/Abstract] OR SARS-CoV-2[Title/Abstract]) AND hasabstract[text])')
    ap.add_argument("--tool", required=True)
    ap.add_argument("--email", required=True)
    ap.add_argument("--api_key", default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    start = parse_ymd(args.mindate)
    end = parse_ymd(args.maxdate)

    # One big count just for info
    total_count, _, _ = esearch(args.term, fmt_ymd(start), fmt_ymd(end), args.tool, args.email, args.api_key)
    print(f"Total PubMed hits in full range: {total_count}")

    # Build safe slices under 10k
    slices = list(split_range_if_needed(start, end, args.term, args.tool, args.email, args.api_key))
    print(f"Planned slices: {len(slices)} (each <= ~{MAX_PER_QUERY} hits)")

    for (s, e, cnt) in slices:
        tag = f"{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}"
        out_path = os.path.join(args.out_dir, f"pubmed_{tag}.jsonl.gz")

        # Skip if already exists and has content
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[skip] {tag} already downloaded: {out_path}")
            continue

        print(f"\n[range] {fmt_ymd(s)} → {fmt_ymd(e)}  hits={cnt}  -> {out_path}")

        count, webenv, query_key = esearch(args.term, fmt_ymd(s), fmt_ymd(e), args.tool, args.email, args.api_key)

        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            for retstart in tqdm(range(0, min(count, 10000), args.batch), desc=f"Fetching {tag}"):
                xml_text = efetch_xml(webenv, query_key, retstart, args.batch, args.tool, args.email, args.api_key)

                # parse XML (retry fetch if parse fails)
                parsed = False
                for _ in range(3):
                    try:
                        root = ET.fromstring(xml_text)
                        parsed = True
                        break
                    except ET.ParseError:
                        print("[warn] XML parse error; refetching this batch...")
                        time.sleep(2.0)
                        xml_text = efetch_xml(webenv, query_key, retstart, args.batch, args.tool, args.email, args.api_key)

                if not parsed:
                    raise ET.ParseError("Failed to parse XML after retries")

                for pa in root.findall(".//PubmedArticle"):
                    rec = parse_article(pa)
                    if rec["text_for_rag"]:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                time.sleep(args.sleep)

    print("\nDone.")
    print(f"All slices are in: {args.out_dir}")
    print("Next step: merge slices into one JSONL (I can give you the merge script if you want).")


if __name__ == "__main__":
    main()
