import argparse
import gzip
import json
import time
import os
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


def get_with_retries(url, params, timeout, max_retries=8, backoff=2.0):
    """
    Robust GET with retries for common transient failures:
    - ChunkedEncodingError (Response ended prematurely)
    - timeouts / connection drops
    - HTTP 429 and 5xx
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            # Retry on 429 / 5xx
            if r.status_code == 429 or (500 <= r.status_code <= 599):
                raise HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r

        except (ChunkedEncodingError, ReadTimeout, ConnectionError, HTTPError) as e:
            last_exc = e
            if attempt == max_retries:
                break

            # exponential backoff: 1,2,4,8,... seconds
            sleep_s = backoff ** (attempt - 1)

            # If 429, respect Retry-After if present
            if isinstance(e, HTTPError) and getattr(e, "response", None) is not None:
                resp = e.response
                if resp is not None and resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            sleep_s = max(sleep_s, float(ra))
                        except ValueError:
                            pass

            print(
                f"[warn] request failed ({type(e).__name__}): "
                f"retry {attempt}/{max_retries} in {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)

        except RequestException as e:
            # other requests errors: do not loop forever, but retry a few times
            last_exc = e
            if attempt == max_retries:
                break
            sleep_s = backoff ** (attempt - 1)
            print(
                f"[warn] request failed ({type(e).__name__}): "
                f"retry {attempt}/{max_retries} in {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)

    # out of retries
    raise last_exc if last_exc else RuntimeError("Unknown request failure")


def esearch(term, mindate, maxdate, tool, email, api_key=""):
    url = f"{EUTILS}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": 0,
        "usehistory": "y",
        "datetype": "pdat",
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

    r = get_with_retries(url, params=params, timeout=180)  # longer timeout for big batches
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


def write_checkpoint(path, retstart):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(retstart))
    except Exception:
        pass


def read_checkpoint(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pubmed_covid_2020_2024.jsonl.gz")
    ap.add_argument("--checkpoint", default="", help="checkpoint file path (default: OUT + .ckpt)")
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint if available")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.34, help="seconds between requests (~3 req/sec)")
    ap.add_argument("--start", type=int, default=0, help="retstart offset (manual resume)")
    ap.add_argument("--mindate", default="2020/01/01")
    ap.add_argument("--maxdate", default="2024/01/01")
    ap.add_argument("--term", default='(COVID-19[All Fields] OR SARS-CoV-2[All Fields])')
    ap.add_argument("--tool", required=True)
    ap.add_argument("--email", required=True)
    ap.add_argument("--api_key", default="")
    args = ap.parse_args()

    ckpt_path = args.checkpoint or (args.out + ".ckpt")

    # Resume priority: --resume uses checkpoint; otherwise use --start
    if args.resume:
        ckpt_val = read_checkpoint(ckpt_path)
        if ckpt_val is not None:
            print(f"[info] resuming from checkpoint: retstart={ckpt_val}")
            args.start = ckpt_val

    count, webenv, query_key = esearch(
        args.term, args.mindate, args.maxdate, args.tool, args.email, args.api_key
    )
    print(f"Total PubMed hits: {count}")

    # Write fresh if start==0, otherwise append
    mode = "wt" if args.start == 0 else "at"

    with gzip.open(args.out, mode, encoding="utf-8") as f:
        for retstart in tqdm(range(args.start, count, args.batch), desc="Fetching batches"):
            try:
                xml_text = efetch_xml(
                    webenv, query_key, retstart, args.batch, args.tool, args.email, args.api_key
                )

                # Sometimes partial downloads create broken XML; retry a couple times
                parsed = False
                for _ in range(3):
                    try:
                        root = ET.fromstring(xml_text)
                        parsed = True
                        break
                    except ET.ParseError:
                        print("[warn] XML parse error; refetching this batch...")
                        time.sleep(2.0)
                        xml_text = efetch_xml(
                            webenv, query_key, retstart, args.batch, args.tool, args.email, args.api_key
                        )

                if not parsed:
                    raise ET.ParseError("Failed to parse XML after retries")

                for pa in root.findall(".//PubmedArticle"):
                    rec = parse_article(pa)
                    if rec["text_for_rag"]:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Checkpoint = next retstart to run
                next_start = retstart + args.batch
                write_checkpoint(ckpt_path, next_start)

                time.sleep(args.sleep)

            except Exception as e:
                print("\n[error] crashed at retstart =", retstart)
                print("Resume with either:")
                print(f"  --start {retstart}")
                print(f"or")
                print(f"  --resume  (checkpoint: {ckpt_path})")
                raise

    print(f"Done. Wrote: {args.out}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
