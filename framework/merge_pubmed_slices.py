import argparse
import gzip
import json
import os
import glob

def iter_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="pubmed_slices_edat")
    ap.add_argument("--out", default="pubmed_covid_2020_2024_edat.jsonl.gz")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.jsonl.gz")))
    if not files:
        raise SystemExit(f"No .jsonl.gz files found in {args.in_dir}")

    seen = set()
    written = 0

    with gzip.open(args.out, "wt", encoding="utf-8") as out_f:
        for fp in files:
            for rec in iter_jsonl_gz(fp):
                pmid = rec.get("pmid") or ""
                # If PMID missing, keep it anyway but avoid key collisions
                key = pmid if pmid else f"NO_PMID::{rec.get('title','')[:80]}::{rec.get('year','')}"
                if key in seen:
                    continue
                seen.add(key)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Input slice files: {len(files)}")
    print(f"Unique records written: {written}")
    print(f"Output: {args.out}")

if __name__ == "__main__":
    main()
