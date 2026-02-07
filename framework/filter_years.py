import gzip, json, argparse, re

def safe_year(rec):
    y = (rec.get("year") or "").strip()
    m = re.match(r"^\d{4}$", y)
    return int(y) if m else None

ap = argparse.ArgumentParser()
ap.add_argument("--infile", default="pubmed_covid_2020_2024_edat.jsonl.gz")
ap.add_argument("--out", default="pubmed_covid_2020_2024_edat_upto_2024.jsonl.gz")
ap.add_argument("--miny", type=int, default=2020)
ap.add_argument("--maxy", type=int, default=2024)
args = ap.parse_args()

kept = 0
total = 0
with gzip.open(args.infile, "rt", encoding="utf-8") as fin, gzip.open(args.out, "wt", encoding="utf-8") as fout:
    for line in fin:
        total += 1
        rec = json.loads(line)
        y = safe_year(rec)
        if y is None:
            continue
        if args.miny <= y <= args.maxy:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

print("Total:", total)
print("Kept:", kept)
print("Output:", args.out)
