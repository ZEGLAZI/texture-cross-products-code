"""Audit the 720 attempted source rows and summarize the original clothing decisions."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("http", "connection", "decode", "geometry", "label_conflict", "duplicate", "dtd_overlap", "retained")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def disposition(row, retained, exclusions):
    status = row["status"]
    if status == "ok":
        if (row["key"], row["class"]) in retained:
            return "retained", "retained"
        note = exclusions[row["key"]]
        if note == "conflicting class annotations":
            return "label_conflict", note
        if note.endswith(".jpg"):
            return "dtd_overlap", note
        if len(note) == 64:
            return "duplicate", note
    if status.startswith("HTTP Error"):
        return "http", status
    if status.startswith("<urlopen error"):
        return "connection", status
    if status.startswith("cannot identify image"):
        return "decode", "undecodable response (original log in clothing/data/attempts.csv)"
    if status in {"aspect", "size"}:
        return "geometry", status
    raise ValueError(f"Unknown disposition {status!r}")


def main(out):
    out.mkdir(exist_ok=True, parents=True)
    data = ROOT / "clothing/data"
    exclusion = dict(json.loads((data / "exclusions.json").read_text()))
    retained = {(r["key"],r["class"]): r for r in read(data / "sources.csv")}
    attempts = read(data / "attempts.csv")
    provider = read(data / "manifest.csv")
    snapshot = {(r["URL"],r["Class Name"]) for r in provider}
    counts = Counter()
    ledger = []
    for row in attempts:
        key, label, url = row["key"], row["class"], row["URL"]
        assert (url,label) in snapshot
        assert hashlib.sha256(("HSMT-textile-v1|"+url).encode()).hexdigest() == key
        category, detail = disposition(row, retained, exclusion)
        counts[label,category] += 1
        ledger.append(dict(key=key, pattern=label, url=url, hostname=urlsplit(url).hostname,
                           outcome=category, detail=detail,
                           crop_sha256=retained.get((key,label),{}).get("sha256_rgb224","")))
    assert len(ledger) == len({(r["key"],r["pattern"]) for r in ledger}) == 720
    assert sum(v for (label,g),v in counts.items() if g == "retained") == len(retained) == 211
    assert 216 == sum(row["status"] == "ok" for row in attempts)
    write(out/"clothing_attempts.csv",ledger)
    classes = sorted({r["pattern"] for r in ledger})
    overview = [dict(pattern=c,attempted=120,**{g:counts[c,g] for g in GROUPS}) for c in classes]
    assert all(sum(r[g] for g in GROUPS)==r["attempted"] for r in overview)
    write(out/"clothing_attrition.csv",overview)
    class_rows = read(ROOT/"clothing/results/by_class.csv")
    effects = []
    for r in class_rows:
        change = float(r["squares"])-float(r["crosses"])
        effects.append(dict(pattern=r["class"],photos=int(r["n_sources"]),squares_pct=float(r["squares"]),
                            products_pct=float(r["crosses"]),difference_pp=change,
                            contribution_to_macro_pp=change/len(class_rows)))
    write(out/"clothing_by_class.csv",effects)
    total = sum(r["contribution_to_macro_pp"] for r in effects)
    details = {}
    for excluded in ({"solid"},{"solid","dotted"}):
        changes = [r["difference_pp"] for r in effects if r["pattern"] not in excluded]
        details["exclude_"+"_".join(sorted(excluded))] = sum(changes)/len(changes)
    details.update(share_solid_dotted_pct=100*sum(r["contribution_to_macro_pp"] for r in effects if r["pattern"] in {"solid","dotted"})/total)
    details["snapshot"] = {"provider_manifest_sha256": hashlib.sha256((data/"manifest.csv").read_bytes()).hexdigest(),
                           "manifest_rows":len(provider), "attempts_sha256": hashlib.sha256((data/"attempts.csv").read_bytes()).hexdigest(),
                           "retained_sources_sha256":hashlib.sha256((data/"sources.csv").read_bytes()).hexdigest(),
                           "snapshot_date":"2026-11-19"}
    saved = read(ROOT/"review2/results/aggregate/selected_settings.csv")
    freq = Counter((r["method"],r["profile"],r["gamma"],r["penalty"]) for r in saved if r["fold"].startswith("Textile_"))
    assert all(sum(v for key,v in freq.items() if key[0]==m)==150 for m in {x[0] for x in freq})
    write(out/"clothing_selection_frequencies.csv",
          [dict(method=k[0],profile=k[1],gamma=k[2],ridge=k[3],folds=v) for k,v in sorted(freq.items())])
    (out/"clothing_summary.json").write_text(json.dumps(details,indent=2)+"\n")
    print(json.dumps(details))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    main(parser.parse_args().output)
