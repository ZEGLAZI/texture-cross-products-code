"""Constituer le lot iNaturalist 2026 avant tout calcul des scores."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import time
from urllib import parse, request, error
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import numpy as np
from PIL import Image
from commun import DATA, digest, read_csv, read_npz, write_csv, write_json
from reference.signatures import signatures

TAXA = {57983: ("tournesol", 9), 967514: ("pissenlit", 12), 55563: ("pâquerette", 10)}
LICENSES = {"cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa"}
FIELDS = (
    "classe",
    "code",
    "taxon_filtre",
    "taxon_observe",
    "nom_taxon",
    "id",
    "photo",
    "compte",
    "date",
    "licence",
    "credit",
    "url",
    "ordre",
)
SIG = ("sha256_jpeg", "sha256_rgb", "phash", "phash_miroir")
API = "https://api.inaturalist.org/v1/observations"


def get(url, timeout=25, attempts=3):
    for step in range(attempts):
        try:
            req = request.Request(
                url, headers={"User-Agent": "HSMT-flower-descriptor-research/1.0"}
            )
            with request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except OSError as exc:
            if (
                isinstance(exc, error.HTTPError) and exc.code in (404, 410)
            ) or step + 1 == attempts:
                raise
            time.sleep(1.2 * 2**step)


def page(taxon, n, store):
    path = store / f"{taxon}_{n}.json"
    if path.is_file():
        return json.loads(path.read_text())
    params = {
        "taxon_id": taxon,
        "quality_grade": "research",
        "term_id": 12,
        "term_value_id": 13,
        "d1": "2015-01-01",
        "d2": "2017-12-31",
        "photo_license": "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa",
        "order_by": "id",
        "order": "asc",
        "per_page": 200,
        "page": n,
    }
    source = json.loads(get(API + "?" + parse.urlencode(params), timeout=50))
    records = []
    for row in source["results"]:
        tax = row.get("taxon") or {}
        photos = [
            {k: p.get(k) for k in ("id", "license_code", "url", "attribution")}
            for p in row.get("photos", [])
        ]
        records.append(
            {
                "id": row["id"],
                "date": row["observed_on"],
                "compte": row["user"]["id"],
                "statut": row["quality_grade"],
                "annotations": [
                    [a["controlled_attribute_id"], a["controlled_value_id"]]
                    for a in row["annotations"]
                ],
                "taxon": tax.get("id"),
                "nom_taxon": tax.get("name"),
                "ancetres": tax.get("ancestor_ids", []),
                "photos": photos,
            }
        )
    result = {"total": source["total_results"], "page": source["page"], "records": records}
    assert result["page"] == n
    temp = path.with_suffix(".part")
    write_json(temp, result)
    temp.replace(path)
    time.sleep(1.05)
    return result


def candidate(taxon, row):
    if (
        row["statut"] != "research"
        or [12, 13] not in row["annotations"]
        or not ("2015-01-01" <= row["date"] <= "2017-12-31")
    ):
        return None
    if row["taxon"] != taxon and taxon not in (row["ancetres"] or []):
        return None
    for p in row["photos"]:
        if p["license_code"] not in LICENSES or not p["url"]:
            continue
        url = re.sub(r"/square\.(?=[^/]+$)", "/medium.", p["url"])
        if "/medium." not in url:
            continue
        key = sha256(f"HSMT-new-observations-2015-2017-v1|{taxon}|{row['id']}".encode()).hexdigest()
        return dict(
            zip(
                FIELDS,
                (
                    TAXA[taxon][0],
                    TAXA[taxon][1],
                    taxon,
                    row["taxon"],
                    row["nom_taxon"],
                    row["id"],
                    p["id"],
                    row["compte"],
                    row["date"],
                    p["license_code"],
                    p["attribution"] or "",
                    url,
                    key,
                ),
            )
        )
    return None


def inventory(cache, data):
    cache.mkdir(exist_ok=True, parents=True)
    data.mkdir(exist_ok=True, parents=True)
    candidates = []
    summary = {}
    filenames = []
    for taxon in TAXA:
        first = page(taxon, 1, cache)
        total = first["total"]
        assert 0 < total <= 10000
        observed = set()
        rows = []
        for n in range(1, (total + 199) // 200 + 1):
            out = first if n == 1 else page(taxon, n, cache)
            assert out["total"] == total
            filenames.append(f"{taxon}_{n}.json")
            for record in out["records"]:
                assert record["id"] not in observed
                observed.add(record["id"])
                entry = candidate(taxon, record)
                if entry:
                    rows.append(entry)
        assert len(observed) == total
        candidates.extend(sorted(rows, key=lambda r: (r["ordre"], r["id"])))
        summary[taxon] = {"observations": total, "candidats": len(rows)}
        print("Recensement", taxon, summary[taxon], flush=True)
    write_csv(data / "candidats.csv", candidates, FIELDS)
    with ZipFile(data / "instantane_inaturalist.zip", "w") as z:
        for name in filenames:
            item = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            z.writestr(
                item, (cache / name).read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=7
            )
    write_json(
        data / "recensement.json",
        {
            "classes": summary,
            "pages": len(filenames),
            "sha_candidats": digest(data / "candidats.csv"),
        },
    )


def download(row, photos):
    path = photos / (str(row["photo"]) + ".jpg")
    try:
        raw = path.read_bytes() if path.exists() else get(row["url"], timeout=14, attempts=2)
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            width, height = im.size
        sig = signatures(raw)
        if not path.exists():
            tmp = path.with_suffix(".part")
            tmp.write_bytes(raw)
            tmp.replace(path)
        return sig, width, height, None
    except (OSError, ValueError) as exc:
        return None, 0, 0, type(exc).__name__


def dist(hashes, pair):
    return min(int(np.bitwise_count(hashes ^ h).min()) for h in pair) if len(hashes) else 65


def select(data, photos):
    photos.mkdir(exist_ok=True, parents=True)
    candidates = read_csv(data / "candidats.csv")
    past = read_csv(data / "observations_anterieures.csv")
    old = read_npz(data / "empreintes_anterieures.npz")
    assert len(old["hashes"]) == 9211
    previous = {key: {r[key] for r in past} for key in ("compte", "id", "photo")}
    seen = {key: set() for key in previous}
    exact = [set(old["sha_j"]), set(old["sha_p"])]
    new = np.zeros((750, 2), np.uint64)
    kept = []
    removed = []
    total = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for taxon in TAXA:
            ranked = [r for r in candidates if int(r["taxon_filtre"]) == taxon]
            n = 0
            next_row = 0
            pending = {}
            reasons = Counter()
            assert [r["ordre"] for r in ranked] == sorted(r["ordre"] for r in ranked)
            for pos, row in enumerate(ranked):
                while next_row < min(pos + 28, len(ranked)):
                    item = ranked[next_row]
                    next_row += 1
                    if (
                        not any(item[k] in previous[k] or item[k] in seen[k] for k in seen)
                        and item["photo"] not in pending
                    ):
                        pending[item["photo"]] = pool.submit(download, item, photos)
                reason = next((k + "_antérieur" for k in previous if row[k] in previous[k]), "")
                if not reason:
                    reason = next((k + "_déjà_retenu" for k in seen if row[k] in seen[k]), "")
                if not reason:
                    future = pending.pop(row["photo"], None)
                    sig, width, height, err = (
                        future.result() if future is not None else download(row, photos)
                    )
                    if err:
                        reason = "image_indisponible"
                    else:
                        pair = np.array([int(sig[k], 16) for k in SIG[2:]], dtype=np.uint64)
                        d0 = dist(old["hashes"], pair)
                        d1 = dist(new[: len(kept)], pair)
                        if min(d0, d1) <= 6 or sig[SIG[0]] in exact[0] or sig[SIG[1]] in exact[1]:
                            reason = "image_identique_ou_proche"
                if reason:
                    removed.append(
                        {"taxon": taxon, "id": row["id"], "photo": row["photo"], "motif": reason}
                    )
                    reasons[reason] += 1
                    continue
                for k in seen:
                    seen[k].add(row[k])
                for i in range(2):
                    exact[i].add(sig[SIG[i]])
                new[len(kept)] = pair
                kept.append(
                    row
                    | sig
                    | {"largeur": width, "hauteur": height, "d_anterieur": d0, "d_lot": d1}
                )
                n += 1
                if n % 100 == 0:
                    print(TAXA[taxon][0], n, "photos", flush=True)
                if n == 200:
                    break
            for future in pending.values():
                future.cancel()
            total[TAXA[taxon][0]] = {"retenues": n, "exclusions": dict(reasons)}
            print(total[TAXA[taxon][0]], flush=True)
    write_csv(
        data / "nouveau_manifeste.csv",
        kept,
        (*FIELDS, *SIG, "largeur", "hauteur", "d_anterieur", "d_lot"),
    )
    write_csv(data / "exclusions.csv", removed, ("taxon", "id", "photo", "motif"))
    assert len(kept) == len(seen["compte"]) == len(seen["id"]) == len(seen["photo"])
    assert not any(previous[k] & seen[k] for k in seen)
    write_json(
        data / "selection.json",
        {
            "classes": total,
            "n": len(kept),
            "comptes": len(seen["compte"]),
            "references": len(old["hashes"]),
            "min_distance_anterieure": min(r["d_anterieur"] for r in kept),
            "min_distance_interne": min(r["d_lot"] for r in kept),
            "sha_manifeste": digest(data / "nouveau_manifeste.csv"),
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["recenser", "choisir", "tout"])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--donnees", type=Path, default=DATA)
    a = ap.parse_args()
    if a.action in ("tout", "recenser"):
        inventory(a.cache, a.donnees)
    if a.action in ("tout", "choisir"):
        select(a.donnees, a.photos)
