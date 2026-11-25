"""Check released feature provenance and deterministic decision examples.

This checks the released arrays, identities and stored decisions. It does not
claim that provider photographs were downloaded or re-extracted in this run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from numerical_release import ReleasedPayload
ARCHIVE_DIGEST = "ad1a3c2cda703498ca2bca2e2f5fa6718ad071931a0d5dbd8869c4f46dfbf385"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check(root: Path) -> dict:
    with ReleasedPayload(root) as bundle:
        release_verification = bundle.verification
        with np.load(io.BytesIO(bundle.read("historical/dtd/results/identites.npz"))) as z:
            dtd_ids, dtd_labels = z["ids"].copy(), z["y"].copy()
        kth_records = list(csv.DictReader(io.StringIO(
            bundle.read("historical/external/data/kth_images.csv").decode("utf-8"))))
        kth_ids = np.asarray([row["chemin"] for row in kth_records])
        identity_counts = {}
        transformer_hashes = {}
        for corpus, ids in (("DTD", dtd_ids), ("KTH", kth_ids)):
            with np.load(root / "pixel_check" / f"{corpus}_colour.npz") as z:
                np.testing.assert_array_equal(z["ids"], ids)
                require(set(("F7", "F8", "rgb")).issubset(z.files), "Missing colour blocks")
                identity_counts[corpus] = len(ids)
            expected = json.loads((root / "pixel_check" / f"{corpus}_transformer_hashes.json").read_text())
            name = f"cross_representation/features/{corpus}_features.npz"
            with np.load(io.BytesIO(bundle.read(name))) as z:
                require(set(z.files) == set(expected), f"{corpus}: transformer fields differ")
                np.testing.assert_array_equal(z["ids"], ids)
                for key, wanted in expected.items():
                    require(array_digest(z[key]) == wanted, f"{corpus}/{key}: array digest differs")
            transformer_hashes[corpus] = expected

    stage_sources = {
        "supervised_resnet50": root / "studies/resnet50/data/DTD_stages.npz",
        "dino_resnet50": root / "additional_features/dino_r50/DTD_stages.npz",
        "dino_vits16": root / "additional_features/dino_vits16/DTD_stages.npz",
    }
    stage_checks = {}
    for name, path in stage_sources.items():
        with np.load(path) as z:
            np.testing.assert_array_equal(z["ids"], dtd_ids)
            stage_checks[name] = {
                "image_identities": len(dtd_ids),
                "stage_shapes": {key: list(z[key].shape) for key in ("layer2", "layer3", "layer4")},
                "sha256": digest(path),
            }

    verified_feature_files = []
    for line in (root / "additional_features/SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        wanted, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), "Unsafe manifest path")
        require(digest(path) == wanted, f"Companion feature fingerprint differs: {relative}")
        verified_feature_files.append(relative)

    catalogue_path = root / "studies/matched_pretraining/results/garment_test_evidence.csv"
    with catalogue_path.open(newline="", encoding="utf-8") as stream:
        catalogue = list(csv.DictReader(stream))
    products = [row["test_product"] for row in catalogue]
    views = sum(int(row["views"]) for row in catalogue)
    motifs = sorted({row["motif"] for row in catalogue})
    require(len(products) == len(set(products)) == 654, "Catalogue product universe differs")
    require(views == 851 and len(motifs) == 6, "Catalogue views or motifs differ")
    for row in catalogue:
        for key, value in row.items():
            if key not in {"test_product", "motif", "views"}:
                require(0 <= int(value) <= int(row["views"]), "Invalid product correctness count")

    classes = sorted({str(name).split("/")[0] for name in dtd_ids})
    require(all(classes[int(label)] == str(name).split("/")[0]
                for name, label in zip(dtd_ids, dtd_labels, strict=True)), "Class index differs")
    with np.load(root / "pixel_check/DTD_predictions.npz") as z:
        pred = {key: z[key].copy() for key in ("ids", "labels", "groups", "test", "H_S", "H_Q")}
    np.testing.assert_array_equal(pred["ids"], dtd_ids[pred["test"]])
    np.testing.assert_array_equal(pred["labels"], dtd_labels[pred["test"]])
    seen = set()
    candidates = {"correction": [], "new_error": []}
    for split in range(len(pred["ids"])):
        for column in sorted(range(len(pred["ids"][split])), key=lambda i: str(pred["ids"][split, i])):
            group = int(pred["groups"][split, column])
            if group in seen:
                continue
            seen.add(group)
            label, squares, full = [int(pred[key][split, column]) for key in ("labels", "H_S", "H_Q")]
            category = "correction" if squares != label and full == label else (
                "new_error" if squares == label and full != label else None)
            if category is None:
                continue
            identifier = str(pred["ids"][split, column])
            candidates[category].append({
                "identifier": identifier, "filename": identifier.split("/")[-1],
                "split": split + 1, "attribute": classes[label],
                "squares": classes[squares], "full": classes[full],
                "identifier_sha256": hashlib.sha256(identifier.encode("utf-8")).hexdigest(),
                "exact_image_group": group, "category": category,
            })
    selected = [row for category in ("correction", "new_error")
                for row in sorted(candidates[category], key=lambda r: r["identifier_sha256"])[:2]]
    expected = [
        ("honeycombed_0088.jpg", 1, "honeycombed", "cracked", "honeycombed"),
        ("wrinkled_0106.jpg", 3, "wrinkled", "cracked", "wrinkled"),
        ("pleated_0063.jpg", 2, "pleated", "pleated", "gauzy"),
        ("bubbly_0135.jpg", 4, "bubbly", "bubbly", "sprinkled"),
    ]
    observed = [tuple(row[key] for key in ("filename", "split", "attribute", "squares", "full"))
                for row in selected]
    require(observed == expected, f"Deterministic selections differ: {observed}")
    require(len(seen) == 5530, "DTD test exact-image universe differs")
    return {
        "status": "passed", "provenance_rows_checked": 6,
        "scope": "Released arrays and stored decisions; no fresh pixel extraction in this check.",
        "historical_numerical_archive_sha256": ARCHIVE_DIGEST,
        "historical_release_verification": release_verification,
        "colour_image_identities": identity_counts,
        "dinov2_arrays_matching_reference_extraction": transformer_hashes,
        "stage_sources": stage_checks,
        "companion_feature_files_verified": verified_feature_files,
        "catalogue": {"products": len(products), "views": views, "motifs": motifs,
                      "evidence_sha256": digest(catalogue_path)},
        "s42": {"test_appearances": int(pred["labels"].size), "exact_image_groups": len(seen),
                "candidate_counts": {key: len(value) for key, value in candidates.items()},
                "rows": selected, "rows_checked": len(selected)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.package.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PASS: six provenance rows and four deterministic DTD examples. {args.report}")


if __name__ == "__main__":
    main()
