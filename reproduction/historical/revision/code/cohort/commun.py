"""Paramètres et formats partagés par le test HSMT."""

import csv
from hashlib import file_digest
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "donnees"
OUT = ROOT / "resultats"
LOCK = "fcf9f1af7e88ec3bbdbf2796f5da95cc6f546443c6d1ed57ffcf1b40bb875574"
SOURCE = "10b57830534d592c1b47bf2d89ecdfe80527263682aa0719e8310e65239df6e3"
LABELS = np.array([10, 12, 9])
NAMES = ("pâquerette", "pissenlit", "tournesol")
CONTROLS = ("additif", "rgb", "f7_rgb", "f8_rgb")


def digest(p):
    with p.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_npz(p):
    with np.load(p, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def read_csv(p):
    with p.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(p, records, fields=None):
    with p.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(p, obj):
    p.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


