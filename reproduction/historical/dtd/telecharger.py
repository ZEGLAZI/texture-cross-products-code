"""Télécharger les archives publiques DTD et DINOv2 avec contrôle d’intégrité."""

import argparse
import shutil
import time
import urllib.request
from pathlib import Path

from preparer import PROTO, empreinte


def fetch(folder):
    folder.mkdir(parents=True, exist_ok=True)
    s = PROTO["sources"]
    items = [
        (name + ".tar.gz", s[name], s[name + "_sha256"])
        for name in ("dtd", "labels", "decaf")
    ]
    items.append(("dino.pth", s["dino"], s["dino_sha256"]))
    items.append(
        (
            "dinov2.zip",
            "https://codeload.github.com/facebookresearch/dinov2/zip/"
            + s["dinov2_commit"],
            "209c4f04fb0d63a869f7ea5c18d61dc7d8196c88cb1cb4a53bf3b0607146685c",
        )
    )
    for name, url, expected in items:
        dest = folder / name
        temp = dest.with_suffix(dest.suffix + ".part")
        if dest.exists() and empreinte(dest) == expected:
            continue
        for attempt in range(3):
            try:
                with (
                    urllib.request.urlopen(url, timeout=120) as r,
                    temp.open("wb") as w,
                ):
                    shutil.copyfileobj(r, w, 1024 * 1024)
                if empreinte(temp) != expected:
                    raise ValueError("Empreinte inattendue pour " + name)
                temp.replace(dest)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        print(name, "vérifié", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dossier", type=Path)
    a = p.parse_args()
    fetch(a.dossier)
