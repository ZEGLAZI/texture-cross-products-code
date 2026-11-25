"""Verify and read the explicitly documented public numerical subset.

The repository exposes files under reproduction/ instead of requiring
an archive download. The original ZIP digest remains a release identifier;
it is never presented as a freshly measured ZIP digest when no ZIP is present.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ARCHIVE_NAME = "data_and_code_r4.zip"
ARCHIVE_SHA256 = "ad1a3c2cda703498ca2bca2e2f5fa6718ad071931a0d5dbd8869c4f46dfbf385"
MANIFEST_SHA256 = "7fa27470dab59573e0933bf1ade04cea903c04483756e9ab5744022350c03283"
PUBLIC_CHANGES_SHA256 = "522462fcfbad14c6bfbcb3ad8da4a289c8e3df024cef49c99b20e74c90a1781b"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_release(root: Path) -> dict:
    root = Path(root).resolve()
    directory = root / "reproduction"
    if directory.is_dir():
        manifest = directory / "SHA256SUMS"
        if digest(manifest) != MANIFEST_SHA256:
            raise ValueError("Historical payload manifest differs")
        changes_path = root / "PUBLIC_PAYLOAD_CHANGES.json"
        if digest(changes_path) != PUBLIC_CHANGES_SHA256:
            raise ValueError("The public-subset change record differs")
        public = json.loads(changes_path.read_text(encoding="utf-8"))
        if (public["original_manifest_entries"] != 796
                or public["original_manifest_sha256"] != MANIFEST_SHA256):
            raise ValueError("The public-subset record names a different historical release")
        changes = {entry["path"]: entry for entry in public["changes"]}
        expected_changes = {"README.md", "historical/revision/README.md", "verify.py",
                            "historical/revision/code/displays_revision.py"}
        if set(changes) != expected_changes or len(public["changes"]) != len(changes):
            raise ValueError("Unexpected public-subset exceptions")
        checked, unchanged, adapted, excluded = 0, 0, 0, []
        manifest_names = set()
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            if name in manifest_names:
                raise ValueError("Repeated historical manifest path")
            manifest_names.add(name)
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Historical manifest path escapes its directory")
            if name in changes:
                entry = changes[name]
                if entry["original_sha256"] != expected:
                    raise ValueError(f"Original fingerprint differs in public-subset record: {name}")
                if entry["action"] == "excluded":
                    if path.exists() or entry["public_sha256"] is not None:
                        raise ValueError(f"Excluded publication file is present: {name}")
                    excluded.append(name)
                    continue
                if entry["action"] != "adapted":
                    raise ValueError(f"Unknown public-subset action: {name}")
                expected = entry["public_sha256"]
                adapted += 1
            else:
                unchanged += 1
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"Historical/public numerical payload differs: {name}")
            checked += 1
        if (len(manifest_names), checked, unchanged, adapted, len(excluded)) != (796, 795, 792, 3, 1):
            raise ValueError("Public numerical subset coverage differs")
        return {"mode": "public_numerical_subset", "original_manifest_entries": 796,
                "files_verified": checked, "unchanged_files_verified": unchanged,
                "adapted_files_verified": adapted, "excluded_publication_files": excluded,
                "original_payload_is_complete_and_byte_identical": False,
                "manifest_sha256": MANIFEST_SHA256,
                "public_change_record_sha256": PUBLIC_CHANGES_SHA256,
                "historical_archive_identifier_sha256": ARCHIVE_SHA256,
                "zip_digest_measured_in_this_check": False}
    archive = root / ARCHIVE_NAME
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("Historical numerical archive differs")
    return {"mode": "original_archive", "archive_sha256": ARCHIVE_SHA256,
            "historical_archive_identifier_sha256": ARCHIVE_SHA256,
            "zip_digest_measured_in_this_check": True}


class ReleasedPayload:
    """Read verified members from the unpacked tree or the original archive."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.verification = verify_release(self.root)
        self.directory = self.root / "reproduction"
        self.archive = None

    def __enter__(self):
        if not self.directory.is_dir():
            self.archive = zipfile.ZipFile(self.root / ARCHIVE_NAME)
        return self

    def read(self, relative: str) -> bytes:
        if self.archive is not None:
            return self.archive.read(relative)
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("Payload path escapes its directory")
        return path.read_bytes()

    def __exit__(self, *args):
        if self.archive is not None:
            self.archive.close()
