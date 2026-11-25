"""Check the code/data distribution before staging it in a fresh repository.

This standard-library-only check verifies every distributed byte, rejects
unexpected files, and restricts documents/archives to the reviewed data assets.
Use --check-git for the initial upload: it requires one fresh root commit,
matching the checked working files, index and every local ref.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
IGNORED_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'runs'}
IGNORED_NAMES = {'.DS_Store', 'Thumbs.db'}
PLOT_PDFS = {
    'reproduction/review2/figures/attributes.pdf',
    'reproduction/review2/figures/centres.pdf',
    'reproduction/review2/figures/margins.pdf',
    'reproduction/review2/figures/selected_contrasts.pdf',
    'studies/ridge_sensitivity/results/ridge_validation.pdf',
    'studies/ridge_sensitivity/results/ridge_validation_publication.pdf',
}
DATA_ZIPS = {
    'reproduction/historical/revision/provenance/inat2026/instantane_inaturalist.zip',
    'reproduction/historical/revision/provenance/plantnet/instantane_gbif.zip',
    'reproduction/historical/revision/results/cohort/instantane_inaturalist.zip',
}
FORBIDDEN_PREFIXES = ('source/', 'reference_20261121/', 'audit/',
                      'studies/publication_checks/')
FORBIDDEN_FILES = {
    'reproduction/historical/revision/code/displays_revision.py',
}
DOCUMENT_SUFFIXES = {'.pdf', '.tex', '.bib', '.bbl', '.sty', '.cls', '.bst', '.dtx', '.ins', '.doc',
                     '.docx', '.odt', '.ppt', '.pptx', '.rtf'}
ARCHIVE_SUFFIXES = {'.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.7z', '.rar'}


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_path(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError(f'Unsafe distribution path: {name}')
    if name in FORBIDDEN_FILES or name.startswith(FORBIDDEN_PREFIXES):
        raise ValueError(f'Excluded publication path: {name}')
    suffix = path.suffix.lower()
    if suffix in DOCUMENT_SUFFIXES and name not in PLOT_PDFS:
        raise ValueError(f'Unreviewed document: {name}')
    if suffix in ARCHIVE_SUFFIXES and name not in DATA_ZIPS:
        raise ValueError(f'Unreviewed archive: {name}')


def read_manifest() -> dict[str, str]:
    manifest = {}
    for number, line in enumerate((ROOT / 'SHA256SUMS').read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'Invalid SHA-256 on manifest line {number}')
        check_path(name)
        if name in manifest or name == 'SHA256SUMS':
            raise ValueError(f'Duplicate or recursive manifest entry: {name}')
        manifest[name] = digest
    if not manifest:
        raise ValueError('The distribution manifest is empty')
    return manifest


def actual_paths() -> set[str]:
    found = set()
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symbolic links are not permitted: {relative}')
        if not path.is_file():
            continue
        if path.name in IGNORED_NAMES or path.suffix in {'.pyc', '.pyo'}:
            continue
        found.add(relative.as_posix())
    return found


def inspect_data_archives() -> int:
    entries = 0
    for name in sorted(DATA_ZIPS):
        with zipfile.ZipFile(ROOT / name) as archive:
            for entry in archive.infolist():
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in entry.filename:
                    raise ValueError(f'Unsafe data archive entry: {name}: {entry.filename}')
                if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError(f'Archive contains a symbolic link: {name}')
                if not entry.is_dir():
                    if path.suffix.lower() != '.json':
                        raise ValueError(f'Unexpected data archive member: {name}: {entry.filename}')
                    entries += 1
    return entries


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', '--no-replace-objects', '-C', str(ROOT), *args], stderr=subprocess.STDOUT)


def inspect_git(expected: set[str]) -> int:
    if not (ROOT / '.git').exists():
        raise ValueError('--check-git requires an initialized repository')
    tracked = {s.decode('utf-8') for s in git('ls-files', '-z').split(b'\0') if s}
    if tracked != expected:
        raise ValueError(f'Git index differs: extra={sorted(tracked-expected)}, missing={sorted(expected-tracked)}')
    if git('diff', '--name-only'):
        raise ValueError('Tracked working files differ from the staged index')
    if git('diff', '--cached', '--name-only'):
        raise ValueError('Staged files differ from the committed release')
    if git('rev-parse', '--is-shallow-repository').strip() != b'false':
        raise ValueError('A shallow clone is not a fresh initial repository')
    commits = git('rev-list', '--all').decode().splitlines()
    if len(commits) != 1:
        raise ValueError('Initial-upload check requires exactly one fresh root commit and no other history')
    commit = commits[0]
    headers = git('cat-file', '-p', commit).split(b'\n\n', 1)[0].splitlines()
    if any(line.startswith(b'parent ') for line in headers):
        raise ValueError('The initial release commit must not have a parent')
    committed = {raw.decode('utf-8') for raw in git('ls-tree', '-rz', '--name-only', commit).split(b'\0') if raw}
    if committed != expected:
        raise ValueError('Committed paths differ from the reviewed distribution')
    for name in committed:
        check_path(name)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-git', action='store_true', help='Initial upload only: require one fresh root commit matching the reviewed files')
    args = parser.parse_args()
    try:
        manifest = read_manifest()
        expected = set(manifest) | {'SHA256SUMS'}
        actual = actual_paths()
        if actual != expected:
            raise ValueError(f'Distribution differs: extra={sorted(actual-expected)}, missing={sorted(expected-actual)}')
        for name, digest in manifest.items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'File changed: {name}')
        entries = inspect_data_archives()
        commits = inspect_git(expected) if args.check_git else None
        print(f'Verified {len(manifest)} files against SHA256SUMS.')
        print(f'Reviewed document inventory: {len(PLOT_PDFS)} standalone numerical plots; {entries} JSON data snapshot members.')
        print('All files match the reviewed code/data distribution; no excluded document path or unlisted file was found.')
        if commits is not None:
            print(f'Git index matches this distribution; checked paths in {commits} local commit(s).')
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        print(f'RELEASE CHECK FAILED: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
