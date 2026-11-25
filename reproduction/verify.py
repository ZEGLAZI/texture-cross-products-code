"""Verify this public numerical subset against its pinned historical manifest.

The original full manifest is preserved. Publication-only exclusions and the
small documentation/integrity adaptations are explicitly checked by the
root verifier, rather than silently ignored or claimed byte-identical.
"""
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from numerical_release import verify_release

report = verify_release(PACKAGE)
print(json.dumps(report, indent=2))
