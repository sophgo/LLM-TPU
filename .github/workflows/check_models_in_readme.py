"""Verify that every subdirectory under ``models/`` is referenced in README.md.

Run from the repository root:

    python .github/workflows/check_models_in_readme.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
README_PATH = REPO_ROOT / "README.md"


def main() -> int:
    if not MODELS_DIR.is_dir():
        print(f"error: models directory not found at {MODELS_DIR}", file=sys.stderr)
        return 2
    if not README_PATH.is_file():
        print(f"error: README.md not found at {README_PATH}", file=sys.stderr)
        return 2

    readme_content = README_PATH.read_text(encoding="utf-8")

    missing = sorted(
        entry.name
        for entry in MODELS_DIR.iterdir()
        if entry.is_dir() and entry.name not in readme_content
    )

    if missing:
        print("Missing model references in README.md:")
        for name in missing:
            print(f"  - {name}")
        return 1

    print("All models are properly referenced in README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
