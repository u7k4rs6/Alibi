"""Block paths that must never be staged, regardless of their content.

Content scanners catch key-shaped strings. This catches a file whose whole
purpose is to hold secrets even when it happens to be empty or obfuscated.
See docs/kickoff/03-security-and-access.md section 3.
"""

from __future__ import annotations

import re
import sys

FORBIDDEN = [
    (re.compile(r"(^|/)\.env($|\.)"), "environment files hold API keys"),
    (re.compile(r"(^|/)raw_responses/"), "raw monitor responses may echo account identifiers"),
    (re.compile(r"\.(pem|key|p12|pfx)$"), "private key material"),
    (re.compile(r"(^|/)kaggle\.json$"), "Kaggle API credentials"),
    (re.compile(r"(^|/)\.netrc$"), "netrc holds plaintext credentials"),
]

ALLOWED = [re.compile(r"(^|/)\.env\.example$")]


def main(paths: list[str]) -> int:
    bad = []
    for path in paths:
        if any(rx.search(path) for rx in ALLOWED):
            continue
        for rx, why in FORBIDDEN:
            if rx.search(path):
                bad.append((path, why))
                break
    for path, why in bad:
        print(f"blocked: {path} ({why})", file=sys.stderr)
    if bad:
        print(
            "\nThese paths are never committed. If a secret already reached a commit, "
            "rotate the key. Do not rewrite history and consider it handled.",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
