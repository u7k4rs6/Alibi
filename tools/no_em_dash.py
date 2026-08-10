"""Reject em dashes in tracked text.

A working agreement for this project rather than a security control. It lives in
the hook so it is enforced rather than remembered.
"""

from __future__ import annotations

import sys

EM_DASH = "—"
TEXT_SUFFIXES = (".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".json", ".cfg", ".lock")


def main(paths: list[str]) -> int:
    hits = []
    for path in paths:
        if not path.endswith(TEXT_SUFFIXES):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, start=1):
                    if EM_DASH in line:
                        hits.append((path, lineno, line.rstrip()))
        except (OSError, UnicodeDecodeError):
            continue
    for path, lineno, line in hits:
        print(f"{path}:{lineno}: em dash: {line}", file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
