"""The sandbox runner. Executed as a fresh subprocess, one per execution.

Nothing in this file is imported by the trainer process. It is spawned by
alibi/env/executor.py and it is the only place model-generated Python is
executed. See docs/kickoff/03-security-and-access.md section 2.

Contract with the parent:

  argv[1]   path to a JSON payload file
  fd 3      write end of a pipe. The last line prefixed with RESULT_MARKER is
            the result. Untrusted code shares the process, so the parent takes
            the last marked line rather than trusting the whole stream.
  stdout    redirected by the parent to a file inside the sandbox directory,
  stderr    so a print bomb is capped by RLIMIT_FSIZE rather than filling a pipe.

The payload's `harness` source is required to append one boolean per test to a
module-level list named __alibi_outcomes__. Outcomes are read back out of the
execution globals even when the harness dies partway through, so a crash after
two passing tests is scored 2/n rather than discarded.
"""

from __future__ import annotations

import json
import sys

RESULT_MARKER = "__ALIBI_RESULT__ "
OUTCOME_MARKER = "__ALIBI_OUTCOME__ "


class _StreamingOutcomes(list):
    """A list that reports each outcome the moment it is appended.

    Without this, a harness killed by the wall clock returns nothing, because
    the runner's result line is written after the candidate finishes. A
    completion that passes two visible tests and then hangs on the third would
    score 0 rather than 2/3, and that fraction is the reward signal, so losing
    it would quietly distort training rather than merely losing a log line.
    """

    def __init__(self, handle) -> None:
        super().__init__()
        self._handle = handle

    def append(self, value) -> None:
        super().append(value)
        try:
            self._handle.write(f"{OUTCOME_MARKER}{len(self) - 1} {int(bool(value))}\n")
            self._handle.flush()
        except (OSError, ValueError):
            # A closed or full descriptor must not turn a scored test into a
            # crash. The final result line is still the primary channel.
            pass

# Blocked at import time inside the sandbox. This is defence in depth and is
# explicitly NOT the security boundary: a determined escape can reach these
# through object graph traversal without an import statement. The boundary is
# the process, the rlimits, and the network namespace. This list exists to turn
# careless generated code into a scored failure instead of a side effect.
DEFAULT_DENIED_IMPORTS = (
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pty",
    "resource",
    "shutil",
    "signal",
    "subprocess",
    "sys",
    "tempfile",
    # Network reaching. With a network namespace active these are already inert.
    # Denied anyway so the block holds on hosts where the namespace is not
    # available, which is the minimum required by the security doc.
    "asyncio",
    "ftplib",
    "http",
    "requests",
    "select",
    "smtplib",
    "socket",
    "socketserver",
    "ssl",
    "telnetlib",
    "urllib",
    "webbrowser",
    "_socket",
    "_ssl",
)


def _confine_open(root: str) -> None:
    """Refuse to open paths outside `root`.

    Language layer confinement, in the same class as the import denylist: it
    turns careless generated code into a scored failure, and it is not a
    boundary. The boundary is the mount namespace built by _sandbox_setup.py,
    on hosts that allow one. This runs as well, not instead, so that a host
    without user namespaces still refuses `open('/etc/passwd')` rather than
    silently reading it.
    """
    import builtins
    import io
    import posixpath

    real_open = builtins.open
    root = posixpath.realpath(root)
    prefix = root.rstrip("/") + "/"

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, int):
            return real_open(file, *args, **kwargs)
        try:
            target = posixpath.realpath(posixpath.join(root, str(file)))
        except (OSError, ValueError):
            raise PermissionError(f"path is not resolvable inside the sandbox: {file!r}") from None
        if target != root and not target.startswith(prefix):
            raise PermissionError(f"path is outside the Alibi sandbox directory: {file!r}")
        return real_open(file, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_open


class _DenyImports:
    """A meta path finder that refuses named top level modules.

    Installed at the front of sys.meta_path. It only fires on a cache miss, so
    the caller also drops the denied names from sys.modules first. Modules
    already holding a direct reference to, say, the os module keep working,
    which is what lets `random` and `numpy` survive the denial of `os`.
    """

    def __init__(self, denied: frozenset[str]) -> None:
        self.denied = denied

    def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
        return None

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".", 1)[0]
        if top in self.denied:
            raise ImportError(f"import of {fullname!r} is blocked by the Alibi sandbox")
        return None


def _apply_limits(limits: dict) -> list[str]:
    """Apply resource limits. Returns the names of limits actually applied."""
    import resource as _resource

    applied = []
    order = [
        ("cpu_seconds", _resource.RLIMIT_CPU),
        ("address_space_bytes", _resource.RLIMIT_AS),
        ("file_size_bytes", _resource.RLIMIT_FSIZE),
        ("processes", _resource.RLIMIT_NPROC),
        ("open_files", _resource.RLIMIT_NOFILE),
        ("core_bytes", _resource.RLIMIT_CORE),
    ]
    for key, which in order:
        value = limits.get(key)
        if value is None:
            continue
        try:
            _resource.setrlimit(which, (value, value))
            applied.append(key)
        except (ValueError, OSError):
            # A limit that cannot be lowered is reported as not applied. It is
            # never silently treated as if it were.
            continue
    return applied


def main() -> int:
    payload = json.loads(open(sys.argv[1], encoding="utf-8").read())

    result_fd = payload.get("result_fd", 3)

    # Warm every import the held out harness needs before the denylist goes up.
    # numpy in particular pulls in os, ctypes and platform lazily, and it is a
    # dependency of the MBPP+ test harness rather than of the candidate code.
    warm_errors = {}
    for name in payload.get("warm_imports", []):
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            warm_errors[name] = f"{type(exc).__name__}: {exc}"

    sys.setrecursionlimit(int(payload.get("recursion_limit", 1000)))

    # Limits are applied before the denylist goes up, because `resource` is
    # itself on the denylist. Applying them here also means the warm imports
    # above are not charged against a budget they did not spend.
    applied = _apply_limits(payload.get("limits", {}))

    if payload.get("confine_open_to_cwd", True):
        _confine_open(payload["sandbox_root"])

    denied = frozenset(payload.get("denied_imports", DEFAULT_DENIED_IMPORTS))
    for name in sorted(denied):
        sys.modules.pop(name, None)
    sys.meta_path.insert(0, _DenyImports(denied))

    handle = open(result_fd, "w", closefd=False, encoding="utf-8")

    source = payload["code"] + "\n\n" + payload["harness"] + "\n"
    globals_dict: dict = {
        "__name__": "__alibi_candidate__",
        "__builtins__": __builtins__,
        # Pre-seeded so a harness that does `__alibi_outcomes__.append(...)`
        # streams. A harness that rebinds the name instead still works, it just
        # loses the streaming property, so the harnesses in alibi/env/tests.py
        # append to this rather than creating their own list.
        "__alibi_outcomes__": _StreamingOutcomes(handle),
    }

    status = "ok"
    detail = ""
    try:
        compiled = compile(source, "<candidate>", "exec")
        exec(compiled, globals_dict)  # noqa: S102 - this is the point of the file
    except BaseException as exc:  # noqa: BLE001 - every failure is a scored outcome
        status = "raised"
        detail = f"{type(exc).__name__}: {exc}"[:2000]

    outcomes = globals_dict.get("__alibi_outcomes__", [])
    if not isinstance(outcomes, list):
        outcomes = []
    outcomes = [bool(x) for x in outcomes]

    # A harness may also leave a JSON serialisable value in __alibi_report__.
    # Used by the synthetic cheat generator to read a reference solution's
    # outputs back out of the sandbox. Anything unserialisable is dropped
    # rather than allowed to fail the whole execution.
    report = globals_dict.get("__alibi_report__")
    try:
        json.dumps(report)
    except (TypeError, ValueError):
        report = None

    record = {
        "status": status,
        "detail": detail,
        "outcomes": outcomes,
        "report": report,
        "limits_applied": applied,
        "warm_import_errors": warm_errors,
    }
    # Written through the same file object built directly on the descriptor.
    # This deliberately avoids `os`, which the denylist above has already
    # removed from sys.modules.
    handle.write(RESULT_MARKER + json.dumps(record) + "\n")
    handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
