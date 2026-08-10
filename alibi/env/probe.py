"""Report which sandbox controls are actually available on this host.

docs/kickoff/03-security-and-access.md section 2 requires that a host which
cannot provide a control says so rather than implying isolation it does not
have. This module is the single place that decides, and its output is embedded
verbatim into every run's env.lock.

It is also the CI step, so a green build on a container that cannot open a
network namespace is not read as a claim that the namespace was active.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field

# Controls without which the sandbox is not established at all. Absence aborts
# the run with exit code 3 rather than degrading. There is no flag to override.
REQUIRED = ("process_isolation", "resource_limits", "temp_dir", "process_group_kill", "import_denylist")

# Controls that strengthen the sandbox where the host allows them. Absence is
# recorded by name in env.lock and does not abort.
PREFERRED = ("network_namespace", "filesystem_namespace")


@dataclass
class Control:
    name: str
    available: bool
    required: bool
    detail: str


@dataclass
class ControlReport:
    controls: list[Control] = field(default_factory=list)

    @property
    def missing_required(self) -> list[str]:
        return sorted(c.name for c in self.controls if c.required and not c.available)

    @property
    def missing_preferred(self) -> list[str]:
        return sorted(c.name for c in self.controls if not c.required and not c.available)

    @property
    def active(self) -> list[str]:
        return sorted(c.name for c in self.controls if c.available)

    def to_dict(self) -> dict:
        return {
            "active": self.active,
            "missing_required": self.missing_required,
            "missing_preferred": self.missing_preferred,
            "detail": {c.name: asdict(c) for c in sorted(self.controls, key=lambda c: c.name)},
        }


def _check_resource_limits() -> Control:
    try:
        import resource
    except ImportError as exc:
        return Control("resource_limits", False, True, f"no resource module: {exc}")
    needed = ["RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE", "RLIMIT_NPROC", "RLIMIT_NOFILE", "RLIMIT_CORE"]
    absent = [n for n in needed if not hasattr(resource, n)]
    if absent:
        return Control("resource_limits", False, True, f"missing limits: {', '.join(absent)}")
    return Control("resource_limits", True, True, "all of " + ", ".join(needed))


def _check_temp_dir() -> Control:
    import tempfile

    try:
        path = tempfile.mkdtemp(prefix="alibi-probe-")
    except OSError as exc:
        return Control("temp_dir", False, True, f"mkdtemp failed: {exc}")
    shutil.rmtree(path, ignore_errors=True)
    return Control("temp_dir", True, True, "tempfile.mkdtemp usable")


def _check_process_group_kill() -> Control:
    import os

    if not hasattr(os, "killpg") or not hasattr(os, "setsid"):
        return Control("process_group_kill", False, True, "os.killpg or os.setsid unavailable")
    return Control("process_group_kill", True, True, "os.setsid plus os.killpg")


def _check_network_namespace() -> Control:
    """Can this host put the sandbox in a network namespace with no interfaces?

    Unprivileged namespace creation needs user namespaces. Containers commonly
    disable them, which is exactly the Kaggle and Colab case called out in the
    security doc.
    """
    unshare = shutil.which("unshare")
    if unshare is None:
        return Control("network_namespace", False, False, "unshare(1) not on PATH")
    try:
        proc = subprocess.run(
            [unshare, "--user", "--map-root-user", "--net", "true"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Control("network_namespace", False, False, f"unshare failed to start: {exc}")
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()[:200]
        return Control("network_namespace", False, False, f"unshare exit {proc.returncode}: {err}")
    return Control("network_namespace", True, False, "unshare --user --map-root-user --net")


def _check_filesystem_namespace() -> Control:
    """Can the sandbox get its own filesystem view instead of the host's?

    A fresh temp cwd, which is all the security doc's table asks for, does not
    stop `open('/etc/passwd')`. A mount namespace plus pivot_root into a
    minimal root does, by making the file absent. Needs the same unprivileged
    user namespace as the network check, plus a known pivot_root syscall number
    for this architecture.

    The check here is cheap. Executor also runs a real execution at
    construction, because unshare(1) returning 0 does not prove pivot_root
    works on this kernel.
    """
    import os

    from alibi.env._sandbox_setup import SYS_PIVOT_ROOT

    machine = os.uname().machine if hasattr(os, "uname") else ""
    if machine not in SYS_PIVOT_ROOT:
        return Control("filesystem_namespace", False, False, f"no pivot_root syscall number known for {machine}")
    unshare = shutil.which("unshare")
    if unshare is None:
        return Control("filesystem_namespace", False, False, "unshare(1) not on PATH")
    try:
        proc = subprocess.run(
            [unshare, "--user", "--map-root-user", "--mount", "true"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Control("filesystem_namespace", False, False, f"unshare failed to start: {exc}")
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()[:200]
        return Control("filesystem_namespace", False, False, f"unshare exit {proc.returncode}: {err}")
    return Control("filesystem_namespace", True, False, "mount namespace plus pivot_root into a minimal root")


def probe() -> ControlReport:
    report = ControlReport()
    report.controls.append(
        Control("process_isolation", True, True, "subprocess with a dedicated runner, never exec in process")
    )
    report.controls.append(
        Control("import_denylist", True, True, "meta path finder in the runner, defence in depth only")
    )
    report.controls.append(_check_resource_limits())
    report.controls.append(_check_temp_dir())
    report.controls.append(_check_process_group_kill())
    report.controls.append(_check_network_namespace())
    report.controls.append(_check_filesystem_namespace())
    return report


def main() -> int:
    report = probe()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    for name in report.missing_preferred:
        print(f"WARN sandbox control unavailable on this host: {name}", file=sys.stderr)
    if report.missing_required:
        for name in report.missing_required:
            print(f"WARN sandbox control unavailable on this host: {name}", file=sys.stderr)
        print("sandbox cannot be established, refusing to execute generated code", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
