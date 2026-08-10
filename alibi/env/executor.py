"""Sandboxed execution of model generated Python.

Every control named in docs/kickoff/03-security-and-access.md section 2 is
either applied here or reported absent by alibi.env.probe. Nothing degrades
silently and there is no flag that turns the sandbox off.

Layering, strongest first:

  1. A separate process, always. The trainer never runs exec or eval on
     generated code.
  2. A network namespace with no interfaces, where the host allows an
     unprivileged one. Recorded as inactive by name when it does not.
  3. resource.setrlimit for CPU, address space, file size, process count,
     descriptors and core dumps, applied inside the runner immediately before
     the candidate code is compiled.
  4. Wall clock timeout enforced by the parent, killing the process group
     rather than the child, so a fork that outlives its parent still dies.
  5. A fresh temp directory as cwd, removed afterwards.
  6. A scrubbed environment with no HOME and no inherited API keys.
  7. An import denylist in the runner. Defence in depth, not a boundary.

A crash, a timeout or a kill is a scored outcome with pass_fraction 0. It is
never an exception that reaches the training loop, per
docs/kickoff/02-technical-architecture.md section 6.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from alibi.env._runner import DEFAULT_DENIED_IMPORTS, OUTCOME_MARKER, RESULT_MARKER
from alibi.env.probe import ControlReport, probe

RUNNER_PATH = Path(__file__).with_name("_runner.py")
SETUP_PATH = Path(__file__).with_name("_sandbox_setup.py")


def _host_paths_the_runner_needs() -> list[str]:
    """Directories that must exist inside the minimal sandbox root.

    Deliberately a short explicit list rather than all of sys.path. Everything
    not named here is absent from the sandbox's filesystem view, which is what
    makes `open('/etc/passwd')` a FileNotFoundError instead of a read.
    """
    import numpy

    paths = {
        "/usr",
        os.path.realpath(sys.base_prefix),
        os.path.dirname(os.path.realpath(sys.executable)),
        # numpy is a dependency of the MBPP+ held out harness, not of the
        # candidate code.
        os.path.dirname(os.path.dirname(os.path.realpath(numpy.__file__))),
        str(RUNNER_PATH.parent.resolve()),
    }
    return sorted(p for p in paths if p and os.path.exists(p))


def _top_level_symlinks() -> dict[str, str]:
    """Top level symlinks such as /lib -> usr/lib, recreated inside the root."""
    links = {}
    for name in ("/lib", "/lib64", "/bin", "/sbin"):
        if os.path.islink(name):
            links[name] = os.readlink(name)
    return links

# Kept here rather than in a config file so that the sandbox's shape is
# readable in one place. Overridable per Executor, never per execution.
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_LIMITS = {
    # RLIMIT_CPU is the backstop for a process that ignores the wall clock, for
    # example one that blocks signals. Set above the wall clock so the wall
    # clock is normally what fires and the two are distinguishable in the logs.
    "cpu_seconds": 10,
    # Address space, not resident memory. RLIMIT_AS caps virtual address space,
    # which is a coarse control at any value.
    #
    # This is a deliberate deviation from the 512 MB in
    # docs/kickoff/03-security-and-access.md section 2, and the doc is the thing
    # that is wrong. The same doc requires the held out harness, the held out
    # harness is EvalPlus's and it opens with `import numpy as np`, and numpy
    # reserves 546 MB of address space on the day one host before executing a
    # single line of candidate code. At 512 MB every held out execution dies
    # with MemoryError before running any test, which would have shown up as a
    # held out pass rate of zero for every completion including correct ones.
    #
    # 1536 MB leaves roughly 960 MB for candidate code, which is close to the
    # headroom the doc intended. The measured numpy figure is recorded here so
    # the choice is checkable rather than asserted.
    "address_space_bytes": 1536 * 1024 * 1024,
    # Small, per the security doc. Also caps a print bomb, because the parent
    # redirects stdout and stderr to files inside the sandbox directory.
    "file_size_bytes": 8 * 1024 * 1024,
    # Blocks fork bombs. Enforced per user namespace on Linux, so a fresh user
    # namespace per execution makes this a genuine per sandbox cap rather than
    # a cap shared with everything else this uid is running. Where the network
    # namespace is unavailable the user namespace is too, and this degrades to
    # a per uid limit. That degradation is recorded in env.lock.
    "processes": 64,
    "open_files": 64,
    "core_bytes": 0,
}

# Imported before the denylist goes up, so that they resolve from the module
# cache afterwards.
#
# Two reasons a module is here. `numpy` is a dependency of the MBPP+ held out
# harness itself. The rest are stdlib modules ordinary MBPP solutions use, and
# several of them reach for a denied module internally at import time: `random`
# does `from os import urandom`, so denying `os` would otherwise turn every
# solution that imports `random` into a sandbox failure that looks like a wrong
# answer.
#
# The security doc prefers a small allowlist if it does not break MBPP
# solutions. This list is the measured version of that preference: it is
# checked against all 378 reference solutions by
# tests/test_executor.py::test_denylist_does_not_break_reference_solutions.
WARM_IMPORTS = (
    "numpy",
    "array",
    "bisect",
    "calendar",
    "cmath",
    "collections",
    "copy",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "operator",
    "queue",
    "random",
    "re",
    "statistics",
    "string",
    "textwrap",
    "typing",
    "unicodedata",
)


class SandboxUnavailable(RuntimeError):
    """Raised when a required control is absent. Exit code 3, never a degrade."""


@dataclass(frozen=True)
class ExecResult:
    """The outcome of one sandboxed execution.

    `outcomes` is one boolean per test in the harness. It is authoritative even
    when `status` is not "ok", because a harness that dies on test 3 of 100 has
    still genuinely reported the first two.
    """

    status: str  # ok | raised | timeout | killed | no_result
    outcomes: list[bool]
    n_tests: int
    detail: str = ""
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    duration_seconds: float = 0.0
    limits_applied: list[str] = field(default_factory=list)
    # Optional structured value a harness left in __alibi_report__. Used to
    # read a reference solution's outputs back out of the sandbox.
    report: object = None

    @property
    def pass_fraction(self) -> float:
        """Fraction of the harness's tests that passed.

        Denominator is the number of tests the harness was asked to run, not
        the number it managed to reach. A timeout after two passes out of a
        hundred scores 0.02, not 1.0.
        """
        if self.n_tests == 0:
            return 0.0
        return sum(self.outcomes) / self.n_tests

    @property
    def all_passed(self) -> bool:
        return self.n_tests > 0 and len(self.outcomes) == self.n_tests and all(self.outcomes)


class Executor:
    """Runs candidate code against a harness, one subprocess per execution.

    Construction probes the host and fails closed. Nothing else in the codebase
    is allowed to execute generated code.
    """

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        limits: dict | None = None,
        denied_imports: tuple[str, ...] = DEFAULT_DENIED_IMPORTS,
        report: ControlReport | None = None,
        confine_open_to_cwd: bool = True,
        self_test: bool = True,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.limits = dict(DEFAULT_LIMITS if limits is None else limits)
        self.denied_imports = tuple(denied_imports)
        self.confine_open_to_cwd = confine_open_to_cwd
        self.report = report if report is not None else probe()
        self.degradations: list[str] = []

        if self.report.missing_required:
            missing = ", ".join(self.report.missing_required)
            raise SandboxUnavailable(
                f"sandbox control unavailable on this host: {missing}. "
                "Refusing to execute generated code. There is no flag to override this."
            )
        self.use_network_namespace = "network_namespace" in self.report.active
        self.use_filesystem_namespace = "filesystem_namespace" in self.report.active
        self._read_only_paths = _host_paths_the_runner_needs() if self.use_filesystem_namespace else []
        self._symlinks = _top_level_symlinks() if self.use_filesystem_namespace else {}

        if self_test:
            self._self_test()

    def _self_test(self) -> None:
        """Prove the configured sandbox actually executes before trusting it.

        A probe can say unshare(1) returns 0 and still leave pivot_root failing
        on this kernel. Without this, that failure would surface as every
        problem scoring zero, which reads like a bad model rather than a broken
        sandbox. A preferred control that fails here is switched off and named.
        A required control that fails here aborts.
        """
        probe_code = "def __alibi_probe__():\n    return 1\n"
        probe_harness = "__alibi_outcomes__ = [__alibi_probe__() == 1]"
        result = self.run(probe_code, probe_harness, 1)
        if result.all_passed:
            return
        if self.use_filesystem_namespace:
            self.degradations.append(
                "filesystem_namespace: self test failed on this host "
                f"({result.status}: {(result.stderr or result.detail).strip()[:300]}), "
                "falling back to language layer confinement in the runner"
            )
            self.use_filesystem_namespace = False
            self._read_only_paths, self._symlinks = [], {}
            for control in self.report.controls:
                if control.name == "filesystem_namespace":
                    control.available = False
                    control.detail = "self test failed at Executor construction"
            result = self.run(probe_code, probe_harness, 1)
            if result.all_passed:
                return
        if self.use_network_namespace:
            self.degradations.append(
                "network_namespace: self test failed on this host "
                f"({result.status}: {(result.stderr or result.detail).strip()[:300]}), "
                "falling back to the import denylist in the runner"
            )
            self.use_network_namespace = False
            for control in self.report.controls:
                if control.name == "network_namespace":
                    control.available = False
                    control.detail = "self test failed at Executor construction"
            result = self.run(probe_code, probe_harness, 1)
            if result.all_passed:
                return
        raise SandboxUnavailable(
            "the sandbox could not execute a trivial program with every optional control "
            f"switched off ({result.status}: {(result.stderr or result.detail).strip()[:500]}). "
            "Refusing to execute generated code."
        )

    def control_summary(self) -> dict:
        """What env.lock records about this executor's actual controls."""
        summary = self.report.to_dict()
        summary["timeout_seconds"] = self.timeout_seconds
        summary["limits"] = dict(sorted(self.limits.items()))
        summary["denied_imports"] = sorted(self.denied_imports)
        summary["network_namespace_in_use"] = self.use_network_namespace
        summary["filesystem_namespace_in_use"] = self.use_filesystem_namespace
        summary["confine_open_to_cwd"] = self.confine_open_to_cwd
        summary["degradations"] = list(self.degradations)
        return summary

    def _command(self, payload_path: Path, workdir: Path) -> list[str]:
        # -s drops the user site directory. -E is deliberately not used and -I
        # is deliberately not used either: both would also discard PYTHONPATH,
        # and PYTHONPATH is how the sandbox is told about exactly one extra
        # directory, the one holding numpy. The environment is scrubbed by
        # passing an explicit env to Popen instead, which is stricter than -E
        # because it whitelists rather than filters.
        runner = [sys.executable, "-s", "-B", str(RUNNER_PATH), str(payload_path)]
        if not self.use_network_namespace:
            return runner

        unshare = shutil.which("unshare") or "unshare"
        flags = ["--user", "--map-root-user", "--net"]
        if not self.use_filesystem_namespace:
            # Network namespace only. The filesystem view is the host's, and
            # confinement falls back to the language layer in the runner, which
            # env.lock records by name.
            return [unshare, *flags, *runner]

        spec_path = workdir / "sandbox_root.json"
        spec_path.write_text(
            json.dumps(
                {
                    "read_only": self._read_only_paths,
                    "read_write": [str(workdir)],
                    "symlinks": self._symlinks,
                    "cwd": str(workdir),
                    "argv": runner,
                }
            ),
            encoding="utf-8",
        )
        return [unshare, *flags, "--mount", sys.executable, "-s", "-B", str(SETUP_PATH), str(spec_path)]

    @staticmethod
    def _scrubbed_env(workdir: Path) -> dict[str, str]:
        """An allowlisted environment. No HOME, no inherited API keys.

        Built from nothing rather than filtered from os.environ, so a variable
        nobody thought of is absent by default instead of present by oversight.

        PYTHONHASHSEED is fixed because docs/kickoff/02-technical-architecture.md
        section 5 puts the executor's determinism in scope even though the
        rollout path's is not.

        PYTHONPATH names exactly one directory, the one holding numpy, which
        the MBPP+ held out harness imports.
        """
        import numpy

        numpy_site = os.path.dirname(os.path.dirname(os.path.realpath(numpy.__file__)))
        return {
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(workdir),
            "PYTHONPATH": numpy_site,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LC_ALL": "C",
            "LANG": "C",
        }

    def run(self, code: str, harness: str, n_tests: int) -> ExecResult:
        """Execute `code` then `harness` in a sandbox and score the outcome.

        `n_tests` is the number of tests the harness was constructed to run. It
        is passed in rather than inferred so that a harness which dies before
        appending anything still scores against the right denominator.
        """
        import time

        workdir = Path(tempfile.mkdtemp(prefix="alibi-exec-"))
        started = time.monotonic()
        read_fd, write_fd = os.pipe()
        # pass_fds keeps the descriptor at its original number rather than
        # remapping it, so the runner is told which number to write to instead
        # of assuming 3.
        os.set_inheritable(write_fd, True)
        proc = None
        try:
            payload_path = workdir / "payload.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "code": code,
                        "harness": harness,
                        "limits": self.limits,
                        "denied_imports": list(self.denied_imports),
                        "warm_imports": list(WARM_IMPORTS),
                        "recursion_limit": 1000,
                        "result_fd": write_fd,
                        "sandbox_root": str(workdir),
                        "confine_open_to_cwd": self.confine_open_to_cwd,
                    }
                ),
                encoding="utf-8",
            )
            out_path = workdir / "stdout.txt"
            err_path = workdir / "stderr.txt"

            with open(out_path, "wb") as out_handle, open(err_path, "wb") as err_handle:
                proc = subprocess.Popen(  # noqa: S603 - argv is fully constructed here
                    self._command(payload_path, workdir),
                    cwd=str(workdir),
                    env=self._scrubbed_env(workdir),
                    stdin=subprocess.DEVNULL,
                    stdout=out_handle,
                    stderr=err_handle,
                    pass_fds=(write_fd,),
                    # New session, so the child leads its own process group and
                    # a fork bomb's children are killed with it.
                    start_new_session=True,
                    close_fds=True,
                )
            os.close(write_fd)
            write_fd = -1

            status = "ok"
            detail = ""
            try:
                proc.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                status = "timeout"
                detail = f"wall clock limit of {self.timeout_seconds}s exceeded"
                self._kill_group(proc)

            raw = self._drain(read_fd)
            duration = time.monotonic() - started

            record = self._parse_result(raw)
            if record is None:
                streamed = self._parse_streamed_outcomes(raw)
                if status == "ok":
                    status = "killed" if (proc.returncode or 0) < 0 else "no_result"
                    if not detail:
                        detail = f"runner produced no result, returncode {proc.returncode}"
                # A killed or timed out execution never wrote its result line.
                # The outcomes it streamed before dying are still real.
                outcomes: list[bool] = streamed
                limits_applied: list[str] = []
                harness_report = None
            else:
                outcomes = record["outcomes"]
                limits_applied = record.get("limits_applied", [])
                harness_report = record.get("report")
                if status == "ok" and record["status"] != "ok":
                    status = record["status"]
                    detail = record.get("detail", "")

            return ExecResult(
                status=status,
                outcomes=outcomes[:n_tests],
                n_tests=n_tests,
                detail=detail,
                stdout=self._read_capped(out_path),
                stderr=self._read_capped(err_path),
                returncode=proc.returncode,
                duration_seconds=duration,
                limits_applied=limits_applied,
                report=harness_report,
            )
        finally:
            if proc is not None and proc.poll() is None:  # pragma: no cover - defensive
                self._kill_group(proc)
            for fd in (read_fd, write_fd):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        """Kill the process group, not just the child.

        A fork bomb's children are reparented the moment their parent dies, so
        killing the child alone leaves them running. start_new_session put the
        child in its own group precisely so this call can reach all of them.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass

    @staticmethod
    def _drain(read_fd: int, cap: int = 4 * 1024 * 1024) -> str:
        chunks = []
        total = 0
        with os.fdopen(os.dup(read_fd), "rb", closefd=True) as handle:
            while total < cap:
                chunk = handle.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
        return b"".join(chunks).decode("utf-8", "replace")

    @staticmethod
    def _parse_result(raw: str) -> dict | None:
        """Take the last marked line.

        Untrusted code shares the process and could write to the descriptor, so
        the first marked line is not trusted to be ours. The runner writes its
        line last, after the candidate has finished.
        """
        for line in reversed(raw.splitlines()):
            if line.startswith(RESULT_MARKER):
                try:
                    record = json.loads(line[len(RESULT_MARKER) :])
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("outcomes"), list):
                    return record
        return None

    @staticmethod
    def _parse_streamed_outcomes(raw: str) -> list[bool]:
        """Rebuild outcomes from the per test lines the runner streamed.

        Used only when no final result line arrived, which means the process
        was killed. Indices are honoured rather than assumed contiguous, and a
        gap truncates the list, because scoring a test that never reported
        would be inventing a number.
        """
        seen: dict[int, bool] = {}
        for line in raw.splitlines():
            if not line.startswith(OUTCOME_MARKER):
                continue
            parts = line[len(OUTCOME_MARKER) :].split()
            if len(parts) != 2:
                continue
            try:
                seen[int(parts[0])] = bool(int(parts[1]))
            except ValueError:
                continue
        outcomes: list[bool] = []
        index = 0
        while index in seen:
            outcomes.append(seen[index])
            index += 1
        return outcomes

    @staticmethod
    def _read_capped(path: Path, cap: int = 8192) -> str:
        try:
            with open(path, "rb") as handle:
                return handle.read(cap).decode("utf-8", "replace")
        except OSError:
            return ""
