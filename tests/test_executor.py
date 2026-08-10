"""Adversarial tests for the sandbox.

docs/kickoff/03-security-and-access.md section 2: "A sandbox with no
adversarial test is a comment, not a control." The four it names are a fork
bomb, an infinite loop, an `open('/etc/passwd')` and a socket connect, and all
four are here as the real thing rather than as a mock.

Each of the four is tested twice where the two layers differ:

  with the default denylist   what a run actually uses, defence in depth
  with the denylist removed   proves the kernel control does the work, since
                              an ImportError from our own denylist would
                              otherwise pass the test while proving nothing

Tests that need a control this host does not have are skipped by name, never
weakened, so a skip is visible in the CI log rather than hidden in a green tick.
"""

from __future__ import annotations

import pytest

from alibi.env.executor import Executor, SandboxUnavailable
from alibi.env.probe import probe
from alibi.env.tests import visible_harness

TRIVIAL_HARNESS = visible_harness(["assert True"])


@pytest.fixture(scope="module")
def executor() -> Executor:
    return Executor()


@pytest.fixture(scope="module")
def unguarded() -> Executor:
    """No import denylist and no language layer path confinement.

    Everything that stops code here is a kernel control: namespaces, rlimits
    and the process group kill. This is the fixture that makes the adversarial
    tests mean something.
    """
    return Executor(denied_imports=(), confine_open_to_cwd=False)


def _assert_scored_not_raised(result) -> None:
    """The contract from architecture doc section 6.

    A crashed or timed out execution is a scored outcome, never an exception
    that kills a run. pass_fraction is 0.0 when something was determinate and
    None when nothing was, and neither is an exception reaching the caller.
    """
    assert result.pass_fraction in (0.0, None)
    assert result.status in {"raised", "timeout", "killed", "no_result", "candidate_error"}


def test_a_candidate_that_does_not_compile_fails_rather_than_being_indeterminate(executor) -> None:
    """The distinction that stops ordinary malformed model output from tripping the halt.

    A 0.5B policy emits code that does not parse all the time. That is a
    definitive statement about the candidate, so every test is a fail. Only the
    sandbox failing to answer is indeterminate.
    """
    result = executor.run("def f(:\n", visible_harness(["assert f() == 1", "assert f() == 2"]), 2)
    assert result.status == "candidate_error"
    assert result.outcomes == ["fail", "fail"]
    assert result.n_indeterminate == 0
    assert result.pass_fraction == 0.0


# --------------------------------------------------------------------------
# The four the security doc names.
# --------------------------------------------------------------------------


def test_fork_bomb_is_contained_by_the_denylist(executor: Executor) -> None:
    code = "import os\nwhile True:\n    os.fork()\n"
    _assert_scored_not_raised(executor.run(code, TRIVIAL_HARNESS, 1))


def test_fork_bomb_is_contained_without_the_denylist(unguarded: Executor) -> None:
    """The real test: RLIMIT_NPROC and the process group kill, not an ImportError.

    With `os` importable the bomb genuinely forks. It is stopped by
    RLIMIT_NPROC, and anything that escapes the child is reaped by killing the
    process group rather than the child alone.
    """
    code = "import os\nwhile True:\n    os.fork()\n"
    result = unguarded.run(code, TRIVIAL_HARNESS, 1)
    _assert_scored_not_raised(result)
    # The host survived: proving this by still being able to run afterwards is
    # a stronger statement than any assertion about the bomb itself.
    assert unguarded.run("def f():\n    return 1\n", visible_harness(["assert f() == 1"]), 1).all_passed


def test_infinite_loop_hits_the_wall_clock(executor: Executor) -> None:
    fast = Executor(timeout_seconds=2.0)
    result = fast.run("while True:\n    pass\n", TRIVIAL_HARNESS, 1)
    assert result.status == "timeout"
    # Nothing was determinate, so the fraction is None rather than 0.0. A
    # hanging sandbox is not a candidate that failed its tests.
    assert result.pass_fraction is None
    assert result.n_indeterminate == 1
    # Killed near the limit rather than merely eventually.
    assert result.duration_seconds < 15.0


def test_infinite_loop_that_ignores_interrupts_still_dies(executor: Executor) -> None:
    """A loop in C that never returns to the interpreter is killed by SIGKILL."""
    fast = Executor(timeout_seconds=2.0, denied_imports=(), confine_open_to_cwd=False)
    code = "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True:\n    pass\n"
    result = fast.run(code, TRIVIAL_HARNESS, 1)
    assert result.status == "timeout"
    assert result.pass_fraction is None
    assert result.n_indeterminate == 1


def test_filesystem_read_outside_the_sandbox_is_refused(executor: Executor) -> None:
    _assert_scored_not_raised(executor.run("open('/etc/passwd').read()\n", TRIVIAL_HARNESS, 1))


def test_filesystem_read_is_refused_by_the_kernel_not_by_our_wrapper(unguarded: Executor) -> None:
    """/etc/passwd is absent from the sandbox's filesystem view, not merely guarded.

    Skipped by name where the host cannot open an unprivileged mount namespace,
    which is the Kaggle and Colab case the security doc calls out. On such a
    host the language layer confinement in the runner is what applies, and
    test_filesystem_read_outside_the_sandbox_is_refused covers it.
    """
    if not unguarded.use_filesystem_namespace:
        pytest.skip("filesystem_namespace unavailable on this host, see alibi.env.probe")
    result = unguarded.run("open('/etc/passwd').read()\n", TRIVIAL_HARNESS, 1)
    _assert_scored_not_raised(result)
    assert "FileNotFoundError" in result.detail, result.detail


def test_writes_outside_the_sandbox_are_refused(unguarded: Executor) -> None:
    if not unguarded.use_filesystem_namespace:
        pytest.skip("filesystem_namespace unavailable on this host, see alibi.env.probe")
    _assert_scored_not_raised(unguarded.run("open('/etc/alibi-was-here','w').write('x')\n", TRIVIAL_HARNESS, 1))


def test_socket_connect_is_refused(executor: Executor) -> None:
    code = "import socket\ns = socket.socket()\ns.connect(('1.1.1.1', 80))\n"
    _assert_scored_not_raised(executor.run(code, TRIVIAL_HARNESS, 1))


def test_socket_connect_is_refused_by_the_network_namespace(unguarded: Executor) -> None:
    """With socket importable, the namespace is what stops the connection."""
    if not unguarded.use_network_namespace:
        pytest.skip("network_namespace unavailable on this host, see alibi.env.probe")
    code = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(3)\n"
        "s.connect(('1.1.1.1', 80))\n"
        "raise AssertionError('the connection succeeded')\n"
    )
    result = unguarded.run(code, TRIVIAL_HARNESS, 1)
    _assert_scored_not_raised(result)
    assert "the connection succeeded" not in result.detail, result.detail
    assert "OSError" in result.detail or "Error" in result.detail, result.detail


def test_dns_resolution_is_refused(unguarded: Executor) -> None:
    if not unguarded.use_network_namespace:
        pytest.skip("network_namespace unavailable on this host, see alibi.env.probe")
    code = "import socket\nsocket.setdefaulttimeout(3)\nsocket.gethostbyname('example.com')\n"
    _assert_scored_not_raised(unguarded.run(code, TRIVIAL_HARNESS, 1))


# --------------------------------------------------------------------------
# The other limits, and the scored-not-raised contract.
# --------------------------------------------------------------------------


def test_memory_bomb_is_scored(unguarded: Executor) -> None:
    _assert_scored_not_raised(unguarded.run("x = bytearray(3 * 1024 * 1024 * 1024)\n", TRIVIAL_HARNESS, 1))


def test_output_flood_is_scored(unguarded: Executor) -> None:
    _assert_scored_not_raised(unguarded.run("while True:\n    print('x' * 4096)\n", TRIVIAL_HARNESS, 1))


def test_blocked_import_is_scored_not_raised(executor: Executor) -> None:
    _assert_scored_not_raised(executor.run("import subprocess\n", TRIVIAL_HARNESS, 1))


def test_syntax_error_is_scored_not_raised(executor: Executor) -> None:
    _assert_scored_not_raised(executor.run("def f(:\n", TRIVIAL_HARNESS, 1))


def test_recursion_limit_is_scored(executor: Executor) -> None:
    """Runaway recursion fails its test rather than escaping the harness.

    The status here is "ok", not "raised": the harness caught the
    RecursionError and recorded a failed test, which is the contract working
    one layer earlier than the other adversarial cases. What matters is the
    fraction.
    """
    code = "def f(n):\n    return f(n + 1)\n"
    result = executor.run(code, visible_harness(["assert f(0) == 1"]), 1)
    assert result.pass_fraction == 0.0
    assert result.outcomes == ["fail"]


def test_partial_progress_is_kept(executor: Executor) -> None:
    """A harness that dies partway still reports the tests it reached."""
    code = "def f(n):\n    if n > 1:\n        while True:\n            pass\n    return n\n"
    result = Executor(timeout_seconds=2.0).run(
        code,
        visible_harness(["assert f(0) == 0", "assert f(1) == 1", "assert f(2) == 2"]),
        3,
    )
    assert result.status == "timeout"
    # Two determinate passes, and the third is indeterminate rather than a
    # fail: nothing was learned about it. The fraction is over determinate
    # tests only, and the indeterminate count travels beside it.
    assert result.outcomes == ["pass", "pass", "indeterminate"]
    assert result.n_indeterminate == 1
    assert result.pass_fraction == pytest.approx(1.0)


def test_environment_is_scrubbed(unguarded: Executor) -> None:
    """No inherited API keys, no HOME."""
    code = (
        "import os\n"
        "leaked = [k for k in os.environ if k not in "
        "{'PATH','TMPDIR','PYTHONPATH','PYTHONHASHSEED','PYTHONDONTWRITEBYTECODE','LC_ALL','LANG','PWD'}]\n"
    )
    harness = "__alibi_outcomes__ = [leaked == [] and 'HOME' not in __import__('os').environ]"
    result = unguarded.run(code, harness, 1)
    assert result.all_passed, (result.status, result.detail, result.report)


def test_executions_do_not_share_state(executor: Executor) -> None:
    """A fresh process and a fresh temp dir per execution."""
    first = executor.run("open('marker.txt','w').write('x')\n", TRIVIAL_HARNESS, 1)
    assert first.all_passed, first.detail
    harness = "__alibi_outcomes__ = [not __alibi_exists__]"
    code = "try:\n    open('marker.txt')\n    __alibi_exists__ = True\nexcept OSError:\n    __alibi_exists__ = False\n"
    assert executor.run(code, harness, 1).all_passed


# --------------------------------------------------------------------------
# Fail closed.
# --------------------------------------------------------------------------


def test_missing_required_control_aborts() -> None:
    """A required control that is absent aborts. There is no override flag."""
    report = probe()
    for control in report.controls:
        if control.name == "resource_limits":
            control.available = False
    with pytest.raises(SandboxUnavailable):
        Executor(report=report)


def test_probe_names_every_missing_control() -> None:
    report = probe()
    named = set(report.active) | set(report.missing_required) | set(report.missing_preferred)
    assert {"process_isolation", "resource_limits", "temp_dir", "process_group_kill"} <= named
    assert {"network_namespace", "filesystem_namespace"} <= named


def test_control_summary_is_serialisable_for_env_lock(executor: Executor) -> None:
    import json

    summary = executor.control_summary()
    json.dumps(summary)
    assert "network_namespace_in_use" in summary
    assert "filesystem_namespace_in_use" in summary
    assert "degradations" in summary
