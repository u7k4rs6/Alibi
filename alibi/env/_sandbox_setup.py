"""Stage one: build a minimal filesystem view, then exec the runner into it.

Runs inside a fresh user, mount and network namespace created by unshare(1).
It replaces the process's root with a tmpfs containing only the paths the
runner actually needs, then pivot_roots into it and execs the real runner.

Why this exists. docs/kickoff/03-security-and-access.md section 2 lists a fresh
temp directory as the filesystem control, but a fresh cwd does not stop
`open('/etc/passwd')`, and section 2 also requires that exact call to be an
adversarial test. A temp cwd cannot satisfy that test. This closes the gap at
the kernel layer where the host allows an unprivileged user namespace, which is
the only layer where it is a real boundary rather than a convention.

Where the host does not allow one, the executor says so by name in env.lock and
falls back to the language layer confinement in _runner.py, which is defence in
depth and is described as such rather than as isolation.

Everything is bind mounted at its original path, so the runner's command line,
cwd and PYTHONPATH are identical whether or not this stage ran.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys

MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_PRIVATE = 1 << 18
MNT_DETACH = 2

# pivot_root(2) is not exported by glibc, so it goes through syscall(2). The
# number is architecture specific and there is no safe guess for an unknown
# machine, so an unrecognised architecture aborts rather than continuing with a
# root it did not actually change.
SYS_PIVOT_ROOT = {"x86_64": 155, "aarch64": 41, "ppc64le": 203, "s390x": 217}

NEW_ROOT = "/tmp/.alibi-root"
OLD_ROOT_NAME = ".oldroot"


class SetupError(RuntimeError):
    pass


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def _mount(libc, source, target, fstype, flags, data=None) -> None:
    rc = libc.mount(
        source.encode() if source else None,
        target.encode(),
        fstype.encode() if fstype else None,
        ctypes.c_ulong(flags),
        data.encode() if data else None,
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise SetupError(f"mount {source} -> {target}: {os.strerror(err)}")


def _bind(libc, source: str, target: str, read_only: bool) -> None:
    if not os.path.exists(source):
        return
    if os.path.isdir(source):
        os.makedirs(target, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb"):
            pass
    _mount(libc, source, target, None, MS_BIND | MS_REC)
    if read_only:
        # A bind mount does not take its flags from the original call. The
        # read only bit has to be set by a second remount, and if that remount
        # fails the mount is writable, so the failure is raised rather than
        # logged.
        _mount(libc, None, target, None, MS_REMOUNT | MS_BIND | MS_RDONLY | MS_REC | MS_NOSUID | MS_NODEV)


def build_root(spec: dict) -> None:
    libc = _libc()
    machine = os.uname().machine
    if machine not in SYS_PIVOT_ROOT:
        raise SetupError(f"no pivot_root syscall number known for architecture {machine}")

    # Detach this namespace's mount propagation, so nothing done here is
    # visible to the host and nothing on the host leaks in later.
    _mount(libc, None, "/", None, MS_REC | MS_PRIVATE)

    os.makedirs(NEW_ROOT, exist_ok=True)
    _mount(libc, "tmpfs", NEW_ROOT, "tmpfs", MS_NOSUID | MS_NODEV, "size=16m,mode=755")

    for source in spec["read_only"]:
        _bind(libc, source, NEW_ROOT + source, read_only=True)
    for source in spec["read_write"]:
        _bind(libc, source, NEW_ROOT + source, read_only=False)

    # Top level symlinks such as /lib -> usr/lib. Recreated rather than bind
    # mounted, because binding a symlink follows it and would mount the target
    # a second time.
    for link, target in spec["symlinks"].items():
        path = NEW_ROOT + link
        if not os.path.lexists(path):
            os.symlink(target, path)

    # A minimal /dev. CPython reads /dev/urandom during startup on some builds
    # and the random module reads it on first use, so its absence would turn
    # ordinary MBPP solutions into sandbox failures.
    dev = NEW_ROOT + "/dev"
    os.makedirs(dev, exist_ok=True)
    _mount(libc, "tmpfs", dev, "tmpfs", MS_NOSUID, "size=1m,mode=755")
    for node in ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random"):
        _bind(libc, node, NEW_ROOT + node, read_only=False)

    old = os.path.join(NEW_ROOT, OLD_ROOT_NAME)
    os.makedirs(old, exist_ok=True)
    rc = libc.syscall(ctypes.c_long(SYS_PIVOT_ROOT[machine]), NEW_ROOT.encode(), old.encode())
    if rc != 0:
        raise SetupError(f"pivot_root: {os.strerror(ctypes.get_errno())}")

    os.chdir("/")
    if libc.umount2(("/" + OLD_ROOT_NAME).encode(), ctypes.c_int(MNT_DETACH)) != 0:
        raise SetupError(f"umount2 old root: {os.strerror(ctypes.get_errno())}")
    os.rmdir("/" + OLD_ROOT_NAME)


def main() -> int:
    spec = json.loads(open(sys.argv[1], encoding="utf-8").read())
    build_root(spec)
    os.chdir(spec["cwd"])
    argv = spec["argv"]
    os.execv(argv[0], argv)
    return 3  # pragma: no cover - execv does not return


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SetupError as exc:
        # Fail closed. The parent sees a non zero exit and a named reason, and
        # scores the execution rather than falling back to a weaker sandbox
        # without saying so.
        print(f"alibi sandbox setup failed: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
