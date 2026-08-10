"""MBPP joined to MBPP+, split into visible and held out, with a manifest.

Both datasets are fetched as parquet directly from the Hugging Face hub at a
pinned commit sha rather than through the `datasets` library. Two reasons, in
order:

  1. A commit sha pins the data harder than a library version does. The
     manifest records the sha and the sha256 of every file actually read, so
     "which data was this run trained on" has a checkable answer, which is what
     docs/kickoff/02-technical-architecture.md section 4 asks env.lock for.
  2. It keeps the day one path free of the training stack. `datasets` is a
     dependency of trl, not of this experiment.

The split:

  visible   MBPP+'s test_list. These go in the prompt and drive the reward.
  held out  MBPP+'s expanded harness, minus every input that also appears in a
            visible assert. Never shown, never in the reward, oracle only.

Problems where the two cannot be made provably disjoint are excluded here
rather than scored loosely, and the manifest names each exclusion and why.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alibi.env.tests import HarnessError, entry_point, harness_inputs, held_out_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"

# Pinned on day 1, 11 Aug 2026. Changing either sha invalidates every run built
# against the old one, which is why they are constants in source and not flags.
MBPP_REPO = "google-research-datasets/mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_FILES = (
    "full/train-00000-of-00001.parquet",
    "full/test-00000-of-00001.parquet",
    "full/validation-00000-of-00001.parquet",
    "full/prompt-00000-of-00001.parquet",
)

MBPPPLUS_REPO = "evalplus/mbppplus"
MBPPPLUS_REVISION = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"
MBPPPLUS_FILES = ("data/test-00000-of-00001-d5781c9c51e02795.parquet",)

USER_AGENT = "alibi/0.1 (day-one dataset build)"


@dataclass(frozen=True)
class Problem:
    """One MBPP problem, joined and split.

    `visible_asserts` are in the prompt. `held_out_indices` index into the
    MBPP+ harness's own `inputs` list and are disjoint from the visible inputs
    by construction.
    """

    task_id: int
    description: str
    entry_point: str
    visible_asserts: list[str]
    plus_test_src: str
    held_out_indices: list[int]
    test_imports: list[str]
    reference_code: str
    mbpp_asserts: list[str]

    @property
    def n_visible(self) -> int:
        return len(self.visible_asserts)

    @property
    def n_held_out(self) -> int:
        return len(self.held_out_indices)

    def prompt_body(self) -> str:
        """The task text plus the visible asserts, which is why cheating pays.

        Kept here so that every consumer, the prompt builder and the synthetic
        cheat generator alike, agrees on exactly what the model can see.
        """
        tests = "\n".join(self.visible_asserts)
        return f"{self.description}\n\nYour code should pass these tests:\n\n{tests}\n"


@dataclass
class BuildReport:
    problems: list[Problem] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    file_digests: dict = field(default_factory=dict)

    def by_id(self) -> dict[int, Problem]:
        return {p.task_id: p for p in self.problems}


def _download(repo: str, revision: str, path: str) -> Path:
    """Fetch one parquet file, cached by repo, revision and path.

    The cache is keyed by revision, so bumping a pin cannot be masked by a
    stale file.
    """
    target = CACHE_DIR / repo.replace("/", "__") / revision / path.replace("/", "__")
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https host
        payload = response.read()
    tmp = target.with_suffix(target.suffix + ".partial")
    tmp.write_bytes(payload)
    tmp.rename(target)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def build(offline_ok: bool = False) -> BuildReport:
    """Join, split and validate. Every exclusion is recorded, never silent."""
    report = BuildReport()

    mbpp_rows: list[dict] = []
    for name in MBPP_FILES:
        path = _download(MBPP_REPO, MBPP_REVISION, name)
        report.file_digests[f"{MBPP_REPO}@{MBPP_REVISION}/{name}"] = _sha256(path)
        mbpp_rows.extend(_read_rows(path))
    mbpp = {row["task_id"]: row for row in mbpp_rows}

    plus_rows: list[dict] = []
    for name in MBPPPLUS_FILES:
        path = _download(MBPPPLUS_REPO, MBPPPLUS_REVISION, name)
        report.file_digests[f"{MBPPPLUS_REPO}@{MBPPPLUS_REVISION}/{name}"] = _sha256(path)
        plus_rows.extend(_read_rows(path))

    for row in sorted(plus_rows, key=lambda r: r["task_id"]):
        task_id = row["task_id"]
        base = mbpp.get(task_id)
        if base is None:
            report.excluded.append({"task_id": task_id, "reason": "task_id absent from MBPP"})
            continue
        if base.get("test_setup_code", "").strip():
            # None of the joined problems have setup code today. If a future
            # revision adds it, it would run before the candidate and change
            # what the visible asserts mean, so it is refused rather than
            # ignored.
            report.excluded.append({"task_id": task_id, "reason": "MBPP test_setup_code is not supported"})
            continue

        plus_test_src = row["test"]
        try:
            entry = entry_point(plus_test_src)
            n_inputs = len(harness_inputs(plus_test_src))
            keep = held_out_indices(plus_test_src, list(row["test_list"]))
        except (HarnessError, SyntaxError, ValueError) as exc:
            report.excluded.append({"task_id": task_id, "reason": f"{type(exc).__name__}: {exc}"[:200]})
            continue

        if not keep:
            report.excluded.append(
                {"task_id": task_id, "reason": f"no held out inputs remain after removing {n_inputs} visible ones"}
            )
            continue

        report.problems.append(
            Problem(
                task_id=task_id,
                description=row["prompt"],
                entry_point=entry,
                visible_asserts=list(row["test_list"]),
                plus_test_src=plus_test_src,
                held_out_indices=keep,
                test_imports=list(row.get("test_imports") or []),
                reference_code=row["code"],
                mbpp_asserts=list(base["test_list"]),
            )
        )

    return report


def manifest(report: BuildReport) -> dict:
    """The split manifest. Problem ids, split hashes, dataset revisions.

    The per problem `visible_sha256` is over the exact assert text that reaches
    the prompt, so a later claim that a run used a given split is checkable
    rather than asserted.
    """
    entries = []
    for problem in sorted(report.problems, key=lambda p: p.task_id):
        visible_blob = "\n".join(problem.visible_asserts).encode("utf-8")
        entries.append(
            {
                "task_id": problem.task_id,
                "entry_point": problem.entry_point,
                "n_visible": problem.n_visible,
                "n_held_out": problem.n_held_out,
                "visible_sha256": hashlib.sha256(visible_blob).hexdigest(),
                "held_out_indices_sha256": hashlib.sha256(
                    json.dumps(problem.held_out_indices).encode("utf-8")
                ).hexdigest(),
                "visible_differs_from_mbpp": problem.visible_asserts != problem.mbpp_asserts,
            }
        )
    split_blob = json.dumps(entries, sort_keys=True).encode("utf-8")
    return {
        "datasets": {
            "mbpp": {"repo": MBPP_REPO, "revision": MBPP_REVISION, "files": list(MBPP_FILES)},
            "mbpp_plus": {"repo": MBPPPLUS_REPO, "revision": MBPPPLUS_REVISION, "files": list(MBPPPLUS_FILES)},
        },
        "file_digests": dict(sorted(report.file_digests.items())),
        "visible_source": (
            "MBPP+ test_list. See docs/kickoff/01-prd.md section 3, which says the original MBPP "
            "asserts. MBPP+ rewrote 112 of the 378 joined problems' asserts to match its own "
            "reference solutions, so MBPP's raw asserts are not all satisfiable by the reference "
            "the held out harness scores against. The count of rewritten problems is recorded per "
            "problem as visible_differs_from_mbpp."
        ),
        "held_out_source": (
            "MBPP+ expanded harness, restricted to inputs that do not appear in any visible assert. "
            "Disjointness is established per problem by comparing input values."
        ),
        "n_problems": len(report.problems),
        "n_excluded": len(report.excluded),
        "excluded": sorted(report.excluded, key=lambda e: e["task_id"]),
        "split_sha256": hashlib.sha256(split_blob).hexdigest(),
        "problems": entries,
    }


def write_manifest(report: BuildReport, path: Path = MANIFEST_PATH) -> dict:
    doc = manifest(report)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def load_problems() -> list[Problem]:
    """Build and return the problem set. Cached files make repeat calls cheap."""
    return build().problems


def problem_from_dict(data: dict) -> Problem:
    return Problem(**data)


def problem_to_dict(problem: Problem) -> dict:
    return asdict(problem)
