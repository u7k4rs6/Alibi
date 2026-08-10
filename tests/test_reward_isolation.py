"""The reward function cannot reach oracle data.

Written on day 1, before alibi/train/reward.py exists, so that it fails loudly
the day someone wires them together rather than passing quietly because there
was nothing to check.

docs/kickoff/02-technical-architecture.md section 3.3: "`ScoredCompletion`
deliberately has no `oracle` field. The oracle verdict is attached downstream in
the logging record, in a separate type. Enforce it with a test that asserts
`reward_fn`'s signature cannot see oracle data, and keep that test even though
it looks silly."

It looks silly. It is the single assertion that keeps the primary metric
meaningful, because an oracle that reaches the reward stops being ground truth
and becomes another thing the policy is optimising against.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "alibi" / "train"
REWARD_PATH = TRAIN_DIR / "reward.py"

# Anything naming these inside reward code is the failure this test exists for.
ORACLE_NAMES = {
    "oracle",
    "CheatOracle",
    "OracleVerdict",
    "structural_check",
    "behavioural_check",
    "held_out_pass_fraction",
    "held_out_indices",
    "held_out_harness",
    "cheated",
}

# Modules reward code must not import, directly or transitively by name.
FORBIDDEN_MODULES = {"alibi.env.oracle", "alibi.data.cheat"}


def _reward_sources() -> list[Path]:
    """The modules that compute a reward.

    Deliberately only reward.py. Architecture doc section 3.3 permits the oracle
    verdict to be attached downstream in the logging record, and
    alibi/train/loop.py is that downstream: it calls the oracle after the reward
    is already a number. Scanning every train module would forbid the logging
    the doc requires, so the invariant is enforced precisely instead:
    reward.py cannot reach the oracle, ScoredCompletion cannot carry it, and no
    call site may pass oracle data into reward_fn. The last of those is
    test_no_call_site_passes_oracle_data_into_the_reward.
    """
    return [REWARD_PATH] if REWARD_PATH.exists() else []


def _train_sources() -> list[Path]:
    if not TRAIN_DIR.exists():
        return []
    return sorted(p for p in TRAIN_DIR.rglob("*.py") if p.name != "__init__.py")


def test_no_call_site_passes_oracle_data_into_the_reward() -> None:
    """The oracle may be logged. It may never be an argument to the reward.

    This is the invariant that survives the reward and the logging living in
    the same file, which they now do.
    """
    offenders = []
    for path in _train_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if name not in {"reward_fn", "reward_breakdown"}:
                continue
            for argument in [*node.args, *[k.value for k in node.keywords]]:
                text = ast.unparse(argument)
                for term in ORACLE_NAMES:
                    if term in text:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}({text})")
    assert not offenders, "oracle data is passed into the reward:\n" + "\n".join(offenders)


def test_the_reward_is_computed_before_the_oracle_is_consulted() -> None:
    """Ordering, in the one module that does both.

    If the oracle were consulted first, a later edit could feed it in without
    any import or signature changing.
    """
    loop = TRAIN_DIR / "loop.py"
    if not loop.exists():
        pytest.skip("alibi/train/loop.py does not exist yet")
    source = loop.read_text(encoding="utf-8")
    reward_at = source.find("reward_breakdown(")
    oracle_at = source.find("oracle.judge(")
    assert reward_at != -1 and oracle_at != -1
    assert reward_at < oracle_at, "the oracle is consulted before the reward is computed"


def test_oracle_module_exists_so_this_test_is_checking_something() -> None:
    """Guards against the test passing because the oracle was renamed away."""
    assert importlib.util.find_spec("alibi.env.oracle") is not None


def test_reward_code_does_not_import_the_oracle() -> None:
    sources = _reward_sources()
    if not sources:
        pytest.skip("alibi/train/ has no modules yet, day 1. This test starts biting on day 2.")
    offenders = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in FORBIDDEN_MODULES:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_MODULES:
                names = ", ".join(a.name for a in node.names)
                offenders.append(f"{path.relative_to(REPO_ROOT)}: from {node.module} import {names}")
    assert not offenders, "reward code imports the oracle:\n" + "\n".join(offenders)


def test_reward_fn_signature_cannot_see_oracle_data() -> None:
    """The literal requirement from architecture doc section 3.3."""
    if not REWARD_PATH.exists():
        pytest.skip("alibi/train/reward.py does not exist yet, day 1.")
    tree = ast.parse(REWARD_PATH.read_text(encoding="utf-8"))
    reward_fns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "reward_fn"
    ]
    assert reward_fns, "alibi/train/reward.py defines no reward_fn"
    for func in reward_fns:
        args = func.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            annotation = ast.unparse(arg.annotation) if arg.annotation else ""
            assert not (ORACLE_NAMES & set(annotation.replace(".", " ").split())), (
                f"reward_fn parameter {arg.arg!r} is annotated {annotation!r}, which names oracle data"
            )
            assert arg.arg.lower() not in {"oracle", "verdict_oracle", "cheat_oracle"}, (
                f"reward_fn takes a parameter named {arg.arg!r}"
            )


def test_reward_code_never_mentions_oracle_attributes() -> None:
    """Catches reaching through a passed object, which a signature check misses."""
    sources = _reward_sources()
    if not sources:
        pytest.skip("alibi/train/ has no modules yet, day 1.")
    offenders = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ORACLE_NAMES:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: .{node.attr}")
            elif isinstance(node, ast.Name) and node.id in ORACLE_NAMES:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.id}")
    assert not offenders, "reward code reaches oracle data:\n" + "\n".join(offenders)


def test_scored_completion_has_no_oracle_field() -> None:
    """ScoredCompletion is what reward_fn receives. It must not carry a verdict."""
    module = importlib.import_module("alibi.train.reward")
    scored = getattr(module, "ScoredCompletion", None)
    if scored is None:
        pytest.skip("ScoredCompletion is not defined yet.")
    fields = set(getattr(scored, "__annotations__", {}))
    assert not (fields & ORACLE_NAMES), f"ScoredCompletion carries oracle data: {sorted(fields & ORACLE_NAMES)}"


def test_the_oracle_takes_no_executor_so_it_cannot_be_handed_one() -> None:
    """CheatOracle scores from fractions, not from the ability to run code.

    A consequence worth locking down: anything holding an oracle cannot also
    execute, and a verdict can be recomputed from stored artifacts with no
    sandbox, which is what makes `alibi verify --no-gpu` possible.
    """
    import inspect

    from alibi.env.oracle import CheatOracle

    params = set(inspect.signature(CheatOracle.__init__).parameters)
    assert "executor" not in params
    assert params <= {"self", "visible_pass_threshold", "held_out_fail_threshold"}
