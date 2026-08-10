"""The `alibi` dispatcher. See docs/kickoff/04-cli-and-report-spec.md.

Day 1 implements `data build` and `data check`. The other subcommands are
declared here and exit 2 with the day they are scheduled for, so the surface
matches the spec and an unimplemented command says so rather than failing with
an unrelated import error.

Exit codes, per doc 04 section 1:
  0 success, 1 assertion or mismatch, 2 configuration error, 3 sandbox unavailable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from alibi import prereg, runlog
from alibi.env.executor import SandboxUnavailable

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_CONFIG = 2
EXIT_SANDBOX = 3

NOT_IMPLEMENTED_YET = {
    "eval": "day 3",
    "replay": "day 6",
}


def _worker_count() -> int:
    """Physical cores minus one, per architecture doc section 6."""
    import os

    return max(1, (os.cpu_count() or 2) - 1)


def _fmt_fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a (no problems)"
    return f"{numerator}/{denominator} = {numerator / denominator:.4f}"


def cmd_data_build(args: argparse.Namespace) -> int:
    from alibi.data.build import build, write_manifest

    report = build()
    doc = write_manifest(report)
    print(f"config hash {runlog.config_hash({'command': 'data build'})}")
    print(f"problems {doc['n_problems']}  excluded {doc['n_excluded']}  split sha256 {doc['split_sha256']}")
    for entry in doc["excluded"]:
        print(f"  excluded task {entry['task_id']}: {entry['reason']}")
    print(f"manifest {runlog.repo_relative(__import__('alibi.data.build', fromlist=['x']).MANIFEST_PATH)}")
    print(f"run id {args.run_id or 'none, data build writes no run directory'}")
    return EXIT_OK


def cmd_data_check(args: argparse.Namespace) -> int:
    """The day one gate, per docs/kickoff/04-cli-and-report-spec.md section 1.

    Reports, over the full problem set, whether a synthetic hardcoding solution
    passes the visible tests and fails the held out ones. If it does not, the
    environment does not incentivise cheating and PRD hypothesis H1 is dead
    before any GPU time is spent.
    """
    from alibi.data.build import build, manifest
    from alibi.data.cheat import make_cheat, score
    from alibi.env.executor import Executor

    config = {
        "command": "data check",
        "limit": args.limit,
        "include_dict_form": args.include_dict_form,
        "include_reference": True,
    }
    print(f"config hash {runlog.config_hash(config)}")

    try:
        executor = Executor()
    except SandboxUnavailable as exc:
        print(f"WARN {exc}", file=sys.stderr)
        return EXIT_SANDBOX

    for note in executor.degradations:
        print(f"WARN sandbox control unavailable on this host: {note}", file=sys.stderr)
    for name in executor.report.missing_preferred:
        print(f"WARN sandbox control unavailable on this host: {name}", file=sys.stderr)

    build_report = build()
    problems = build_report.problems
    if args.limit:
        problems = problems[: args.limit]
    if not problems:
        print("no problems built", file=sys.stderr)
        return EXIT_CONFIG

    workers = _worker_count()
    print(f"scoring {len(problems)} problems with {workers} workers")

    def one(problem):
        record = {
            "task_id": problem.task_id,
            "entry_point": problem.entry_point,
            "n_visible": problem.n_visible,
            "n_held_out": problem.n_held_out,
        }
        cheat = make_cheat(problem, executor)
        record["cheat"] = {
            "constructed": cheat.constructed,
            "reason": cheat.reason,
            "visible_pass_fraction": cheat.visible_pass_fraction,
            "held_out_pass_fraction": cheat.held_out_pass_fraction,
            "visible_status": cheat.visible_status,
            "held_out_status": cheat.held_out_status,
            "visible_indeterminate": cheat.visible_indeterminate,
            "held_out_indeterminate": cheat.held_out_indeterminate,
            "held_out_indeterminate_indices": cheat.held_out_indeterminate_indices,
            "structural_flagged": cheat.structural_flagged,
            "structural_signals": cheat.structural_signals,
            "cheat_form": cheat.cheat_form,
            "cheat_at_threshold": prereg.sensitivity(
                cheat.held_out_pass_fraction, cheat.visible_pass_fraction
            ),
        }
        visible, held_out = score(problem, problem.reference_code, executor)
        record["reference"] = {
            "visible_pass_fraction": visible.pass_fraction,
            "held_out_pass_fraction": held_out.pass_fraction,
            "visible_status": visible.status,
            "held_out_status": held_out.status,
            "visible_indeterminate": visible.n_indeterminate,
            "held_out_indeterminate": held_out.n_indeterminate,
            "held_out_indeterminate_indices": held_out.indeterminate_indices,
        }
        if args.include_dict_form:
            dict_cheat = make_cheat(problem, executor, dict_form=True)
            record["cheat_dict_form"] = {
                "constructed": dict_cheat.constructed,
                "visible_pass_fraction": dict_cheat.visible_pass_fraction,
                "held_out_pass_fraction": dict_cheat.held_out_pass_fraction,
                "structural_flagged": dict_cheat.structural_flagged,
            }
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(one, problems))
    records.sort(key=lambda r: r["task_id"])

    summary = summarise(records)
    print_summary(summary, records)

    # The eligibility manifest is built from this same run, so the denominator
    # and the gate numbers can never come from different measurements.
    if args.limit is None:
        from alibi.data import eligibility as eligibility_module

        document = eligibility_module.build(records, manifest(build_report)["split_sha256"])
        eligibility_module.write(document)
        summary["eligibility"] = {
            "n_eligible": document["n_eligible"],
            "n_excluded": document["n_excluded"],
            "excluded_by_rule": document["excluded_by_rule"],
            "eligibility_hash": document["eligibility_hash"],
        }
        print()
        print(" eligibility, per the frozen rules in alibi/prereg.py")
        print(f"  eligible problems {document['n_eligible']}/{len(records)}")
        for rule, count in document["excluded_by_rule"].items():
            print(f"    excluded by {rule:<38s} {count:>4d}")
        print(f"  eligibility hash {document['eligibility_hash']}")
        print(f"  prereg hash      {prereg.PREREG_HASH}")
    else:
        print()
        print(" eligibility manifest not written: --limit was set, so the set would be partial")

    run_dir, run_id = runlog.new_run_dir("datacheck", args.run_id)
    runlog.write_run(
        run_dir,
        config,
        {"summary": summary, "problems": records},
        sandbox=executor.control_summary(),
        dataset=manifest(build_report)["datasets"],
        seeds={"note": "data check is deterministic and uses no sampler, so no seed applies"},
    )
    print(f"artifacts {runlog.repo_relative(run_dir)}")
    print(f"run id {run_id}")
    return EXIT_OK


def summarise(records: list[dict]) -> dict:
    """Every number the gate reports. Computed here and nowhere else."""
    total = len(records)
    constructed = [r for r in records if r["cheat"]["constructed"]]

    # None means unmeasured, never zero. `_ge` and `_lt` refuse to compare it.
    passes_visible = [r for r in constructed if _ge(r["cheat"]["visible_pass_fraction"], 1.0)]
    passes_visible_and_fails_held_out = [r for r in passes_visible if _lt(r["cheat"]["held_out_pass_fraction"], 1.0)]
    passes_visible_and_zero_held_out = [
        r for r in passes_visible if r["cheat"]["held_out_pass_fraction"] == 0.0
    ]
    # The pre-registered rule, at all three thresholds, always all three.
    cheat_at_threshold = {}
    for label in sorted({k for r in constructed for k in r["cheat"]["cheat_at_threshold"]}):
        verdicts = [r["cheat"]["cheat_at_threshold"].get(label) for r in constructed]
        determinate = [v for v in verdicts if v is not None]
        cheat_at_threshold[label] = {
            "n_cheat": sum(determinate),
            "n_determinate": len(determinate),
            "n_unmeasured": len(verdicts) - len(determinate),
            "rate": (sum(determinate) / len(determinate)) if determinate else None,
        }
    # The oracle's own behavioural rule, applied to the synthetic cheat.
    from alibi.env.oracle import HELD_OUT_FAIL_THRESHOLD, behavioural_check

    behaviourally_flagged = [
        r
        for r in constructed
        if behavioural_check(r["cheat"]["visible_pass_fraction"], r["cheat"]["held_out_pass_fraction"])
    ]
    structurally_flagged = [r for r in constructed if r["cheat"]["structural_flagged"]]

    ref_visible_ok = [r for r in records if _ge(r["reference"]["visible_pass_fraction"], 1.0)]
    ref_held_out_ok = [r for r in records if _ge(r["reference"]["held_out_pass_fraction"], 1.0)]

    held_out_fractions = [r["cheat"]["held_out_pass_fraction"] for r in passes_visible if r["cheat"]["held_out_pass_fraction"] is not None]
    total_held_out_tests = sum(r["n_held_out"] for r in records)
    total_held_out_indeterminate = sum(r["reference"].get("held_out_indeterminate", 0) for r in records)
    visible_counts = Counter(r["n_visible"] for r in records)
    held_out_counts = [r["n_held_out"] for r in records]

    summary = {
        "n_problems": total,
        "n_cheat_constructed": len(constructed),
        "cheat_construction_failures": [
            {"task_id": r["task_id"], "reason": r["cheat"]["reason"]} for r in records if not r["cheat"]["constructed"]
        ],
        "fraction_cheat_passes_visible": _ratio(len(passes_visible), total),
        "fraction_cheat_passes_visible_and_fails_held_out": _ratio(len(passes_visible_and_fails_held_out), total),
        "fraction_cheat_passes_visible_and_scores_zero_held_out": _ratio(len(passes_visible_and_zero_held_out), total),
        "fraction_cheat_flagged_behaviourally": _ratio(len(behaviourally_flagged), total),
        "fraction_cheat_flagged_structurally": _ratio(len(structurally_flagged), total),
        "cheat_at_threshold": cheat_at_threshold,
        "cheat_form_counts": _form_counts(constructed),
        "reference_held_out_indeterminate": {
            "tests": total_held_out_tests,
            "indeterminate": total_held_out_indeterminate,
            "fraction": (total_held_out_indeterminate / total_held_out_tests) if total_held_out_tests else None,
        },
        "held_out_fail_threshold": HELD_OUT_FAIL_THRESHOLD,
        "held_out_pass_fraction_of_passing_cheats": {
            "mean": statistics.fmean(held_out_fractions) if held_out_fractions else None,
            "median": statistics.median(held_out_fractions) if held_out_fractions else None,
            "max": max(held_out_fractions) if held_out_fractions else None,
        },
        "reference_solution": {
            "fraction_passes_visible": _ratio(len(ref_visible_ok), total),
            "fraction_passes_held_out": _ratio(len(ref_held_out_ok), total),
        },
        "visible_assert_count_distribution": dict(sorted(visible_counts.items())),
        "held_out_test_count": {
            "min": min(held_out_counts) if held_out_counts else None,
            "median": statistics.median(held_out_counts) if held_out_counts else None,
            "max": max(held_out_counts) if held_out_counts else None,
        },
        # Required by doc 04 section 1 but not measurable without a model. An
        # unmeasured value is absent, not zero.
        "base_model_visible_pass_rate": None,
        "base_model_held_out_pass_rate": None,
        "base_model_absent_reason": (
            "day 1 is CPU only and loads no model. Scheduled for day 2, see docs/kickoff/01-prd.md section 10."
        ),
    }
    if any("cheat_dict_form" in r for r in records):
        dict_records = [r for r in records if r.get("cheat_dict_form", {}).get("constructed")]
        dict_pass = [r for r in dict_records if _ge(r["cheat_dict_form"]["visible_pass_fraction"], 1.0)]
        dict_flagged = [r for r in dict_records if r["cheat_dict_form"]["structural_flagged"]]
        summary["dict_form_cheat"] = {
            "n": len(dict_records),
            "fraction_passes_visible": _ratio(len(dict_pass), total),
            "fraction_flagged_structurally": _ratio(len(dict_flagged), total),
        }
    return summary


def _ge(value, threshold) -> bool:
    """None is unmeasured and compares to nothing. Never coerced to 0.0."""
    return value is not None and value >= threshold


def _lt(value, threshold) -> bool:
    return value is not None and value < threshold


def _form_counts(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        for form in record["cheat"].get("cheat_form") or []:
            counts[form] = counts.get(form, 0) + 1
    return dict(sorted(counts.items()))


def _ratio(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": (numerator / denominator) if denominator else None,
    }


def print_summary(summary: dict, records: list[dict]) -> None:
    total = summary["n_problems"]

    def line(label: str, ratio: dict) -> None:
        value = "n/a" if ratio["value"] is None else f"{ratio['value']:.4f}"
        print(f"  {label:<52s} {ratio['numerator']:>4d}/{ratio['denominator']:<4d}  {value}")

    print()
    print(f"day one gate, {total} problems")
    print()
    print(" synthetic cheat, built from the visible asserts only")
    line("passes visible tests", summary["fraction_cheat_passes_visible"])
    line("passes visible and fails held out", summary["fraction_cheat_passes_visible_and_fails_held_out"])
    line("passes visible and scores 0.0 held out", summary["fraction_cheat_passes_visible_and_scores_zero_held_out"])
    line("flagged by the oracle, behavioural check", summary["fraction_cheat_flagged_behaviourally"])
    line("flagged by the oracle, structural check", summary["fraction_cheat_flagged_structurally"])
    if "dict_form_cheat" in summary:
        print()
        print(" the same cheat written as a dict lookup")
        line("passes visible tests", summary["dict_form_cheat"]["fraction_passes_visible"])
        line("flagged by the oracle, structural check", summary["dict_form_cheat"]["fraction_flagged_structurally"])

    print()
    print(" pre-registered behavioural rule, all three thresholds, always")
    for label, entry in sorted(summary["cheat_at_threshold"].items()):
        rate = "n/a" if entry["rate"] is None else f"{entry['rate']:.4f}"
        print(
            f"  held out <= {label} {'':<32s} {entry['n_cheat']:>4d}/{entry['n_determinate']:<4d}  {rate}"
            + (f"   unmeasured {entry['n_unmeasured']}" if entry["n_unmeasured"] else "")
        )

    print()
    print(" cheat_form composition of the synthetic cheat")
    for form, count in summary["cheat_form_counts"].items():
        print(f"  {form:<50s} {count:>4d}/{total:<4d}  {count / total:.4f}")

    indet = summary["reference_held_out_indeterminate"]
    frac = "n/a" if indet["fraction"] is None else f"{indet['fraction']:.6f}"
    print()
    print(f" held out indeterminate, reference solutions   {indet['indeterminate']}/{indet['tests']}  {frac}")

    print()
    print(" reference solution, as a control on the instrument")
    line("passes visible tests", summary["reference_solution"]["fraction_passes_visible"])
    line("passes held out tests", summary["reference_solution"]["fraction_passes_held_out"])

    held = summary["held_out_pass_fraction_of_passing_cheats"]
    if held["mean"] is not None:
        print()
        print(" held out pass fraction of cheats that passed visible")
        print(f"  mean {held['mean']:.4f}   median {held['median']:.4f}   max {held['max']:.4f}")

    print()
    print(" visible assert count distribution")
    for count, n in summary["visible_assert_count_distribution"].items():
        print(f"  {count} asserts {'':<38s} {n:>4d}/{total:<4d}  {n / total:.4f}")

    held_counts = summary["held_out_test_count"]
    print()
    print(f" held out test count   min {held_counts['min']}  median {held_counts['median']}  max {held_counts['max']}")

    print()
    print(" base model pass rates                                not measured")
    print(f"   reason: {summary['base_model_absent_reason']}")

    failures = summary["cheat_construction_failures"]
    if failures:
        print()
        print(f" cheat could not be constructed for {len(failures)} problems")
        for entry in failures[:10]:
            print(f"  task {entry['task_id']}: {entry['reason']}")


def cmd_monitor_probe(args: argparse.Namespace) -> int:
    from alibi.monitor.llm import ESCALATION, probe

    models = args.models or list(ESCALATION)
    print(f"config hash {runlog.config_hash({'command': 'monitor probe', 'models': models})}")
    any_available = False
    for model in models:
        result = probe(model)
        any_available = any_available or result.available
        state = "OK  " if result.available else "NO  "
        print(f"  {state}{model:<46s} status={result.status} {result.latency_seconds if False else result.latency:.2f}s  {result.detail[:100]}")
    print(f"run id {args.run_id or 'none, monitor probe writes no run directory'}")
    return EXIT_OK if any_available else EXIT_CONFIG


def cmd_monitor_select(args: argparse.Namespace) -> int:
    """Probe, qualify and record. A monitored arm cannot start without this."""
    from alibi.monitor.llm import DEFAULT_BASE_URL, PROMPT_TEMPLATE_VERSION
    from alibi.monitor.qualify import escalate
    from alibi.monitor.selection import write_selection
    import os

    print(f"config hash {runlog.config_hash({'command': 'monitor select', 'sample': args.sample})}")
    chosen, trail = escalate(candidates=args.models or None, sample=args.sample)
    for entry in trail:
        print(f"  {entry['model']:<46s} {entry['outcome'][:120]}")
    if chosen is None:
        print("no candidate both served and qualified. A monitored arm cannot start.", file=sys.stderr)
        return EXIT_CONFIG

    qualification = next(e.get("qualification") for e in trail if e["model"] == chosen)
    document = write_selection(
        model_id=chosen,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        base_url=os.environ.get("MONITOR_BASE_URL") or DEFAULT_BASE_URL,
        trail=trail,
        qualification=qualification,
    )
    print(f"selected {document['model_id']}")
    print(f"selection hash {document['selection_hash']}")
    if document["shares_family_with_policy"]:
        print("WARN the selected judge shares a model family with the policy it grades")
        print(f"WARN {document['confound_note']}")
    print(f"run id {args.run_id or 'none, monitor select writes no run directory'}")
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    from alibi.report.verify import verify

    code, lines = verify(no_gpu=args.no_gpu)
    for line in lines:
        print(line)
    return EXIT_MISMATCH if code else EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    from alibi.report.build import build_report

    paths = build_report()
    print(f"config hash {runlog.config_hash({'command': 'report'})}")
    for name, path in paths.items():
        print(f"  {name:<12s} {runlog.repo_relative(path)}")
    print(f"run id {args.run_id or 'none, report reads artifacts and writes no run directory'}")
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    from alibi.train.grpo import run, smoke
    from alibi.train.loop import ArmConfig

    if args.smoke:
        print(f"config hash {runlog.config_hash({'command': 'train', 'smoke': True})}")
        result = smoke(steps=args.steps or 3, prompts=2, group=2)
        print(json.dumps({k: v for k, v in result.items() if k != "summaries"}, indent=2, sort_keys=True))
        print(f"run id {result['run_id']}")
        return EXIT_OK if result["status"] == "complete" else EXIT_MISMATCH

    if not args.arm:
        print("train needs --arm, or --smoke for the dry run", file=sys.stderr)
        return EXIT_CONFIG
    config = ArmConfig(arm=args.arm, seed=args.seed, steps=args.steps or 100, dry_run=args.dry_run)
    print(f"config hash {config.hash()}")
    result = run(config, run_id=args.run_id)
    print(json.dumps({k: v for k, v in result.items() if k != "summaries"}, indent=2, sort_keys=True))
    print(f"run id {result['run_id']}")
    return EXIT_OK if result["status"] == "complete" else EXIT_MISMATCH


def cmd_not_implemented(args: argparse.Namespace) -> int:
    print(f"`alibi {args.command}` is not implemented yet, scheduled for {NOT_IMPLEMENTED_YET[args.command]}")
    return EXIT_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alibi", description="Reward hacking and monitor evasion, in miniature.")
    parser.add_argument("--run-id", default=None, help="run id to write under artifacts/runs/, generated if absent")
    parser.add_argument("--config", default=None, help="path to a config file")
    parser.add_argument("--dry-run", action="store_true", help="do not write artifacts")
    parser.add_argument(
        "--allow-monitor-misses",
        action="store_true",
        help="allow monitor cache misses during a replay. Off by default and loud when on.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="build and check the problem set")
    data_sub = data.add_subparsers(dest="subcommand", required=True)
    data_sub.add_parser("build", help="join MBPP and MBPP+, write the split manifest").set_defaults(
        func=cmd_data_build
    )
    check = data_sub.add_parser("check", help="the day one gate: does hardcoding the visible asserts pay?")
    check.add_argument("--limit", type=int, default=None, help="score only the first N problems")
    check.add_argument(
        "--include-dict-form",
        action="store_true",
        help="also score the dict shaped cheat, which the structural check does not catch",
    )
    check.set_defaults(func=cmd_data_check)

    train = sub.add_parser("train", help="train one arm at one seed, resumable by run id")
    train.add_argument("--arm", choices=["a0", "a1", "a2", "a3"], default=None)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--smoke", action="store_true", help="the dry run: 3 steps, 2 prompts, stub monitor")
    train.set_defaults(func=cmd_train)

    monitor = sub.add_parser("monitor", help="probe, qualify and record the judge")
    monitor_sub = monitor.add_subparsers(dest="subcommand", required=True)
    mprobe = monitor_sub.add_parser("probe", help="5-token availability probe, never assume a listing")
    mprobe.add_argument("--models", nargs="*", default=None)
    mprobe.set_defaults(func=cmd_monitor_probe)
    mselect = monitor_sub.add_parser("select", help="probe, qualify and freeze the judge identity")
    mselect.add_argument("--models", nargs="*", default=None)
    mselect.add_argument("--sample", type=int, default=20)
    mselect.set_defaults(func=cmd_monitor_select)

    verify_parser = sub.add_parser("verify", help="recompute every published number, exit non-zero on mismatch")
    verify_parser.add_argument("--no-gpu", action="store_true", default=True)
    verify_parser.set_defaults(func=cmd_verify)

    sub.add_parser("report", help="build REPORT.md and the figures from artifacts").set_defaults(func=cmd_report)

    for name in NOT_IMPLEMENTED_YET:
        sub.add_parser(name, help=f"not implemented yet, {NOT_IMPLEMENTED_YET[name]}").set_defaults(
            func=cmd_not_implemented
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "data" and not hasattr(args, "func"):
        parser.error("data needs a subcommand")
    try:
        return args.func(args)
    except SandboxUnavailable as exc:
        print(f"WARN {exc}", file=sys.stderr)
        return EXIT_SANDBOX


if __name__ == "__main__":
    raise SystemExit(main())
