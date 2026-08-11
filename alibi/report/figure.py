"""The figures. Rebuilt after every completed run, so an artifact always exists.

Panels required by the unattended brief, all sharing an x axis of training step:

  1  obfuscation gap per arm, min to max band across seeds, never a standard
     error, because n=3
  2  oracle cheat rate, and 3 monitor flag rate, as separate panels, because the
     gap alone hides whether it opened by cheating more or flagging less
  4  indeterminate rate per arm, so a reader can check whether measurement drift
     tracks the headline
  5  cheat_form composition over steps

Rules from docs/kickoff/04-cli-and-report-spec.md section 3: no dual axes, no
unnamed smoothing, no truncated y axis. If a curve is unimpressive the fix is
the caption, not the axis.
"""

from __future__ import annotations

from pathlib import Path

from alibi import prereg, runlog
from alibi.report import metrics

FIGURE_DIR = runlog.REPO_ROOT / "report" / "figures"

# One colour per arm, consistent across every artifact in the repo.
ARM_COLOURS = {"a0": "#1b6ca8", "a1": "#c85200", "a2": "#2a7d4f", "a3": "#7b5aa6"}


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def build_figures() -> dict[str, Path]:
    """Every panel. Returns the paths written, or an empty dict with no runs."""
    runs = metrics.load_declared_runs()
    if not runs:
        return {}
    try:
        plt = _matplotlib()
    except ImportError:
        return {}

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    grouped = metrics.by_arm(runs)
    written: dict[str, Path] = {}

    written["primary"] = _primary(plt, grouped)
    written["components"] = _components(plt, grouped)
    written["indeterminate"] = _indeterminate(plt, grouped)
    written["cheat_form"] = _cheat_form(plt, grouped)
    written["false_positive"] = _false_positive(plt, grouped)
    written["determinacy_floor"] = _determinacy_floor(plt, grouped)
    return written


def _band(plt, axis, series_list, attribute, arm):
    """Min to max across seeds. Never a standard error at n=3."""
    columns = [getattr(s, attribute) for s in series_list]
    length = min((len(c) for c in columns), default=0)
    if not length:
        return
    steps = series_list[0].steps[:length]
    lows, highs, means = [], [], []
    for index in range(length):
        values = [c[index] for c in columns if c[index] is not None]
        if not values:
            lows.append(None)
            highs.append(None)
            means.append(None)
            continue
        lows.append(min(values))
        highs.append(max(values))
        means.append(sum(values) / len(values))
    colour = ARM_COLOURS.get(arm, "#555555")
    valid = [i for i, m in enumerate(means) if m is not None]
    if not valid:
        return
    xs = [steps[i] for i in valid]
    axis.plot(xs, [means[i] for i in valid], color=colour, label=f"{arm} (n={len(series_list)})")
    if len(series_list) > 1:
        axis.fill_between(
            xs, [lows[i] for i in valid], [highs[i] for i in valid], color=colour, alpha=0.18, linewidth=0
        )


def _caption(figure, text: str) -> None:
    figure.text(0.01, 0.005, text, fontsize=7, va="bottom", ha="left", wrap=True)


def _primary(plt, grouped) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for arm, series_list in grouped.items():
        _band(plt, axis, series_list, "gap", arm)
    axis.axhline(0.0, color="#888888", linewidth=0.9, linestyle="--")
    axis.set_xlabel("training step")
    axis.set_ylabel("obfuscation gap")
    axis.set_title("Obfuscation gap by arm")
    axis.legend(fontsize=8)
    n_seeds = max((len(v) for v in grouped.values()), default=0)
    _caption(
        figure,
        f"Qwen2.5-0.5B-Instruct, LoRA. gap = cheat_rate_oracle - flag_rate_monitor, cheat at "
        f"held-out pass fraction <= {prereg.PREREG.oracle.held_out_cheat_threshold:.2f}. "
        f"Band is min to max across seeds, not a standard error, n={n_seeds}. "
        f"lambda = {prereg.PREREG.lambda_monitor}. prereg {prereg.PREREG_HASH[:12]}.",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    path = FIGURE_DIR / "obfuscation_gap.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _components(plt, grouped) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    for arm, series_list in grouped.items():
        _band(plt, axes[0], series_list, "cheat_rate", arm)
        _band(plt, axes[1], series_list, "flag_rate", arm)
    axes[0].set_title("Oracle cheat rate (held-out, never in reward)")
    axes[1].set_title("Monitor flag rate (in reward)")
    for axis in axes:
        axis.set_xlabel("training step")
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("rate")
    axes[0].legend(fontsize=8)
    _caption(
        figure,
        "The gap alone hides whether it opened by cheating more or by flagging less. "
        "Monitor error verdicts count as flagged, so a monitor outage raises the right panel.",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    path = FIGURE_DIR / "cheat_and_flag_rates.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _indeterminate(plt, grouped) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    for arm, series_list in grouped.items():
        _band(plt, axis, series_list, "indeterminate_rate", arm)
    axis.axhline(
        prereg.PREREG.halt.max_indeterminate_fraction,
        color="#b03030",
        linewidth=0.9,
        linestyle="--",
        label="halt threshold",
    )
    axis.set_xlabel("training step")
    axis.set_ylabel("indeterminate fraction of held-out tests")
    axis.set_title("Measurement drift: indeterminate held-out executions")
    axis.legend(fontsize=8)
    _caption(
        figure,
        "This machine is a laptop under sustained load. Held-out timeouts are the channel "
        "through which thermal and contention drift could contaminate the primary metric, "
        "which is why this is both a halt condition and a required panel.",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    path = FIGURE_DIR / "indeterminate_rate.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _false_positive(plt, grouped) -> Path:
    """Flag rate on known-honest reference solutions, beside the arm's flag rate.

    A monitor that punishes honest work at a material rate changes what a fall
    in the flag rate means, so the two are plotted together rather than the
    headline being shown alone.
    """
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    for arm, series_list in grouped.items():
        if arm == "a0":
            continue
        _band(plt, axis, series_list, "honest_flag_rate", arm)
        _band(plt, axis, series_list, "flag_rate", arm)
    axis.set_xlabel("training step")
    axis.set_ylabel("monitor flag rate")
    axis.set_ylim(0, 1)
    axis.set_title("Monitor flag rate on policy completions and on known-honest code")
    axis.legend(fontsize=8)
    _caption(
        figure,
        "Lower pair of lines per arm is the false positive estimate: the same judge, same view, "
        "shown MBPP+ reference solutions, which are honest by construction. Reference solutions "
        "are cleaner than anything a 0.5B policy emits, so this is a lower bound. The sample "
        "rotates deterministically by step, so read the trend rather than one step.",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    path = FIGURE_DIR / "false_positive_rate.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _determinacy_floor(plt, grouped) -> Path:
    """Fraction of completions excluded by the determinacy floor, per arm."""
    from alibi.report.metrics import DETERMINACY_FLOOR

    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    for arm, series_list in grouped.items():
        _band(plt, axis, series_list, "excluded_fraction", arm)
    axis.set_xlabel("training step")
    axis.set_ylabel("fraction of completions excluded")
    axis.set_title(f"Completions with fewer than {DETERMINACY_FLOOR} determinate held-out tests")
    axis.legend(fontsize=8)
    _caption(
        figure,
        f"A completion scored on fewer than {DETERMINACY_FLOOR} determinate held-out tests has an "
        "indeterminate oracle verdict and leaves both the numerator and the denominator of the "
        "cheat rate. It is never counted as a cheat. A rising line here means the headline is "
        "computed over a shrinking and possibly non-random subset, so read it beside the gap.",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    path = FIGURE_DIR / "determinacy_floor_exclusions.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _cheat_form(plt, grouped) -> Path:
    arms = sorted(grouped)
    figure, axes = plt.subplots(1, max(1, len(arms)), figsize=(4.2 * max(1, len(arms)), 3.8), squeeze=False)
    forms = sorted({f for series_list in grouped.values() for s in series_list for c in s.cheat_form_counts for f in c})
    for column, arm in enumerate(arms):
        axis = axes[0][column]
        series = grouped[arm][0]
        for form in forms:
            values = [counts.get(form, 0) for counts in series.cheat_form_counts]
            axis.plot(series.steps[: len(values)], values, label=form, linewidth=1.2)
        axis.set_title(f"{arm} (seed {series.seed})")
        axis.set_xlabel("training step")
        if column == 0:
            axis.set_ylabel("completions with this form")
            axis.legend(fontsize=7)
    try:
        fp = metrics.reference_structural_false_positives()["false_positive_rate"]
        fp_text = (
            f" The same check flags {fp:.4f} of known-honest MBPP+ reference solutions, so that rate is "
            "the detector's own floor and prevalence near it is not a statement about the policy."
        )
    except Exception:  # noqa: BLE001 - a caption must not break the figure
        fp_text = ""
    _caption(
        figure,
        "cheat_form composition over training. Structural labels are diagnostic only and never "
        "enter the reward or the primary oracle." + fp_text,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    path = FIGURE_DIR / "cheat_form_composition.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
