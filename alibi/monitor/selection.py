"""The chosen monitor, recorded as a hashed declaration.

`alibi/prereg.py` is frozen and carries no monitor model id, so the monitor
identity sits **outside** the pre-registration hash. That is a real gap in the
registration and it is stated rather than papered over: a reader comparing two
runs must check this file's hash as well as the prereg hash to know they were
judged by the same thing.

Once written, the monitor identity and prompt template version are frozen for
the run matrix. `alibi.halt` refuses a run whose monitor selection hash differs
from the one recorded at the start.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SELECTION_PATH = Path(__file__).resolve().parent / "selection.json"


def selection_hash(document: dict) -> str:
    payload = {
        "model_id": document["model_id"],
        "prompt_template_version": document["prompt_template_version"],
        "base_url": document["base_url"],
        "monitor_name": document["monitor_name"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_selection(
    model_id: str,
    prompt_template_version: str,
    base_url: str,
    trail: list[dict],
    qualification: dict | None,
    monitor_name: str = "llm",
) -> dict:
    from alibi.monitor.llm import shares_family_with_policy

    document = {
        "monitor_name": monitor_name,
        "model_id": model_id,
        "prompt_template_version": prompt_template_version,
        "base_url": base_url,
        "selected_utc": datetime.now(timezone.utc).isoformat(),
        "shares_family_with_policy": shares_family_with_policy(model_id),
        "confound_note": (
            "This judge shares a model family with the Qwen2.5-0.5B policy it grades. Shared "
            "tokenizer, pretraining data and failure modes mean the monitor may be unusually "
            "good or unusually blind to exactly this policy's habits. This does not invalidate "
            "the monitored arms, and it must be stated in the report rather than discovered by "
            "a reader."
        )
        if shares_family_with_policy(model_id)
        else None,
        "qualification": qualification,
        "escalation_trail": trail,
        "outside_prereg_hash": (
            "alibi/prereg.py carries no monitor model id, so this selection is not covered by the "
            "pre-registration hash. Compare this file's selection_hash as well."
        ),
    }
    document["selection_hash"] = selection_hash(document)
    SELECTION_PATH.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def load_selection() -> dict | None:
    if not SELECTION_PATH.exists():
        return None
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def selected_model_id() -> str | None:
    document = load_selection()
    return document["model_id"] if document else None
