from __future__ import annotations

import os

from modules.dataset import SymptomsDataset


def format_duration(seconds: float) -> str:
    # Format as HH:MM:SS.s (two decimals)
    m, s = divmod(seconds, 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def next_run_id(artifact_dir: str) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    max_id = 0
    for name in os.listdir(artifact_dir):
        # look for patterns like *_123.* or run_123.*
        parts = name.split("_")
        if parts:
            tail = parts[-1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            if len(digits) == 3:
                try:
                    max_id = max(max_id, int(digits))
                except ValueError:
                    pass
    return f"{max_id + 1:03d}"


def print_dataset_stats(name: str, ds: SymptomsDataset) -> None:
    from collections import Counter
    from main import console  # Move import here to avoid circular dependency
    console.rule(f"[bold]{name} stats")
    console.log({
        "size": len(ds),
        "conditions": dict(Counter(ds.conditions)),
        "avg_cuis_per_doc": (sum(len(x) for x in (ds._concept_ids or [[]])) / max(1, len(ds))) if ds._concept_ids else 0.0,
    })
    if ds._concept_ids:
        for i in range(min(3, len(ds))):
            console.log({"symptoms": ds.symptoms[i], "condition": ds.conditions[i], "cui_ids": ds._concept_ids[i]})
