"""Publish accepted outputs to HuggingFace as a new dataset (and optional audit sidecar)."""
from __future__ import annotations

from datasets import Dataset
from rich.console import Console

from .config import Config
from .db import Store
from .trace import trace_event

console = Console()


def publish_dataset(cfg: Config, store: Store) -> None:
    rows = store.list_outputs()
    if not rows:
        console.print("  no accepted outputs to publish.")
        trace_event(store, "publish", "publish.skip", message="no outputs")
        return
    trace_event(store, "publish", "publish.start", message=f"{len(rows)} rows → {cfg.output.dataset}")
    console.print(f"  publishing {len(rows)} accepted outputs to {cfg.output.dataset}")
    main_records = [
        {"question": r["question"], "answer": r["answer"], "topic": r["topic"]}
        for r in rows
    ]
    ds = Dataset.from_list(main_records)
    ds.push_to_hub(cfg.output.dataset, private=cfg.output.private)

    if cfg.output.push_audit_sidecar:
        sidecar_records = []
        for r in rows:
            rec = {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "topic": r["topic"],
                "parent_seed_ids": r["parent_seed_ids"],
                "parent_fingerprints": r["parent_fingerprints"],
                "brief_id": r["brief_id"] if "brief_id" in r.keys() else None,
                "scaffold_id": r["scaffold_id"] if "scaffold_id" in r.keys() else None,
                "embedding": r["embedding"],
                "minhash": r["minhash"],
                "audit_json": r["audit_json"],
                "clean_accept": int(r["clean_accept"]),
                "refinement_rounds": int(r["refinement_rounds"]),
                "accepted_at": float(r["accepted_at"]),
            }
            if "merge_mode" in r.keys():
                rec["merge_mode"] = r["merge_mode"]
            sidecar_records.append(rec)
        sidecar_name = f"{cfg.output.dataset}-audit"
        console.print(f"  publishing audit sidecar to {sidecar_name}")
        Dataset.from_list(sidecar_records).push_to_hub(sidecar_name, private=cfg.output.private)
    trace_event(store, "publish", "publish.done", message="hub push complete")
