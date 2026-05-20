"""Small local web UI for configuring and running lineage synthesis."""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from .config import Config, load_config
from .lineage import run_lineage

RUNS: dict[str, "RunState"] = {}
RUNS_LOCK = threading.Lock()


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)


def _load_config_dict(path: str) -> dict[str, Any]:
    cfg = load_config(path)
    return cfg.model_dump(mode="json")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


class RunState:
    def __init__(self, run_id: str, config: dict[str, Any], run_dir: Path):
        self.run_id = run_id
        self.run_dir = run_dir
        self.lock = threading.RLock()
        self.data: dict[str, Any] = {
            "id": run_id,
            "status": "queued",
            "created_at": int(time.time()),
            "started_at": None,
            "finished_at": None,
            "config": config,
            "summary": {},
            "seeds": {},
            "seed_order": [],
            "rows": [],
            "events": [],
            "upload": {"status": "idle", "dataset": None, "error": None},
            "error": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.data, default=str))

    def save(self) -> None:
        path = self.run_dir / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2, ensure_ascii=False), encoding="utf-8")

    def update_event(self, event: dict[str, Any]) -> None:
        event = _json_safe(event)
        with self.lock:
            self.data["events"].append(event)
            if len(self.data["events"]) > 1000:
                self.data["events"] = self.data["events"][-1000:]

            kind = event.get("type")
            seed_id = event.get("seed_id")
            if kind == "run_started":
                self.data["status"] = "running"
                self.data["started_at"] = event.get("time")
                self.data["summary"].update(
                    {
                        "input_dataset": event.get("input_dataset"),
                        "input_split": event.get("input_split"),
                        "output_dataset": event.get("output_dataset"),
                        "seed_count": event.get("seed_count"),
                        "source_iteration": event.get("source_iteration"),
                        "current_iteration": event.get("current_iteration"),
                    }
                )
            elif kind == "seed_started":
                row = event.get("row") or {}
                seed = {
                    "seed_id": seed_id,
                    "lineage_id": event.get("lineage_id"),
                    "index": event.get("seed_index"),
                    "status": "running",
                    "stage": "queued",
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "rows": [row] if row else [],
                    "steps": [],
                }
                self.data["seeds"][seed_id] = seed
                self.data["seed_order"].append(seed_id)
                if row:
                    self.data["rows"].append(row)
            elif kind == "stage_started" and seed_id in self.data["seeds"]:
                seed = self.data["seeds"][seed_id]
                seed["status"] = "running"
                seed["stage"] = event.get("stage")
                seed["steps"].append(
                    {
                        "iteration": event.get("iteration"),
                        "stage": event.get("stage"),
                        "status": "running",
                        "candidate_count": event.get("candidate_count"),
                        "started_at": event.get("time"),
                    }
                )
            elif kind == "stage_done" and seed_id in self.data["seeds"]:
                seed = self.data["seeds"][seed_id]
                seed["stage"] = f"{event.get('stage')}_done"
                for step in reversed(seed["steps"]):
                    if step.get("iteration") == event.get("iteration") and step.get("stage") == event.get("stage"):
                        step["status"] = "done"
                        step["finished_at"] = event.get("time")
                        if "candidate_count" in event:
                            step["candidate_count"] = event.get("candidate_count")
                        if "candidates" in event:
                            step["candidates"] = event.get("candidates")
                        if "raw_response" in event:
                            step["raw_response"] = event.get("raw_response")
                        break
            elif kind in {"iteration_accepted", "iteration_rejected"} and seed_id in self.data["seeds"]:
                seed = self.data["seeds"][seed_id]
                row = event.get("row") or {}
                row["artifacts"] = event.get("artifacts") or {}
                seed["rows"].append(row)
                seed["stage"] = "accepted" if kind == "iteration_accepted" else "rejected"
                seed["last_iteration"] = event.get("iteration")
                self.data["rows"].append(row)
            elif kind == "seed_done" and seed_id in self.data["seeds"]:
                self.data["seeds"][seed_id]["status"] = event.get("status") or "done"
                self.data["seeds"][seed_id]["stage"] = "done"
            elif kind == "seed_error" and seed_id in self.data["seeds"]:
                self.data["seeds"][seed_id]["status"] = "error"
                self.data["seeds"][seed_id]["stage"] = "error"
                self.data["seeds"][seed_id]["error"] = event.get("error")
            elif kind == "upload_started":
                self.data["upload"] = {"status": "running", "dataset": event.get("dataset"), "error": None}
                self.data["status"] = "uploading"
            elif kind == "upload_done":
                self.data["upload"] = {"status": "done", "dataset": event.get("dataset"), "error": None}
            elif kind == "run_done":
                self.data["summary"] = event.get("summary") or self.data["summary"]
                self.data["status"] = "done"
                self.data["finished_at"] = event.get("time")
        self.save()

    def fail(self, exc: BaseException) -> None:
        with self.lock:
            self.data["status"] = "error"
            self.data["finished_at"] = int(time.time())
            self.data["error"] = str(exc)
            self.data["traceback"] = traceback.format_exc()
            if self.data.get("upload", {}).get("status") == "running":
                self.data["upload"]["status"] = "error"
                self.data["upload"]["error"] = str(exc)
        self.save()


def start_run(payload: dict[str, Any], base_dir: Path) -> RunState:
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = base_dir / run_id
    config_data = payload.get("config") or {}
    overrides = payload.get("overrides") or {}
    credentials = payload.get("credentials") or {}
    openai_api_key = str(credentials.get("openai_api_key") or "").strip()
    hf_token = str(credentials.get("hf_token") or "").strip()

    if overrides.get("local_path"):
        config_data.setdefault("output", {})["local_path"] = overrides["local_path"]
    else:
        config_data.setdefault("output", {})["local_path"] = str(run_dir / "lineage.jsonl")
    if overrides.get("summary_path"):
        config_data.setdefault("output", {})["summary_path"] = overrides["summary_path"]
    else:
        config_data.setdefault("output", {})["summary_path"] = str(run_dir / "summary.json")

    cfg = Config.model_validate(config_data)
    run_state = RunState(run_id, cfg.model_dump(mode="json"), run_dir)
    _write_yaml(run_dir / "config.yaml", run_state.data["config"])
    with RUNS_LOCK:
        RUNS[run_id] = run_state

    def worker() -> None:
        previous_env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "HF_TOKEN": os.environ.get("HF_TOKEN"),
            "HUGGINGFACE_HUB_TOKEN": os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        }
        try:
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            cfg_obj = Config.model_validate(run_state.data["config"])
            run_lineage(
                cfg_obj,
                num_seeds=overrides.get("num_seeds"),
                seed=overrides.get("seed"),
                push_to_hub=overrides.get("push_to_hub"),
                dataset_id=overrides.get("dataset_id") or None,
                event_sink=run_state.update_event,
            )
        except BaseException as exc:
            run_state.fail(exc)
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    threading.Thread(target=worker, daemon=True).start()
    return run_state


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>atomicmath</title>
  <style>
    :root {
      color-scheme: light;
      --bg:#eceee9;
      --panel:#fffefa;
      --panel-soft:#f6f2ea;
      --field:#fffdf8;
      --ink:#23251f;
      --muted:#686b61;
      --quiet:#87867a;
      --border:#d6d0c4;
      --border-strong:#aaa191;
      --accent:#315c46;
      --accent-2:#9a553f;
      --gold:#a77b2d;
      --header:#26362e;
      --accent-ink:#ffffff;
      --ok:#3e6b51;
      --bad:#9d4238;
      --warn:#8b6429;
      --shadow:0 10px 28px rgba(44, 39, 31, .08);
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    html { min-width: 320px; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
    }
    header {
      min-height: 64px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:16px;
      padding:0 22px;
      border-bottom:1px solid #17251e;
      background:var(--header);
      color:#fffefa;
      position:sticky;
      top:0;
      z-index:2;
      box-shadow:0 8px 24px rgba(31, 39, 32, .14);
    }
    .brand { display:grid; gap:2px; min-width:0; }
    .brand span { color:#d7dfd3; font-size:12px; font-weight:650; }
    .actions { display:flex; align-items:center; gap:8px; }
    h1 { font-size:18px; margin:0; font-weight:760; letter-spacing:0; }
    button {
      min-height:36px;
      border:1px solid var(--border-strong);
      background:#f9f5ed;
      color:var(--ink);
      border-radius:6px;
      padding:8px 12px;
      font-weight:700;
      cursor:pointer;
      white-space:nowrap;
    }
    button:hover { border-color:var(--accent); }
    header button { border-color:#6e806f; background:#eef0e8; }
    button.primary { background:var(--accent-2); border-color:var(--accent-2); color:var(--accent-ink); }
    button:disabled { opacity:.55; cursor:not-allowed; }
    main {
      display:grid;
      grid-template-columns:minmax(360px, 430px) minmax(0, 1fr);
      gap:18px;
      padding:18px;
      max-width:1720px;
      margin:0 auto;
    }
    section {
      background:var(--panel);
      border:1px solid var(--border);
      border-radius:8px;
      min-width:0;
      box-shadow:var(--shadow);
      overflow:hidden;
      border-top:4px solid var(--accent);
    }
    main > section:nth-child(2) { border-top-color:var(--accent-2); }
    main > section:first-child { align-self:start; position:sticky; top:82px; max-height:calc(100vh - 100px); overflow:auto; }
    .panel-title {
      padding:14px 16px;
      border-bottom:1px solid var(--border);
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:12px;
      background:var(--panel-soft);
    }
    .panel-title h2 { font-size:13px; margin:0; text-transform:uppercase; color:#39362f; letter-spacing:.04em; }
    .form { padding:16px; display:grid; gap:14px; }
    .form-block {
      border:1px solid var(--border);
      border-left:4px solid var(--accent);
      border-radius:8px;
      padding:12px;
      background:#fbf8f1;
      display:grid;
      gap:12px;
    }
    .form-block:nth-of-type(1) { border-left-color:var(--accent-2); }
    .form-block:nth-of-type(2) { border-left-color:var(--accent); }
    .form-block:nth-of-type(3) { border-left-color:var(--gold); }
    .form-block:nth-of-type(4) { border-left-color:#596a5f; }
    .form-block:nth-of-type(5) { border-left-color:#7a6b56; }
    .block-title {
      font-size:12px;
      font-weight:850;
      letter-spacing:.055em;
      text-transform:uppercase;
      color:#343930;
      display:flex;
      align-items:center;
      gap:8px;
    }
    .block-title::before {
      content:"";
      width:8px;
      height:8px;
      border-radius:50%;
      background:var(--accent);
      display:inline-block;
    }
    label {
      display:grid;
      gap:6px;
      font-size:11px;
      font-weight:760;
      color:#4d493f;
      text-transform:uppercase;
      letter-spacing:.035em;
      min-width:0;
    }
    input, textarea, select {
      width:100%;
      min-width:0;
      border:1px solid var(--border);
      border-radius:6px;
      padding:9px 10px;
      font: inherit;
      font-size:13px;
      background:var(--field);
      color:var(--ink);
      outline:none;
      transition:border-color .12s ease, box-shadow .12s ease, background .12s ease;
    }
    input:focus, textarea:focus, select:focus {
      border-color:var(--accent);
      box-shadow:0 0 0 3px rgba(53, 92, 71, .12);
      background:#fff;
    }
    textarea {
      min-height:220px;
      font-family:var(--mono);
      font-size:12px;
      line-height:1.55;
      resize:vertical;
    }
    .advanced {
      border:1px solid var(--border);
      border-radius:8px;
      background:#fbf8f1;
      padding:10px 12px;
    }
    .advanced summary {
      cursor:pointer;
      font-size:12px;
      font-weight:850;
      letter-spacing:.055em;
      text-transform:uppercase;
      color:#343930;
    }
    .advanced .hint { margin:10px 0; }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; min-width:0; }
    .grid3 { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:12px; min-width:0; }
    .checks { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .check {
      min-height:38px;
      display:flex;
      align-items:center;
      gap:8px;
      border:1px solid var(--border);
      border-radius:6px;
      padding:8px 10px;
      background:var(--panel-soft);
      font-size:12px;
      font-weight:700;
      text-transform:none;
      letter-spacing:0;
      color:var(--ink);
    }
    .check:hover { border-color:var(--border-strong); background:#fffaf0; }
    .check input { width:auto; accent-color:var(--accent-2); }
    .hint { color:var(--muted); font-size:12px; line-height:1.45; overflow:hidden; text-overflow:ellipsis; }
    .tabs {
      display:flex;
      gap:4px;
      padding:12px 14px 0;
      border-bottom:1px solid var(--border);
      background:var(--panel-soft);
    }
    .tab {
      border-color:transparent;
      background:transparent;
      border-bottom-left-radius:0;
      border-bottom-right-radius:0;
      color:var(--muted);
      padding:9px 12px;
    }
    .tab.active {
      background:var(--panel);
      border-color:var(--border);
      border-bottom-color:var(--panel);
      color:var(--ink);
      transform:translateY(1px);
    }
    .content { padding:16px; }
    .statusbar { display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); gap:10px; margin-bottom:14px; }
    .metric { border:1px solid var(--border); border-top:3px solid var(--gold); border-radius:8px; padding:12px; background:var(--panel-soft); min-width:0; }
    .metric b {
      display:block;
      font-size:21px;
      line-height:1.1;
      font-weight:760;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }
    .metric span { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; font-weight:760; }
    .seed-list { display:grid; gap:8px; max-height:calc(100vh - 260px); overflow:auto; padding-right:4px; }
    .seed {
      display:grid;
      grid-template-columns:86px minmax(0,1fr) 68px;
      gap:12px;
      align-items:center;
      border:1px solid var(--border);
      border-radius:8px;
      padding:11px;
      cursor:pointer;
      background:var(--field);
      min-width:0;
    }
    .seed:hover { border-color:var(--border-strong); background:#fff; }
    .seed.active { border-color:var(--accent); box-shadow:0 0 0 3px rgba(53, 92, 71, .12); background:#fff; }
    .pill {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-height:24px;
      border-radius:999px;
      padding:3px 9px;
      font-size:11px;
      font-weight:800;
      text-align:center;
      background:#ebe7de;
      color:#4c473c;
      white-space:nowrap;
    }
    .pill.running,.pill.uploading { background:#f0e1bd; color:#725314; }
    .pill.done,.pill.accepted,.pill.dry_run { background:#e2ecdf; color:var(--ok); }
    .pill.rejected,.pill.error { background:#f1dfdc; color:var(--bad); }
    .seed-title { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-weight:760; font-size:13px; }
    .seed-sub { color:var(--muted); font-size:12px; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .detail-grid { display:grid; grid-template-columns:minmax(0, 1fr) minmax(320px, .9fr); gap:14px; align-items:start; }
    .box { border:1px solid var(--border); border-radius:8px; background:var(--field); padding:14px; min-width:0; }
    .box h3 { margin:0 0 10px; font-size:12px; text-transform:uppercase; letter-spacing:.045em; color:#4b473e; }
    pre {
      white-space:pre-wrap;
      word-break:break-word;
      margin:0;
      font:12px/1.55 var(--mono);
      color:#2d2b27;
    }
    .steps { display:grid; gap:10px; }
    .step { border:1px solid var(--border); border-radius:8px; padding:11px; background:var(--panel-soft); }
    .step-head { display:flex; justify-content:space-between; align-items:center; gap:10px; font-weight:800; margin-bottom:8px; }
    .candidate { border-top:1px solid var(--border); padding-top:10px; margin-top:10px; }
    .candidate b { font-size:13px; }
    .small { font-size:12px; color:var(--muted); min-width:0; }
    .error { color:var(--bad); font-weight:800; }
    .hidden { display:none !important; }
    @media (max-width: 1120px) {
      main { grid-template-columns:1fr; }
      main > section:first-child { position:static; max-height:none; }
      .detail-grid { grid-template-columns:1fr; }
    }
    @media (max-width: 720px) {
      header { align-items:flex-start; flex-direction:column; padding:12px 14px; }
      .actions { width:100%; display:grid; grid-template-columns:1fr 1fr; gap:8px; }
      main { padding:10px; gap:10px; }
      .grid2, .grid3, .checks, .statusbar { grid-template-columns:1fr; }
      .seed { grid-template-columns:78px minmax(0,1fr); }
      .seed > .small { display:none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>atomicmath</h1>
      <span>lineage runner</span>
    </div>
    <div class="actions">
      <button id="reloadConfigBtn">Reload config</button>
      <button id="startBtn" class="primary">Start run</button>
    </div>
  </header>
  <main>
    <section>
      <div class="panel-title"><h2>Configuration</h2><span id="configPath" class="hint"></span></div>
      <div class="form">
        <div class="form-block">
          <div class="block-title">Credentials</div>
          <div class="grid2">
            <label>OpenAI API key<input id="openai_api_key" type="password" autocomplete="off" placeholder="Uses environment if blank"></label>
            <label>HF token<input id="hf_token" type="password" autocomplete="off" placeholder="Uses environment if blank"></label>
          </div>
          <div class="hint">Keys are used only for this run and are not saved into the config, run state, or uploaded dataset.</div>
        </div>

        <div class="form-block">
          <div class="block-title">Source Dataset</div>
          <div class="grid2">
            <label>Input dataset<input id="input_dataset"></label>
            <label>Dataset config<input id="input_config_name"></label>
          </div>
          <div class="grid3">
            <label>Split<input id="input_split"></label>
            <label>Question field<input id="input_question_field"></label>
            <label>Answer field<input id="input_answer_field"></label>
          </div>
          <div class="grid3">
            <label>Iteration field<input id="input_iteration_field"></label>
            <label>Memory field<input id="input_memory_field"></label>
            <label>Seed rows<input id="input_max_seeds" type="number" min="1"></label>
          </div>
          <div class="grid2">
            <label>Sampling seed<input id="input_seed" type="number"></label>
            <label>Target iteration<input id="current_iteration_preview" readonly placeholder="Inferred at run time"></label>
          </div>
        </div>

        <div class="form-block">
          <div class="block-title">Generation</div>
          <div class="grid3">
            <label>Generator model<input id="models_generator"></label>
            <label>Refiner model<input id="models_refiner"></label>
            <label>Judge model<input id="models_judge"></label>
          </div>
          <div class="grid2">
            <label>Candidates / parent<input id="lineage_candidates_per_iteration" type="number" min="1" max="8"></label>
            <label>Memory items<input id="lineage_max_memory_items" type="number" min="0"></label>
          </div>
        </div>

        <div class="form-block">
          <div class="block-title">Quality Gate</div>
          <div class="grid3">
            <label>Min novelty<input id="lineage_min_novelty" type="number" min="0" max="1" step="0.01"></label>
            <label>Min depth<input id="lineage_min_depth" type="number" min="0" max="1" step="0.01"></label>
            <label>Min non-stitched<input id="lineage_min_non_stitched" type="number" min="0" max="1" step="0.01"></label>
          </div>
          <div class="checks">
            <label class="check"><input id="lineage_continue_on_rejected" type="checkbox"> Continue on rejected</label>
            <label class="check"><input id="runtime_dry_run" type="checkbox"> Dry run</label>
          </div>
        </div>

        <div class="form-block">
          <div class="block-title">Output</div>
          <div class="grid2">
            <label>Output dataset<input id="output_dataset"></label>
            <label>Output split<input id="output_split"></label>
          </div>
          <div class="checks">
            <label class="check"><input id="output_push_to_hub" type="checkbox"> Upload after run</label>
            <label class="check"><input id="output_private" type="checkbox"> Private dataset</label>
            <label class="check"><input id="output_append_if_same_dataset" type="checkbox"> Append if same dataset</label>
          </div>
        </div>

        <details class="advanced">
          <summary>Advanced raw config</summary>
          <div class="hint">Grouped controls are applied to this JSON before a run starts.</div>
          <textarea id="advanced"></textarea>
        </details>
      </div>
    </section>
    <section>
      <div class="panel-title">
        <h2>Run Status</h2>
        <span id="runLabel" class="hint">No run started</span>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="overview">Overview</button>
        <button class="tab" data-tab="details">Question Details</button>
        <button class="tab" data-tab="events">Events</button>
      </div>
      <div id="overview" class="content">
        <div class="statusbar">
          <div class="metric"><b id="mStatus">idle</b><span>Status</span></div>
          <div class="metric"><b id="mSeeds">0</b><span>Seeds</span></div>
          <div class="metric"><b id="mIteration">-</b><span>Iteration</span></div>
          <div class="metric"><b id="mDone">0</b><span>Done</span></div>
          <div class="metric"><b id="mRows">0</b><span>Rows</span></div>
          <div class="metric"><b id="mUpload">idle</b><span>Upload</span></div>
        </div>
        <div id="errorBox" class="box error hidden"></div>
        <div class="seed-list" id="seedList"></div>
      </div>
      <div id="details" class="content hidden">
        <div id="detailEmpty" class="hint">Select a seed from the overview list.</div>
        <div id="detailBody" class="detail-grid hidden"></div>
      </div>
      <div id="events" class="content hidden">
        <div class="box"><pre id="eventLog"></pre></div>
      </div>
    </section>
  </main>
  <script>
    let config = null;
    let currentRun = null;
    let pollTimer = null;
    let selectedSeed = null;

    const $ = (id) => document.getElementById(id);
    const fields = [
      ["input.dataset", "input_dataset", "str"],
      ["input.config_name", "input_config_name", "str"],
      ["input.split", "input_split", "str"],
      ["input.question_field", "input_question_field", "str"],
      ["input.answer_field", "input_answer_field", "str"],
      ["input.iteration_field", "input_iteration_field", "str"],
      ["input.memory_field", "input_memory_field", "str"],
      ["input.max_seeds", "input_max_seeds", "int"],
      ["input.seed", "input_seed", "int"],
      ["lineage.candidates_per_iteration", "lineage_candidates_per_iteration", "int"],
      ["lineage.max_memory_items", "lineage_max_memory_items", "int"],
      ["lineage.min_novelty", "lineage_min_novelty", "float"],
      ["lineage.min_depth", "lineage_min_depth", "float"],
      ["lineage.min_non_stitched", "lineage_min_non_stitched", "float"],
      ["models.generator", "models_generator", "str"],
      ["models.refiner", "models_refiner", "str"],
      ["models.judge", "models_judge", "str"],
      ["output.dataset", "output_dataset", "str"],
      ["output.split", "output_split", "str"],
      ["output.push_to_hub", "output_push_to_hub", "bool"],
      ["output.private", "output_private", "bool"],
      ["output.append_if_same_dataset", "output_append_if_same_dataset", "bool"],
      ["lineage.continue_on_rejected", "lineage_continue_on_rejected", "bool"],
      ["runtime.dry_run", "runtime_dry_run", "bool"]
    ];

    function getPath(obj, path) {
      return path.split(".").reduce((acc, key) => acc && acc[key], obj);
    }
    function setPath(obj, path, value) {
      const parts = path.split(".");
      let cur = obj;
      for (let i = 0; i < parts.length - 1; i++) {
        cur[parts[i]] = cur[parts[i]] || {};
        cur = cur[parts[i]];
      }
      cur[parts[parts.length - 1]] = value;
    }
    function clone(obj) { return JSON.parse(JSON.stringify(obj)); }
    function escapeHtml(s) {
      return String(s ?? "").replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function fillForm() {
      for (const [path, id, type] of fields) {
        const el = $(id);
        const val = getPath(config, path);
        if (!el) continue;
        if (type === "bool") el.checked = Boolean(val);
        else el.value = val ?? "";
      }
      $("advanced").value = JSON.stringify(config, null, 2);
    }
    function readFormConfig() {
      let cfg;
      try {
        cfg = JSON.parse($("advanced").value);
      } catch (err) {
        throw new Error("Advanced config is not valid JSON: " + err.message);
      }
      for (const [path, id, type] of fields) {
        const el = $(id);
        let value;
        if (type === "bool") value = el.checked;
        else if (type === "int") value = parseInt(el.value || "0", 10);
        else if (type === "float") value = parseFloat(el.value || "0");
        else value = el.value;
        setPath(cfg, path, value);
      }
      return cfg;
    }
    async function loadConfig() {
      const res = await fetch("/api/config");
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      config = data.config;
      $("configPath").textContent = data.config_path;
      fillForm();
    }
    async function startRun() {
      let cfg;
      try {
        cfg = readFormConfig();
      } catch (err) {
        alert(err.message);
        return;
      }
      $("startBtn").disabled = true;
      const openaiApiKey = $("openai_api_key").value;
      const hfToken = $("hf_token").value;
      $("openai_api_key").value = "";
      $("hf_token").value = "";
      const res = await fetch("/api/runs", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          config: cfg,
          overrides: {
            num_seeds: cfg.input.max_seeds,
            seed: cfg.input.seed,
            push_to_hub: cfg.output.push_to_hub,
            dataset_id: cfg.output.dataset
          },
          credentials: {
            openai_api_key: openaiApiKey,
            hf_token: hfToken
          }
        })
      });
      if (!res.ok) {
        $("startBtn").disabled = false;
        alert(await res.text());
        return;
      }
      const data = await res.json();
      currentRun = data.id;
      selectedSeed = null;
      $("runLabel").textContent = currentRun;
      poll();
      pollTimer = setInterval(poll, 1500);
    }
    async function poll() {
      if (!currentRun) return;
      const res = await fetch(`/api/runs/${currentRun}`);
      if (!res.ok) return;
      const state = await res.json();
      renderState(state);
      if (["done", "error"].includes(state.status)) {
        clearInterval(pollTimer);
        $("startBtn").disabled = false;
      }
    }
    function renderState(state) {
      const seeds = state.seed_order.map(id => state.seeds[id]).filter(Boolean);
      const done = seeds.filter(s => ["done", "dry_run", "rejected", "error"].includes(s.status)).length;
      $("mStatus").textContent = state.status;
      $("mSeeds").textContent = seeds.length || state.summary.seed_count || 0;
      $("mIteration").textContent = state.summary.current_iteration ?? "-";
      $("current_iteration_preview").value = state.summary.current_iteration ? `iteration ${state.summary.current_iteration}` : "";
      $("mDone").textContent = done;
      $("mRows").textContent = state.rows.length;
      $("mUpload").textContent = state.upload?.status || "idle";
      if (state.error) {
        $("errorBox").classList.remove("hidden");
        $("errorBox").textContent = state.error;
      } else {
        $("errorBox").classList.add("hidden");
      }
      $("seedList").innerHTML = seeds.map(seed => {
        const title = seed.question || seed.seed_id;
        return `<div class="seed ${seed.seed_id === selectedSeed ? "active" : ""}" data-seed="${escapeHtml(seed.seed_id)}">
          <div class="pill ${escapeHtml(seed.status)}">${escapeHtml(seed.status)}</div>
          <div><div class="seed-title">${escapeHtml(title)}</div><div class="seed-sub">seed ${escapeHtml(seed.seed_id)} · ${escapeHtml(seed.stage || "")}</div></div>
          <div class="small">${(seed.rows || []).length} rows</div>
        </div>`;
      }).join("");
      document.querySelectorAll(".seed").forEach(el => {
        el.onclick = () => {
          selectedSeed = el.dataset.seed;
          document.querySelector('[data-tab="details"]').click();
          renderState(state);
        };
      });
      $("eventLog").textContent = (state.events || []).slice(-200).map(e => JSON.stringify(e)).join("\n");
      renderDetails(state);
    }
    function renderDetails(state) {
      const seed = selectedSeed ? state.seeds[selectedSeed] : null;
      if (!seed) {
        $("detailEmpty").classList.remove("hidden");
        $("detailBody").classList.add("hidden");
        return;
      }
      $("detailEmpty").classList.add("hidden");
      $("detailBody").classList.remove("hidden");
      const rows = seed.rows || [];
      const generated = rows.filter(r => r.role === "generated" || r.role === "rejected");
      const steps = seed.steps || [];
      $("detailBody").innerHTML = `
        <div class="box">
          <h3>Seed</h3>
          <pre>${escapeHtml(seed.question)}</pre>
          <h3 style="margin-top:12px">Seed Answer</h3>
          <pre>${escapeHtml(seed.answer)}</pre>
          <h3 style="margin-top:12px">Generated Iterations</h3>
          ${generated.map(row => `
            <div class="candidate">
              <div class="step-head"><span>Iteration ${escapeHtml(row.iteration)} · ${escapeHtml(row.role)}</span><span class="pill ${row.accepted ? "done" : "rejected"}">${row.accepted ? "accepted" : "rejected"}</span></div>
              <pre>${escapeHtml(row.question)}</pre>
              <div class="small" style="margin-top:8px">Answer</div>
              <pre>${escapeHtml(row.answer)}</pre>
              <div class="small" style="margin-top:8px">Scores</div>
              <pre>${escapeHtml(row.scores_json || "{}")}</pre>
            </div>
          `).join("") || '<div class="hint">No generated row yet.</div>'}
        </div>
        <div class="box">
          <h3>Synthesis Steps</h3>
          <div class="steps">
          ${steps.map(step => `
            <div class="step">
              <div class="step-head"><span>Iter ${escapeHtml(step.iteration)} · ${escapeHtml(step.stage)}</span><span class="pill ${escapeHtml(step.status)}">${escapeHtml(step.status)}</span></div>
              <div class="small">Candidates: ${escapeHtml(step.candidate_count ?? "")}</div>
              ${(step.candidates || []).map(c => `
                <div class="candidate">
                  <b>${escapeHtml(c.label)} · ${escapeHtml(c.decision || "")}</b>
                  <div class="small">${escapeHtml(c.failure_kind || "")}</div>
                  <pre>${escapeHtml(c.transformation || c.question || "")}</pre>
                  ${c.judge_notes ? `<div class="small" style="margin-top:8px">Judge notes</div><pre>${escapeHtml(c.judge_notes)}</pre>` : ""}
                  <pre>${escapeHtml(JSON.stringify(c.scores || {}, null, 2))}</pre>
                </div>
              `).join("")}
              ${step.raw_response ? `<div class="candidate"><b>Raw judge response</b><pre>${escapeHtml(step.raw_response)}</pre></div>` : ""}
            </div>
          `).join("") || '<div class="hint">No steps yet.</div>'}
          </div>
        </div>`;
    }
    document.querySelectorAll(".tab").forEach(tab => {
      tab.onclick = () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        for (const id of ["overview", "details", "events"]) $(id).classList.toggle("hidden", id !== tab.dataset.tab);
      };
    });
    $("reloadConfigBtn").onclick = loadConfig;
    $("startBtn").onclick = startRun;
    fields.forEach(([path, id, type]) => {
      const el = $(id);
      if (!el) return;
      el.onchange = () => {
        try {
          const cfg = readFormConfig();
          $("advanced").value = JSON.stringify(cfg, null, 2);
        } catch (_) {}
      };
    });
    loadConfig().catch(err => alert(err.message));
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AtomicMathWeb/0.1"

    @property
    def app(self) -> "AtomicMathServer":
        return self.server  # type: ignore[return-value]

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(HTML, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/config":
            query = parse_qs(parsed.query)
            config_path = query.get("path", [self.app.config_path])[0]
            try:
                self._send_json({"config_path": config_path, "config": _load_config_dict(config_path)})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/runs":
            with RUNS_LOCK:
                runs = [run.snapshot() for run in RUNS.values()]
            runs.sort(key=lambda item: item.get("created_at") or 0, reverse=True)
            self._send_json({"runs": runs})
            return
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.rsplit("/", 1)[-1]
            with RUNS_LOCK:
                run = RUNS.get(run_id)
            if run is None:
                self._send_json({"error": "unknown run"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(run.snapshot())
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            try:
                payload = self._read_json()
                run = start_run(payload, self.app.run_dir)
                self._send_json({"id": run.run_id, "state": run.snapshot()}, status=HTTPStatus.CREATED)
            except Exception as exc:
                self._send_json({"error": str(exc), "traceback": traceback.format_exc()}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}")


class AtomicMathServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], config_path: str, run_dir: Path):
        super().__init__(server_address, Handler)
        self.config_path = config_path
        self.run_dir = run_dir


def serve(config_path: str, host: str = "127.0.0.1", port: int = 8765, run_dir: str = "./out/web_runs") -> None:
    server = AtomicMathServer((host, port), config_path=config_path, run_dir=Path(run_dir))
    print(f"atomicmath web UI: http://{host}:{port}/", flush=True)
    server.serve_forever()
