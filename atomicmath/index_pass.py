"""Per-seed indexer: question embedding + MinHash (+ optional solution trace for debugging)."""
from __future__ import annotations

from rich.console import Console
from rich.progress import Progress

from .config import Config
from .db import Store
from .llm import EmbeddingClient
from .minhash_util import minhash_signature
from .trace import trace_event

console = Console()


def _is_litellm_auth_failure(exc: BaseException) -> bool:
    """True when the failure is missing/invalid API credentials (retrying all seeds is pointless)."""
    mod = getattr(exc.__class__, "__module__", "") or ""
    name = exc.__class__.__name__
    if "litellm" in mod and name in ("AuthenticationError", "BadRequestError"):
        return True
    s = str(exc).lower().replace(" ", "")
    if "authenticationerror" in s:
        return True
    if "api_key" in s and ("must be set" in s or "environment variable" in s or "not set" in s):
        return True
    return False


def index_seeds(cfg: Config, store: Store, embedder: EmbeddingClient) -> None:
    pending = store.list_pending_seeds()
    if not pending:
        console.print("  all seeds already indexed.")
        return
    console.print(f"  indexing {len(pending)} seeds")
    trace_event(store, "index", "index.batch.start", message=f"{len(pending)} seeds pending")
    with Progress() as prog:
        t = prog.add_task("[green]indexing", total=len(pending))
        for row in pending:
            sid = row["id"]
            q, a = row["question"], row["answer"]
            topic_raw = row["topic_raw"] if "topic_raw" in row.keys() else ""
            sol = row["solution_text"] if "solution_text" in row.keys() else ""
            try:
                _index_one(cfg, store, embedder, sid, q, a, topic_raw, solution_text=sol or "")
            except Exception as e:
                console.print(f"  [red]index failed for {sid}[/]: {e}")
                if _is_litellm_auth_failure(e):
                    trace_event(
                        store,
                        "index",
                        "index.abort",
                        seed_id=sid,
                        message="stopped indexing — API authentication failed",
                    )
                    raise RuntimeError(
                        "Indexing aborted: LiteLLM authentication failed (see error above). "
                        "Fix env vars (e.g. export OPENAI_API_KEY=...) then re-run; pending seeds stay unindexed."
                    ) from e
            prog.advance(t)


def _index_one(
    cfg: Config,
    store: Store,
    embedder: EmbeddingClient,
    sid: str,
    q: str,
    a: str,
    topic_raw: str = "",
    *,
    solution_text: str = "",
) -> None:
    trace_event(
        store,
        "index",
        "index.seed.start",
        seed_id=sid,
        message="embedding + MinHash",
        payload={"question": q, "answer": a, "topic_raw": topic_raw or None},
    )
    qtext = q
    truncated = False
    if len(qtext) > 16000:
        qtext = qtext[:8000]
        truncated = True
    trace_event(
        store,
        "index",
        "index.embed",
        seed_id=sid,
        payload={
            "truncated": truncated,
            "char_len": len(qtext),
            "question": qtext,
            "embedder_model": cfg.models.embedder,
        },
    )
    emb = embedder.embed(qtext)
    trace_event(store, "index", "index.embed.done", seed_id=sid, payload={"dim": len(emb)})

    trace_event(store, "index", "index.hash", seed_id=sid, message="MinHash")
    mh = minhash_signature(q, a)

    sol = (solution_text or "").strip()
    trace_blob = sol[:12000] if sol else a
    store.write_signals(
        seed_id=sid,
        embedding=emb,
        minhash=mh,
        trace=trace_blob,
        fingerprint="",
        fingerprint_embedding=[],
    )
    if truncated:
        with store._conn() as c:
            c.execute("UPDATE seeds SET embed_truncated = 1 WHERE id = ?", (sid,))
    store.mark_indexed(sid)
    trace_event(
        store,
        "index",
        "index.seed.done",
        seed_id=sid,
        payload={"minhash_perm_count": len(mh)},
    )
