"""SQLite storage for seeds, signals, retrosynthesis tables, outputs, and logs."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS seeds (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    topic_raw TEXT NOT NULL,
    topic_norm TEXT,
    solution_text TEXT NOT NULL DEFAULT '',
    eligible INTEGER NOT NULL DEFAULT 1,
    indexed INTEGER NOT NULL DEFAULT 0,
    embed_truncated INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS signals (
    seed_id TEXT PRIMARY KEY,
    embedding TEXT NOT NULL,
    minhash TEXT NOT NULL,
    trace TEXT,
    fingerprint TEXT NOT NULL DEFAULT '',
    fingerprint_embedding TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (seed_id) REFERENCES seeds(id)
);

CREATE INDEX IF NOT EXISTS idx_seeds_topic ON seeds(topic_norm) WHERE eligible = 1;

CREATE TABLE IF NOT EXISTS briefs (
    id TEXT PRIMARY KEY,
    brief_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS scaffolds (
    id TEXT PRIMARY KEY,
    brief_id TEXT NOT NULL,
    scaffold_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (brief_id) REFERENCES briefs(id)
);

CREATE TABLE IF NOT EXISTS coverage (
    primary_concept TEXT NOT NULL,
    secondary_concept TEXT NOT NULL,
    accept_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (primary_concept, secondary_concept)
);

CREATE TABLE IF NOT EXISTS exemplars (
    id TEXT PRIMARY KEY,
    seed_id TEXT NOT NULL UNIQUE,
    scaffold_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (seed_id) REFERENCES seeds(id)
);

CREATE TABLE IF NOT EXISTS outputs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    topic TEXT NOT NULL,
    parent_seed_ids TEXT NOT NULL,
    parent_fingerprints TEXT NOT NULL,
    brief_id TEXT,
    scaffold_id TEXT,
    embedding TEXT NOT NULL,
    minhash TEXT NOT NULL,
    audit_json TEXT NOT NULL,
    accepted_at REAL NOT NULL,
    clean_accept INTEGER NOT NULL,
    refinement_rounds INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outputs_topic ON outputs(topic);

CREATE TABLE IF NOT EXISTS run_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS round_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    topic TEXT,
    merge_mode TEXT,
    outcome TEXT NOT NULL,
    failure_kind TEXT,
    output_id TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    phase TEXT NOT NULL,
    step TEXT NOT NULL,
    seed_id TEXT,
    attempt INTEGER,
    message TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_pipeline_events_id ON pipeline_events(id);

CREATE TABLE IF NOT EXISTS seed_hinges (
    id TEXT PRIMARY KEY,
    seed_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    hinge_text TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (seed_id) REFERENCES seeds(id)
);
CREATE INDEX IF NOT EXISTS idx_seed_hinges_seed_id ON seed_hinges(seed_id);

CREATE TABLE IF NOT EXISTS mutation_episodes (
    id TEXT PRIMARY KEY,
    seed_id TEXT NOT NULL,
    hinge_ids TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    mutation_used TEXT NOT NULL DEFAULT '',
    new_question TEXT,
    answer TEXT,
    short_solution TEXT,
    result TEXT NOT NULL,
    failure_kind TEXT,
    scores_json TEXT NOT NULL DEFAULT '{}',
    plan_json TEXT NOT NULL DEFAULT '{}',
    candidate_json TEXT NOT NULL DEFAULT '{}',
    story TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (seed_id) REFERENCES seeds(id)
);
CREATE INDEX IF NOT EXISTS idx_mutation_episodes_seed_id ON mutation_episodes(seed_id);
CREATE INDEX IF NOT EXISTS idx_mutation_episodes_result ON mutation_episodes(result);

CREATE TABLE IF NOT EXISTS mutation_experiences (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL DEFAULT 'global',
    kind TEXT NOT NULL,
    topic_norm TEXT NOT NULL DEFAULT '',
    failure_kind TEXT,
    mutation_used TEXT NOT NULL DEFAULT '',
    lesson TEXT NOT NULL,
    source_episode_ids TEXT NOT NULL DEFAULT '[]',
    source_count INTEGER NOT NULL DEFAULT 0,
    weight REAL NOT NULL DEFAULT 1.0,
    active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mutation_experiences_kind ON mutation_experiences(kind, active);
CREATE INDEX IF NOT EXISTS idx_mutation_experiences_topic ON mutation_experiences(topic_norm, kind, active);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            self._ensure_schema_migrations(c)

    def _ensure_schema_migrations(self, c: sqlite3.Connection) -> None:
        """Lightweight additive migrations for existing DB files."""
        info = c.execute("PRAGMA table_info(seeds)").fetchall()
        cols = {row[1] for row in info}
        if cols and "solution_text" not in cols:
            c.execute("ALTER TABLE seeds ADD COLUMN solution_text TEXT NOT NULL DEFAULT ''")
        out_info = c.execute("PRAGMA table_info(outputs)").fetchall()
        if not out_info:
            return
        out_cols = {row[1] for row in out_info}
        if "brief_id" not in out_cols:
            c.execute("ALTER TABLE outputs ADD COLUMN brief_id TEXT")
        if "scaffold_id" not in out_cols:
            c.execute("ALTER TABLE outputs ADD COLUMN scaffold_id TEXT")
        mut_info = c.execute("PRAGMA table_info(mutation_episodes)").fetchall()
        mut_cols = {row[1] for row in mut_info}
        if mut_cols and "plan_json" not in mut_cols:
            c.execute("ALTER TABLE mutation_episodes ADD COLUMN plan_json TEXT NOT NULL DEFAULT '{}'")
        if mut_cols and "candidate_json" not in mut_cols:
            c.execute("ALTER TABLE mutation_episodes ADD COLUMN candidate_json TEXT NOT NULL DEFAULT '{}'")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    # --- run state -----------------------------------------------------------

    def state_get(self, key: str, default: Any = None) -> Any:
        with self._conn() as c:
            r = c.execute("SELECT value FROM run_state WHERE key = ?", (key,)).fetchone()
            if r is None:
                return default
            return json.loads(r["value"])

    def state_set(self, key: str, value: Any) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO run_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    # --- seeds ---------------------------------------------------------------

    def upsert_seed(
        self,
        seed_id: str,
        question: str,
        answer: str,
        topic_raw: str,
        solution_text: str = "",
    ) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO seeds(id,question,answer,topic_raw,solution_text) VALUES(?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     question=excluded.question,
                     answer=excluded.answer,
                     topic_raw=excluded.topic_raw,
                     solution_text=excluded.solution_text""",
                (seed_id, question, answer, topic_raw, solution_text or ""),
            )

    def set_topic_norm(self, seed_id: str, topic_norm: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE seeds SET topic_norm = ? WHERE id = ?", (topic_norm, seed_id))

    def set_eligible(self, seed_id: str, eligible: bool) -> None:
        with self._conn() as c:
            c.execute("UPDATE seeds SET eligible = ? WHERE id = ?", (1 if eligible else 0, seed_id))

    def mark_indexed(self, seed_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE seeds SET indexed = 1 WHERE id = ?", (seed_id,))

    def list_pending_seeds(self, limit: int | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM seeds WHERE indexed = 0 AND eligible = 1"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._conn() as c:
            return list(c.execute(q).fetchall())

    def count_pending_index_seeds(self) -> int:
        with self._conn() as c:
            return int(
                c.execute("SELECT COUNT(*) FROM seeds WHERE indexed = 0 AND eligible = 1").fetchone()[0]
            )

    def count_indexed_eligible_seeds(self) -> int:
        with self._conn() as c:
            return int(
                c.execute(
                    "SELECT COUNT(*) FROM seeds s JOIN signals g ON s.id = g.seed_id "
                    "WHERE s.eligible = 1 AND s.indexed = 1"
                ).fetchone()[0]
            )

    def list_indexed_seeds(self) -> list[sqlite3.Row]:
        with self._conn() as c:
            return list(
                c.execute(
                    "SELECT s.*, sg.embedding, sg.minhash, sg.trace "
                    "FROM seeds s JOIN signals sg ON s.id = sg.seed_id "
                    "WHERE s.eligible = 1"
                ).fetchall()
            )

    def list_indexed_seeds_for_bootstrap(self, limit: int) -> list[sqlite3.Row]:
        """Indexed eligible seeds with non-trivial reference solution text."""
        lim = max(1, int(limit))
        with self._conn() as c:
            return list(
                c.execute(
                    "SELECT s.id, s.question, s.answer, s.topic_raw, s.solution_text, s.topic_norm "
                    "FROM seeds s JOIN signals g ON s.id = g.seed_id "
                    "WHERE s.eligible = 1 AND s.indexed = 1 AND length(trim(s.solution_text)) >= 20 "
                    "ORDER BY s.id LIMIT ?",
                    (lim,),
                ).fetchall()
            )

    def get_seed(self, seed_id: str) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute("SELECT * FROM seeds WHERE id = ?", (seed_id,)).fetchone()

    def list_mutation_seed_rows(self, limit: int | None = None) -> list[sqlite3.Row]:
        q = (
            "SELECT * FROM seeds WHERE eligible = 1 AND length(trim(solution_text)) >= 20 "
            "ORDER BY id"
        )
        params: tuple[Any, ...] = ()
        if limit is not None:
            q += " LIMIT ?"
            params = (max(1, int(limit)),)
        with self._conn() as c:
            return list(c.execute(q, params).fetchall())

    def topic_counts(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT topic_norm, COUNT(*) FROM seeds WHERE eligible = 1 AND topic_norm IS NOT NULL GROUP BY topic_norm"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    # --- signals -------------------------------------------------------------

    def write_signals(
        self,
        seed_id: str,
        *,
        embedding: list[float],
        minhash: list[int],
        trace: str | None,
        fingerprint: str = "",
        fingerprint_embedding: list[float] | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO signals(seed_id,embedding,minhash,trace,fingerprint,fingerprint_embedding)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(seed_id) DO UPDATE SET
                     embedding=excluded.embedding,
                     minhash=excluded.minhash,
                     trace=excluded.trace,
                     fingerprint=excluded.fingerprint,
                     fingerprint_embedding=excluded.fingerprint_embedding""",
                (
                    seed_id,
                    json.dumps(embedding),
                    json.dumps(minhash),
                    trace,
                    fingerprint,
                    json.dumps(fingerprint_embedding or []),
                ),
            )

    # --- briefs / scaffolds / coverage / exemplars -------------------------

    def insert_brief(self, brief_id: str, brief_json: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO briefs(id, brief_json, created_at) VALUES(?,?,?)
                   ON CONFLICT(id) DO UPDATE SET brief_json=excluded.brief_json""",
                (brief_id, brief_json, time.time()),
            )

    def insert_scaffold(self, scaffold_id: str, brief_id: str, scaffold_json: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO scaffolds(id, brief_id, scaffold_json, created_at) VALUES(?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     brief_id=excluded.brief_id,
                     scaffold_json=excluded.scaffold_json""",
                (scaffold_id, brief_id, scaffold_json, time.time()),
            )

    def get_coverage(self, primary: str, secondary: str) -> int:
        with self._conn() as c:
            r = c.execute(
                "SELECT accept_count FROM coverage WHERE primary_concept = ? AND secondary_concept = ?",
                (primary, secondary),
            ).fetchone()
        return int(r[0]) if r else 0

    def bump_coverage(self, primary: str, secondary: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO coverage(primary_concept, secondary_concept, accept_count)
                   VALUES(?,?,1)
                   ON CONFLICT(primary_concept, secondary_concept)
                   DO UPDATE SET accept_count = accept_count + 1""",
                (primary, secondary),
            )

    def all_coverage(self) -> dict[tuple[str, str], int]:
        with self._conn() as c:
            rows = c.execute("SELECT primary_concept, secondary_concept, accept_count FROM coverage").fetchall()
        return {(r[0], r[1]): int(r[2]) for r in rows}

    def count_exemplars(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM exemplars").fetchone()[0])

    def has_exemplar_for_seed(self, seed_id: str) -> bool:
        with self._conn() as c:
            r = c.execute("SELECT 1 FROM exemplars WHERE seed_id = ? LIMIT 1", (seed_id,)).fetchone()
        return r is not None

    def insert_exemplar(self, exemplar_id: str, seed_id: str, scaffold_json: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO exemplars(id, seed_id, scaffold_json, created_at) VALUES(?,?,?,?)
                   ON CONFLICT(seed_id) DO UPDATE SET
                     id=excluded.id,
                     scaffold_json=excluded.scaffold_json,
                     created_at=excluded.created_at""",
                (exemplar_id, seed_id, scaffold_json, time.time()),
            )

    def list_exemplar_rows(self) -> list[sqlite3.Row]:
        with self._conn() as c:
            return list(c.execute("SELECT id, seed_id, scaffold_json FROM exemplars").fetchall())

    def list_exemplar_profile_rows(self) -> list[sqlite3.Row]:
        """Exemplar scaffolds plus seed topic metadata for corpus-profile sampling."""
        with self._conn() as c:
            return list(
                c.execute(
                    """SELECT e.id, e.seed_id, e.scaffold_json, s.topic_norm, s.topic_raw
                       FROM exemplars e JOIN seeds s ON e.seed_id = s.id
                       WHERE s.eligible = 1"""
                ).fetchall()
            )

    # --- outputs -------------------------------------------------------------

    def insert_output(self, **kwargs: Any) -> None:
        cols = (
            "id,question,answer,topic,parent_seed_ids,parent_fingerprints,brief_id,scaffold_id,"
            "embedding,minhash,audit_json,accepted_at,clean_accept,refinement_rounds"
        )
        with self._conn() as c:
            # Support legacy DBs that still require merge_mode column.
            pragma = c.execute("PRAGMA table_info(outputs)").fetchall()
            names = {row[1] for row in pragma}
            if "merge_mode" in names:
                c.execute(
                    f"""INSERT INTO outputs({cols},merge_mode)
                        VALUES(:id,:question,:answer,:topic,:parent_seed_ids,:parent_fingerprints,
                               :brief_id,:scaffold_id,:embedding,:minhash,:audit_json,:accepted_at,
                               :clean_accept,:refinement_rounds,:merge_mode)""",
                    {**kwargs, "merge_mode": kwargs.get("merge_mode", "")},
                )
            else:
                c.execute(
                    f"""INSERT INTO outputs({cols})
                        VALUES(:id,:question,:answer,:topic,:parent_seed_ids,:parent_fingerprints,
                               :brief_id,:scaffold_id,:embedding,:minhash,:audit_json,:accepted_at,
                               :clean_accept,:refinement_rounds)""",
                    kwargs,
                )

    def output_count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM outputs").fetchone()[0]

    def output_topic_counts(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute("SELECT topic, COUNT(*) FROM outputs GROUP BY topic").fetchall()
        return {r[0]: r[1] for r in rows}

    def list_outputs(self) -> list[sqlite3.Row]:
        with self._conn() as c:
            return list(c.execute("SELECT * FROM outputs").fetchall())

    # --- round log -----------------------------------------------------------

    def log_round(
        self,
        timestamp: float,
        topic: str | None,
        merge_mode: str | None,
        outcome: str,
        failure_kind: str | None = None,
        output_id: str | None = None,
        payload: dict | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO round_log(timestamp,topic,merge_mode,outcome,failure_kind,output_id,payload)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    timestamp,
                    topic,
                    merge_mode,
                    outcome,
                    failure_kind,
                    output_id,
                    json.dumps(payload) if payload else None,
                ),
            )

    # --- live dashboard / trace ----------------------------------------------

    def clear_pipeline_events(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM pipeline_events")

    def log_pipeline_event(
        self,
        phase: str,
        step: str,
        *,
        seed_id: str | None = None,
        attempt: int | None = None,
        message: str | None = None,
        payload: dict | None = None,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO pipeline_events(ts, phase, step, seed_id, attempt, message, payload)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    time.time(),
                    phase,
                    step,
                    seed_id,
                    attempt,
                    message,
                    json.dumps(payload, ensure_ascii=False) if payload else None,
                ),
            )
            return int(cur.lastrowid)

    def list_pipeline_events(self, after_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        lim = max(1, min(int(limit), 2000))
        with self._conn() as c:
            rows = c.execute(
                """SELECT id, ts, phase, step, seed_id, attempt, message, payload
                   FROM pipeline_events WHERE id > ? ORDER BY id ASC LIMIT ?""",
                (int(after_id), lim),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            pl = r["payload"]
            out.append(
                {
                    "id": r["id"],
                    "ts": r["ts"],
                    "phase": r["phase"],
                    "step": r["step"],
                    "seed_id": r["seed_id"],
                    "attempt": r["attempt"],
                    "message": r["message"],
                    "payload": json.loads(pl) if pl else None,
                }
            )
        return out

    def dashboard_counts(self) -> dict[str, int]:
        with self._conn() as c:
            n_seeds = c.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
            n_indexed = c.execute("SELECT COUNT(*) FROM seeds WHERE indexed = 1").fetchone()[0]
            n_eligible = c.execute("SELECT COUNT(*) FROM seeds WHERE eligible = 1").fetchone()[0]
            n_outputs = c.execute("SELECT COUNT(*) FROM outputs").fetchone()[0]
            n_events = c.execute("SELECT COUNT(*) FROM pipeline_events").fetchone()[0]
            last_ev = c.execute("SELECT MAX(id) FROM pipeline_events").fetchone()[0]
            n_ex = c.execute("SELECT COUNT(*) FROM exemplars").fetchone()[0]
        return {
            "seeds_total": int(n_seeds),
            "seeds_indexed": int(n_indexed),
            "seeds_eligible": int(n_eligible),
            "outputs": int(n_outputs),
            "exemplars": int(n_ex),
            "events_total": int(n_events),
            "last_event_id": int(last_ev or 0),
        }

    # --- single-question mutation -------------------------------------------

    def delete_seed_hinges(self, seed_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM seed_hinges WHERE seed_id = ?", (seed_id,))

    def insert_seed_hinge(
        self,
        *,
        hinge_id: str,
        seed_id: str,
        ordinal: int,
        hinge_text: str,
        label: str,
        model: str,
        prompt_version: str,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO seed_hinges(
                       id, seed_id, ordinal, hinge_text, label, model, prompt_version, created_at
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       ordinal=excluded.ordinal,
                       hinge_text=excluded.hinge_text,
                       label=excluded.label,
                       model=excluded.model,
                       prompt_version=excluded.prompt_version""",
                (
                    hinge_id,
                    seed_id,
                    int(ordinal),
                    hinge_text,
                    label,
                    model,
                    prompt_version,
                    time.time(),
                ),
            )

    def list_seed_hinges(self, seed_id: str) -> list[sqlite3.Row]:
        with self._conn() as c:
            return list(
                c.execute(
                    "SELECT * FROM seed_hinges WHERE seed_id = ? ORDER BY ordinal, id",
                    (seed_id,),
                ).fetchall()
            )

    def count_seed_hinges(self, seed_id: str) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM seed_hinges WHERE seed_id = ?", (seed_id,)).fetchone()[0])

    def insert_mutation_episode(
        self,
        *,
        episode_id: str,
        seed_id: str,
        hinge_ids: list[str],
        prompt_text: str,
        mutation_used: str = "",
        new_question: str | None = None,
        answer: str | None = None,
        short_solution: str | None = None,
        result: str = "pending",
        failure_kind: str | None = None,
        scores: dict | None = None,
        plan: dict | None = None,
        candidate: dict | None = None,
        story: str = "",
    ) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                """INSERT INTO mutation_episodes(
                       id, seed_id, hinge_ids, prompt_text, mutation_used, new_question, answer,
                       short_solution, result, failure_kind, scores_json, plan_json, candidate_json,
                       story, created_at, updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                       mutation_used=excluded.mutation_used,
                       new_question=excluded.new_question,
                       answer=excluded.answer,
                       short_solution=excluded.short_solution,
                       result=excluded.result,
                       failure_kind=excluded.failure_kind,
                       scores_json=excluded.scores_json,
                       plan_json=excluded.plan_json,
                       candidate_json=excluded.candidate_json,
                       story=excluded.story,
                       updated_at=excluded.updated_at""",
                (
                    episode_id,
                    seed_id,
                    json.dumps(hinge_ids),
                    prompt_text,
                    mutation_used,
                    new_question,
                    answer,
                    short_solution,
                    result,
                    failure_kind,
                    json.dumps(scores or {}, ensure_ascii=False),
                    json.dumps(plan or {}, ensure_ascii=False),
                    json.dumps(candidate or {}, ensure_ascii=False),
                    story,
                    now,
                    now,
                ),
            )

    def update_mutation_episode(
        self,
        episode_id: str,
        *,
        result: str,
        failure_kind: str | None,
        scores: dict,
        story: str,
    ) -> None:
        with self._conn() as c:
            c.execute(
                """UPDATE mutation_episodes
                   SET result = ?, failure_kind = ?, scores_json = ?, story = ?, updated_at = ?
                   WHERE id = ?""",
                (
                    result,
                    failure_kind,
                    json.dumps(scores, ensure_ascii=False),
                    story,
                    time.time(),
                    episode_id,
                ),
            )

    def get_mutation_episode(self, episode_id: str) -> sqlite3.Row | None:
        with self._conn() as c:
            return c.execute("SELECT * FROM mutation_episodes WHERE id = ?", (episode_id,)).fetchone()

    def list_pending_mutation_episodes(self, limit: int | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM mutation_episodes WHERE result = 'pending' ORDER BY created_at"
        params: tuple[Any, ...] = ()
        if limit is not None:
            q += " LIMIT ?"
            params = (max(1, int(limit)),)
        with self._conn() as c:
            return list(c.execute(q, params).fetchall())

    def list_mutation_stories(
        self,
        *,
        seed_id: str | None = None,
        result: str | None = None,
        limit: int = 5,
    ) -> list[sqlite3.Row]:
        clauses = ["story != ''"]
        params: list[Any] = []
        if seed_id is not None:
            clauses.append("seed_id = ?")
            params.append(seed_id)
        if result is not None:
            clauses.append("result = ?")
            params.append(result)
        q = (
            "SELECT * FROM mutation_episodes WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_at DESC LIMIT ?"
        )
        params.append(max(1, int(limit)))
        with self._conn() as c:
            return list(c.execute(q, tuple(params)).fetchall())

    def upsert_mutation_experience(
        self,
        *,
        experience_id: str,
        kind: str,
        lesson: str,
        source_episode_id: str,
        topic_norm: str = "",
        failure_kind: str | None = None,
        mutation_used: str = "",
        scope: str = "global",
        weight_delta: float = 1.0,
    ) -> None:
        now = time.time()
        with self._conn() as c:
            existing = c.execute(
                "SELECT * FROM mutation_experiences WHERE id = ?",
                (experience_id,),
            ).fetchone()
            if existing is None:
                c.execute(
                    """INSERT INTO mutation_experiences(
                           id, scope, kind, topic_norm, failure_kind, mutation_used, lesson,
                           source_episode_ids, source_count, weight, active, created_at, updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        experience_id,
                        scope,
                        kind,
                        topic_norm or "",
                        failure_kind,
                        mutation_used or "",
                        lesson,
                        json.dumps([source_episode_id], ensure_ascii=False),
                        1,
                        float(weight_delta),
                        1,
                        now,
                        now,
                    ),
                )
                return

            try:
                source_ids = json.loads(existing["source_episode_ids"] or "[]")
            except Exception:
                source_ids = []
            is_new_source = bool(source_episode_id) and source_episode_id not in source_ids
            if is_new_source:
                source_ids.append(source_episode_id)
            c.execute(
                """UPDATE mutation_experiences
                   SET topic_norm = COALESCE(NULLIF(topic_norm, ''), ?),
                       failure_kind = COALESCE(failure_kind, ?),
                       mutation_used = COALESCE(NULLIF(mutation_used, ''), ?),
                       source_episode_ids = ?,
                       source_count = ?,
                       weight = weight + ?,
                       active = 1,
                       updated_at = ?
                   WHERE id = ?""",
                (
                    topic_norm or "",
                    failure_kind,
                    mutation_used or "",
                    json.dumps(source_ids, ensure_ascii=False),
                    len(source_ids),
                    float(weight_delta) if is_new_source else 0.0,
                    now,
                    experience_id,
                ),
            )

    def list_mutation_experiences(
        self,
        *,
        kind: str | None = None,
        topic_norm: str | None = None,
        limit: int = 10,
        active_only: bool = True,
        prioritize_topic: bool = True,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if active_only:
            clauses.append("active = 1")
        if kind and kind != "all":
            clauses.append("kind = ?")
            params.append(kind)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        if topic_norm and prioritize_topic:
            order = (
                " ORDER BY CASE WHEN topic_norm = ? THEN 0 WHEN topic_norm = '' THEN 1 ELSE 2 END,"
                " weight DESC, source_count DESC, updated_at DESC"
            )
            params.append(topic_norm)
        else:
            order = " ORDER BY weight DESC, source_count DESC, updated_at DESC"
        q = "SELECT * FROM mutation_experiences" + where + order + " LIMIT ?"
        params.append(max(1, int(limit)))
        with self._conn() as c:
            return list(c.execute(q, tuple(params)).fetchall())

    def count_mutation_experiences(self, *, active_only: bool = False) -> int:
        q = "SELECT COUNT(*) FROM mutation_experiences"
        params: tuple[Any, ...] = ()
        if active_only:
            q += " WHERE active = 1"
        with self._conn() as c:
            return int(c.execute(q, params).fetchone()[0])

    def prune_mutation_experiences(self, *, max_active: int) -> int:
        max_active = max(1, int(max_active))
        with self._conn() as c:
            rows = c.execute(
                """SELECT id FROM mutation_experiences
                   WHERE active = 1
                   ORDER BY weight DESC, source_count DESC, updated_at DESC""",
            ).fetchall()
            if len(rows) <= max_active:
                return 0
            deactivate = [r["id"] for r in rows[max_active:]]
            c.executemany(
                "UPDATE mutation_experiences SET active = 0, updated_at = ? WHERE id = ?",
                [(time.time(), experience_id) for experience_id in deactivate],
            )
            return len(deactivate)
