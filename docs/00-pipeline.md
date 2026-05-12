# atomicmath pipeline (current)

This document matches the code in `atomicmath/`. The flow is **retrosynthesis**: sample a concept brief → compose a scaffold → realize a problem → verify → publish.

## 1. Data flow

1. **Ingest** (`ingest.py`): HuggingFace `datasets.load_dataset`; optional `input.config_name`; filters (language, problem type, country, etc.); writes `seeds` with `solution_text` from `input.solution_field`. Required dataset columns are only `question_field` and `solution_field`. Optional fields (`answer_field`, `topic_field`, `language_field`, `problem_type_field`, `country_field`) are ignored when absent. If `topic_field` is absent, `input.default_topic` is used. For MathNet, `language` is often blank — a non-empty `language` column must match `language_filter`; empty means rely on `filters.language` ASCII heuristic only. `problem_type_filter` entries like `answer only` also match MathNet’s `final answer only`.
2. **Topic normalization** (`ingest.normalize_topics`): maps `topic_raw` → `topic_norm`; clusters with embeddings + LLM labels when there are many distinct raw topics; marks seeds in topics with fewer than `filters.min_topic_size` rows as ineligible.
3. **Index** (`index_pass.py`): per eligible seed — question embedding, MinHash on `(question, answer)`, `signals.trace` = solution excerpt (or answer). Legacy columns `signals.fingerprint` / `fingerprint_embedding` stay empty (no LLM fingerprint step).
4. **Sanity** (`sanity.py`): placeholder; always passes.
5. **Exemplar pool** (`exemplars.py`): LLM reverse-engineers a scaffold JSON from each indexed seed with enough `solution_text`; stored in `exemplars` (one row per `seed_id`).
6. **Synthesis loop** (`pipeline.py`): `curriculum.sample_brief` → `exemplars.retrieve` → `composer.design` → persist `briefs` / `scaffolds` → `realizer.write` (best-of-K) → `verify_correctness` → moderate `quality.judge_quality` gate → `NoveltyIndex` (MinHash + cosine vs seeds + prior outputs) → on accept `insert_output` + `bump_coverage(primary_concept, secondary_concept)`. Correctness first checks symbolic/string canonical equality; if that fails and `gate.answer_equivalence_fallback` is true, a small LLM (`gate.answer_equivalence_model`) judges whether textual final answers are mathematically equivalent. The quality gate is not an IMO filter; it rejects clearly routine, direct-formula, exercise-like problems while allowing accessible contest-style outputs.
7. **Publish** (`publish.py`): main HF dataset `(question, answer, topic)`; optional audit sidecar with ids, embeddings, minhash, `brief_id`, `scaffold_id`, `audit_json`.

There is **no** difficulty band, **no** parent-seed sampler, **no** procedural fingerprint LLM, **no** pair-rating tensor in the schema.

## 2. SQLite tables (conceptual)

| Table | Role |
|-------|------|
| `seeds` | Ingested problems + `solution_text`, eligibility, indexed flag |
| `signals` | `embedding`, `minhash`, `trace` per seed |
| `briefs`, `scaffolds` | One row per synthesis attempt that reached composer |
| `coverage` | Accept counts per ordered `(primary_concept, secondary_concept)` |
| `exemplars` | Bootstrapped scaffold JSON keyed by `seed_id` |
| `outputs` | Accepted synthesized rows (`brief_id`, `scaffold_id`, empty `parent_*` JSON arrays) |
| `round_log`, `pipeline_events`, `run_state` | Logging and resumability |

## 3. Config surface

See `examples/config.example.yaml`. Unknown top-level YAML keys are ignored (`Config.model_config.extra = "ignore"`) so old `sampler:` / `merge_mix:` blocks do not break loads.

## 4. CLI

```bash
atomicmath run --config path/to/config.yaml
atomicmath run --config path/to/config.yaml --dry-run
atomicmath watch --config path/to/config.yaml --port 8765
```

## 5. Costs

Indexing is **one embedding + MinHash per seed** (cheap). Exemplar bootstrap is **one LLM JSON call per seed** (capped). Each synthesis round is composer + realizer (K generations) + verifiers + quality judge + novelty. Use `--dry-run` to validate ingest and index without spending on bootstrap/synthesis.
