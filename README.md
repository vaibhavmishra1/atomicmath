# atomicmath

Generate novel contest-style math problems from a Hugging Face seed dataset using a **retrosynthesis** loop: concept brief → scaffold → realized problem, with correctness and novelty gates.

## Install

```bash
pip install -e .
```

Set API keys (as required by your `models.*` entries), e.g.:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...   # for Hub push / gated datasets
```

## Run

```bash
atomicmath run --config examples/config.example.yaml
atomicmath run --config examples/config.example.yaml --dry-run
```

Progress and caches live in the SQLite file and `./cache/` from your config. See [`docs/00-pipeline.md`](docs/00-pipeline.md) for stages and tables.

## Costs

Bootstrap and synthesis are LLM-heavy. Use `--dry-run` to validate ingest and indexing only; lower `runtime.target_count` and `input.max_seeds` for smoke tests.
