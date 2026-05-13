# atomicmath

`atomicmath` is an experimental pipeline for creating new contest-style math
problems from solved seed problems.

The current main idea is not to decorate a problem with extra conditions or
longer calculations. The pipeline tries to identify the mathematical places
where a student is likely to get stuck, then asks a model to transform the
problem so that one of those bottlenecks is preserved but appears in a sharper
or less obvious form.

In this repo those bottlenecks are called **hinges**.

## Core Idea

A seed problem contains more than a topic label. It contains a small piece of
math logic that makes the problem work:

- a hidden equality case;
- a divisibility obstruction;
- a conjugate/root/eigenvalue trap;
- an extremal condition;
- a representation shift;
- a cyclic-order or parity mistake;
- a condition that must become tight.

The mutation pipeline tries to extract those hinges from the seed question and
solution, then generate a new problem that keeps the important hinge while
changing the surface structure.

The target is:

```text
same mathematical bottleneck
different discovery path
similar difficulty
more creative problem shape
short solution
no stitched-on downstream task
```

## Current Mutation Pipeline

```text
Hugging Face seed dataset
        |
        v
ingest solved seed problems
        |
        v
extract hinge notes
        |
        v
build mutation prompt
  - seed question
  - seed solution
  - hinge notes
  - transformation examples
  - global success memory
  - global failure memory
        |
        v
plan + generate in one LLM call
        |
        v
candidate problem
        |
        v
judge
  - correctness
  - hinge preservation
  - mutation quality
  - sharpness
  - novelty
  - non-stitched structure
  - solution economy
        |
        v
accepted / rejected episode
        |
        v
distill global mutation memory
        |
        v
optional publish to Hugging Face
```

## Hinge Extraction

For each seed question, the extractor writes 2-3 self-contained hinge notes.

A hinge note should explain:

- what concept or trick is being tested;
- why students are likely to fail there;
- what triggers the hinge;
- what the hinge resolves in the solution;
- how the idea could be pushed toward the edge of solvability;
- what mutations would become artificial or boring.

The hinge is not the thing being directly mutated. It is the guardrail that
keeps the new problem mathematically related to the seed.

## Generation

Generation is a single LLM call that performs both planning and problem writing.

The model receives:

- the original question;
- the original answer and solution;
- extracted hinge notes;
- examples of strong and weak transformations;
- global memory of previous successes and failures.

It must return a structured candidate containing:

- the chosen transformation;
- what got mutated;
- why the mutation is nontrivial;
- the new question;
- answer;
- short solution;
- why the result is sharper;
- why it is not stitched;
- why it is not a direct sibling of the seed.

The prompt intentionally does not force a fixed mutation type. The model is
asked to reject weak ideas first, then choose a transformation that changes what
the solver must notice.

## Global Mutation Memory

The system keeps two layers of memory:

```text
mutation_episodes
  raw log of every attempt
  seed, prompt, candidate, scores, accept/reject story

mutation_experiences
  distilled global memory
  compact lessons about what worked and what failed
```

This is global memory, not question-wise memory. A new seed can benefit from
lessons learned on older seeds.

Only a small top-K memory set is shown to the generator:

- top success memories;
- top failure memories;
- topic-matched memories are prioritized;
- old/low-weight memories are kept in storage but not necessarily shown.

This prevents the prompt from growing forever while still allowing the system to
learn from previous attempts.

See [`docs/03-global-mutation-memory.md`](docs/03-global-mutation-memory.md).

## Storage

The pipeline stores progress in SQLite. The most relevant mutation tables are:

- `seeds`: ingested seed problems;
- `seed_hinges`: extracted hinge notes;
- `mutation_episodes`: every generated candidate and judge result;
- `mutation_experiences`: distilled global success/failure memory.

The default example config writes to:

```text
./atomicmath_math500.db
./cache/
```

## Install

```bash
pip install -e .
```

Set API keys required by your configured models and upload target:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...
```

## Configuration

The example config uses `HuggingFaceH4/MATH-500`:

```bash
examples/config.example.yaml
```

Important mutation settings:

```yaml
mutation:
  extraction_model: "openai/gpt-5-mini"
  generation_model: "openai/gpt-5-mini"
  judge_model: "openai/gpt-5-mini"
  global_memory_enabled: true
  global_success_memory_limit: 5
  global_failure_memory_limit: 5
  global_memory_max_active: 300
```

## Common Commands

Inspect or backfill global memory:

```bash
python3 -m atomicmath.cli mutate memory \
  --config examples/config.example.yaml \
  --backfill \
  --limit 10
```

Run a small end-to-end mutation probe:

```bash
python3 scripts/probe_mutation_generation.py \
  --config examples/config.example.yaml \
  --limit 10 \
  --n 1
```

Print the exact generation prompt for a seed:

```bash
python3 -m atomicmath.cli mutate build-prompt \
  --config examples/config.example.yaml \
  --seed-id <seed_id>
```

Generate candidates for one seed:

```bash
python3 -m atomicmath.cli mutate generate \
  --config examples/config.example.yaml \
  --seed-id <seed_id> \
  --n 1 \
  --judge
```

Publish accepted mutation outputs to Hugging Face:

```bash
python3 -m atomicmath.cli mutate publish \
  --config examples/config.example.yaml \
  --dataset vibhuiitj/math500-output_atomicmath
```

## Older Retrosynthesis Path

The repo also contains an older composer/realizer pipeline:

```text
ingest -> normalize topics -> index seeds -> bootstrap exemplars
      -> compose scaffold -> realize candidate -> verify -> publish
```

Run it with:

```bash
atomicmath run --config examples/config.example.yaml
atomicmath run --config examples/config.example.yaml --dry-run
```

See [`docs/00-pipeline.md`](docs/00-pipeline.md) for that path.

## Cost Notes

The mutation probe uses multiple LLM calls per seed:

- hinge extraction;
- plan + generate;
- correctness verification when enabled;
- quality judging.

Use small `--limit` values first. The SQLite DB and cache let you inspect and
continue runs without throwing away previous work.
