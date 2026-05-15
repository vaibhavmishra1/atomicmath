# atomicmath

`atomicmath` is a math problem synthesis system for generating contest-style
problems from solved seed problems.

The system is built around structural transformation rather than surface-level
rewriting. It identifies the mathematical bottlenecks that make a seed problem
work, then generates new problems that preserve those bottlenecks while changing
how the solver discovers them.

The system represents these bottlenecks as **hinges**.

## Synthesis Objective

A seed problem contains more than a topic label. It contains a small piece of
math logic that makes the problem work:

- a hidden equality case;
- a divisibility obstruction;
- a conjugate/root/eigenvalue trap;
- an extremal condition;
- a representation shift;
- a cyclic-order or parity mistake;
- a condition that must become tight.

The mutation pipeline extracts these hinges from the seed question and solution,
then generates a new problem that preserves the important mathematical logic
while changing the problem structure.

The synthesis criteria are:

```text
same mathematical bottleneck
different discovery path
similar difficulty
more creative problem shape
short solution
no stitched-on downstream task
```

## Mutation Pipeline

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

Each hinge note explains:

- what concept or trick is being tested;
- why students are likely to fail there;
- what triggers the hinge;
- what the hinge resolves in the solution;
- how the idea can be pushed toward the edge of solvability;
- what mutations would become artificial or boring.

A hinge is not the object being directly mutated. It is the guardrail that keeps
the new problem mathematically aligned with the seed.

## Generation

Generation is a single LLM call that performs both transformation planning and
problem writing.

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

The prompt does not force a fixed mutation type. It requires the model to reject
weak directions first, then choose a transformation that changes what the solver
must notice.

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

This is global memory, not question-wise memory. New seeds benefit from lessons
learned on previous generations.

Only a small top-K memory set is shown to the generator:

- top success memories;
- top failure memories;
- topic-matched memories are prioritized;
- old or low-weight memories remain in storage without being promoted into the
  active prompt context.

This keeps the prompt bounded while preserving reusable knowledge from prior
generations.

See [`docs/03-global-mutation-memory.md`](docs/03-global-mutation-memory.md).

## MathNet 100 Benchmark

The current benchmark compares two generation paths on the same 100 sampled
MathNet United States seed problems:

- **Direct baseline**: seed question + solution -> direct generation.
- **atomicmath**: seed question + solution -> hinge extraction -> mutation prompt
  with global memory -> generation.

Both outputs are evaluated with the same seed-relative comparison judge. This is
the apples-to-apples metric used for the headline acceptance rate.

| Metric | Direct baseline | atomicmath |
| --- | ---: | ---: |
| Total seeds | 100 | 100 |
| Generated | 96 | 95 |
| Judged | 96 | 94 |
| Accepted | 35 | 34 |
| Accepted rate | 35.0% | 34.0% |
| Correctness rate | 96.9% | 88.3% |
| Generation success rate | 96.0% | 95.0% |
| Judge success rate | 96.0% | 94.0% |
| Mean MinHash overlap | 0.047 | 0.017 |
| Mean embedding cosine | 0.603 | 0.612 |
| Mean depth score | 0.482 | 0.564 |
| Mean contest score | 0.628 | 0.678 |
| Mean novelty score | 0.659 | 0.482 |
| Mean seed alignment | 0.628 | 0.790 |
| Mean non-stitched score | 0.981 | 0.990 |
| Mean solution economy | 0.889 | 0.880 |
| Mean routine score | 0.690 | 0.652 |

Failure breakdown under the shared comparison judge:

| Failure kind | Direct baseline | atomicmath |
| --- | ---: | ---: |
| accepted | 35 | 34 |
| generation/error | 4 | 6 |
| incorrect | 3 | 11 |
| near_paraphrase | 6 | 23 |
| routine | 17 | 8 |
| weak_quality | 35 | 18 |

The shared judge result is currently a near tie on accepted rate. atomicmath
improves depth, contest score, seed alignment, non-stitched structure, and
routine score, but loses ground on correctness and near-paraphrase rejections.
This suggests the hinge path is producing more purposeful transformations, but
the generation/judging loop still needs stricter correctness control and better
anti-sibling pressure.

atomicmath also reports its own hinge-aware internal judge:

| Internal atomicmath metric | Value |
| --- | ---: |
| Accepted | 82 / 100 |
| Rejected | 18 / 100 |
| Internal accepted rate | 82.0% |
| Mean hinge preservation | 0.945 |
| Mean mutation quality | 0.883 |
| Mean sharpness | 0.827 |
| Mean atomic novelty | 0.800 |
| Mean atomic non-stitched score | 0.996 |
| Mean atomic solution economy | 0.898 |

The gap between internal acceptance (82%) and shared-judge acceptance (34%) is
important. The internal judge is useful for measuring whether the pipeline thinks
it preserved and transformed the hinge, but the comparison judge is stricter for
external quality. The next optimization target is to align the internal
atomicmath judge more closely with the shared comparison judge.

Published benchmark datasets:

- [`vibhuiitj/mathnet-direct-baseline_100`](https://huggingface.co/datasets/vibhuiitj/mathnet-direct-baseline_100)
- [`vibhuiitj/mathnet-atomicmath_100`](https://huggingface.co/datasets/vibhuiitj/mathnet-atomicmath_100)

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

## Retrosynthesis Path

The repository also includes a composer/realizer pipeline:

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

## Operational Notes

The mutation probe uses multiple LLM calls per seed:

- hinge extraction;
- plan + generate;
- correctness verification when enabled;
- quality judging.

Use small `--limit` values for initial validation. The SQLite DB and cache make
runs inspectable and resumable.
