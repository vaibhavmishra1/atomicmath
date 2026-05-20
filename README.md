# atomicmath

`atomicmath` is an iterative math problem synthesis pipeline.

It starts from solved seed problems and advances the dataset one inferred
iteration at a time:

```text
latest dataset rows -> generate variants -> refine variants -> judge variants
                    -> select one new row for the next iteration
```

The goal is to study whether new mathematical problem structure can emerge over
time from repeated generation, criticism, selection, and memory. The system is
not built around topic labels or hand-written hinge graphs. It operates directly
on question-answer pairs and records the full artifact trail for every generated
iteration.

## What It Does

For each sampled parent problem, atomicmath:

1. Loads a Hugging Face dataset with configurable question/answer fields.
2. Checks the configured `iteration` column.
3. Uses rows from the current max iteration as parents.
4. If no iteration column exists, treats the dataset as raw seeds and targets
   iteration `1`.
5. Generates several transformed candidate problems for that next iteration.
6. Refines all candidates together for correctness and compactness.
7. Judges candidates for correctness, novelty, depth, seed alignment,
   non-stitched structure, and solution economy.
8. Saves the accepted or rejected next-iteration row with artifacts and memory.
9. Optionally appends the new rows back to the same Hugging Face dataset or
   publishes them to a separate dataset.

Each saved row includes:

- `question`
- `answer`
- `iteration`
- `role` (`generated` or `rejected`)
- `lineage_id`
- `parent_id`
- `source_question`
- `source_answer`
- `memory`
- `memory_json`
- `scores_json`
- `artifacts_json`

`artifacts_json` stores raw generation, refinement, judgement, candidate, and
memory context for that iteration. The `memory` and `memory_json` fields are
written into the dataset itself, so the next run can read prior success/failure
lessons from parent rows without hidden external state.

## Install

```bash
pip install -e .
```

Set the provider keys required by your configured models:

```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...
```

## Configure

The only required input columns are a question field and an answer field. Their
names are configurable:

```yaml
input:
  dataset: "ShadenA/MathNet"
  config_name: "United_States"
  split: "train"
  question_field: "question"
  answer_field: "answer"
  iteration_field: "iteration"
  memory_field: "memory"
  max_seeds: 100
  seed: 42
```

The sampling seed makes selected examples stable across runs.

Output can be local only or pushed to Hugging Face:

```yaml
output:
  dataset: "vibhuiitj/atomicmath-lineage"
  split: "train"
  private: false
  push_to_hub: false
  append_if_same_dataset: true
  local_path: "./out/lineage/atomicmath_lineage.jsonl"
  summary_path: "./out/lineage/atomicmath_lineage_summary.json"
```

If `output.dataset` is the same as `input.dataset`, generated questions are
appended to the original dataset before upload. If it differs, atomicmath creates
or updates the output dataset with the new current-iteration rows.

## Run

Dry run seed sampling:

```bash
python3 -m atomicmath.cli run \
  --config examples/config.example.yaml \
  --num-seeds 10 \
  --dry-run
```

Generate lineages locally:

```bash
python3 -m atomicmath.cli run \
  --config examples/config.example.yaml \
  --num-seeds 10
```

Generate and upload to Hugging Face:

```bash
python3 -m atomicmath.cli run \
  --config examples/config.example.yaml \
  --num-seeds 100 \
  --push-to-hub \
  --dataset-id vibhuiitj/atomicmath-lineage
```

Launch the local web UI:

```bash
python3 -m atomicmath.cli web \
  --config examples/config.example.yaml \
  --port 8765
```

The UI lets you enter OpenAI/Hugging Face credentials for the run, edit grouped
dataset/model/quality/output settings, and inspect each parent row as queued,
running, done, rejected, or errored. It expands finished rows into generation,
refinement, judgement, selected candidate, scores, and raw artifact context.

## Pipeline

```text
Hugging Face dataset
        |
        v
deterministic seed sample
        |
        v
infer source max iteration and target current iteration
        |
        v
generator creates multiple transformed candidates
        |
        v
refiner improves all candidates together
        |
        v
judge scores correctness, novelty, depth, alignment, compactness
        |
        v
selector picks the next accepted question
        |
        v
iteration row + artifacts + memory
```

## Design Notes

The prompts use loose sectioned text rather than strict JSON. The program parses
standard headings such as `Problem:`, `Answer:`, `Solution:`, and judge score
fields. Raw responses are still saved, so failed parsing or weak candidates are
inspectable after a run.

Memory is dataset-native. Each generated or rejected row stores a compact
`memory` field plus detailed `memory_json` and `artifacts_json`. On the next run,
atomicmath reads memory from rows in the latest accepted iteration and uses it as
context for producing the next iteration.
