# Global Mutation Memory

The mutation path now separates raw history from reusable memory.

```text
mutation_episodes
  raw per-attempt log:
  seed, hinges, prompt, candidate, scores, accept/reject story

mutation_experiences
  distilled global memory:
  compact success/failure lessons with topic, failure kind, weight, source episodes

generation prompt
  top-K global success memories
  top-K global failure memories
  current seed question + solution + hinges
```

This replaces question-wise memory as the main learning signal. A seed no longer needs
previous attempts on itself to benefit from memory; any judged mutation can teach future
generations.

## Write Path

After `atomicmath mutate judge` scores an episode:

1. `mutation_episodes` is updated with result, scores, and judge story.
2. The story is promoted into `mutation_experiences`.
3. The experience receives:
   - `kind`: `success` or `failure`
   - `topic_norm`
   - `failure_kind`
   - `mutation_used`
   - compact lesson text
   - source episode IDs
   - score-derived weight
4. Active memories are pruned to `mutation.global_memory_max_active`.

## Read Path

When building the next generation prompt, the pipeline loads:

- `mutation.global_success_memory_limit` success memories
- `mutation.global_failure_memory_limit` failure memories

Topic-matched memories are prioritized, but memory is still global. The model sees only
this small top-K working set, not the full history.

## Useful Commands

```bash
python3 -m atomicmath.cli mutate memory \
  --config examples/config.example.yaml \
  --backfill \
  --limit 10
```

Backfill promotes already judged `mutation_episodes` into global memory.

```bash
python3 -m atomicmath.cli mutate memory \
  --config examples/config.example.yaml \
  --kind failure \
  --limit 10
```

Inspect failure memories currently being used as avoidance guidance.
