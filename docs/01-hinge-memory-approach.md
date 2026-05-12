# Hinge Memory Approach

This document describes the proposed next-generation synthesis approach for `atomicmath`.
It is not the current production pipeline. The current code still follows the
retrosynthesis flow documented in [`00-pipeline.md`](00-pipeline.md).

## Objective

Given solved seed problems, generate new math problems that preserve the real
mathematical difficulty of the seeds while becoming more creative and less
surface-derivative.

The system should not decorate old questions, add routine calculation, or stitch
multiple problems together. It should identify the student-facing bottlenecks in
solutions, then synthesize a new problem where one such bottleneck is pushed close
to the edge of solvability.

## Core Idea

The unit of memory is a **hinge story**.

A hinge is a concept, trick, lemma, rule, or piece of mathematical logic where a
student is likely to get stuck. It should describe the actual reasoning move, not
just the broad topic.

Examples of broad topics that are too weak:

- `geometry`
- `polynomials`
- `number theory`
- `angle chasing`

Examples of useful hinges:

- `An interior equality point under a global nonnegativity condition forces a
  tangency or repeated-root condition.`
- `An inscribed angle must use the arc not containing the angle vertex; choosing
  the visually shorter arc gives the wrong answer.`
- `A tangent-chord angle can be transferred into an inscribed angle equality,
  which then unlocks triangle similarity.`

The hinge note must be self-sufficient. A future generator should understand the
hinge without needing the original question in front of it.

## Hinge Story Format

The extraction prompt should ask for 2-3 hinges from each solved question.
Each hinge should be written as loose structured text, not strict JSON.

Recommended fields:

```text
HINGE_NAME:
Short memorable name.

ABSTRACT_SUMMARY:
What the hinge is in reusable mathematical language.

WHAT_IT_TESTS:
The real concept or logic being tested.

WHY_STUDENTS_FAIL:
The common mistake, missing observation, or false shortcut.

GENERAL_SOLVING_IDEA:
How this kind of hinge is usually unlocked.

EDGE_OF_SOLVABILITY:
How to sharpen this hinge without adding boring length or computation.

FAILURE_MODES:
How generated questions using this hinge usually become bad.
```

The original question and solution should still be stored as provenance, but the
hinge definition should not depend on them.

## Local Spine

For each source question, the extracted hinges form a **local spine**.

The local spine records how the solution moves from one hinge to the next:

```text
h1 -> h2 -> h3
```

This is not a global ontology. It is only the internal reasoning flow of one
question.

Example:

```text
global nonnegativity + equality at an interior point
  -> infer tangency / double-root behavior
  -> factor and extract the parameter
  -> certify positivity with a square factor
```

The local spine is useful because it preserves coherence. A new generated problem
should usually borrow a clean spine from one source question instead of combining
many unrelated hinges.

## Cross-Question Retrieval

Cross-question matching should be based on hinge-story embeddings.

The goal is not to prove that two hinges are identical. The goal is to find
nearby or compatible hinge stories from other questions that might provide a
creative twist.

For a source question:

1. Extract its 2-3 hinge stories.
2. Embed each hinge story.
3. Retrieve similar hinge stories from other questions.
4. Choose one related question or hinge neighborhood as a twist source.

This avoids the hardest version of canonicalization. We do not need to decide
whether two hinges are exactly the same before trying generation.

## Generation Prompt Shape

The generator should see two separate sections, not one flattened blob.

```text
SOURCE Q1 HINGE SPINE:
h1 -> h2 -> h3

RELATED Q2 HINGE SPINE:
h4 -> h5 -> h6

TASK:
Create one new math problem inspired by these hinge stories.
Choose one primary hinge as the backbone.
Optionally borrow one twist from the related hinge spine.
Do not stitch multiple problems together.
Do not add lengthy routine computation.
The challenge must be logical, not merely long.
Push the primary hinge closer to the edge of solvability.

RETURN:
- Problem
- Answer
- Short solution
- Primary hinge being tested
- Borrowed twist, if any
- Why the result is sharper but not bloated
```

The model has freedom, but only within a controlled frame:

- It may sharpen the base hinge.
- It may borrow one compatible twist.
- It may reframe the same logic in a new configuration.
- It should not merge all supplied hinges equally.

## Persistent Memory

The proposed persistent objects are:

### `hinge_story`

A self-contained note describing one difficult mathematical move.

Conceptual fields:

```json
{
  "hinge_id": "h_001",
  "question_id": "q_001",
  "domain": "geometry",
  "hinge_text": "...",
  "embedding": [...],
  "source_question": "...",
  "source_solution": "..."
}
```

### `local_spine_edge`

An intra-question edge showing how hinge stories connect inside one solved
problem.

```json
{
  "question_id": "q_001",
  "from": "h_001",
  "to": "h_002",
  "relation": "unlocks"
}
```

### `cross_question_affinity`

A soft learned edge between hinge stories from different questions.

This edge does not mean the hinges are identical. It means they have been useful
or plausible together during retrieval/generation.

```json
{
  "from": "h_001",
  "to": "h_884",
  "affinity": 0.72,
  "successes": 5,
  "failures": 2,
  "notes": "Works when h_001 is primary and h_884 is only a twist."
}
```

### `synthesis_episode_memory`

A record of attempted generation from hinge spines.

```json
{
  "episode_id": "ep_001",
  "source_question_id": "q_001",
  "related_question_id": "q_019",
  "hinges_used": ["h_001", "h_002", "h_884"],
  "result": "accepted",
  "failure_mode": null,
  "lesson": "The borrowed tangent-chord hinge sharpened the arc-choice trap without adding a long calculation."
}
```

## Feedback Loop

Generation should update memory.

After every attempt, judge:

- Was the primary hinge necessary?
- Was the borrowed hinge meaningful or decorative?
- Did the problem become sharper or just longer?
- Was the solution coherent and short?
- Did the problem become ambiguous or stitched?

Update `cross_question_affinity` and `synthesis_episode_memory` accordingly.

Successful combinations get stronger. Bad combinations get weaker and store
their failure modes.

## Verification Gates

A generated problem should not be accepted just because it is novel.

Minimum gates:

1. **Correctness**
   The answer and short solution must be mathematically valid.

2. **Hinge necessity**
   The intended primary hinge must be central. If the problem can be solved
   cleanly without it, reject.

3. **No stitching**
   The question should feel like one coherent problem, not two problems glued
   together.

4. **No boring complexity**
   Added difficulty should come from a sharper logical trap, not long arithmetic
   or many routine steps.

5. **Novelty**
   The generated problem should not be a paraphrase, number swap, or direct
   scaffold copy.

## MVP Plan

The first useful prototype should stay small:

1. Extract 2-3 hinge stories per MathNet question.
2. Store hinge stories and local spines.
3. Embed hinge stories.
4. For one selected source question, retrieve one related question by hinge
   similarity.
5. Prompt the generator with `Q1 hinge spine` and `Q2 hinge spine`.
6. Generate 3 candidate problems.
7. Manually inspect coherence, hinge necessity, and non-stitching.
8. Store the attempt in synthesis memory.

Only after this works should we build canonical hinge nodes or a larger graph.

## Design Principle

The system should optimize for:

```text
one clean mathematical bottleneck
+ one optional compatible twist
+ short coherent solution
+ no decorative complexity
```

It should avoid:

```text
surface paraphrases
+ arbitrary theorem stacking
+ long computations
+ multipart stitched tasks
+ vague topic-level similarity
```

