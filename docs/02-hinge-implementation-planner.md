# Single-Question Mutation Planner

This is the current implementation plan for the hinge/mutation idea.

Important correction: this version does **not** build a hinge graph, embed
hinges, retrieve similar hinges, canonicalize hinge nodes, or compose hinges
from multiple questions.

Version 2 transforms **one solved question at a time**.

The input is:

```text
original question
original solution
hinge notes explaining what the problem really tests
transformation examples
success mutation stories
failure mutation stories
combined plan-and-generate instructions
```

The output is:

```text
new question
answer
short solution
mutation used
conceptual delta
reason for mutation
why sharper
why not stitched
```

## Objective

Given one solved seed problem, generate a new math problem that preserves the
seed's real mathematical bottleneck while changing the surface form, structure,
or edge case enough to become genuinely new.

The generator should not decorate the old problem, add long calculations, or
stitch extra tasks onto it. It should understand what the original problem is
testing, then mutate the problem in a controlled way.

The hinge is not the object being mutated directly. The hinge is the guardrail:
it tells the model what important mathematical logic must remain alive in the
new problem.

## Core Prompt Shape

The pipeline now uses one combined prompt after hinge extraction:

1. Internally plan a transformation, reject weak ideas, and choose a direction.
2. Generate the final problem in the same API call.

The combined prompt should look like this:

```text
QUESTION:
<original problem>

SOLUTION:
<original solution>

HINGES:
<2-3 hinge notes explaining the important mathematical bottlenecks>

TRANSFORMATION EXAMPLES:
Cow/barn example:
<simple non-math example showing what a good or bad transformation means>
Math interpretation:
<how this applies to math problems>

SUCCESS MUTATION STORIES:
<examples of previous mutations that worked, what changed, and why>

FAILURE MUTATION STORIES:
<examples of previous mutations that failed, what changed, and why>

TASK:
Brainstorm several transformations.
Reject weak transformations.
Choose one ambitious but solvable transformation.
Explain the conceptual delta and preserved hinge.
```

The final generator then receives the chosen transformation plan and writes the
new question.

```text
QUESTION + SOLUTION + HINGES
TRANSFORMATION PLAN

TASK:
Generate one new math problem from the transformation plan.
Return answer, short solution, preserved hinge, conceptual delta, why sharper,
why not stitched, and why not a direct sibling of the seed.
```

## Why Use Cow/Barn Examples

The model often misunderstands "mutation" as:

- changing numbers;
- adding a condition;
- making the statement longer;
- adding an extra theorem;
- changing the topic completely.

Cow/barn examples teach mutation in a low-stakes domain before the model applies
the idea to math.

Example:

```text
Original: A cow is tied to a barn and can graze in a circular region.

Bad mutation: Add two more cows, three fences, and ask for total grass.
Why bad: This increases bookkeeping but does not create a sharper idea.

Good mutation: Move the cow's rope anchor from the outside wall to a corner.
Why good: The same grazing-radius idea remains, but the accessible region now
depends on an edge case of the geometry.
```

This makes the desired behavior concrete: mutate the structure, not the length.

## Transformation Examples

These examples teach the planner what good and bad transformation behavior looks
like. They are **not** fixed categories that the model must choose from.

### 1. Parameter Edge Mutation

Cow/barn example:

```text
Original: A cow tied to the middle of a barn wall grazes a semicircle.
Mutation: Tie the cow exactly at a corner, where the accessible region changes
from a clean semicircle to a boundary-sensitive sector.
```

Math interpretation:

```text
Move a condition to the boundary where a hidden equality case, degeneracy,
extremum, repeated root, tangent case, parity boundary, or equality condition
becomes decisive.
```

Use when:

- the original solution depends on equality;
- the important move is recognizing a boundary case;
- the original has a parameter that can be pushed to a threshold.

Avoid:

- ugly parameter values;
- long casework;
- making the problem unsolvable by crossing the boundary.

### 2. Representation Shift Mutation

Cow/barn example:

```text
Original: A cow's grazing area is described by a rope length.
Mutation: Describe the same constraint through a fence shadow or a gate path,
so the solver must translate the situation before using the same idea.
```

Math interpretation:

```text
Keep the same core logic but express it through a different representation:
polynomial roots instead of inequalities, vectors instead of coordinates,
geometry instead of algebra, recurrence instead of closed form, or graph
condition instead of number condition.
```

Use when:

- the original hinge is too visible;
- the problem can become more creative by hiding the same structure;
- the solution remains short after translation.

Avoid:

- changing to a representation that requires unrelated machinery;
- making translation the whole problem;
- losing the original mathematical bottleneck.

### 3. Constraint Swap Mutation

Cow/barn example:

```text
Original: The cow has a fixed rope and the barn position is fixed.
Mutation: The grazing area is fixed, and the rope length must be determined.
```

Math interpretation:

```text
Invert the role of known and unknown quantities while preserving the same
reasoning engine. For example, instead of proving a maximum, ask for the
condition under which equality is possible.
```

Use when:

- the original problem solves forward from assumptions to value;
- reversing the direction makes the hidden condition more important;
- the same hinge becomes necessary in the reverse direction.

Avoid:

- turning the problem into a routine algebra solve;
- introducing multiple independent unknowns;
- making the answer non-unique without adding a clean uniqueness condition.

### 4. Trap Reorientation Mutation

Cow/barn example:

```text
Original: The tempting mistake is to use the full circular grazing area.
Mutation: Make the tempting mistake a different one: the cow can reach around
one corner but not the other.
```

Math interpretation:

```text
Preserve the kind of student mistake, but change where the trap appears. The new
problem should still punish the same false shortcut, but not in the same visual
or algebraic form.
```

Use when:

- the original problem has a clear common wrong move;
- the hinge is mainly about rejecting a tempting shortcut;
- the new version can expose that shortcut less obviously.

Avoid:

- artificial trick wording;
- gotcha ambiguity;
- traps that depend on hidden conventions.

### 5. Minimal Twist Mutation

Cow/barn example:

```text
Original: A cow tied to a barn grazes an area.
Mutation: Add one gate that opens only if the cow reaches it; this changes the
same grazing logic without creating a second problem.
```

Math interpretation:

```text
Add exactly one secondary condition that makes the original hinge interact with
a new but lightweight idea.
```

Use when:

- the seed is mathematically clean but too direct;
- one small condition can make the hinge less obvious;
- the solution still has one main path.

Avoid:

- adding an unrelated theorem;
- adding a second final question;
- adding a long verification after the main idea is done.

## Hinge Notes

The hinge notes should explain what the seed problem actually tests.

They should not just say:

```text
inequality
polynomial
geometry
number theory
```

They should say things like:

```text
The equation tests whether the solver recognizes that global nonnegativity plus
equality at an interior point forces tangency or repeated-root behavior.
```

or:

```text
The problem tests the ability to choose the arc intercepted by an inscribed
angle, not simply the visually shorter arc between two labeled vertices.
```

Recommended hinge note format:

```text
HINGE_NAME:
Short name.

WHAT_THE_PROBLEM_TESTS:
The important mathematical concept, trick, rule, lemma, or logic.

WHY_THIS_IS_NONTRIVIAL:
Why students are likely to miss it.

COMMON_WRONG_MOVE:
The tempting but incorrect approach.

HOW_THE_SOLUTION_UNLOCKS:
The general solving move.

WHAT_A_GOOD_MUTATION_SHOULD_PRESERVE:
What must remain true for the new problem to still test this hinge.
```

## Success Mutation Story

A success story teaches the model what worked before.

Format:

```text
SOURCE HINGE:
<what the old problem tested>

TRANSFORMATION USED:
<free-form transformation name>

WHAT GOT MUTATED:
<specific structural change>

WHY IT WORKED:
<why the new problem became sharper without becoming longer>

WHAT TO REUSE:
<general lesson>
```

Example:

```text
SOURCE HINGE:
Global nonnegativity plus equality at an interior point forces a double-root
condition.

TRANSFORMATION USED:
Reverse the hidden equality condition.

WHAT GOT MUTATED:
Instead of giving the exponent and asking for positivity, the mutated problem
gave positivity and asked which exponent can make equality occur at a specified
interior point.

WHY IT WORKED:
The repeated-root idea became necessary earlier. The problem did not add a new
theorem or long computation.

WHAT TO REUSE:
Reverse the direction of the condition when it makes the hinge unavoidable.
```

## Failure Mutation Story

A failure story teaches what not to do.

Format:

```text
SOURCE HINGE:
<what the old problem tested>

TRANSFORMATION USED:
<free-form transformation name>

WHAT GOT MUTATED:
<specific structural change>

WHY IT FAILED:
<stitched, routine, ambiguous, too long, hinge disappeared, incorrect, etc.>

WHAT TO AVOID:
<general lesson>
```

Example:

```text
SOURCE HINGE:
Inscribed angle requires choosing the arc not containing the angle vertex.

MUTATION USED:
Minimal Twist Mutation.

WHAT GOT MUTATED:
Added a second circle and asked for another angle after solving the first one.

WHY IT FAILED:
The new part was only attached after the original hinge was finished. It became
two problems glued together.

WHAT TO AVOID:
Do not add downstream tasks after the main hinge has already been resolved.
```

## Generator Output Contract

The generator should return structured text or JSON.

Recommended JSON for easier storage:

```json
{
  "new_question": "...",
  "answer": "...",
  "short_solution": "...",
  "mutation_used": "Parameter Edge Mutation",
  "what_got_mutated": "...",
  "reason_for_mutation": "...",
  "primary_hinge_preserved": "...",
  "why_problem_is_sharper": "...",
  "why_not_stitched": "...",
  "risk_notes": "..."
}
```

Strict JSON is useful for the final generator output, but not required for hinge
notes. Hinge extraction can stay loose and story-like.

## Quality Gate

After generation, judge the candidate on these dimensions:

```text
correctness
primary hinge preserved
mutation is meaningful
problem is sharper
problem is not stitched
solution is not bloated
novelty from the original seed
```

Suggested judge output:

```json
{
  "pass": true,
  "correctness": 0.0,
  "hinge_preservation": 0.0,
  "mutation_quality": 0.0,
  "sharpness": 0.0,
  "non_stitched": 0.0,
  "solution_economy": 0.0,
  "novelty": 0.0,
  "failure_kind": null,
  "reason": "..."
}
```

Reject if:

- the answer or solution is wrong;
- the original hinge disappears;
- the mutation is just number swapping;
- the mutation adds routine length;
- the problem has multiple glued tasks;
- the final solution needs an unrelated heavy theorem;
- the candidate is a near paraphrase of the seed.

## Data Model

For Version 1, keep the schema small.

### `seed_hinges`

Stores hinge notes extracted from one seed.

```sql
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
```

No embeddings.

No canonical hinge IDs.

No cross-question edges.

### `mutation_episodes`

Stores each mutation attempt.

```sql
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
    story TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    FOREIGN KEY (seed_id) REFERENCES seeds(id)
);
```

`story` should contain either the success mutation story or failure mutation
story generated after judging.

## Implementation Phases

### Phase 1: Hinge Extraction for One Seed

Build:

```text
atomicmath/mutation_hinges.py
```

CLI:

```text
atomicmath mutate extract-hinges --config examples/config.example.yaml --seed-id <id>
```

Behavior:

- load one seed question and solution;
- ask the model for 2-3 hinge notes;
- store hinge notes in `seed_hinges`;
- print them for inspection.

Acceptance:

- hinge notes are self-contained;
- each note explains what the original problem tests;
- the notes mention common wrong moves and what a mutation should preserve.

### Phase 2: Transformation Planner + Prompt Builder

Build:

```text
atomicmath/mutation_prompt.py
```

CLI:

```text
atomicmath mutate build-prompt --config ... --seed-id <id>
```

Behavior:

- load original question;
- load original solution;
- load seed hinges;
- insert transformation examples;
- insert recent success/failure stories if available;
- ask the same generation call to reject boring transformations;
- produce the final generation prompt.

Acceptance:

- prompt follows the shape in this document;
- transformation examples are included before the task;
- success/failure memory is short and relevant;
- the returned JSON contains discarded transformations and a clear conceptual delta;
- prompt is readable enough to manually inspect.

### Phase 3: Mutation Generator

Build:

```text
atomicmath/mutation_generator.py
```

CLI:

```text
atomicmath mutate generate --config ... --seed-id <id> --n 3
```

Behavior:

- build transformation plan;
- build final generation prompt;
- call generator model;
- return candidate JSON;
- store raw candidate in `mutation_episodes` with `result='pending'`.

Acceptance:

- each candidate lists the chosen transformation;
- each candidate explains what got mutated;
- each candidate explains the conceptual delta;
- each candidate explains why sharper and why not stitched;
- each candidate explains why it is not a direct sibling of the seed;
- output includes answer and short solution.

### Phase 4: Mutation Judge

Build:

```text
atomicmath/mutation_quality.py
```

CLI:

```text
atomicmath mutate judge --config ... --episode-id <id>
```

Behavior:

- use existing correctness verifier where possible;
- add mutation-specific quality judge;
- compare candidate against original question for novelty;
- update episode result as accepted/rejected;
- write success or failure mutation story.

Acceptance:

- judge catches number swaps;
- judge catches stitched additions;
- judge catches missing hinge;
- accepted candidates have a useful success story.

### Phase 5: Small Batch Probe

CLI:

```text
atomicmath mutate probe --config ... --limit 20 --n 3
```

Behavior:

- choose 20 seeds with solutions;
- extract hinges if missing;
- generate 3 mutations each;
- judge all candidates;
- print accepted/rejected summary.

Manual review target:

```text
60 generated candidates
at least 8-10 worth reading seriously
at least 3 genuinely promising
low stitched-output rate
```

If this fails, improve the mutation examples and judge prompt before adding more
infrastructure.

## Files To Add

```text
atomicmath/mutation_hinges.py
atomicmath/mutation_prompt.py
atomicmath/mutation_generator.py
atomicmath/mutation_quality.py
scripts/probe_mutation_generation.py
docs/03-mutation-prompts.md
```

## Files To Modify

```text
atomicmath/db.py
atomicmath/config.py
atomicmath/cli.py
examples/config.example.yaml
docs/README.md
```

## What We Are Explicitly Not Building Yet

Not in Version 1:

- hinge embeddings;
- similar hinge retrieval;
- HNSW;
- hinge graph;
- canonical hinge clusters;
- cross-question affinity;
- multi-question composition;
- neural-net-like hinge layers.

Those ideas may return later, but only after single-question mutation works.

## Success Definition

This version is successful if, for a single solved seed question, the system can:

1. explain the real mathematical hinge;
2. reject weak transformations before choosing a strong one;
3. generate a new problem that is not just a paraphrase;
4. preserve the important mathematical bottleneck;
5. make the problem sharper without making it longer or stitched;
6. produce a correct short solution;
7. store a success/failure story that improves the next mutation attempt.

The core loop is:

```text
question + solution
  -> hinge notes
  -> combined plan+generate call with examples and memory
  -> mutation judge
  -> success/failure story
  -> next attempt improves
```
