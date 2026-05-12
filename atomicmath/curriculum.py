"""Concept-pair brief sampling (coverage-biased, no difficulty bands)."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TypeVar

from .config import Config
from .db import Store
from .taxonomy import Taxonomy

T = TypeVar("T")


@dataclass(frozen=True)
class Brief:
    primary_concept: str
    secondary_concept: str
    answer_form: str
    topic: str  # coarse output topic (taxonomy topic label)


def _pair_weight(cfg: Config, store: Store, p: str, s: str) -> float:
    n = store.get_coverage(p, s)
    sm = max(1e-6, cfg.curriculum.coverage_smoothing)
    w = 1.0 / (sm + float(n)) ** float(cfg.curriculum.coverage_power)
    return w


def _weighted_choice(items: list[T], weights: list[float], rng: random.Random) -> T:
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += max(0.0, weight)
        if r <= acc:
            return item
    return items[-1]


def _static_fallback_brief(cfg: Config, store: Store, tax: Taxonomy, rng: random.Random) -> Brief:
    concepts = list(tax.concepts)
    if len(concepts) < 2:
        c0 = concepts[0] if concepts else "integer"
        return Brief(
            primary_concept=c0,
            secondary_concept=c0,
            answer_form=rng.choice(tax.question_forms),
            topic=rng.choice(tax.topics),
        )

    pairs: list[tuple[str, str]] = []
    weights: list[float] = []
    for i, a in enumerate(concepts):
        for b in concepts[i + 1 :]:
            pairs.append((a, b))
            w = _pair_weight(cfg, store, a, b)
            weights.append(w)

    if rng.random() < cfg.curriculum.epsilon_uniform_pair and pairs:
        p, s = rng.choice(pairs)
    elif pairs:
        p, s = _weighted_choice(pairs, weights, rng)
    else:
        p, s = concepts[0], concepts[1]

    answer_form = rng.choice(tax.question_forms)
    topic = rng.choice(tax.topics)
    return Brief(primary_concept=p, secondary_concept=s, answer_form=answer_form, topic=topic)


def _choice_from_counter(counter: Counter[str], rng: random.Random, fallback: str) -> str:
    if not counter:
        return fallback
    items = list(counter.keys())
    weights = [float(counter[x]) for x in items]
    return _weighted_choice(items, weights, rng)


def sample_brief(cfg: Config, store: Store, tax: Taxonomy, rng: random.Random) -> Brief:
    """Sample from the bootstrapped seed profile.

    The default taxonomy is now only a fallback vocabulary. Once exemplars exist,
    the curriculum samples observed concept pairs, answer forms, and seed topics
    from the ingested corpus, so generation stays close to the seed distribution.
    """
    rows = store.list_exemplar_profile_rows()
    if not rows:
        return _static_fallback_brief(cfg, store, tax, rng)

    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_answer_forms: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    pair_topics: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    global_answer_forms: Counter[str] = Counter()
    global_topics: Counter[str] = Counter()

    valid_concepts = set(tax.concepts)
    valid_forms = set(tax.question_forms)
    for row in rows:
        try:
            scaffold = json.loads(row["scaffold_json"])
        except Exception:
            continue
        primary = str(scaffold.get("primary_concept", "")).strip()
        secondary = str(scaffold.get("secondary_concept", "")).strip()
        answer_form = str(scaffold.get("answer_form", "")).strip()
        topic = str(row["topic_norm"] or row["topic_raw"] or "uncategorized").strip()
        if primary not in valid_concepts or secondary not in valid_concepts:
            continue
        if answer_form not in valid_forms:
            answer_form = "FIND_VALUE" if "FIND_VALUE" in valid_forms else (tax.question_forms[0] if tax.question_forms else "FIND_VALUE")
        pair = (primary, secondary)
        pair_counts[pair] += 1
        pair_answer_forms[pair][answer_form] += 1
        pair_topics[pair][topic] += 1
        global_answer_forms[answer_form] += 1
        global_topics[topic] += 1

    if not pair_counts:
        return _static_fallback_brief(cfg, store, tax, rng)

    pairs = list(pair_counts.keys())
    if rng.random() < cfg.curriculum.epsilon_uniform_pair:
        primary, secondary = rng.choice(pairs)
    else:
        # Observed frequency keeps us on-distribution; coverage weight pushes away from exhausted pairs.
        weights = [
            float(pair_counts[pair]) * _pair_weight(cfg, store, pair[0], pair[1])
            for pair in pairs
        ]
        primary, secondary = _weighted_choice(pairs, weights, rng)

    pair = (primary, secondary)
    answer_form = _choice_from_counter(
        pair_answer_forms[pair],
        rng,
        fallback=_choice_from_counter(global_answer_forms, rng, "FIND_VALUE"),
    )
    topic = _choice_from_counter(
        pair_topics[pair],
        rng,
        fallback=_choice_from_counter(global_topics, rng, "uncategorized"),
    )
    return Brief(
        primary_concept=primary,
        secondary_concept=secondary,
        answer_form=answer_form,
        topic=topic,
    )
