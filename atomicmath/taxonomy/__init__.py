from __future__ import annotations

import json
from importlib import resources
from pathlib import Path


class Taxonomy:
    def __init__(self, topics: list[str], concepts: list[str], bond_types: list[str], question_forms: list[str]):
        self.topics = topics
        self.concepts = concepts  # CONCEPT-kind atoms; TOPIC names are kept separately
        self.bond_types = bond_types
        self.question_forms = question_forms
        self._topic_set = set(topics)
        self._concept_set = set(concepts)
        self._bond_set = set(bond_types)
        self._qform_set = set(question_forms)
        # Atom name space = topics ∪ concepts (for node-name validity check)
        self._all_atoms = self._topic_set | self._concept_set
        # Stable indices for tensor lookups
        self.concept_index: dict[str, int] = {c: i for i, c in enumerate(concepts)}
        self.topic_index: dict[str, int] = {t: i for i, t in enumerate(topics)}

    def is_valid_atom(self, name: str) -> bool:
        return name in self._all_atoms

    def is_valid_concept(self, name: str) -> bool:
        return name in self._concept_set

    def is_valid_topic(self, name: str) -> bool:
        return name in self._topic_set

    def is_valid_bond(self, name: str) -> bool:
        return name in self._bond_set

    def is_valid_question_form(self, name: str) -> bool:
        return name in self._qform_set

    def kind_of(self, name: str) -> str | None:
        if name in self._topic_set:
            return "TOPIC"
        if name in self._concept_set:
            return "CONCEPT"
        return None

    def n_concepts(self) -> int:
        return len(self.concepts)


def load_taxonomy(path: str = "default") -> Taxonomy:
    if path == "default":
        with resources.files("atomicmath.taxonomy").joinpath("default.json").open() as f:
            data = json.load(f)
    else:
        with open(Path(path)) as f:
            data = json.load(f)
    return Taxonomy(
        topics=data["topics"],
        concepts=data["concepts"],
        bond_types=data["bond_types"],
        question_forms=data["question_forms"],
    )
