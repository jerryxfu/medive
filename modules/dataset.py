from __future__ import annotations

import csv
import os
import random
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split

# Suppress tqdm.rich experimental warning (in case downstream imports use it)
warnings.filterwarnings("ignore", message="rich is experimental/alpha")


@dataclass
class Example:
    # renamed fields: text -> symptoms, label -> condition
    symptoms: str
    condition: str


class SymptomsDataset:
    def __init__(self, symptoms: List[str], conditions: List[str]):
        assert len(symptoms) == len(conditions), "symptoms and conditions must align"
        self.symptoms = symptoms
        self.conditions = conditions
        self._concept_ids: Optional[List[List[int]]] = None

    @classmethod
    def from_examples(cls, examples: List[Example]) -> "SymptomsDataset":
        return cls([e.symptoms for e in examples], [e.condition for e in examples])

    def __len__(self) -> int:
        return len(self.symptoms)

    def attach_concepts(self, concept_ids: List[List[int]]) -> None:
        if len(concept_ids) != len(self.symptoms):
            raise ValueError("concept_ids length mismatch dataset")
        self._concept_ids = concept_ids

    def concept_ids_at(self, idx: int) -> List[int] | None:
        if self._concept_ids is None:
            return None
        return self._concept_ids[idx]

    def train_val_test_split(self, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42, stratify: bool = True) -> Tuple[
        "SymptomsDataset", "SymptomsDataset", "SymptomsDataset"]:
        assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and (val_ratio + test_ratio) < 1
        n = len(self)

        def subset(idxs: List[int]) -> "SymptomsDataset":
            ds = SymptomsDataset([self.symptoms[i] for i in idxs], [self.conditions[i] for i in idxs])
            if self._concept_ids is not None:
                ds.attach_concepts([self._concept_ids[i] for i in idxs])
            return ds

        if not stratify:
            idxs = list(range(n))
            random.Random(seed).shuffle(idxs)
            n_val = int(n * val_ratio)
            n_test = int(n * test_ratio)
            val_idx = idxs[:n_val]
            test_idx = idxs[n_val:n_val + n_test]
            train_idx = idxs[n_val + n_test:]
            return subset(train_idx), subset(val_idx), subset(test_idx)

        X = list(range(n))
        y = list(self.conditions)
        try:
            temp_ratio = val_ratio + test_ratio
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=temp_ratio, random_state=seed, stratify=y
            )
            val_share_of_temp = val_ratio / temp_ratio
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1.0 - val_share_of_temp), random_state=seed, stratify=y_temp
            )
            return subset(X_train), subset(X_val), subset(X_test)
        except ValueError:
            idxs = list(range(n))
            random.Random(seed).shuffle(idxs)
            n_val = int(n * val_ratio)
            n_test = int(n * test_ratio)
            val_idx = idxs[:n_val]
            test_idx = idxs[n_val:n_val + n_test]
            train_idx = idxs[n_val + n_test:]
            return subset(train_idx), subset(val_idx), subset(test_idx)


def load_examples_from_csv(path: str, symptoms_col: str = "symptoms", condition_col: str = "condition", delimiter: str = ",") -> List[Example]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("CSV file must have a header row with columns including 'symptoms' and 'condition'.")
        if symptoms_col not in reader.fieldnames or condition_col not in reader.fieldnames:
            raise ValueError(f"CSV missing required columns: '{symptoms_col}', '{condition_col}'. Found: {reader.fieldnames}")
        for row in reader:
            s = (row.get(symptoms_col) or "").strip()
            c = (row.get(condition_col) or "").strip()
            if s and c:
                examples.append(Example(symptoms=s, condition=c))
    return examples


def load_examples_from_path(path: str) -> List[Example]:
    ext = os.path.splitext(path)[1].lower()
    if ext != ".csv":
        raise ValueError("Only .csv datasets are supported.")
    return load_examples_from_csv(path)
