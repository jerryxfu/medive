from __future__ import annotations

import csv
import os
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple

from tqdm.rich import tqdm

# Suppress tqdm.rich experimental warning
warnings.filterwarnings("ignore", message="rich is experimental/alpha")


@dataclass
class Example:
    text: str
    label: str


class SymptomsDataset:
    def __init__(self, texts: List[str], labels: List[str]):
        assert len(texts) == len(labels), "texts and labels must align"
        self.texts = texts
        self.labels = labels

    @classmethod
    def from_examples(cls, examples: List[Example]) -> "SymptomsDataset":
        return cls([e.text for e in examples], [e.label for e in examples])

    def __len__(self) -> int:
        return len(self.texts)

    def train_val_test_split(self, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> Tuple[
        "SymptomsDataset", "SymptomsDataset", "SymptomsDataset"]:
        assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and (val_ratio + test_ratio) < 1
        n = len(self)
        idxs = list(range(n))
        random.Random(seed).shuffle(idxs)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        val_idx = idxs[:n_val]
        test_idx = idxs[n_val:n_val + n_test]
        train_idx = idxs[n_val + n_test:]

        def subset(idxs: List[int]) -> "SymptomsDataset":
            return SymptomsDataset([self.texts[i] for i in idxs], [self.labels[i] for i in idxs])

        return subset(train_idx), subset(val_idx), subset(test_idx)


# Canonical symptom lexicon
LEXICON: Dict[str, List[str]] = {
    "influenza": [
        "fever",
        "chills",
        "body aches",
        "fatigue",
        "dry cough",
        "sore throat",
        "headache",
        "runny nose",
        "loss of appetite",
    ],
    "common_cold": [
        "runny nose",
        "sneezing",
        "sore throat",
        "mild cough",
        "congestion",
        "mild headache",
        "watery eyes",
    ],
    "migraine": [
        "headache",
        "throbbing pain",
        "nausea",
        "vomiting",
        "sensitivity to light",
        "sensitivity to sound",
        "visual aura",
        "dizziness",
    ],
    "food_poisoning": [
        "nausea",
        "vomiting",
        "diarrhea",
        "stomach cramps",
        "fever",
        "fatigue",
        "dehydration",
        "loss of appetite",
    ],
    "allergy": [
        "sneezing",
        "itchy eyes",
        "runny nose",
        "nasal congestion",
        "rash",
        "hives",
        "watery eyes",
        "coughing",
    ],
    "myocardial_infarction_heart_attack": [
        "chest pain or pressure",
        "shortness of breath",
        "nausea",
        "vomiting",
        "fatigue",
        "lightheadedness",
        "palpitations",
        "sweating",
        "anxiety",
    ],
    "dehydration": [
        "dizziness",
        "lightheadedness",
        "fatigue",
        "headache",
        "nausea",
        "dry mouth",
        "dark urine",
        "confusion",
    ],
    "low_blood_pressure_hypotension": [
        "dizziness",
        "lightheadedness",
        "fatigue",
        "blurred vision",
        "confusion",
        "fainting",
        "loss of balance",
    ],
    "stroke": [
        "sudden difficulty speaking or understanding",
        "sudden numbness or weakness",
        "loss of balance or coordination",
        "sudden vision changes",
        "confusion",
        "sudden severe headache",
        "seizures",
        "facial droop",
    ],
    "pneumonia": [
        "cough (productive or dry)",
        "fever",
        "chills",
        "shortness of breath",
        "fatigue",
        "chest pain",
        "sweating",
        "cyanosis",
        "loss of appetite",
    ],
    "vasovagal_syncope": [
        "dizziness",
        "lightheadedness",
        "nausea",
        "fainting",
        "blurred vision",
        "palpitations",
        "pallor",
        "sweating",
    ],
    "urinary_tract_infection": [
        "burning sensation when urinating",
        "frequent urination",
        "urgency to urinate",
        "lower abdominal pain",
        "cloudy urine",
        "mild fever",
    ],
    "gastroenteritis": [
        "nausea",
        "vomiting",
        "diarrhea",
        "stomach cramps",
        "fever",
        "fatigue",
        "loss of appetite",
        "dehydration",
    ],
    "covid_19": [
        "fever",
        "dry cough",
        "fatigue",
        "loss of taste or smell",
        "shortness of breath",
        "sore throat",
        "headache",
        "muscle aches",
        "chills",
    ],
    "asthma_attack": [
        "shortness of breath",
        "wheezing",
        "chest tightness",
        "coughing",
        "difficulty speaking in full sentences",
        "anxiety",
    ],
    "appendicitis": [
        "abdominal pain (lower right quadrant)",
        "nausea",
        "vomiting",
        "loss of appetite",
        "fever",
        "abdominal tenderness",
        "constipation or diarrhea",
    ],
    "hypertension_high_blood_pressure_crisis": [
        "severe headache",
        "blurred vision",
        "chest pain",
        "shortness of breath",
        "nosebleeds",
        "confusion",
        "anxiety",
    ],
}


def get_default_classes() -> List[str]:
    # Single source of truth for default demo classes
    return list(LEXICON.keys())


def generate_synthetic_dataset(n_samples: int, class_names: List[str], seed: int = 42) -> List[Example]:
    """
    Create a synthetic dataset mapping symptom phrases to conditions.
    """
    rng = random.Random(seed)

    def synthesize_for_class(label: str) -> str:
        vocab = LEXICON.get(label)
        if not vocab:
            vocab = [f"symptom_{label}_0", f"symptom_{label}_1", f"symptom_{label}_2"]
        core = rng.sample(vocab, k=min(3, len(vocab)))
        noise_pool = list({w for words in LEXICON.values() for w in words if w not in core})
        noise = rng.sample(noise_pool, k=min(2, len(noise_pool))) if noise_pool else []
        items = core + noise
        rng.shuffle(items)
        templates = [
            "Patient reports {}.",
            "Symptoms include {}.",
            "Noted {} over the last 48 hours.",
            "Complaints: {}.",
        ]
        t = rng.choice(templates)
        return t.format(", ".join(items))

    examples: List[Example] = []
    per_class = max(1, n_samples // max(1, len(class_names)))

    # Progress bar for synthesis
    pbar = tqdm(total=n_samples, desc="[data] synth", leave=False, dynamic_ncols=True)

    for c in class_names:
        for _ in range(per_class):
            examples.append(Example(text=synthesize_for_class(c), label=c))
            if len(examples) <= n_samples:
                pbar.update(1)

    # Top up to n_samples, if needed
    while len(examples) < n_samples:
        c = rng.choice(class_names)
        examples.append(Example(text=synthesize_for_class(c), label=c))
        pbar.update(1)

    pbar.close()

    rng.shuffle(examples)
    return examples


def load_examples_from_csv(path: str, text_col: str = "text", label_col: str = "label", delimiter: str = ",") -> List[Example]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("CSV file must have a header row with columns including 'text' and 'label'.")
        if text_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise ValueError(f"CSV missing required columns: '{text_col}', '{label_col}'. Found: {reader.fieldnames}")
        for row in reader:
            t = (row.get(text_col) or "").strip()
            y = (row.get(label_col) or "").strip()
            if t and y:
                examples.append(Example(text=t, label=y))
    return examples


def load_examples_from_path(path: str) -> List[Example]:
    ext = os.path.splitext(path)[1].lower()
    if ext != ".csv":
        raise ValueError("Only .csv datasets are supported. Ensure your file ends with .csv")
    return load_examples_from_csv(path)


# region CLI dataset generation
def write_examples_to_csv(examples: List["Example"], path: str, text_col: str = "text", label_col: str = "label") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([text_col, label_col])
        for ex in examples:
            writer.writerow([ex.text, ex.label])


def generate_demo_csv(out_path: str = os.path.join("data", "demo.csv"), n_samples: int = 200, class_names: List[str] | None = None, seed: int = 42) -> str:
    if class_names is None:
        class_names = get_default_classes()
    examples = generate_synthetic_dataset(n_samples=n_samples, class_names=class_names, seed=seed)
    write_examples_to_csv(examples, out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a synthetic symptoms CSV dataset.")
    parser.add_argument("--out", type=str, default=os.path.join("data", "demo.csv"), help="Output CSV path (default: data/demo.csv)")
    parser.add_argument("--n", type=int, default=200, help="Number of samples to generate (default: 200)")
    parser.add_argument(
        "--classes",
        type=str,
        default=",".join(get_default_classes()),
        help="Comma-separated class names",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    out_path = generate_demo_csv(out_path=args.out, n_samples=args.n, class_names=classes, seed=args.seed)
    print(f"Wrote synthetic CSV to {out_path} ({args.n} rows, classes={classes})")
# endregion
