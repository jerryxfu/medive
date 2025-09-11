from __future__ import annotations

"""Web data harvesting for symptom -> condition dataset creation.

Sources: Public domain condition pages (e.g., MedlinePlus). Verify terms of use.
This module keeps requests polite and volume minimal. Intended for small-scale
prototyping, not crawling.
"""
from dataclasses import dataclass
from typing import List, Dict, Iterable, Optional, Tuple
import time
import re
import os
import csv
import random
import requests
from bs4 import BeautifulSoup

USER_AGENT = "medive-research/0.1 (+github)"
DEFAULT_TIMEOUT = 10
SLEEP_BETWEEN = 1.0  # seconds between requests

# Simple heuristic to decide if a bullet item is a symptom phrase
SYMPTOM_PATTERN = re.compile(r"[a-z0-9 ,\-()]+", re.IGNORECASE)


@dataclass
class ConditionPage:
    condition_id: str  # label used in dataset (snake_case)
    url: str
    css_selector: str | None = None  # optional narrower scope


DEFAULT_PAGES: List[ConditionPage] = [
    ConditionPage("influenza", "https://medlineplus.gov/flu.html"),
    ConditionPage("common_cold", "https://medlineplus.gov/commoncold.html"),
    ConditionPage("migraine", "https://medlineplus.gov/migraine.html"),
    ConditionPage("food_poisoning", "https://medlineplus.gov/foodpoisoning.html"),
    ConditionPage("allergy", "https://medlineplus.gov/allergy.html"),
]


def fetch_page(url: str) -> str | None:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception:
        return None


def extract_symptom_candidates(html: str, scope_selector: str | None = None) -> List[str]:
    """Extract short <li> items that look like symptom phrases.

    Heuristic: keep short list items that match a simple character pattern and
    contain at least one symptom keyword. Deduplicates while preserving order.
    """
    soup = BeautifulSoup(html, "lxml")
    scope = soup.select_one(scope_selector) if scope_selector else None
    scope = scope or soup

    items: List[str] = []
    for li in scope.find_all("li"):
        # Get text with spaces between nodes; strip edges, then collapse internal whitespace
        raw = li.get_text(" ", True)
        txt = re.sub(r"\s+", " ", raw).strip()
        if not txt:
            continue
        # Filter out very long paragraphs or navigational content
        if len(txt) > 120:
            continue
        low = txt.lower()
        if SYMPTOM_PATTERN.fullmatch(low) and any(
            k in low for k in [
                "pain", "ache", "fever", "cough", "nausea", "vomit", "headache", "fatigue",
                "rash", "sneeze", "throat", "nose", "runny", "shortness", "breath", "chest",
                "congestion", "diarrhea", "dizziness", "lightheaded",
            ]
        ):
            items.append(low)

    # Deduplicate preserving order
    seen = set()
    uniq: List[str] = []
    for s in items:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def harvest(pages: Optional[Iterable[ConditionPage]] = None, delay: float = SLEEP_BETWEEN) -> Dict[str, List[str]]:
    """Fetch each page and return {condition_id: [raw symptom phrases]}.

    Network-friendly: sleeps `delay` seconds between requests.
    """
    pages = list(pages) if pages is not None else list(DEFAULT_PAGES)
    out: Dict[str, List[str]] = {}
    for p in pages:
        html = fetch_page(p.url)
        if html is None:
            time.sleep(delay)
            continue
        phrases = extract_symptom_candidates(html, p.css_selector)
        if phrases:
            out[p.condition_id] = phrases
        time.sleep(delay)
    return out


def _aggregate_symptoms(symptoms: List[str], joiner: str = ",") -> str:
    # join with commas, no extra spaces: "a,b,c"
    cleaned = [re.sub(r"\s+", " ", s.strip()) for s in symptoms if s.strip()]
    return joiner.join(cleaned)


def _make_negatives(data: Dict[str, List[str]], k: int = 1, joiner: str = ",", aggregate: bool = True, seed: int = 42) -> List[Tuple[str, str]]:
    """Create simple negative pairs by mismatching symptoms with other conditions.

    Returns list of (symptoms_text, condition_id) negatives.
    """
    rnd = random.Random(seed)
    conds = list(data.keys())
    negatives: List[Tuple[str, str]] = []
    if not conds:
        return negatives
    for cond in conds:
        # pick k other conditions uniformly
        others = [c for c in conds if c != cond]
        if not others:
            continue
        choices = [others[rnd.randrange(len(others))] for _ in range(max(0, k))]
        for other in choices:
            symp_list = data[other]
            text = _aggregate_symptoms(symp_list, joiner) if aggregate else (symp_list[rnd.randrange(len(symp_list))] if symp_list else "")
            if text:
                negatives.append((text, cond))
    return negatives


def build_examples_from_web(pages: Optional[Iterable[ConditionPage]] = None, aggregate: bool = True, joiner: str = ",",
                             include_negatives: bool = False, negatives_per_condition: int = 1) -> List["Example"]:
    """Return Examples(symptoms, condition) built from web pages.

    aggregate=True will join bullet phrases into a single comma-separated string per condition.
    include_negatives adds mismatched (symptoms, condition) examples.
    """
    data = harvest(pages)
    from modules.dataset import Example  # late import to avoid heavy deps at module import
    examples: List[Example] = []
    for cond, symps in data.items():
        if aggregate:
            text = _aggregate_symptoms(symps, joiner)
            if text:
                examples.append(Example(symptoms=text, condition=cond))
        else:
            for s in symps:
                if s:
                    examples.append(Example(symptoms=s, condition=cond))
    if include_negatives and data:
        neg_rows = _make_negatives(data, k=max(0, negatives_per_condition), joiner=joiner, aggregate=aggregate)
        for s, cond in neg_rows:
            examples.append(Example(symptoms=s, condition=cond))
    return examples


def export_csv_from_web(out_path: str, pages: Optional[Iterable[ConditionPage]] = None, *, aggregate: bool = True, joiner: str = ",",
                         include_negatives: bool = False, negatives_per_condition: int = 1) -> int:
    """Harvest and write CSV with header symptoms,condition.

    - aggregate=True: write one row per condition with symptoms joined as "a,b,c".
      aggregate=False: write one row per bullet phrase.
    - include_negatives adds mismatched rows (fast to skip when False).

    Returns number of rows written.
    """
    data = harvest(pages)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows: List[Tuple[str, str]] = []  # (symptoms, condition)

    for cond, symps in data.items():
        if aggregate:
            text = _aggregate_symptoms(symps, joiner)
            if text:
                rows.append((text, cond))
        else:
            for s in symps:
                if s:
                    rows.append((s, cond))

    if include_negatives and data:
        rows.extend(_make_negatives(data, k=max(0, negatives_per_condition), joiner=joiner, aggregate=aggregate))

    count = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symptoms", "condition"])  # align with modules.dataset
        for s, cond in rows:
            w.writerow([s, cond])
            count += 1
    return count


if __name__ == "__main__":
    out_file = os.path.join("data", "web.csv")
    rows = export_csv_from_web(out_file, aggregate=True, include_negatives=False)
    print(f"Wrote {rows} rows to {out_file}")
