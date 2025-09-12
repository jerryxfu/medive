from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Iterable

# -----------------
# Normalization
# -----------------
_norm_re = re.compile(r"[^a-z0-9]+")


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = _norm_re.sub(" ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class ConceptExtractionConfig:
    mrconso_path: str = os.path.join(os.getcwd(), "MRCONSO.RRF")
    max_concepts_per_doc: int = 32
    min_score: float = 0.0  # unused for exact matcher; kept for API compatibility
    cache_path: str | None = None  # jsonl cache (one doc per line)
    lowercase: bool = True  # kept for API compatibility
    include_all_eng_terms: bool = False  # if False, use only preferred English terms (ISPREF=Y)
    max_ngram: int = 6  # longest n-gram to consider during matching
    include_ambiguous_mappings: bool = True  # keep all CUIs for a term instead of first-seen only
    prebuilt_gazetteer_path: str | None = None  # path to prebuilt gazetteer JSON file


class UmlsConceptExtractor:
    def __init__(self, cfg: ConceptExtractionConfig | None = None) -> None:
        self.cfg = cfg or ConceptExtractionConfig()
        self._gazetteer: Dict[str, List[str]] = {}
        self._cache: Dict[str, List[str]] = {}
        self._used_spans: set[str] = set()  # spans actually matched in processed texts
        self._load_gazetteer()
        if self.cfg.cache_path and os.path.exists(self.cfg.cache_path):
            with open(self.cfg.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self._cache[row["text"]] = row.get("cuis", [])

    def _load_gazetteer(self) -> None:
        # Try to load prebuilt gazetteer first
        if self.cfg.prebuilt_gazetteer_path and os.path.exists(self.cfg.prebuilt_gazetteer_path):
            with open(self.cfg.prebuilt_gazetteer_path, "r", encoding="utf-8") as f:
                self._gazetteer = json.load(f)
            if not self._gazetteer:
                raise RuntimeError("Prebuilt gazetteer is empty")
            return

        # Fall back to building from MRCONSO.RRF
        path = self.cfg.mrconso_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"MRCONSO.RRF not found at {path}")

        # MRCONSO fields (pipe-delimited). STR is at index 14 in UMLS 2024AB; be robust to variable columns.
        # We keep only English entries (LAT='ENG'). If include_all_eng_terms=False, keep ISPREF='Y'.
        term2cuis: Dict[str, List[str]] = {}
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 15:
                    continue
                CUI = parts[0]
                LAT = parts[1]
                ISPREF = parts[6] if len(parts) > 7 else "N"
                STR = parts[14] if len(parts) > 15 else ""
                if LAT != "ENG":
                    continue
                if not self.cfg.include_all_eng_terms and ISPREF != "Y":
                    continue
                if not STR:
                    continue
                term = normalize(STR)
                if not term:
                    continue
                if self.cfg.include_ambiguous_mappings:
                    lst = term2cuis.get(term)
                    if lst is None:
                        term2cuis[term] = [CUI]
                    elif CUI not in lst:
                        lst.append(CUI)
                else:
                    # Keep first seen mapping to avoid flipping between CUIs for ambiguous strings
                    if term not in term2cuis:
                        term2cuis[term] = [CUI]
        if not term2cuis:
            raise RuntimeError("No English terms loaded from MRCONSO; check file or filters.")
        self._gazetteer = term2cuis

    def save_gazetteer(self, path: str) -> None:
        """Save the processed gazetteer to a JSON file for reuse."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._gazetteer, f, indent=2, ensure_ascii=False)

    def get_used_gazetteer_subset(self) -> Dict[str, List[str]]:
        """Return a dict of only spans that were matched at least once."""
        if not self._used_spans:
            return {}
        return {k: self._gazetteer[k] for k in self._used_spans if k in self._gazetteer}

    def _extract_doc(self, text: str) -> List[str]:
        if text in self._cache:
            return list(self._cache[text])
        norm_text = normalize(text)
        tokens = norm_text.split()

        found: Dict[str, float] = {}  # CUI -> weighted count (salience proxy)

        token_len = len(tokens)
        max_n = max(1, int(self.cfg.max_ngram))
        # Sliding window n-gram exact matching (token-based)
        for i in range(token_len):
            for n in range(min(max_n, token_len - i), 0, -1):  # prefer longer spans
                span = " ".join(tokens[i:i + n])
                cuis = self._gazetteer.get(span)
                if cuis:
                    # Weight by n-gram length to favor more specific phrases
                    weight = float(n)
                    for cui in cuis:
                        found[cui] = found.get(cui, 0.0) + weight
                    self._used_spans.add(span)
                    break  # skip overlaps starting at same i

        # Rank CUIs by weighted count desc, then CUI id for tie-breaker
        ranked = sorted(found.items(), key=lambda x: (-x[1], x[0]))
        cuis = [c for c, _ in ranked[: self.cfg.max_concepts_per_doc]]
        if self.cfg.cache_path:
            with open(self.cfg.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"text": text, "cuis": cuis}, ensure_ascii=False) + "\n")
        self._cache[text] = cuis
        return cuis

    def batch_extract(self, texts: Iterable[str]) -> List[List[str]]:
        return [self._extract_doc(t) for t in texts]


# Vocabulary utilities (unchanged)
SPECIAL_TOKENS = ["[PAD]", "[NO_CUI]", "[UNK_CUI]"]
PAD_ID = 0
NO_CUI_ID = 1
UNK_CUI_ID = 2


def build_cui_vocab(concept_lists: List[List[str]], min_freq: int = 1, max_size: int = 8000) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for seq in concept_lists:
        for c in seq:
            freq[c] = freq.get(c, 0) + 1
    items = [(c, f) for c, f in freq.items() if f >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for c, _ in items[: max(0, max_size - len(SPECIAL_TOKENS))]:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def map_concepts_to_ids(concept_lists: List[List[str]], vocab: Dict[str, int]) -> List[List[int]]:
    out: List[List[int]] = []
    for seq in concept_lists:
        ids = [vocab.get(c, UNK_CUI_ID) for c in (seq if seq else ["[NO_CUI]"])]
        out.append(ids or [NO_CUI_ID])
    return out
