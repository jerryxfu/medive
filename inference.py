from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from modules.embedding import TextEmbeddingEncoder
from modules.models import HybridTextCUIClassifier
from modules.concepts import UmlsConceptExtractor, ConceptExtractionConfig, map_concepts_to_ids, PAD_ID, NO_CUI_ID, UNK_CUI_ID

console = Console()
run_id = "001"

ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
RUN_SUMMARY = os.path.join(ARTIFACTS_DIR, f"run_summary_{run_id}.json")
ENCODER_WEIGHTS = os.path.join(ARTIFACTS_DIR, f"encoder_state_{run_id}.pt")
CLASSIFIER_WEIGHTS = os.path.join(ARTIFACTS_DIR, f"classifier_state_{run_id}.pt")
CUI_VOCAB_PATH = os.path.join(ARTIFACTS_DIR, f"cui_vocab_{run_id}.json")


def load_run_config(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"run_summary.json not found at {path}. Please run training first to generate artifacts."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("config", {})
    classes = cfg.get("classes")
    if not classes:
        raise ValueError("Class names not found in run_summary.json under config.classes")
    return cfg


def load_cui_vocab(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CUI vocab not found at {path}. Train first to generate it.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_models(cfg: Dict, device: str, cui_vocab: Dict[str, int]) -> Tuple[TextEmbeddingEncoder, HybridTextCUIClassifier, List[str]]:
    classes: List[str] = list(cfg["classes"])  # ensure list copy
    hf_model_name: str = str(cfg.get("encoder"))
    max_seq_len: int = int(cfg.get("max_seq_len", 128))
    mlp_hidden_dim: int = int(cfg.get("mlp_hidden_dim", 256))
    mlp_dropout: float = float(cfg.get("mlp_dropout", 0.1))
    cui_emb_dim: int = int(cfg.get("cui_emb_dim", 128))

    with console.status(f"Initializing text encoder ({hf_model_name})"):
        encoder = TextEmbeddingEncoder(
            hf_model_name=hf_model_name,
            max_seq_len=max_seq_len,
            fine_tune=False,  # inference only
            freeze_layer_norm=True,
            device=device,
            seed=42,
        )

    with console.status("Initializing hybrid classifier"):
        classifier = HybridTextCUIClassifier(
            text_dim=encoder.output_dim,
            cui_vocab_size=len(cui_vocab),
            cui_emb_dim=cui_emb_dim,
            hidden_dim=mlp_hidden_dim,
            num_classes=len(classes),
            dropout=mlp_dropout,
        )
        encoder.model.eval()
        classifier.eval()

    if not (os.path.exists(ENCODER_WEIGHTS) and os.path.exists(CLASSIFIER_WEIGHTS)):
        raise FileNotFoundError(
            "Model weights not found in artifacts/. Train first to create encoder_state.pt and classifier_state.pt"
        )

    with console.status("Loading encoder weights"):
        enc_state = torch.load(ENCODER_WEIGHTS, map_location=device)
        encoder.model.load_state_dict(enc_state)

    with console.status("Loading classifier weights"):
        clf_state = torch.load(CLASSIFIER_WEIGHTS, map_location=device)
        classifier.load_state_dict(clf_state)

    with console.status(f"Moving models to {device}"):
        encoder.model.to(device)
        classifier.to(device)

    return encoder, classifier, classes


def _pad_concepts(seqs: List[List[int]], pad_id: int = PAD_ID) -> Tuple[torch.Tensor, torch.Tensor]:
    if not seqs:
        return torch.zeros(0, 0, dtype=torch.long), torch.zeros(0, 0, dtype=torch.long)
    max_len = max(len(s) for s in seqs) if any(len(s) > 0 for s in seqs) else 0
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        # Treat empty or [NO_CUI] as no-concept: keep all pads and zero mask
        if not s or (len(s) == 1 and s[0] == NO_CUI_ID):
            console.log(f"Sequence {i} has no CUIs extracted; using all-pad input.")
            continue
        t = torch.tensor(s[:max_len], dtype=torch.long)
        out[i, : t.numel()] = t
        # mask 1 only for real concept IDs (exclude UNK_CUI)
        mask[i, : t.numel()] = (t != UNK_CUI_ID).long()
    return out, mask


def extract_and_map_cuis(
    texts: List[str],
    min_score: float,
    max_per_doc: int,
    vocab: Dict[str, int],
    extractor: UmlsConceptExtractor | None = None,
) -> List[List[int]]:
    if extractor is None:
        extractor = UmlsConceptExtractor(ConceptExtractionConfig(min_score=min_score, max_concepts_per_doc=max_per_doc))
    concept_lists = extractor.batch_extract(texts)
    return map_concepts_to_ids(concept_lists, vocab)


def predict_with_confidence(
    encoder: TextEmbeddingEncoder,
    classifier: HybridTextCUIClassifier,
    texts: List[str],
    cui_ids: List[List[int]],
    device: str,
    batch_size: int = 32,
) -> Tuple[List[int], List[List[float]]]:
    preds: List[int] = []
    probs_all: List[List[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            tokens = encoder.tokenize(batch)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            seqs = cui_ids[i: i + batch_size]
            c_ids, c_mask = _pad_concepts(seqs)
            c_ids = c_ids.to(device)
            c_mask = c_mask.to(device)
            emb = encoder.forward(tokens)
            logits = classifier(emb, c_ids, c_mask)
            probs = F.softmax(logits, dim=-1)
            top = torch.argmax(probs, dim=-1)
            preds.extend(top.detach().cpu().tolist())
            probs_all.extend(probs.detach().cpu().tolist())
    return preds, probs_all


def show_results(text: str, classes: List[str], pred_idx: int, probs: List[float], topk: int = 10) -> None:
    pairs = sorted([(i, p) for i, p in enumerate(probs)], key=lambda x: x[1], reverse=True)[:topk]
    header = f"Prediction: {classes[pred_idx]} (confidence {probs[pred_idx] * 100:.1f}%)"
    console.rule(header)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Rank", justify="right")
    table.add_column("Class")
    table.add_column("Confidence", justify="right")
    for r, (i, p) in enumerate(pairs, start=1):
        table.add_row(str(r), classes[i], f"{p * 100:.2f}%")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive symptom -> diagnosis prediction (hybrid)")
    parser.add_argument("--text", type=str, default=None, help="Single text to classify (skip interactive mode)")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--topk", type=int, default=10, help="Show top-k classes")
    args = parser.parse_args()

    cfg = load_run_config(RUN_SUMMARY)
    cui_vocab = load_cui_vocab(CUI_VOCAB_PATH)
    encoder, classifier, classes = build_models(cfg, args.device, cui_vocab)

    min_score = float(cfg.get("min_cui_score"))
    max_cuis = int(cfg.get("max_cuis_per_doc"))
    subset_gazetteer_path = cfg.get("gazetteer_subset_path")

    # Build or load extractor once
    with console.status("Loading UMLS gazetteer (subset if available)"):
        extractor_cfg = ConceptExtractionConfig(
            max_concepts_per_doc=max_cuis,
            min_score=min_score,
            prebuilt_gazetteer_path=subset_gazetteer_path if subset_gazetteer_path and os.path.exists(subset_gazetteer_path) else None,
        )
        extractor = UmlsConceptExtractor(extractor_cfg)
        if subset_gazetteer_path and not os.path.exists(subset_gazetteer_path):
            console.log("[yellow]Subset gazetteer path in config not found; fell back to full MRCONSO parse.[/yellow]")

    if args.text:
        cui_ids = extract_and_map_cuis([args.text], min_score, max_cuis, cui_vocab, extractor=extractor)
        preds, probs = predict_with_confidence(encoder, classifier, [args.text], cui_ids, device=args.device)
        show_results(args.text, classes, preds[0], probs[0], topk=args.topk)
        return

    console.print("Type symptoms and press Enter. Type 'q/quit/exit' to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting.")
            break
        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            break

        with console.status(f"Extracting CUIs for '{text}'"):
            cui_ids = extract_and_map_cuis([text], min_score, max_cuis, cui_vocab, extractor=extractor)
        with console.status(f"Predicting diagnosis for '{text}'"):
            preds, probs = predict_with_confidence(encoder, classifier, [text], cui_ids, device=args.device)
        show_results(text, classes, preds[0], probs[0], topk=args.topk)


if __name__ == "__main__":
    main()
