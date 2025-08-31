from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from embedding import TextEmbeddingEncoder
from models import MLPClassifier

console = Console()

ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
RUN_SUMMARY = os.path.join(ARTIFACTS_DIR, "run_summary.json")
ENCODER_WEIGHTS = os.path.join(ARTIFACTS_DIR, "encoder_state.pt")
CLASSIFIER_WEIGHTS = os.path.join(ARTIFACTS_DIR, "classifier_state.pt")


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


def build_models(cfg: Dict, device: str) -> Tuple[TextEmbeddingEncoder, MLPClassifier, List[str]]:
    classes: List[str] = list(cfg["classes"])  # ensure list copy
    hf_model_name: str = str(cfg.get("encoder"))
    max_seq_len: int = int(cfg.get("max_seq_len", 64))
    mlp_hidden_dim: int = int(cfg.get("mlp_hidden_dim", 256))
    mlp_dropout: float = float(cfg.get("mlp_dropout", 0.1))

    encoder = TextEmbeddingEncoder(
        hf_model_name=hf_model_name,
        max_seq_len=max_seq_len,
        fine_tune=False,  # inference only
        freeze_layer_norm=True,
        device=device,
        seed=42,
    )
    classifier = MLPClassifier(
        input_dim=encoder.output_dim,
        hidden_dim=mlp_hidden_dim,
        num_classes=len(classes),
        dropout=mlp_dropout,
    )
    encoder.model.eval()
    classifier.eval()

    # Load weights
    if not (os.path.exists(ENCODER_WEIGHTS) and os.path.exists(CLASSIFIER_WEIGHTS)):
        raise FileNotFoundError(
            "Model weights not found in artifacts/. Train first to create encoder_state.pt and classifier_state.pt"
        )

    enc_state = torch.load(ENCODER_WEIGHTS, map_location=device)
    clf_state = torch.load(CLASSIFIER_WEIGHTS, map_location=device)
    encoder.model.load_state_dict(enc_state)
    classifier.load_state_dict(clf_state)

    encoder.model.to(device)
    classifier.to(device)
    return encoder, classifier, classes


def predict_with_confidence(
    encoder: TextEmbeddingEncoder, classifier: MLPClassifier, texts: List[str], device: str, batch_size: int = 32
) -> Tuple[List[int], List[List[float]]]:
    preds: List[int] = []
    probs_all: List[List[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            tokens = encoder.tokenize(batch)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            emb = encoder.forward(tokens)
            logits = classifier(emb)
            probs = F.softmax(logits, dim=-1)
            top = torch.argmax(probs, dim=-1)
            preds.extend(top.detach().cpu().tolist())
            probs_all.extend(probs.detach().cpu().tolist())
    return preds, probs_all


def show_results(text: str, classes: List[str], pred_idx: int, probs: List[float], topk: int = 3) -> None:
    # Build top-k pairs
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
    parser = argparse.ArgumentParser(description="Interactive symptom -> diagnosis prediction")
    parser.add_argument("--text", type=str, default=None, help="Single text to classify (skip interactive mode)")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--topk", type=int, default=3, help="Show top-k classes")
    args = parser.parse_args()

    cfg = load_run_config(RUN_SUMMARY)
    encoder, classifier, classes = build_models(cfg, args.device)

    if args.text:
        preds, probs = predict_with_confidence(encoder, classifier, [args.text], device=args.device)
        show_results(args.text, classes, preds[0], probs[0], topk=args.topk)
        return

    console.print("Type symptoms and press Enter. Type 'quit' to exit.")
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
        preds, probs = predict_with_confidence(encoder, classifier, [text], device=args.device)
        show_results(text, classes, preds[0], probs[0], topk=args.topk)


if __name__ == "__main__":
    main()
