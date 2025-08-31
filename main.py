from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Dict, List, Any, Tuple

import torch
from rich.console import Console
from rich.table import Table
from rich.traceback import install as rich_traceback_install

# Suppress tqdm.rich experimental warning
warnings.filterwarnings("ignore", message="rich is experimental/alpha")

# Nicer errors
rich_traceback_install(show_locals=False)
console = Console()
try:  # Access to a protected member _log_render of a class
    console._log_render.omit_repeated_times = False  # always show time bracket
except Exception:
    pass

# -------------------------
# Configuration
# -------------------------
SEED = 42
OUTPUT_DIR = os.path.join(os.getcwd(), "artifacts")

# Data config
DATASET_PATH = os.path.join(os.getcwd(), "data", "demo.csv")  # CSV with header: text,label
N_SAMPLES = 600  # synthetic samples if no CSV
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DEFAULT_CLASS_NAMES = [
    "influenza", "common_cold", "migraine", "food_poisoning", "allergy"
]

# Embedding config
HF_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MAX_SEQUENCE_LENGTH = 128
FINE_TUNE_ENCODER = True  # if True, encoder params are updated during training
FREEZE_LAYER_NORM = True  # keep layer norms trainable even when freezing encoder

# Model config (MLP classifier)
MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.1

# Training config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 16
BATCH_SIZE = 64
LEARNING_RATE = 3e-5  # joint fine-tuning often needs a smaller LR
WEIGHT_DECAY = 1e-4  # L2 regularization
WARMUP_RATIO = 0.06  # proportion of training for linear LR warmup
GRAD_ACCUM_STEPS = 1  # gradient accumulation steps
MAX_GRAD_NORM = 1.0  # gradient clipping
LABEL_SMOOTHING = 0.05  # 0.0 to disable

# Logging/preview
NUM_PREDICTION_SAMPLES = 5  # number of test samples to show in preview
LOG_EVERY = 50  # log training stats every N steps

# ---------------
# Orchestration
# ---------------
from dataset import (
    generate_synthetic_dataset,
    SymptomsDataset,
    load_examples_from_path,
    Example,
)
from embedding import TextEmbeddingEncoder
from models import MLPClassifier
from train import TorchTrainer, evaluate_predictions


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_label_mappings(class_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {c: i for i, c in enumerate(class_names)}
    id2label = {i: c for c, i in label2id.items()}
    return label2id, id2label


def preview_samples(texts: List[str], preds: List[int], id2label: Dict[int, str], k: int = 5) -> List[Dict[str, Any]]:
    return [{"symptoms": t, "pred": id2label[int(p)]} for t, p in list(zip(texts, preds))[:k]]


def derive_classes_from_examples(examples: List[Example]) -> List[str]:
    seen: List[str] = []
    seen_set = set()
    for ex in examples:
        y = ex.label
        if y and y not in seen_set:
            seen.append(y)
            seen_set.add(y)
    return seen


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    ensure_output_dir(OUTPUT_DIR)

    use_csv = os.path.exists(DATASET_PATH)
    if use_csv:  # THERE IS CSV
        with console.status(f"[bold cyan]\\[data][/bold cyan] Loading dataset from CSV: {os.path.relpath(DATASET_PATH)}"):
            examples = load_examples_from_path(DATASET_PATH)
        console.log(f"[bold cyan]\\[data][/bold cyan] Loaded {len(examples)} rows from CSV")
        class_names = derive_classes_from_examples(examples)
        if not class_names:
            console.print("[bold red]\\[data][/bold red] No labels found in CSV. Ensure the 'label' column has non-empty values.")
            sys.exit(1)

    else:  # NO CSV, USE SYNTHETIC
        console.log("[bold red]\\[data][/bold red] No CSV found; generating synthetic dataset")
        examples = generate_synthetic_dataset(n_samples=N_SAMPLES, class_names=DEFAULT_CLASS_NAMES, seed=SEED)
        console.log(f"[bold cyan]\\[data][/bold cyan] Generated {len(examples)} examples")
        class_names = DEFAULT_CLASS_NAMES

    dataset = SymptomsDataset.from_examples(examples)
    train_ds, val_ds, test_ds = dataset.train_val_test_split(val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED)
    console.log(f"[bold cyan]\\[data][/bold cyan] Split -> train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    label2id, id2label = build_label_mappings(class_names)

    console.rule("Config")

    # Encoder
    with console.status(f"[bold blue]\\[embedding][/bold blue] Loading encoder: {HF_MODEL_NAME}"):
        encoder = TextEmbeddingEncoder(
            hf_model_name=HF_MODEL_NAME,
            max_seq_len=MAX_SEQUENCE_LENGTH,
            fine_tune=FINE_TUNE_ENCODER,
            freeze_layer_norm=FREEZE_LAYER_NORM,
            device=DEVICE,
            seed=SEED,
        )
    console.log(f"[bold blue]\\[embedding][/bold blue] Encoder loaded {HF_MODEL_NAME}. Output dim: {encoder.output_dim}")

    # Model
    with console.status("[bold magenta]\\[model][/bold magenta] Building MLP classifier..."):
        model = MLPClassifier(input_dim=encoder.output_dim, hidden_dim=MLP_HIDDEN_DIM, num_classes=len(class_names), dropout=MLP_DROPOUT)
    console.log("[bold magenta]\\[model][/bold magenta] Model built MLP classifier.")

    console.rule("Training")

    # Trainer
    console.log(f"[bold yellow]\\[train][/bold yellow] Starting training on device={DEVICE}")
    trainer = TorchTrainer(
        encoder=encoder,
        classifier=model,
        label2id=label2id,
        id2label=id2label,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        log_every=LOG_EVERY,
        seed=SEED,
        label_smoothing=LABEL_SMOOTHING,
    )

    trainer.fit(train_ds, val_ds)

    # Save best weights for inference
    try:
        enc_path = os.path.join(OUTPUT_DIR, "encoder_state.pt")
        clf_path = os.path.join(OUTPUT_DIR, "classifier_state.pt")
        torch.save(encoder.model.state_dict(), enc_path)
        torch.save(model.state_dict(), clf_path)
        console.log(f"[bold green]\\[save][/bold green] Saved encoder -> {enc_path}")
        console.log(f"[bold green]\\[save][/bold green] Saved classifier -> {clf_path}")
    except Exception as e:
        console.log(f"[bold red]\\[save][/bold red] Failed to save model weights: {e}")

    # Evaluation
    console.log("[bold yellow]\\[eval][/bold yellow] Evaluating on test set...")
    test_texts, test_labels = test_ds.texts, [label2id[y] for y in test_ds.labels]
    test_logits = trainer.predict_logits(test_texts)
    test_preds = [int(torch.tensor(logit).argmax().item()) for logit in test_logits]
    metrics = evaluate_predictions(test_labels, test_preds, num_classes=len(class_names))

    # region Print metrics table
    metrics_table = Table(show_header=True, header_style="bold")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_row("accuracy", f"{metrics.get('accuracy', 0.0):.4f}")
    metrics_table.add_row("macro_f1", f"{metrics.get('macro_f1', 0.0):.4f}")
    # endregion
    console.print(metrics_table)

    # region Inference preview
    console.log("[bold yellow]\\[preview][/bold yellow] Generating inference preview...")
    preview = preview_samples(test_ds.texts, test_preds, id2label, k=NUM_PREDICTION_SAMPLES)
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", width=3)
    table.add_column("Symptoms", overflow="fold")
    table.add_column("Pred")
    for i, row in enumerate(preview, start=1):
        table.add_row(str(i), row["symptoms"], row["pred"])
    # endregion
    console.print(table)

    summary = {
        "config": {
            "encoder": HF_MODEL_NAME,
            "fine_tune_encoder": FINE_TUNE_ENCODER,
            "max_seq_len": MAX_SEQUENCE_LENGTH,
            "embed_dim": encoder.output_dim,
            "mlp_hidden_dim": MLP_HIDDEN_DIM,
            "mlp_dropout": MLP_DROPOUT,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "label_smoothing": LABEL_SMOOTHING,
            "classes": class_names,
        },
        "metrics": metrics,
        "preview": preview,
    }

    out_path = os.path.join(OUTPUT_DIR, "run_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    console.log(f"[bold green]\\[done][/bold green] Summary saved to {out_path}")


if __name__ == "__main__":
    main()
