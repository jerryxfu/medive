from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Tuple

import torch
from rich.console import Console
from rich.table import Table
from rich.traceback import install as rich_traceback_install

from utils import next_run_id, print_dataset_stats
from modules.evaluator import evaluate_model, compute_metrics

warnings.filterwarnings("ignore", message="rich is experimental/alpha")  # suppress experimental warning
rich_traceback_install(show_locals=False)
console = Console()
try:  # Access to a protected member _log_render of a class
    console._log_render.omit_repeated_times = False  # always show time bracket
except Exception:
    pass

# --------------------
# Configuration
# --------------------
SEED = 42
OUTPUT_DIR = os.path.join(os.getcwd(), "artifacts")
LOG_EVERY = 50

# Data config
DATASET_PATH = os.path.join(os.getcwd(), "data", "positives-f800-50k.csv")
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Embedding config
HF_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MAX_SEQUENCE_LENGTH = 128
FINE_TUNE_ENCODER = True
FREEZE_LAYER_NORM = True

# Concept (CUI) config
CUI_EMB_DIM = 256  # dimension of CUI embeddings
MAX_CUIS_PER_DOC = 32  # max CUIs to extract per document
MIN_CUI_SCORE = 0.80  # minimum confidence score to keep CUI (not used for exact matcher)
CUI_VOCAB_MAX_SIZE = 12000  # max size of CUI vocabulary (most frequent)

# Model config
MLP_HIDDEN_DIM = 256  # hidden layer dimension in MLP head
MLP_DROPOUT = 0.15  # dropout in MLP head

# Training config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 6
BATCH_SIZE = 64  # multiples of 8 for modern GPUs and mixed precision (320 safe on 24GB VRAM)
LEARNING_RATE = 3e-5
ENCODER_LEARNING_RATE = 3e-5
HEAD_LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4  # L2 regularization
WARMUP_RATIO = 0.06  # proportion of training for linear LR warmup
GRAD_ACCUM_STEPS = 1  # gradient accumulation steps
MAX_GRAD_NORM = 1.0  # gradient clipping
LABEL_SMOOTHING = 0.04  # 0.0 to disable

# --------------------
# Orchestration
# --------------------
from modules.dataset import (
    SymptomsDataset,
    load_examples_from_path,
    Example,
)
from modules.embedding import TextEmbeddingEncoder
from modules.models import HybridTextCUIClassifier
from modules.train import TorchTrainer
from modules.concepts import (
    UmlsConceptExtractor,
    ConceptExtractionConfig,
    build_cui_vocab,
    map_concepts_to_ids,
)


# label -> id and id -> label dictionaries for training
def _label_mappings(class_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    with console.status(f"[bold blue]\\[label_mappings][/bold blue] Building label mappings..."):
        label2id = {cls: index for (index, cls) in enumerate(class_names)}
        id2label = {index: cls for (cls, index) in label2id.items()}
        return label2id, id2label


# Derive class names from examples in order of first appearance
def _derive_classes_from_examples(examples: List[Example]) -> List[str]:
    with console.status(f"[bold blue]\\[classes][/bold blue] Deriving classes from examples..."):
        seen: List[str] = []
        seen_set = set()
        for example in examples:
            y = example.condition
            if y and y not in seen_set:
                seen.append(y)
                seen_set.add(y)
        # list(dict.fromkeys(example.condition for example in examples if example.condition)) # Python 3.7+ preserves order
        return seen


# Extract concepts for all texts in dataset, build vocab, map to ids, attach to dataset
def _attach_concepts_to_dataset(dataset: SymptomsDataset, vocab_path: str, run_id: str) -> Tuple[Dict[str, int], List[List[int]]]:
    gazetteer_path = os.path.join(OUTPUT_DIR, f"gazetteer_{run_id}.json")

    extractor = UmlsConceptExtractor(
        ConceptExtractionConfig(max_concepts_per_doc=MAX_CUIS_PER_DOC, min_score=MIN_CUI_SCORE)
    )

    # Run extraction first (this populates extractor._used_spans)
    concept_lists = extractor.batch_extract(dataset.symptoms)

    # Build vocab and map
    vocab = build_cui_vocab(concept_lists, max_size=CUI_VOCAB_MAX_SIZE)
    cui_ids = map_concepts_to_ids(concept_lists, vocab)
    dataset.attach_concepts(cui_ids)

    # Subset gazetteer containing only matched spans
    subset = extractor.get_used_gazetteer_subset()
    if subset:
        payload = subset
        console.log(f"Gazetteer subset spans: {len(subset)} (saving subset)")
    else:
        # Fallback (should not normally happen unless no matches)
        payload = extractor._gazetteer
        console.log(f"No spans tracked; saving full gazetteer with {len(payload)} entries")

    # Save compact to reduce size
    with console.status(f"[bold blue]\\[gazetteer][/bold blue] Writing gazetteer subset to {gazetteer_path}..."):
        with open(gazetteer_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    return vocab, cui_ids


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_id = next_run_id(OUTPUT_DIR)

    # Load CSV
    examples = load_examples_from_path(DATASET_PATH)
    class_names = _derive_classes_from_examples(examples)
    if not class_names:
        raise RuntimeError("No conditions found in CSV. Ensure the 'condition' column has non-empty values.")
    console.log({"dataset_path": DATASET_PATH, "num_examples": len(examples), "classes": class_names})

    dataset = SymptomsDataset.from_examples(examples)
    train_ds, val_ds, test_ds = dataset.train_val_test_split(val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED)

    label2id, id2label = _label_mappings(class_names)

    # Concept extraction on union for consistent vocab
    union_symptoms = train_ds.symptoms + val_ds.symptoms + test_ds.symptoms
    union_conditions = train_ds.conditions + val_ds.conditions + test_ds.conditions
    union = SymptomsDataset(union_symptoms, union_conditions)
    vocab_path = os.path.join(OUTPUT_DIR, f"cui_vocab_{run_id}.json")
    vocab, union_cui_ids = _attach_concepts_to_dataset(union, vocab_path, run_id)
    train_ds.attach_concepts(union_cui_ids[: len(train_ds)])
    val_ds.attach_concepts(union_cui_ids[len(train_ds): len(train_ds) + len(val_ds)])
    test_ds.attach_concepts(union_cui_ids[len(train_ds) + len(val_ds):])

    print_dataset_stats("Train", train_ds)
    print_dataset_stats("Val", val_ds)
    print_dataset_stats("Test", test_ds)

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
        console.log(f"Using encoder: {HF_MODEL_NAME} of dimension {encoder.output_dim} on device {DEVICE}")

    model = HybridTextCUIClassifier(
        text_dim=encoder.output_dim,
        cui_vocab_size=len(vocab),
        cui_emb_dim=CUI_EMB_DIM,
        hidden_dim=MLP_HIDDEN_DIM,
        num_classes=len(class_names),
        dropout=MLP_DROPOUT,
    )

    # Trainer
    trainer = TorchTrainer(
        encoder=encoder,
        classifier=model,
        label2id=label2id,
        id2label=id2label,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        encoder_lr=ENCODER_LEARNING_RATE,
        head_lr=HEAD_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        seed=SEED,
        label_smoothing=LABEL_SMOOTHING,
    )

    trainer.fit(train_ds, val_ds)

    # Save weights
    encoder_path = os.path.join(OUTPUT_DIR, f"encoder_state_{run_id}.pt")
    classifier_path = os.path.join(OUTPUT_DIR, f"classifier_state_{run_id}.pt")
    torch.save(encoder.model.state_dict(), encoder_path)
    torch.save(model.state_dict(), classifier_path)

    # Evaluation
    test_symptoms, test_conditions = test_ds.symptoms, [label2id[y] for y in test_ds.conditions]
    test_logits = trainer.predict_logits(test_symptoms, test_concepts=test_ds._concept_ids)
    test_preds = [int(torch.tensor(logit).argmax().item()) for logit in test_logits]

    # Basic metrics for backwards compatibility
    metrics = evaluate_model(test_conditions, test_preds, num_classes=len(class_names))

    metrics_table = Table(show_header=True, header_style="bold")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_row("accuracy", f"{metrics.get('accuracy', 0.0):.4f}")
    metrics_table.add_row("macro_f1", f"{metrics.get('macro_f1', 0.0):.4f}")
    console.print(metrics_table)

    preview = [{"symptoms": t, "pred": id2label[int(p)]} for t, p in list(zip(test_ds.symptoms, test_preds))[:5]]
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", width=3)
    table.add_column("Symptoms", overflow="fold")
    table.add_column("Pred")
    for i, row in enumerate(preview, start=1):
        table.add_row(str(i), row["symptoms"], row["pred"])
    console.print(table)

    # Extract embeddings for visualization
    console.log(f"[bold blue]\\[embeddings][/bold blue] Extracting test embeddings...")
    test_embeddings = trainer.extract_embeddings(test_symptoms)

    # Comprehensive analysis with confusion matrix and embedding plots
    console.log(f"[bold blue]\\[analysis][/bold blue] Running comprehensive analysis...")
    compute_metrics(
        test_symptoms=test_symptoms,
        test_conditions=test_conditions,
        test_logits=test_logits,
        test_embeddings=test_embeddings,
        class_names=class_names,
        output_dir=OUTPUT_DIR,
        run_id=run_id
    )

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
            "encoder_learning_rate": ENCODER_LEARNING_RATE,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "label_smoothing": LABEL_SMOOTHING,
            "classes": class_names,
            "cui_emb_dim": CUI_EMB_DIM,
            "max_cuis_per_doc": MAX_CUIS_PER_DOC,
            "min_cui_score": MIN_CUI_SCORE,
            "cui_vocab_size": len(vocab),
            "gazetteer_subset_path": os.path.join(OUTPUT_DIR, f"gazetteer_{run_id}.json"),
            "run_id": run_id,
        },
        "metrics": metrics,
    }

    out_path = os.path.join(OUTPUT_DIR, f"run_summary_{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    console.log(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
