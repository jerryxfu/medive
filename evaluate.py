"""
Evaluation script using evaluator.py

Usage:
    python run_eval.py --run_id 001
    python run_eval.py --run_id 002 --data_path data/custom_test.csv
"""

import argparse
import json
import os
from typing import Dict, List

import torch
from rich.console import Console

from modules.concepts import UmlsConceptExtractor, ConceptExtractionConfig, map_concepts_to_ids
from modules.dataset import SymptomsDataset, load_examples_from_path
from modules.embedding import TextEmbeddingEncoder
from modules.evaluator import compute_metrics
from modules.models import HybridTextCUIClassifier
from modules.train import TorchTrainer

console = Console()
try:  # Access to a protected member _log_render of a class
    console._log_render.omit_repeated_times = False  # always show time bracket
except Exception:
    pass


def load_run_config(artifacts_dir: str, run_id: str) -> Dict:
    """Load configuration from run summary."""
    summary_path = os.path.join(artifacts_dir, f"run_summary_{run_id}.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Run summary not found: {summary_path}")

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    return summary['config']


def load_cui_vocab(artifacts_dir: str, run_id: str) -> Dict[str, int]:
    """Load CUI vocabulary."""
    vocab_path = os.path.join(artifacts_dir, f"cui_vocab_{run_id}.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"CUI vocab not found: {vocab_path}")

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    return vocab


def load_models(config: Dict, vocab: Dict[str, int], artifacts_dir: str, run_id: str, device: str):
    """Load trained encoder and classifier models."""
    # Load encoder
    encoder = TextEmbeddingEncoder(
        hf_model_name=config['encoder'],
        max_seq_len=config['max_seq_len'],
        fine_tune=config['fine_tune_encoder'],
        freeze_layer_norm=True,  # Default for inference
        device=device,
        seed=42
    )

    # Load encoder weights
    encoder_path = os.path.join(artifacts_dir, f"encoder_state_{run_id}.pt")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder weights not found: {encoder_path}")

    encoder.model.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.model.eval()

    # Load classifier
    classifier = HybridTextCUIClassifier(
        text_dim=config['embed_dim'],
        cui_vocab_size=len(vocab),
        cui_emb_dim=config['cui_emb_dim'],
        hidden_dim=config['mlp_hidden_dim'],
        num_classes=len(config['classes']),
        dropout=config['mlp_dropout']
    )

    # Load classifier weights
    classifier_path = os.path.join(artifacts_dir, f"classifier_state_{run_id}.pt")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier weights not found: {classifier_path}")

    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    classifier.to(device)

    return encoder, classifier


def extract_concepts_for_evaluation(symptoms: List[str], vocab: Dict[str, int], config: Dict) -> List[List[int]]:
    """Extract and map concepts for the test data using the generated gazetteer subset."""
    console.status("[bold blue]\\[concepts][/bold blue] Extracting concepts for evaluation...")

    # Use the gazetteer subset if available (same as inference.py)
    gazetteer_subset_path = config.get('gazetteer_subset_path')

    extractor_config = ConceptExtractionConfig(
        max_concepts_per_doc=config['max_cuis_per_doc'],
        min_score=config['min_cui_score'],
        prebuilt_gazetteer_path=gazetteer_subset_path if gazetteer_subset_path and os.path.exists(gazetteer_subset_path) else None,
    )

    extractor = UmlsConceptExtractor(extractor_config)

    if gazetteer_subset_path and not os.path.exists(gazetteer_subset_path):
        console.log("[yellow]Gazetteer subset not found; falling back to full MRCONSO parse.[/yellow]")
    elif gazetteer_subset_path and os.path.exists(gazetteer_subset_path):
        console.log(f"[green]Using gazetteer subset from: {gazetteer_subset_path}[/green]")
    else:
        console.log("[yellow]No gazetteer subset path configured; using full MRCONSO parse.[/yellow]")

    # Extract concepts
    concept_lists = extractor.batch_extract(symptoms)

    # Map to IDs using existing vocabulary
    cui_ids = map_concepts_to_ids(concept_lists, vocab)

    console.log(f"[blue]Extracted concepts for {len(symptoms)} symptoms[/blue]")

    return cui_ids


def run_evaluation(
    run_id: str,
    artifacts_dir: str = "artifacts",
    data_path: str = None,
    device: str = None
):
    """Run comprehensive evaluation on pre-trained models."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    console.log(f"[bold green]\\[eval][/bold green] Running evaluation for run_id: {run_id}")
    console.log(f"[blue]Using device: {device}[/blue]")

    # Load configuration
    config = load_run_config(artifacts_dir, run_id)
    vocab = load_cui_vocab(artifacts_dir, run_id)

    # Load test data
    if data_path is None:
        data_path = config.get('dataset_path', os.path.join(os.getcwd(), "data", "positives-f800-15k.csv"))

    console.status(f"[blue]Loading data from: {data_path}[/blue]")
    examples = load_examples_from_path(data_path)

    # Create dataset and get test split (same seed as training)
    dataset = SymptomsDataset.from_examples(examples)
    _, _, test_ds = dataset.train_val_test_split(
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    console.log(f"[blue]Test set size: {len(test_ds)}[/blue]")

    # Load models
    encoder, classifier = load_models(config, vocab, artifacts_dir, run_id, device)

    # Create label mappings
    class_names = config['classes']
    label2id = {cls: idx for idx, cls in enumerate(class_names)}
    id2label = {idx: cls for cls, idx in label2id.items()}

    # Extract concepts for test data
    test_cui_ids = extract_concepts_for_evaluation(test_ds.symptoms, vocab, config)
    test_ds.attach_concepts(test_cui_ids)

    # Create trainer for inference
    trainer = TorchTrainer(
        encoder=encoder,
        classifier=classifier,
        label2id=label2id,
        id2label=id2label,
        device=device,
        batch_size=64,  # Use reasonable batch size for inference
        lr=1e-5,  # Not used for inference
        epochs=1  # Not used for inference
    )

    # Get predictions and logits
    test_symptoms = test_ds.symptoms
    test_conditions = [label2id[condition] for condition in test_ds.conditions]
    test_logits = trainer.predict_logits(test_symptoms, test_concepts=test_ds._concept_ids)

    # Extract embeddings
    console.status("[bold blue]\\[embeddings][/bold blue] Extracting embeddings...")
    test_embeddings = trainer.extract_embeddings(test_symptoms)
    console.log(f"[bold blue]\\[embeddings][/bold blue] Extracted embeddings shape: {test_embeddings.shape}")

    # Run comprehensive evaluation
    console.status("[bold green]\\[evaluation][/bold green] Running evaluation...")
    analysis_results = compute_metrics(
        test_symptoms=test_symptoms,
        test_conditions=test_conditions,
        test_logits=test_logits,
        test_embeddings=test_embeddings,
        class_names=class_names,
        output_dir=artifacts_dir,
        run_id=run_id
    )

    return analysis_results


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on pre-trained models")
    parser.add_argument("--run_id", required=True, help="Run ID (e.g., '001')")
    parser.add_argument("--artifacts_dir", default="artifacts", help="Path to artifacts directory")
    parser.add_argument("--data_path", help="Path to test data CSV (optional, uses training data by default)")
    parser.add_argument("--device", help="Device to use (cuda/cpu, auto-detected by default)")

    args = parser.parse_args()

    try:
        run_evaluation(
            run_id=args.run_id,
            artifacts_dir=args.artifacts_dir,
            data_path=args.data_path,
            device=args.device
        )
    except Exception as e:
        console.log(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
