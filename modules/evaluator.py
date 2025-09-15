from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from rich.console import Console
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

console = Console()
try:  # Access to a protected member _log_render of a class
    console._log_render.omit_repeated_times = False  # always show time bracket
except Exception:
    pass


def evaluate_model(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    """
    Basic evaluation function that returns key metrics.
    Called by main.py for backwards compatibility.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }


class ModelEvaluator:
    """
    Comprehensive model evaluation.
    """

    def __init__(self, output_dir: str, run_id: str):
        self.output_dir = output_dir
        self.run_id = run_id
        self.plots_dir = os.path.join(output_dir, f"plots_{run_id}")
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate_predictions(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_logits: List[List[float]],
        class_names: List[str],
        symptoms: List[str],
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluation of model predictions.

        Args:
            y_true: True labels (class indices)
            y_pred: Predicted labels (class indices)
            y_logits: Raw model outputs/logits
            class_names: List of class names
            symptoms: Input texts for evaluation
            save_plots: Whether to save visualization plots

        Returns:
            Dictionary containing all evaluation results
        """
        results = {}

        # Basic metrics
        results['metrics'] = self._compute_detailed_metrics(y_true, y_pred, class_names)

        # Confusion matrix
        results['confusion_matrix'] = self._evaluate_confusion_matrix(
            y_true, y_pred, class_names, save_plots
        )

        # Per-class evaluation
        results['per_class_evaluation'] = self._per_class_evaluation(
            y_true, y_pred, class_names
        )

        # Prediction confidence evaluation
        results['confidence_evaluation'] = self._confidence_evaluation(
            y_true, y_pred, y_logits, save_plots
        )

        # Error evaluation
        results['error_evaluation'] = self._error_evaluation(
            y_true, y_pred, symptoms, class_names
        )

        # Save detailed results
        self._save_metrics(results)

        return results

    def evaluate_embeddings(
        self,
        embeddings: np.ndarray,
        labels: List[int],
        class_names: List[str],
        symptoms: List[str],
        method: str = 'pca_umap',
        save_plots: bool = True,
        pca_components: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate and visualize embeddings using PCA preprocessing then UMAP clustering.

        Args:
            embeddings: High-dimensional embeddings [N, D]
            labels: True class labels
            class_names: List of class names
            symptoms: Input texts for annotation
            method: Dimensionality reduction method (default 'pca_umap')
            save_plots: Whether to save plots
            pca_components: Number of PCA components for preprocessing

        Returns:
            Dictionary with embedding evaluation results
        """
        with console.status(f"[bold blue]\\[embeddings][/bold blue] Evaluating embeddings using PCA → UMAP..."):
            # Apply PCA to reduce noise and dimensionality
            with console.status("[bold blue]\\[embeddings][/bold blue] Applying PCA dimensionality reduction..."):
                n_components = min(pca_components, embeddings.shape[1], embeddings.shape[0] - 1)
                pca = PCA(n_components=n_components, random_state=42)
                embeddings_pca = pca.fit_transform(embeddings)

                console.log(
                    f"[blue]PCA reduced from {embeddings.shape[1]} to {embeddings_pca.shape[1]} dimensions (explained variance: {pca.explained_variance_ratio_.sum():.3f})[/blue]")

            # Apply UMAP to the PCA-reduced embeddings
            console.status("[bold blue]\\[embeddings][/bold blue] Applying UMAP dimensionality reduction...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            embeddings_2d = reducer.fit_transform(embeddings_pca)

            console.log(f"[blue]UMAP reduced to 2 dimensions[/blue]")

        # Create visualization
        if save_plots:
            self._plot_embeddings_2d(
                embeddings_2d, labels, class_names
            )

        # Compute embedding statistics
        results = {
            'method': 'pca_umap',
            'pca_components': n_components,
            'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
            'original_dim': embeddings.shape[1],
            'pca_dim': embeddings_pca.shape[1],
            'reduced_embeddings': embeddings_2d.tolist(),
            'embedding_stats': self._compute_embedding_stats(embeddings, labels, class_names)
        }

        return results

    def _compute_detailed_metrics(self, y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # Per-class metrics
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(class_names))))
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(class_names))))
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(class_names))))

        metrics['per_class'] = {
            class_names[i]: {
                'f1': float(f1_per_class[i]),
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i])
            }
            for i in range(len(class_names))
        }

        return metrics

    def _evaluate_confusion_matrix(self, y_true: List[int], y_pred: List[int], class_names: List[str], save_plots: bool) -> Dict[str, Any]:
        """Generate and evaluate confusion matrix."""
        with console.status("[bold blue]\\[evaluation][/bold blue] Evaluating confusion matrix..."):
            cm = confusion_matrix(y_true, y_pred)

            if save_plots:
                # Plot confusion matrix
                plt.figure(figsize=(max(8, len(class_names)), max(6, int(len(class_names) * 0.8))))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names
                )
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'), dpi=56, bbox_inches='tight')
                plt.close()

                # Normalized confusion matrix
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                plt.figure(figsize=(max(8, len(class_names)), max(6, int(len(class_names) * 0.8))))
                sns.heatmap(
                    cm_norm,
                    annot=True,
                    fmt='.2f',
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names
                )
                plt.title('Normalized Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix_normalized.png'), dpi=56, bbox_inches='tight')
                plt.close()

            return {
                'matrix': cm.tolist(),
                'normalized_matrix': (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
            }

    def _per_class_evaluation(self, y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict[str, Any]:
        """Evaluate performance per class."""
        with console.status("[bold blue]\\[evaluation][/bold blue] Evaluating per-class performance..."):
            evaluation = {}

            for i, class_name in enumerate(class_names):
                true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == i and p == i)
                false_positives = sum(1 for t, p in zip(y_true, y_pred) if t != i and p == i)
                false_negatives = sum(1 for t, p in zip(y_true, y_pred) if t == i and p != i)
                true_negatives = sum(1 for t, p in zip(y_true, y_pred) if t != i and p != i)

                evaluation[class_name] = {
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives,
                    'true_negatives': true_negatives,
                    'support': sum(1 for t in y_true if t == i)
                }

            return evaluation

    def _confidence_evaluation(self, y_true: List[int], y_pred: List[int], y_logits: List[List[float]], save_plots: bool) -> Dict[str, Any]:
        """Evaluate prediction confidence."""
        with console.status("[bold blue]\\[evaluation][/bold blue] Evaluating prediction confidence..."):
            # Convert logits to probabilities
            logits_array = np.array(y_logits)
            probs = torch.softmax(torch.tensor(logits_array), dim=1).numpy()

            # Max probability (confidence) for each prediction
            max_probs = np.max(probs, axis=1)

            # Confidence by correctness
            correct_mask = np.array(y_true) == np.array(y_pred)
            correct_confidences = max_probs[correct_mask]
            incorrect_confidences = max_probs[~correct_mask]

            if save_plots:
                # Plot confidence distribution
                plt.figure(figsize=(11, 6))
                plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct predictions', color='green')
                plt.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect predictions', color='red')
                plt.xlabel('Prediction Confidence')
                plt.ylabel('Count')
                plt.title('Prediction Confidence Distribution')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()

            return {
                'mean_confidence_correct': float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
                'mean_confidence_incorrect': float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
                'std_confidence_correct': float(np.std(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
                'std_confidence_incorrect': float(np.std(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
            }

    def _error_evaluation(self, y_true: List[int], y_pred: List[int], symptoms: List[str], class_names: List[str]) -> Dict[str, Any]:
        """Evaluate prediction errors."""
        with console.status("[bold blue]\\[evaluation][/bold blue] Evaluating error..."):
            errors = []

            for i, (true_label, pred_label, symptom) in enumerate(zip(y_true, y_pred, symptoms)):
                if true_label != pred_label:
                    errors.append({
                        'index': i,
                        'symptom': symptom,
                        'true_class': class_names[true_label],
                        'predicted_class': class_names[pred_label]
                    })

            # Most common error types
            error_types = {}
            for error in errors:
                error_type = f"{error['true_class']} -> {error['predicted_class']}"
                error_types[error_type] = error_types.get(error_type, 0) + 1

            # Sort by frequency
            most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)

            return {
                'total_errors': len(errors),
                'error_rate': len(errors) / len(y_true),
                'most_common_error_types': most_common_errors[:10],  # Top 10
                'sample_errors': errors[:20]  # First 20 errors for manual inspection
            }

    def _plot_embeddings_2d(self, embeddings_2d, labels, class_names):
        with console.status("[bold blue]\\[embeddings][/bold blue] Generating plots..."):
            num_classes = len(class_names)
            colors = plt.cm.tab20(np.linspace(0, 1, min(num_classes, 20)))

            fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=False)

            unique_labels = sorted(set(labels))
            for idx, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    color=colors[idx % len(colors)],
                    label=class_names[label],
                    alpha=0.6,
                    s=10,
                )

            ax.set_xlabel('Reduced embedding space component 1')
            ax.set_ylabel('Reduced embedding space component 2')
            ax.set_title('Text Embeddings Visualization (PCA → UMAP)')

            # Legend outside the plot
            ax.legend(
                ncol=3, fontsize='small', loc='upper left',
                bbox_to_anchor=(1.01, 1), borderaxespad=0.
            )

            plt.savefig(os.path.join(self.plots_dir, 'embeddings.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def _compute_embedding_stats(self, embeddings: np.ndarray, labels: List[int], class_names: List[str]) -> Dict[str, Any]:
        """Compute statistics about embeddings."""
        with console.status("[bold blue]\\[embeddings][/bold blue] Computing embedding statistics..."):
            stats = {
                'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
            }

            # Per-class embedding statistics
            per_class_stats = {}
            for i, class_name in enumerate(class_names):
                class_mask = np.array(labels) == i
                if np.any(class_mask):
                    class_embeddings = embeddings[class_mask]
                    per_class_stats[class_name] = {
                        'count': int(np.sum(class_mask)),
                        'mean_norm': float(np.mean(np.linalg.norm(class_embeddings, axis=1))),
                        'std_norm': float(np.std(np.linalg.norm(class_embeddings, axis=1)))
                    }

            stats['per_class'] = per_class_stats
            return stats

    def _save_metrics(self, results: Dict[str, Any]):
        """Save metrics results to JSON."""
        output_path = os.path.join(self.output_dir, f"eval_metrics_{self.run_id}.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        console.log(f"Eval metrics saved to {output_path}")

    def print_summary_table(self, metrics: Dict[str, Any]):
        """Print a summary table of key metrics."""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        # Overall metrics
        table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
        table.add_row("Macro F1", f"{metrics['macro_f1']:.4f}")
        table.add_row("Weighted F1", f"{metrics['weighted_f1']:.4f}")
        table.add_row("Macro Precision", f"{metrics['macro_precision']:.4f}")
        table.add_row("Macro Recall", f"{metrics['macro_recall']:.4f}")

        console.print("\n")
        console.print(table)
        console.print("\n")


def compute_metrics(
    test_symptoms: List[str],
    test_conditions: List[int],
    test_logits: List[List[float]],
    test_embeddings: Optional[np.ndarray],
    class_names: List[str],
    output_dir: str,
    run_id: str
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation.

    This is the main entry point for evaluation.
    """
    evaluator = ModelEvaluator(output_dir, run_id)

    # Get predictions from logits
    test_preds = [int(torch.tensor(logit).argmax().item()) for logit in test_logits]

    with console.status(f"[bold green]\\[evaluation][/bold green] Running model evaluation..."):
        # Evaluate predictions
        prediction_results = evaluator.evaluate_predictions(
            y_true=test_conditions,
            y_pred=test_preds,
            y_logits=test_logits,
            class_names=class_names,
            symptoms=test_symptoms,
            save_plots=True
        )

    # Print summary
    evaluator.print_summary_table(prediction_results['metrics'])

    # Evaluate embeddings if provided
    embedding_results = {}
    if test_embeddings is not None:
        with console.status(f"[bold blue]\\[embeddings][/bold blue] Evaluating embeddings..."):
            # Use PCA followed by UMAP
            try:
                embedding_results = evaluator.evaluate_embeddings(
                    embeddings=test_embeddings,
                    labels=test_conditions,
                    class_names=class_names,
                    symptoms=test_symptoms,
                    save_plots=True
                )
            except Exception as e:
                console.log(f"[red]Warning: PCA → UMAP evaluation failed: {e}[/red]")

    console.log(f"[bold green]\\[evaluation][/bold green] Evaluation complete, plots saved to: {evaluator.plots_dir}")

    return {
        'predictions': prediction_results,
        'embeddings': embedding_results
    }
