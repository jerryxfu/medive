from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.rich import tqdm
from transformers import get_linear_schedule_with_warmup

from .concepts import PAD_ID, NO_CUI_ID, UNK_CUI_ID
from .embedding import TextEmbeddingEncoder
from utils import format_duration

console = Console()
try:  # Access to a protected member _log_render of a class
    console._log_render.omit_repeated_times = False  # always show time bracket
except Exception:
    pass


class TextLabelDataset(Dataset):
    # hold symptom texts and integer condition IDs
    def __init__(self, symptoms: List[str], condition_ids: List[int], concept_ids: List[List[int]]):
        assert len(symptoms) == len(condition_ids) == len(concept_ids)
        self.symptoms = symptoms
        self.condition_ids = condition_ids
        self.concepts = concept_ids

    def __len__(self) -> int:
        return len(self.symptoms)

    def __getitem__(self, idx: int) -> Tuple[str, int, List[int]]:
        return self.symptoms[idx], int(self.condition_ids[idx]), list(self.concepts[idx])


@dataclass
class _Batch:
    tokens: Dict[str, torch.Tensor]
    labels: torch.Tensor
    concept_ids: torch.Tensor  # [B,Lc]
    concept_mask: torch.Tensor  # [B,Lc]


def pad_concepts(seqs: List[List[int]], pad_id: int = PAD_ID, max_len: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if not seqs:
        return torch.zeros(0, 0, dtype=torch.long), torch.zeros(0, 0, dtype=torch.long)
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        # Treat empty or [NO_CUI] as no concept signal: keep pad, keep mask zeros
        if not s or (len(s) == 1 and s[0] == NO_CUI_ID):
            console.log(f"Sequence {i} has no CUIs extracted; using all-pad input.")
            continue
        trunc = s[:max_len]
        t = torch.tensor(trunc, dtype=torch.long)
        out[i, :len(trunc)] = t
        # mask 1 only for real concept IDs (exclude UNK_CUI)
        valid = (t != UNK_CUI_ID).long()
        mask[i, :len(trunc)] = valid
    return out, mask


def collate_fn(examples, encoder: TextEmbeddingEncoder) -> _Batch:
    symptom_texts = [t for t, _, _ in examples]
    condition_ids = torch.tensor([y for _, y, _ in examples], dtype=torch.long)
    tokens = encoder.tokenize(symptom_texts)
    seqs = [c for _, _, c in examples]
    concept_ids, concept_mask = pad_concepts(seqs)
    return _Batch(tokens=tokens, labels=condition_ids, concept_ids=concept_ids, concept_mask=concept_mask)


def evaluate_predictions(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes))))
    return {"accuracy": acc, "macro_f1": macro_f1}


class TorchTrainer:
    def __init__(
        self,
        encoder: TextEmbeddingEncoder,
        classifier: nn.Module,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        device: str = "cpu",
        batch_size: int = 32,
        lr: float = 3e-5,
        weight_decay: float = 1e-4,
        epochs: int = 5,
        warmup_ratio: float = 0.06,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        label_smoothing: float = 0.0,
        encoder_lr: float | None = None,
        head_lr: float | None = None,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier
        self.label2id = label2id
        self.id2label = id2label
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_grad_norm = max_grad_norm

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.encoder.model.train(True)
        self.classifier.train(True)
        self.encoder.model.to(self.device)
        self.classifier.to(self.device)

        # AMP setup
        self.use_amp = (self.device.type == "cuda")
        from torch.amp import GradScaler  # local import for clarity
        self.scaler = GradScaler(enabled=self.use_amp)

        # split encoder and head learning rates if provided
        encoder_lr = float(encoder_lr) if encoder_lr is not None else float(self.lr)
        head_lr = float(head_lr) if head_lr is not None else float(self.lr)

        # select parameters to exclude from weight decay
        no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight")

        def split_params(named_params):
            decay, no_decay = [], []
            for n, p in named_params:
                if not p.requires_grad:
                    continue
                if any(k in n for k in no_decay_keys):
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        encoder_decay, encoder_no_decay = split_params(self.encoder.model.named_parameters())
        head_decay, head_no_decay = split_params(self.classifier.named_parameters())

        param_groups = []
        if encoder_decay:  # encoder with weight decay
            param_groups.append({"params": encoder_decay, "lr": encoder_lr, "weight_decay": self.weight_decay})
        if encoder_no_decay:  # encoder without weight decay
            param_groups.append({"params": encoder_no_decay, "lr": encoder_lr, "weight_decay": 0.0})
        if head_decay:  # head with weight decay
            param_groups.append({"params": head_decay, "lr": head_lr, "weight_decay": self.weight_decay})
        if head_no_decay:  # head without weight decay
            param_groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})
        if not param_groups:  # all params frozen
            param_groups.append({"params": [], "lr": self.lr, "weight_decay": self.weight_decay})

        self.optimizer = torch.optim.AdamW(param_groups)

        # Label smoothing to improve generalization and reduce overconfidence
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

        self._best_state = None

    def make_loader(self, symptoms: List[str], condition_ids: List[int], concept_ids: List[List[int]], shuffle: bool) -> DataLoader:
        dataset = TextLabelDataset(symptoms, condition_ids, concept_ids)

        def collate(examples):
            return collate_fn(examples, self.encoder)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=collate)

    def fit(self, train_ds, val_ds) -> None:
        train_condition_ids = [self.label2id[y] for y in train_ds.conditions]
        val_condition_ids = [self.label2id[y] for y in val_ds.conditions]
        assert train_ds._concept_ids is not None and val_ds._concept_ids is not None, "Concept IDs must be attached before training."
        train_loader = self.make_loader(train_ds.symptoms, train_condition_ids, train_ds._concept_ids, shuffle=True)
        val_loader = self.make_loader(val_ds.symptoms, val_condition_ids, val_ds._concept_ids, shuffle=False)

        steps_per_epoch = max(1, math.ceil(len(train_loader) / self.grad_accum_steps))
        total_steps = steps_per_epoch * self.epochs
        warmup_steps = int(self.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        global_step = 0
        best_val = -1.0
        total_start = perf_counter()
        for epoch in range(1, self.epochs + 1):
            epoch_start = perf_counter()
            self.encoder.model.train(True)
            self.classifier.train(True)
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs} \\[train]", leave=False, dynamic_ncols=True, smoothing=0.1)
            for step, batch in enumerate(pbar, start=1):
                # Move batch to device
                batch.tokens = {k: v.to(self.device) for k, v in batch.tokens.items()}
                batch.labels = batch.labels.to(self.device)
                batch.concept_ids = batch.concept_ids.to(self.device)
                batch.concept_mask = batch.concept_mask.to(self.device)

                # Forward with AMP
                with torch.set_grad_enabled(True):
                    with autocast(device_type="cuda", enabled=self.use_amp):
                        emb = self.encoder.forward(batch.tokens)
                        logits = self.classifier(emb, batch.concept_ids, batch.concept_mask)
                        loss = self.criterion(logits, batch.labels) / self.grad_accum_steps

                # Backward (scaled if AMP enabled)
                self.scaler.scale(loss).backward()
                running_loss += float(loss.item())

                if step % self.grad_accum_steps == 0:
                    # Unscale gradients from AMP and clip
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_([p for p in self.encoder.model.parameters() if p.requires_grad], self.max_grad_norm)

                    # Optimizer step via scaler, then scheduler gated on real step
                    scale_before = float(self.scaler.get_scale())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = float(self.scaler.get_scale())
                    did_step = scale_after >= scale_before  # if overflow occurred, scale decreases and step was skipped

                    if did_step:
                        scheduler.step()
                        global_step += 1

                    self.optimizer.zero_grad(set_to_none=True)

                    pbar.set_postfix({  # Progress bar update
                        "loss": f"{running_loss / max(1, step):.4f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        "step": f"{global_step}/{total_steps}",
                    })
            pbar.close()

            # Validation
            val_acc, val_f1 = self._evaluate_loader(val_loader)
            epoch_sec = perf_counter() - epoch_start
            console.log(
                f"[bold cyan]\\[val][/bold cyan] Epoch {epoch}: accuracy={val_acc:.4f} macro_f1={val_f1:.4f} | time={format_duration(epoch_sec)}"
            )
            if val_f1 > best_val:
                best_val = val_f1
                self._best_state = {
                    "encoder": {k: v.detach().cpu() for k, v in self.encoder.model.state_dict().items()},
                    "classifier": {k: v.detach().cpu() for k, v in self.classifier.state_dict().items()},
                }

        # Restore best
        if self._best_state is not None:
            self.encoder.model.load_state_dict(self._best_state["encoder"])
            self.classifier.load_state_dict(self._best_state["classifier"])

        total_sec = perf_counter() - total_start
        console.log(f"[bold green]\\[train][/bold green] Training complete! time={format_duration(total_sec)}")

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> Tuple[float, float]:
        self.encoder.model.eval()
        self.classifier.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        pbar = tqdm(loader, desc="[eval]", leave=False, dynamic_ncols=True)
        for batch in pbar:
            batch.tokens = {k: v.to(self.device) for k, v in batch.tokens.items()}
            batch.labels = batch.labels.to(self.device)
            batch.concept_ids = batch.concept_ids.to(self.device)
            batch.concept_mask = batch.concept_mask.to(self.device)

            # Eval forward with AMP
            with autocast(device_type="cuda", enabled=self.use_amp):
                embeddings = self.encoder.forward(batch.tokens)
                logits = self.classifier(embeddings, batch.concept_ids, batch.concept_mask)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(batch.labels.detach().cpu().tolist())
        pbar.close()
        metrics = evaluate_predictions(all_labels, all_preds, num_classes=len(self.id2label))
        return metrics["accuracy"], metrics["macro_f1"]

    @torch.no_grad()
    def predict_logits(self, symptoms: List[str], test_concepts: List[List[int]], batch_size: Optional[int] = None) -> List[List[float]]:
        self.encoder.model.eval()
        self.classifier.eval()
        if batch_size is None:
            batch_size = self.batch_size
        logits_out: List[List[float]] = []
        total_batches = (len(symptoms) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(symptoms), batch_size), total=total_batches, desc="[predict]", leave=False, dynamic_ncols=True)
        for i in pbar:
            batch_symptoms = symptoms[i:i + batch_size]
            tokens = self.encoder.tokenize(batch_symptoms)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            seqs = test_concepts[i:i + batch_size]
            c_ids, c_mask = pad_concepts(seqs)
            concept_ids_tensor = c_ids.to(self.device)
            concept_mask_tensor = c_mask.to(self.device)
            with autocast(device_type="cuda", enabled=self.use_amp):
                emb = self.encoder.forward(tokens)
                logits = self.classifier(emb, concept_ids_tensor, concept_mask_tensor)
            logits_out.extend(logits.detach().cpu().tolist())
        pbar.close()
        return logits_out

    @torch.no_grad()
    def extract_embeddings(self, symptoms: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Extract text embeddings for analysis and visualization."""
        self.encoder.model.eval()
        if batch_size is None:
            batch_size = self.batch_size
        embeddings_out = []
        total_batches = (len(symptoms) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(symptoms), batch_size), total=total_batches, desc="[extract_embeddings]", leave=False, dynamic_ncols=True)
        for i in pbar:
            batch_symptoms = symptoms[i:i + batch_size]
            tokens = self.encoder.tokenize(batch_symptoms)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with autocast(device_type="cuda", enabled=self.use_amp):
                emb = self.encoder.forward(tokens)
            embeddings_out.append(emb.detach().cpu().numpy())
        pbar.close()
        return np.vstack(embeddings_out)
