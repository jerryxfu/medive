from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from rich.console import Console
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.rich import tqdm
from transformers import get_linear_schedule_with_warmup

from embedding import TextEmbeddingEncoder

console = Console()
try:  # Access to a protected member _log_render of a class
    console._log_render.omit_repeated_times = False  # always show time bracket
except Exception:
    pass


def _format_duration(seconds: float) -> str:
    # Format as HH:MM:SS.s (two decimals)
    m, s = divmod(seconds, 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}"


class _TextLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


@dataclass
class _Batch:
    tokens: Dict[str, torch.Tensor]
    labels: torch.Tensor


def _collate_fn(examples: List[Tuple[str, int]], encoder: TextEmbeddingEncoder) -> _Batch:
    texts = [t for t, _ in examples]
    labels = torch.tensor([y for _, y in examples], dtype=torch.long)
    tokens = encoder.tokenize(texts)
    return _Batch(tokens=tokens, labels=labels)


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
        log_every: int = 50,
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
        self.log_every = log_every

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

    def _make_loader(self, texts: List[str], labels: List[int], shuffle: bool) -> DataLoader:
        dataset = _TextLabelDataset(texts, labels)

        # Need closure over encoder for tokenization in collate
        def collate(examples):
            return _collate_fn(examples, self.encoder)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=collate)

    def fit(self, train_ds, val_ds) -> None:
        train_labels = [self.label2id[y] for y in train_ds.labels]
        val_labels = [self.label2id[y] for y in val_ds.labels]
        train_loader = self._make_loader(train_ds.texts, train_labels, shuffle=True)
        val_loader = self._make_loader(val_ds.texts, val_labels, shuffle=False)

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

                # Forward with AMP
                with torch.set_grad_enabled(True):
                    with autocast(device_type="cuda", enabled=self.use_amp):
                        emb = self.encoder.forward(batch.tokens)
                        logits = self.classifier(emb)
                        loss = self.criterion(logits, batch.labels) / self.grad_accum_steps

                # Backward (scaled if AMP enabled)
                self.scaler.scale(loss).backward()
                running_loss += float(loss.item())

                if step % self.grad_accum_steps == 0:
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_([p for p in self.encoder.model.parameters() if p.requires_grad], self.max_grad_norm)

                    # Optimizer step via scaler, then scheduler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # Update progress bar with smoothed loss and scheduler step math
                    pbar.set_postfix({
                        "loss": f"{running_loss / max(1, step):.4f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        "step": f"{global_step}/{total_steps}",
                    })
            pbar.close()

            # Validation
            val_acc, val_f1 = self._evaluate_loader(val_loader)
            epoch_sec = perf_counter() - epoch_start
            console.log(
                f"[bold cyan]\\[val][/bold cyan] Epoch {epoch}: accuracy={val_acc:.4f} macro_f1={val_f1:.4f} "
                f"| time={_format_duration(epoch_sec)}"
            )
            if val_f1 > best_val:
                best_val = val_f1
                self._best_state = {
                    "encoder": {k: v.detach().cpu() for k, v in self.encoder.model.state_dict().items()},
                    "classifier": {k: v.detach().cpu() for k, v in self.classifier.state_dict().items()},
                }

        # Restore best
        if self._best_state is not None:
            self.encoder.model.load_state_dict(self._best_state["encoder"])  # type: ignore[arg-type]
            self.classifier.load_state_dict(self._best_state["classifier"])  # type: ignore[arg-type]

        total_sec = perf_counter() - total_start
        console.log(f"[bold green]\\[train][/bold green] Training complete! time={_format_duration(total_sec)}")

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
            from torch.amp import autocast
            with autocast(device_type="cuda", enabled=self.use_amp):
                emb = self.encoder.forward(batch.tokens)
                logits = self.classifier(emb)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(batch.labels.detach().cpu().tolist())
        pbar.close()
        metrics = evaluate_predictions(all_labels, all_preds, num_classes=len(self.id2label))
        return metrics["accuracy"], metrics["macro_f1"]

    @torch.no_grad()
    def predict_logits(self, texts: List[str]) -> List[List[float]]:
        self.encoder.model.eval()
        self.classifier.eval()
        logits_out: List[List[float]] = []
        # Iterate over batches with a progress bar
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(range(0, len(texts), self.batch_size), total=total_batches, desc="[predict]", leave=False, dynamic_ncols=True)
        for i in pbar:
            batch_texts = texts[i:i + self.batch_size]
            tokens = self.encoder.tokenize(batch_texts)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            from torch.amp import autocast
            with autocast(device_type="cuda", enabled=self.use_amp):
                emb = self.encoder.forward(tokens)
                logits = self.classifier(emb)
            logits_out.extend(logits.detach().cpu().tolist())
        pbar.close()
        return logits_out
