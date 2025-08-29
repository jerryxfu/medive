from __future__ import annotations

from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModel


class TextEmbeddingEncoder:
    def __init__(
        self,
        hf_model_name: str,
        max_seq_len: int = 64,
        fine_tune: bool = True,
        freeze_layer_norm: bool = True,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.model = AutoModel.from_pretrained(hf_model_name)
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.model.to(self.device)

        self.output_dim = int(getattr(self.model.config, "hidden_size", 768))

        self.set_trainable(fine_tune=fine_tune, freeze_layer_norm=freeze_layer_norm)

    def set_trainable(self, fine_tune: bool, freeze_layer_norm: bool = True) -> None:
        """
        Control which encoder params are trainable.
        - fine_tune True: all params trainable.
        - fine_tune False: freeze all except layer norms if freeze_layer_norm is True.
        """
        for p in self.model.parameters():
            p.requires_grad = fine_tune
        if not fine_tune and freeze_layer_norm:
            for name, module in self.model.named_modules():
                if any(tag in name.lower() for tag in ["layernorm", "layer_norm", "ln"]):
                    for p in module.parameters():
                        p.requires_grad = True

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Prefer mean pooling over CLS for robustness across models
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        return pooled
