# Medical Inference via Vector Embeddings

MEDIVE is a personal research project aiming to use vector embeddings to encode symptom and condition meaning.

## Quick start

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
2. Download MRCONSO.RRF from UMLS and place it in the root.
3. Run training:

    ```bash
    python main.py
    ```
4. After training, run inference on custom symptoms text:

    ```bash
    python infer.py
    ```

## Features

- Hybrid classification head combining text embeddings and UMLS CUI embeddings.
- Uses mixed precision (AMP) for faster training on GPU.
- Early stopping based on validation macro-F1.
- Configurable via constants at the top of `main.py`.
- Reproducible runs with all settings saved to `artifacts/run_summary.json`.

## Files

- `main.py` – orchestrates training and evaluation
- `modules/dataset.py` – dataset and CSV loading (symptoms + concept IDs)
- `modules/concepts.py` – CUI extraction from MRCONSO.RRF
- `modules/embedding.py` – text encoder
- `modules/models.py` – classifier
- `modules/train.py` – training loop
- `infer.py` – inference script
- `webdata.py` – minimal web harvester for building datasets from public pages

## Architecture Overview

```
                        ┌─────────────────┐
                        │   Input Text    │
                        │ CSV: symptoms,  │
                        │      condition  │
                        └────────┬────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │       Text Preprocessing      │
                 │ - Lowercase                   │
                 │ - Remove non-alphanum chars   │
                 │ - Collapse spaces             │
                 └───────────────┬───────────────┘
                                 │
                ┌────────────────┴───────────────┐
                │       CUI Extraction           │
                │ - Load MRCONSO.RRF             │
                │ - Filter LAT=ENG, ISPREF=Y     │
                │ - Build gazetteer: term→CUI    │
                │ - Tokenize text, slide n-grams │
                │ - Exact match n-grams → CUIs   │
                │ - Rank by count, keep top k    │
                └───────────────┬────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │   CUI ID Mapping / Padding    │
                │ - Map CUIs → integer IDs      │
                │ - Special tokens: [PAD],      │
                │   [NO_CUI], [UNK_CUI]         │
                │ - Pad to MAX_CUIS_PER_DOC     │
                └───────────────┬───────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                   Encoding                    │
        │                                               │
        │ Text Pathway:                                 │
        │ - Tokenize & pad                              │
        │ - SAPBERT encoder → last_hidden_state         │
        │ - Masked mean pooling → text embedding [B, Dt]│
        │                                               │
        │ CUI Pathway:                                  │
        │ - nn.Embedding(CUI_IDs)                       │
        │ - Masked mean pooling → CUI embedding [B, Dc] │
        │ - Dropout                                     │
        └───────────────┬───────────────┬───────────────┘
                        │               │
                        │               │
                        └─────Fusion────┘
                                │
                 ┌──────────────┴───────────────┐
                 │     Concatenate Text + CUI   │
                 │       [B, Dt+Dc]             │
                 │       MLP → logits [B, C]    │
                 └──────────────┬───────────────┘
                                │
                ┌───────────────┴───────────────┐
                │         Training Loop         │
                │ - AdamW + scheduler           │
                │ - Label smoothing             │
                │ - AMP + gradient clipping     │
                │ - Early stopping by macro-F1  │
                └───────────────┬───────────────┘
                                │
                ┌───────────────┴───────────────┐
                │       Evaluation / Inference  │
                │ - Reuse MRCONSO pipeline      │
                │ - Map CUIs → IDs              │
                │ - Compute embeddings          │
                │ - Classifier logits → labels  │
                │ - Metrics: accuracy, macro-F1 │
                └───────────────────────────────┘
```

- Batch size: B
- Token sequence length (after tokenizer): T
- Text encoder hidden dim: H (e.g. 768)
- Max CUIs per doc (padded/truncated): Lc (e.g. 32)
- CUI embedding dim: Dc (e.g. 128)
- Number of classes: C
- Raw dataset rows: N
- mask: 1 for CUIs, 0 for padding

## Step-by-step Architecture

**1. Input and split**

- Input: CSV at data/demo.csv with columns symptoms,condition.
- Load all rows, infer class set from conditions.
- Stratified split into train/val/test via sklearn.

**2. Concept extraction (UMLS CUIs) with MRCONSO.RRF**

- Load MRCONSO.RRF from repo root and build a gazetteer:
    - Keep LAT=ENG rows; by default ISPREF=Y only (preferred English terms).
    - Normalize terms: lowercase, keep a–z/0–9, collapse spaces.
    - Map normalized term → CUI (first-seen CUI kept for ambiguous strings).

- For each input symptoms text:
    - Normalize, tokenize, slide n-grams up to max_ngram (default 6).
    - Exact match n-grams against the gazetteer; count matches per CUI.
    - Rank CUIs by count desc; keep top MAX_CUIS_PER_DOC (default 32).

- Build a CUI vocabulary from the union of all extracted CUIs (train+val+test):
    - Special tokens: [PAD]=0, [NO_CUI]=1, [UNK_CUI]=2.
    - Map each text’s CUIs to integer IDs; if none found → [NO_CUI].

- Attach the CUI id lists back to each split.

**3. Text encoder**

- Model: Hugging Face encoder (SapBERT) via TextEmbeddingEncoder.
- Tokenization with truncation/padding to MAX_SEQUENCE_LENGTH.
- Forward: last_hidden_state → mean pooling (mask-aware) → text embedding (size Dt, e.g., 768).
- Trainability: fine_tune (True/False) plus optional layer norm freezing.

**4. Hybrid classifier**

- Concept pathway:
    - nn.Embedding(cui_vocab_size, CUI_EMB_DIM) with padding_idx=[PAD].
    - Pad variable-length CUI ID lists per batch to [B, Lc]; create mask [B, Lc].
    - Pool CUI embeddings with masked mean → [B, Dc].
    - Dropout on pooled CUI vector.
- Fusion + head:
    - Concatenate [text_vec (Dt); cui_vec (Dc)] → [B, Dt+Dc].
    - MLP: Linear → ReLU → Dropout → Linear → logits [B, num_classes].

**5. Batching and collation**

- Collate returns:
    - tokens: input_ids, attention_mask (HF tensors).
    - labels (condition IDs): LongTensor [B].
    - concept_ids: LongTensor [B, Lc], concept_mask: LongTensor [B, Lc].

**6. Training loop (TorchTrainer)**

- Optimizer: AdamW with separate param groups for encoder/head and no-decay on LayerNorm/bias.
- Scheduler: linear with warmup (warmup_ratio).
- Loss: CrossEntropy with label smoothing (default 0.10).
- AMP: autocast + GradScaler on CUDA.
- Gradient clipping for encoder and head.
- Early best tracking: keep best by validation macro-F1; restore best weights after training.
- Artifacts saved to artifacts/: encoder_state.pt, classifier_state.pt, cui_vocab.json, run_summary.json.

**7. Evaluation**

- On val during training for model selection.
- On test after training:
    - Reuse attached concept IDs.
    - Predict logits in batches, argmax to class IDs.
    - Metrics: accuracy, macro-F1.

**8. Inference (infer.py)**

- Load run_summary.json config, cui_vocab.json, and model weights.
- Build encoder + HybridClassifier to match training config.
- For each input symptoms text:
    - Extract CUIs using the same MRCONSO gazetteer (same normalization and n-gram settings).
    - Map CUIs to IDs with saved vocabulary; pad and mask.
    - Tokenize text, compute embeddings, run classifier, softmax for confidences.
    - Display top-k classes and confidences.

**Key configuration knobs**

- MRCONSO extraction:
    - include_all_eng_terms (default False for ISPREF=Y only).
    - max_ngram (default 6).
    - max_concepts_per_doc (default 32).
- Model:
    - HF_MODEL_NAME, MAX_SEQUENCE_LENGTH, CUI_EMB_DIM, MLP_HIDDEN_DIM, DROPOUT.
- Training:
    - EPOCHS, BATCH_SIZE, learning rates (encoder/head), WEIGHT_DECAY, WARMUP_RATIO, LABEL_SMOOTHING.

**Data contracts and fallbacks**

- If a symptoms text yields no CUIs, it gets [NO_CUI] so the concept pathway always has input.
- Ambiguity in MRCONSO strings is resolved by first-seen CUI (simple heuristic).
- Vocab is built from the training-time union; unseen CUIs map to [UNK_CUI] at inference.

**Artifacts and reproducibility**

- All important settings are recorded in artifacts/run_summary.json.
- cui_vocab.json ensures consistent CUI ID mapping between train and inference.
