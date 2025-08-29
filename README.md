# Biomedical Symptom → Condition Prototype

Predict medical conditions (or urgency) from lists of symptoms using a biomedical encoder (e.g., SapBERT) + MLP classifier, with optional joint fine-tuning.

- Artifacts: saved to `./artifacts`

## Quickstart

1) Configure a Python environment (e.g. `python -m venv venv`), then activate it.

2) Install dependencies:

```
pip install -r requirements.txt
```

Refer to https://pytorch.org/get-started/locally/ for CUDA-specific PyTorch wheels if needed.

3) Run training and evaluation:

```
python main.py
```

## Run configuration

All configuration is in `main.py`.

Outputs include metrics and a preview in `artifacts/run_summary.json`.

## Data

You can train on:

- A CSV dataset (recommended): place at `data/demo.csv` or set `DATASET_PATH` in `main.py`.
- Synthetic data: if no CSV is found, a synthetic dataset is generated.

CSV format (UTF-8):

- Header: `text,label`
- `text`: free-text symptoms
- `label`: machine-friendly class id (snake_case), e.g. `influenza`, `stroke`, `myocardial_infarction_heart_attack`

Built-in class ids include (non-exhaustive):
`influenza`, `common_cold`, `migraine`, `food_poisoning`, `allergy`, `myocardial_infarction_heart_attack`, `dehydration`, `low_blood_pressure_hypotension`,
`stroke`, `pneumonia`, `vasovagal_syncope`, `urinary_tract_infection`, `gastroenteritis`, `covid_19`, `asthma_attack`, `appendicitis`,
`hypertension_high_blood_pressure_crisis`.

## Dataset generation CLI

Use the CLI to generate a demo CSV

```
python dataset.py --out data\demo.csv --n 400 --seed 42
```

Custom subset of classes:

```
python dataset.py --out data/cardio.csv --n 300 --classes "stroke,myocardial_infarction_heart_attack,low_blood_pressure_hypotension" --seed 1
```

Notes:

- `--classes` is a comma-separated list of class ids (snake_case).
- `main.py` auto-loads `data/demo.csv` if present; otherwise it falls back to synthetic data.

## Expanding

- Encoders: modify `embedding.py` (pooling, mixed precision, LoRA, etc.)
- Models: try alternative heads or GNNs in `models.py` - todo!
