"""
Split a wide binary symptom dataset into positive and negative sets.

Input format (CSV):
Condition, fever, cough, headache, loss_of_smell, ...
Influenza, 1, 1, 0, 0, ...
COVID-19, 1, 1, 1, 1, ...

Outputs:
- positives.csv: rows of (condition, "symptom_a,symptom_b,...")
- negatives.csv (via --include-negatives): rows of (condition, "symptom_x,symptom_y,...")
"""

import argparse
import warnings

import pandas as pd
from tqdm.rich import tqdm

warnings.filterwarnings("ignore", message="rich is experimental/alpha")  # suppress experimental warning


def main():
    parser = argparse.ArgumentParser(description="Split dataset into positive/negative symptom sets")
    parser.add_argument("--input", required=True, help="Input CSV file (wide binary)")
    parser.add_argument("--positives", default="positives.csv", help="Output file for positives")
    parser.add_argument("--negatives", default="negatives.csv", help="Output file for negatives")
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="write negatives CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    condition_col = df.columns[0]  # first column is condition
    symptom_cols = df.columns[1:]  # rest are symptoms

    positives = []
    negatives = [] if args.include_negatives else None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        condition = row[condition_col]
        pos_syms = [sym for sym in symptom_cols if row[sym] == 1]
        if pos_syms:
            positives.append({"condition": condition, "symptoms": ",".join(pos_syms)})

        if args.include_negatives:
            neg_syms = [sym for sym in symptom_cols if row[sym] == 0]
            if neg_syms:
                negatives.append({"condition": condition, "symptoms": ",".join(neg_syms)})  # type: ignore[arg-type]

    pd.DataFrame(positives).to_csv(args.positives, index=False)
    print(f"Saved {len(positives)} positives to {args.positives}")

    if args.include_negatives and negatives is not None:
        pd.DataFrame(negatives).to_csv(args.negatives, index=False)
        print(f"Saved {len(negatives)} negatives to {args.negatives}")


if __name__ == "__main__":
    main()
