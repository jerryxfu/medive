#!/usr/bin/env python3
"""
Reduce dataset size by keeping a percentage of each class.
Useful for creating smaller datasets for development testing while maintaining class distribution.

Usage:
    python reduce_dataset.py --input data/positives.csv --output data/positives_small.csv --keep-percent 20
    python reduce_dataset.py --input data/positives.csv --output data/positives_small.csv --target-size 1000
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Reduce dataset size by keeping a percentage of each class")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keep-percent", type=float, help="Percentage of each class to keep (e.g., 20 = keep 20% of each class)")
    group.add_argument("--target-size", type=int, help="Target number of rows in output")

    parser.add_argument("--preserve-header", action="store_true", default=True, help="Always keep the header row")
    parser.add_argument("--class-column", default="condition", help="Column name containing class labels (default: condition)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    original_size = len(df)
    print(f"Original dataset size: {original_size} rows")

    # Check if class column exists
    if args.class_column not in df.columns:
        print(f"Error: Class column '{args.class_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return

    # Get class distribution
    class_counts = df[args.class_column].value_counts()
    print(f"\nOriginal class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    if args.keep_percent:
        # Keep specified percentage of each class
        keep_percent = args.keep_percent / 100
        if keep_percent <= 0 or keep_percent > 100:
            print(f"Error: keep-percent must be between 0 and 100")
            return

        print(f"\nKeeping {args.keep_percent}% of each class")
        reduced_dfs = []

        for class_name in class_counts.index:
            class_df = df[df[args.class_column] == class_name]
            n_keep = max(1, int(len(class_df) * keep_percent))
            sampled_df = class_df.sample(n=n_keep, random_state=42)
            reduced_dfs.append(sampled_df)

        reduced_df = pd.concat(reduced_dfs, ignore_index=True).sample(frac=1, random_state=42)

    elif args.target_size:
        # Calculate percentage needed to achieve target size
        target = args.target_size
        if target >= original_size:
            print(f"Target size {target} is >= original size {original_size}, copying all rows")
            reduced_df = df.copy()
        else:
            target_percent = target / original_size
            print(f"Target size: {target} rows, keeping {target_percent * 100:.1f}% of each class")

            reduced_dfs = []
            total_kept = 0

            for class_name in class_counts.index:
                class_df = df[df[args.class_column] == class_name]
                n_keep = max(1, int(len(class_df) * target_percent))
                sampled_df = class_df.sample(n=n_keep, random_state=42)
                reduced_dfs.append(sampled_df)
                total_kept += len(sampled_df)

            reduced_df = pd.concat(reduced_dfs, ignore_index=True).sample(frac=1, random_state=42)

            # If we're over target, trim randomly
            if len(reduced_df) > target:
                reduced_df = reduced_df.sample(n=target, random_state=42)

    new_size = len(reduced_df)
    reduction_percent = (1 - new_size / original_size) * 100

    print(f"\nReduced dataset size: {new_size} rows ({reduction_percent:.1f}% reduction)")

    # Show new class distribution
    new_class_counts = reduced_df[args.class_column].value_counts()
    print(f"\nNew class distribution:")
    for class_name, count in new_class_counts.items():
        original_count = class_counts[class_name]
        kept_percent = (count / original_count) * 100
        print(f"  {class_name}: {count} ({kept_percent:.1f}% of original)")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save reduced dataset
    print(f"\nSaving to {output_path}...")
    reduced_df.to_csv(output_path, index=False)

    print("Done!")


if __name__ == "__main__":
    main()
