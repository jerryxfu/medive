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
    num_classes = len(class_counts)
    print(f"\nOriginal class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    if args.keep_percent:
        # Keep specified percentage of each class
        keep_percent = args.keep_percent / 100
        if keep_percent <= 0 or keep_percent > 100:
            print(f"Error: keep-percent must be between 0 and 100")
            return

        print(f"\nKeeping {args.keep_percent}% of each class (minimum 1 sample per class)")
        reduced_dfs = []

        for class_name in class_counts.index:
            class_df = df[df[args.class_column] == class_name]
            calculated_keep = int(len(class_df) * keep_percent)
            # Force at least 1 sample per class, even if percentage would give 0
            n_keep = max(1, calculated_keep)

            sampled_df = class_df.sample(n=n_keep, random_state=42)
            reduced_dfs.append(sampled_df)

        reduced_df = pd.concat(reduced_dfs, ignore_index=True).sample(frac=1, random_state=42)

    elif args.target_size:
        # Validate target size can accommodate all classes
        target = args.target_size
        if target < num_classes:
            print(f"Error: Target size {target} is smaller than number of classes ({num_classes})")
            print(f"⚠️ Minimum target size must be {num_classes} to ensure at least 1 sample per class")
            return

        if target >= original_size:
            print(f"Target size {target} is >= original size {original_size}, copying all rows")
            reduced_df = df.copy()
        else:
            print(f"Target size: {target} rows (guaranteeing at least 1 sample per class)")

            reduced_dfs = []
            remaining_target = target - num_classes  # Reserve 1 slot per class

            # First, ensure each class gets at least 1 sample
            for class_name in class_counts.index:
                class_df = df[df[args.class_column] == class_name]
                sampled_df = class_df.sample(n=1, random_state=42)
                reduced_dfs.append(sampled_df)

            # Then distribute remaining slots proportionally
            if remaining_target > 0:
                for class_name in class_counts.index:
                    class_df = df[df[args.class_column] == class_name]
                    # Skip the already sampled row
                    remaining_class_df = class_df.drop(reduced_dfs[list(class_counts.index).index(class_name)].index)

                    if len(remaining_class_df) > 0:
                        # Calculate additional samples for this class
                        class_proportion = class_counts[class_name] / original_size
                        additional_samples = int(remaining_target * class_proportion)
                        additional_samples = min(additional_samples, len(remaining_class_df))

                        if additional_samples > 0:
                            additional_df = remaining_class_df.sample(n=additional_samples, random_state=42)
                            reduced_dfs[list(class_counts.index).index(class_name)] = pd.concat([
                                reduced_dfs[list(class_counts.index).index(class_name)],
                                additional_df
                            ])

            reduced_df = pd.concat(reduced_dfs, ignore_index=True).sample(frac=1, random_state=42)

            # Final trim if we're slightly over target (due to rounding)
            if len(reduced_df) > target:
                # Randomly remove excess, but ensure we keep at least 1 per class
                excess = len(reduced_df) - target
                for _ in range(excess):
                    # Find classes with more than 1 sample
                    current_counts = reduced_df[args.class_column].value_counts()
                    removable_classes = current_counts[current_counts > 1].index
                    if len(removable_classes) > 0:
                        # Remove one sample from a random class that has more than 1
                        remove_class = pd.Series(removable_classes).sample(1, random_state=42).iloc[0]
                        class_indices = reduced_df[reduced_df[args.class_column] == remove_class].index
                        remove_idx = pd.Series(class_indices).sample(1, random_state=42).iloc[0]
                        reduced_df = reduced_df.drop(remove_idx).reset_index(drop=True)

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

    # Verify no class was completely removed
    missing_classes = set(class_counts.index) - set(new_class_counts.index)
    if missing_classes:
        print(f"\n⚠️ WARNING: The following classes were completely removed: {missing_classes}")
    else:
        print(f"\n✅ All {num_classes} classes preserved in reduced dataset")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save reduced dataset
    print(f"\nSaving to {output_path}...")
    reduced_df.to_csv(output_path, index=False)

    print("Done!")


if __name__ == "__main__":
    main()
