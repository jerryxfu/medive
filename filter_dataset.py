"""
Filter dataset by keeping only classes that meet a minimum percentage representation.
Removes classes that fall below the specified threshold.

Usage:
    python filter_datasets.py --input data/positives.csv --output data/filtered_positives.csv --min-percent 5.0
    python filter_datasets.py --input data/positives.csv --output data/filtered_positives.csv --min-count 100
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Filter dataset by keeping only classes with sufficient representation")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--min-percent", type=float,
                       help="Minimum percentage representation required for a class to be kept (e.g., 5.0 = keep classes with at least 5% of total samples)")
    group.add_argument("--min-count", type=int, help="Minimum absolute count required for a class to be kept")

    parser.add_argument("--class-column", default="condition", help="Column name containing class labels (default: condition)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information about filtering")

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
    print(f"Original number of classes: {num_classes}")

    if args.verbose:
        print(f"\nOriginal class distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / original_size) * 100
            print(f"  {class_name}: {count} ({percentage:.2f}%)")

    # Determine which classes to keep
    classes_to_keep = []
    classes_to_remove = []

    if args.min_percent:
        min_threshold = args.min_percent
        print(f"\nFiltering classes with less than {min_threshold}% representation...")

        for class_name, count in class_counts.items():
            percentage = (count / original_size) * 100
            if percentage >= min_threshold:
                classes_to_keep.append(class_name)
            else:
                classes_to_remove.append((class_name, count, percentage))

        if args.verbose and classes_to_remove:
            print(f"\nClasses being removed (below {min_threshold}% threshold):")
            for class_name, count, percentage in classes_to_remove:
                print(f"  {class_name}: {count} ({percentage:.2f}%)")

    elif args.min_count:
        min_threshold = args.min_count
        print(f"\nFiltering classes with less than {min_threshold} samples...")

        for class_name, count in class_counts.items():
            if count >= min_threshold:
                classes_to_keep.append(class_name)
            else:
                classes_to_remove.append((class_name, count))

        if args.verbose and classes_to_remove:
            print(f"\nClasses being removed (below {min_threshold} samples threshold):")
            for class_name, count in classes_to_remove:
                percentage = (count / original_size) * 100
                print(f"  {class_name}: {count} ({percentage:.2f}%)")

    # Filter the dataset
    if len(classes_to_keep) == 0:
        print(f"\nError: No classes meet the specified threshold!")
        print(f"All classes would be removed.")
        return

    print(f"\nKeeping {len(classes_to_keep)} out of {num_classes} classes")
    filtered_df = df[df[args.class_column].isin(classes_to_keep)]

    new_size = len(filtered_df)
    reduction_percent = (1 - new_size / original_size) * 100
    classes_removed = num_classes - len(classes_to_keep)

    print(f"\nFiltered dataset size: {new_size} rows ({reduction_percent:.1f}% reduction)")
    print(f"Classes removed: {classes_removed}")

    # Show final class distribution
    new_class_counts = filtered_df[args.class_column].value_counts()
    print(f"\nFinal class distribution:")
    for class_name, count in new_class_counts.items():
        percentage = (count / new_size) * 100
        original_percentage = (class_counts[class_name] / original_size) * 100
        print(f"  {class_name}: {count} ({percentage:.2f}% of filtered, {original_percentage:.2f}% of original)")

    # Save filtered dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n✅ Saving filtered dataset to {output_path}...")
    filtered_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
