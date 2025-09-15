"""
Augment dataset by introducing slight variations in symptom phrasing.
Creates new rows with paraphrased symptoms while preserving semantic meaning.

Usage:
    python augment_dataset.py --input data/positives.csv --output data/augmented_positives.csv --multiplier 2
    python augment_dataset.py --input data/positives.csv --output data/augmented_positives.csv --target-size 50000
"""

import argparse
import random
import re
from pathlib import Path

import pandas as pd
from tqdm.rich import tqdm
import warnings

warnings.filterwarnings("ignore", message="rich is experimental/alpha")


class SymptomAugmenter:
    """Generate variations of symptom phrases while preserving meaning."""

    def __init__(self):
        # Synonym mappings for common medical terms
        self.synonyms = {
            # Pain variations
            "pain": ["ache", "discomfort", "soreness", "hurt"],
            "ache": ["pain", "discomfort", "soreness"],
            "discomfort": ["pain", "ache", "uneasiness"],
            "soreness": ["pain", "ache", "tenderness"],

            # Body parts
            "stomach": ["abdomen", "belly", "tummy"],
            "abdomen": ["stomach", "belly"],
            "belly": ["stomach", "abdomen"],
            "head": ["skull", "cranium"],
            "chest": ["thorax"],
            "throat": ["pharynx"],

            # Conditions
            "fever": ["high temperature", "elevated temperature", "pyrexia"],
            "headache": ["head pain", "cephalgia"],
            "nausea": ["queasiness", "sick feeling", "upset stomach"],
            "fatigue": ["tiredness", "exhaustion", "weakness"],
            "tiredness": ["fatigue", "exhaustion", "weariness"],
            "weakness": ["fatigue", "feebleness"],
            "dizziness": ["lightheadedness", "vertigo"],
            "swelling": ["edema", "puffiness", "inflammation"],
            "rash": ["skin eruption", "dermatitis"],
            "cough": ["coughing", "hacking"],
            "breathing": ["respiration"],
            "shortness of breath": ["dyspnea", "breathlessness"],
            "difficulty breathing": ["trouble breathing", "labored breathing"],

            # Descriptors
            "severe": ["intense", "sharp", "acute", "extreme"],
            "mild": ["slight", "minor", "gentle"],
            "chronic": ["persistent", "ongoing", "long-term"],
            "sudden": ["abrupt", "rapid", "quick"],
            "gradual": ["slow", "progressive"],
            "frequent": ["regular", "repeated", "recurring"],
            "occasional": ["intermittent", "sporadic"],

            # Qualifiers
            "excessive": ["too much", "increased", "elevated"],
            "decreased": ["reduced", "lowered", "diminished"],
            "difficulty": ["trouble", "problems with"],
            "inability": ["unable to", "cannot"],
            "loss of": ["absence of", "lack of"],
        }

        # Phrase transformations
        self.transformations = [
            # Add descriptors
            (r'\b(pain)\b', lambda m: random.choice([
                f"mild {m.group(1)}", f"moderate {m.group(1)}", f"sharp {m.group(1)}", m.group(1)
            ])),

            # Rephrase "pain in X" to "X pain"
            (r'pain in ([\w\s]+)', lambda m: random.choice([
                f"pain in {m.group(1)}", f"{m.group(1)} pain", f"{m.group(1)} discomfort"
            ])),

            # Add "feeling of" to some symptoms
            (r'\b(nausea|dizziness|weakness|fatigue)\b', lambda m: random.choice([
                m.group(1), f"feeling of {m.group(1)}", f"sensation of {m.group(1)}"
            ])),

            # Rephrase "difficulty X" variations
            (r'difficulty ([\w\s]+)', lambda m: random.choice([
                f"difficulty {m.group(1)}", f"trouble {m.group(1)}", f"problems {m.group(1)}"
            ])),
        ]

    def get_synonym(self, word: str) -> str:
        """Get a random synonym for a word, or return the original if no synonym exists."""
        word_lower = word.lower()
        if word_lower in self.synonyms:
            return random.choice(self.synonyms[word_lower])
        return word

    def apply_synonym_substitution(self, text: str) -> str:
        """Apply synonym substitution to text."""
        words = text.split()
        result = []

        for word in words:
            # Remove punctuation for lookup but preserve it
            clean_word = re.sub(r'[^\w\s]', '', word).lower()

            # Check for multi-word phrases first
            for phrase_len in [3, 2]:  # Check longer phrases first
                if len(result) >= phrase_len - 1:
                    phrase_words = [re.sub(r'[^\w\s]', '', w).lower() for w in words[len(result) - phrase_len + 1:len(result) + 1]]
                    phrase = ' '.join(phrase_words)

                    if phrase in self.synonyms:
                        # Replace the phrase
                        synonym = random.choice(self.synonyms[phrase])
                        # Remove previous words and add the synonym
                        for _ in range(phrase_len - 1):
                            result.pop()
                        result.append(synonym)
                        break
            else:
                # Single word substitution
                if random.random() < 0.3:  # 30% chance to substitute
                    synonym = self.get_synonym(clean_word)
                    if synonym != clean_word:
                        # Preserve original capitalization and punctuation
                        if word[0].isupper():
                            synonym = synonym.capitalize()
                        # Add back punctuation
                        punctuation = re.findall(r'[^\w\s]', word)
                        if punctuation:
                            synonym += ''.join(punctuation)
                        result.append(synonym)
                    else:
                        result.append(word)
                else:
                    result.append(word)

        return ' '.join(result)

    def apply_transformations(self, text: str) -> str:
        """Apply phrase transformations to text."""
        result = text
        for pattern, replacement in self.transformations:
            if random.random() < 0.2:  # 20% chance to apply each transformation
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    def augment_symptoms(self, symptoms_text: str) -> str:
        """Generate a variation of the symptoms text."""
        if not symptoms_text or not symptoms_text.strip():
            return symptoms_text

        # Split symptoms by comma
        symptoms = [s.strip() for s in symptoms_text.split(',')]
        augmented_symptoms = []

        for symptom in symptoms:
            if not symptom:
                continue

            # Apply transformations
            augmented = symptom

            # Apply synonym substitution
            if random.random() < 0.6:  # 60% chance
                augmented = self.apply_synonym_substitution(augmented)

            # Apply phrase transformations
            if random.random() < 0.4:  # 40% chance
                augmented = self.apply_transformations(augmented)

            # Small chance to reorder if multiple symptoms
            if len(symptoms) > 1 and random.random() < 0.1:  # 10% chance
                continue  # Skip this symptom (will be added later if not already included)

            augmented_symptoms.append(augmented)

        # Randomly reorder symptoms
        if len(augmented_symptoms) > 1 and random.random() < 0.3:  # 30% chance
            random.shuffle(augmented_symptoms)

        return ','.join(augmented_symptoms)


def main():
    parser = argparse.ArgumentParser(description="Augment dataset with variations in symptom phrasing")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--multiplier", type=float,
                       help="Multiply dataset size by this factor (e.g., 2.0 = double the size)")
    group.add_argument("--target-size", type=int,
                       help="Target total number of rows in output (including originals)")

    parser.add_argument("--include-originals", action="store_true", default=True,
                        help="Include original rows in output (default: True)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--symptoms-column", default="symptoms",
                        help="Column name containing symptoms (default: symptoms)")
    parser.add_argument("--condition-column", default="condition",
                        help="Column name containing condition (default: condition)")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    original_size = len(df)
    print(f"Original dataset size: {original_size} rows")

    # Validate columns
    if args.symptoms_column not in df.columns:
        print(f"Error: Symptoms column '{args.symptoms_column}' not found")
        print(f"Available columns: {list(df.columns)}")
        return

    if args.condition_column not in df.columns:
        print(f"Error: Condition column '{args.condition_column}' not found")
        print(f"Available columns: {list(df.columns)}")
        return

    # Calculate target augmentations
    if args.multiplier:
        target_total = int(original_size * args.multiplier)
        if args.include_originals:
            n_augmentations = target_total - original_size
        else:
            n_augmentations = target_total
    else:  # target_size
        if args.include_originals:
            n_augmentations = args.target_size - original_size
        else:
            n_augmentations = args.target_size

    if n_augmentations <= 0:
        print(f"Error: No augmentations needed. Target would result in {n_augmentations} new rows.")
        return

    print(f"Generating {n_augmentations} augmented samples...")

    # Initialize augmenter
    augmenter = SymptomAugmenter()

    # Generate augmented data
    augmented_rows = []

    # Calculate how many augmentations per original row
    augmentations_per_row = n_augmentations / original_size

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting samples"):
        condition = row[args.condition_column]
        original_symptoms = row[args.symptoms_column]

        # Determine number of augmentations for this row
        base_augs = int(augmentations_per_row)
        extra_aug = 1 if random.random() < (augmentations_per_row - base_augs) else 0
        n_augs_for_row = base_augs + extra_aug

        # Generate augmentations for this row
        for _ in range(n_augs_for_row):
            augmented_symptoms = augmenter.augment_symptoms(original_symptoms)

            # Create new row
            new_row = row.copy()
            new_row[args.symptoms_column] = augmented_symptoms
            augmented_rows.append(new_row)

    # Combine original and augmented data
    if args.include_originals:
        final_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    else:
        final_df = pd.DataFrame(augmented_rows)

    # Shuffle the final dataset
    final_df = final_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    final_size = len(final_df)
    augmentation_ratio = final_size / original_size

    print(f"\nAugmentation complete!")
    print(f"Original size: {original_size} rows")
    print(f"Augmented size: {len(augmented_rows)} new rows")
    print(f"Final size: {final_size} rows ({augmentation_ratio:.1f}x original)")

    # Show some examples
    print(f"\nExample augmentations:")
    sample_indices = random.sample(range(len(augmented_rows)), min(3, len(augmented_rows)))
    for i, aug_idx in enumerate(sample_indices):
        aug_row = augmented_rows[aug_idx]
        condition = aug_row[args.condition_column]

        # Find original for comparison
        orig_row = df[df[args.condition_column] == condition].iloc[0]

        print(f"\n{i+1}. {condition}")
        print(f"   Original: {orig_row[args.symptoms_column]}")
        print(f"   Augmented: {aug_row[args.symptoms_column]}")

    # Save augmented dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n✅ Saving augmented dataset to {output_path}...")
    final_df.to_csv(output_path, index=False)

    print("Done!")


if __name__ == "__main__":
    main()
