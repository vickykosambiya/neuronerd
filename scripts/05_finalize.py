"""
05_finalize.py - Dataset Finalization and Train/Test Split

Converts cleaned data to Alpaca JSONL format and splits into
train (95%) and test (5%) sets for Unsloth fine-tuning.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CLEANED_DIR,
    FINAL_DIR,
    TRAIN_SPLIT,
)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def validate_alpaca_format(entries: List[Dict]) -> tuple[List[Dict], int]:
    """Ensure all entries conform to Alpaca format."""
    valid = []
    fixed = 0
    
    for entry in entries:
        alpaca_entry = {
            "instruction": str(entry.get("instruction", "")).strip(),
            "input": str(entry.get("input", "")).strip(),
            "output": str(entry.get("output", "")).strip(),
        }
        
        # Validate non-empty instruction and output
        if alpaca_entry["instruction"] and alpaca_entry["output"]:
            valid.append(alpaca_entry)
        else:
            fixed += 1
    
    return valid, fixed


def shuffle_and_split(
    entries: List[Dict], 
    train_ratio: float = TRAIN_SPLIT,
    seed: int = 42
) -> tuple[List[Dict], List[Dict]]:
    """Shuffle and split data into train and test sets."""
    # Set seed for reproducibility
    random.seed(seed)
    
    # Shuffle
    shuffled = entries.copy()
    random.shuffle(shuffled)
    
    # Calculate split point
    split_idx = int(len(shuffled) * train_ratio)
    
    train_set = shuffled[:split_idx]
    test_set = shuffled[split_idx:]
    
    return train_set, test_set


def save_jsonl(entries: List[Dict], file_path: str):
    """Save entries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def calculate_stats(entries: List[Dict]) -> Dict:
    """Calculate dataset statistics."""
    instruction_lengths = [len(e["instruction"].split()) for e in entries]
    output_lengths = [len(e["output"].split()) for e in entries]
    
    return {
        "count": len(entries),
        "avg_instruction_words": sum(instruction_lengths) / len(entries) if entries else 0,
        "avg_output_words": sum(output_lengths) / len(entries) if entries else 0,
        "min_output_words": min(output_lengths) if output_lengths else 0,
        "max_output_words": max(output_lengths) if output_lengths else 0,
    }


def main(validate_only: bool = False):
    """Main finalization function."""
    print("=" * 60)
    print("NeuroNerd Dataset Pipeline - Step 5: Finalization")
    print("=" * 60)
    
    # Setup
    os.makedirs(FINAL_DIR, exist_ok=True)
    
    # Find cleaned data
    cleaned_file = os.path.join(CLEANED_DIR, "cleaned_qa.jsonl")
    
    if not os.path.exists(cleaned_file):
        print(f"\nNo cleaned data found at '{cleaned_file}'!")
        print("Please run 04_quality_control.py first.")
        return
    
    print(f"\nLoading data from: {cleaned_file}")
    
    # Load data
    entries = load_jsonl(cleaned_file)
    print(f"Loaded {len(entries)} entries")
    
    # Validate format
    entries, removed = validate_alpaca_format(entries)
    if removed > 0:
        print(f"Removed {removed} invalid entries during format validation")
    
    if validate_only:
        print("\n*** VALIDATION ONLY MODE ***")
        stats = calculate_stats(entries)
        print(f"\nDataset Statistics:")
        print(f"  Total entries: {stats['count']}")
        print(f"  Avg instruction length: {stats['avg_instruction_words']:.1f} words")
        print(f"  Avg output length: {stats['avg_output_words']:.1f} words")
        print(f"  Output range: {stats['min_output_words']} - {stats['max_output_words']} words")
        return
    
    # Shuffle and split
    print(f"\nSplitting data: {TRAIN_SPLIT*100:.0f}% train / {(1-TRAIN_SPLIT)*100:.0f}% test")
    train_set, test_set = shuffle_and_split(entries)
    
    print(f"  Train set: {len(train_set)} entries")
    print(f"  Test set: {len(test_set)} entries")
    
    # Save files
    train_file = os.path.join(FINAL_DIR, "train.jsonl")
    test_file = os.path.join(FINAL_DIR, "test.jsonl")
    
    save_jsonl(train_set, train_file)
    save_jsonl(test_set, test_file)
    
    # Calculate and save statistics
    train_stats = calculate_stats(train_set)
    test_stats = calculate_stats(test_set)
    
    full_stats = {
        "train": train_stats,
        "test": test_stats,
        "total_entries": len(entries),
        "train_ratio": TRAIN_SPLIT,
        "test_ratio": 1 - TRAIN_SPLIT,
    }
    
    stats_file = os.path.join(FINAL_DIR, "_dataset_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(full_stats, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Dataset finalization complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Train: {train_file} ({len(train_set)} entries)")
    print(f"  Test:  {test_file} ({len(test_set)} entries)")
    print(f"\nDataset Statistics:")
    print(f"  Average question length: {train_stats['avg_instruction_words']:.1f} words")
    print(f"  Average answer length: {train_stats['avg_output_words']:.1f} words")
    print("\n" + "=" * 60)
    print("✓ Ready for Unsloth fine-tuning!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finalize dataset for Unsloth")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't split")
    args = parser.parse_args()
    
    main(validate_only=args.validate_only)
