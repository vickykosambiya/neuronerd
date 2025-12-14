"""
04_quality_control.py - Data Quality Filtering

Filters generated Q&A data to remove:
- Invalid JSON entries
- Short/lazy answers
- AI refusals
- Duplicates
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from config import (
    GENERATED_DIR,
    CLEANED_DIR,
    MIN_ANSWER_WORDS,
    MAX_ANSWER_WORDS,
    REFUSAL_PHRASES,
)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file, handling malformed lines."""
    valid_entries = []
    invalid_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                valid_entries.append(entry)
            except json.JSONDecodeError:
                invalid_count += 1
    
    return valid_entries, invalid_count


def validate_fields(entries: List[Dict]) -> tuple[List[Dict], int]:
    """Ensure all entries have required fields."""
    valid = []
    removed = 0
    
    for entry in entries:
        if isinstance(entry, dict):
            if "instruction" in entry and "output" in entry:
                # Ensure input field exists
                if "input" not in entry:
                    entry["input"] = ""
                valid.append(entry)
            else:
                removed += 1
        else:
            removed += 1
    
    return valid, removed


def filter_short_answers(entries: List[Dict], min_words: int = MIN_ANSWER_WORDS) -> tuple[List[Dict], int]:
    """Remove entries with answers that are too short."""
    valid = []
    removed = 0
    
    for entry in entries:
        output = entry.get("output", "")
        word_count = len(output.split())
        
        if word_count >= min_words:
            valid.append(entry)
        else:
            removed += 1
    
    return valid, removed


def filter_long_answers(entries: List[Dict], max_words: int = MAX_ANSWER_WORDS) -> tuple[List[Dict], int]:
    """Remove entries with answers that are suspiciously long (may be errors)."""
    valid = []
    removed = 0
    
    for entry in entries:
        output = entry.get("output", "")
        word_count = len(output.split())
        
        if word_count <= max_words:
            valid.append(entry)
        else:
            removed += 1
    
    return valid, removed


def filter_refusals(entries: List[Dict], phrases: List[str] = REFUSAL_PHRASES) -> tuple[List[Dict], int]:
    """Remove entries containing AI refusal phrases."""
    valid = []
    removed = 0
    
    for entry in entries:
        instruction = entry.get("instruction", "").lower()
        output = entry.get("output", "").lower()
        combined = instruction + " " + output
        
        is_refusal = False
        for phrase in phrases:
            if phrase.lower() in combined:
                is_refusal = True
                break
        
        if not is_refusal:
            valid.append(entry)
        else:
            removed += 1
    
    return valid, removed


def filter_text_references(entries: List[Dict]) -> tuple[List[Dict], int]:
    """Remove entries that reference 'the text' or 'the passage'."""
    reference_patterns = [
        r'\baccording to the (text|passage|excerpt)\b',
        r'\bin (this|the) (text|passage|excerpt)\b',
        r'\bthe (text|passage|excerpt) (states|mentions|describes|explains)\b',
        r'\bas (stated|mentioned|described) in the (text|passage)\b',
    ]
    
    valid = []
    removed = 0
    
    for entry in entries:
        instruction = entry.get("instruction", "")
        output = entry.get("output", "")
        combined = instruction + " " + output
        
        has_reference = False
        for pattern in reference_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                has_reference = True
                break
        
        if not has_reference:
            valid.append(entry)
        else:
            removed += 1
    
    return valid, removed


def deduplicate(entries: List[Dict]) -> tuple[List[Dict], int]:
    """Remove duplicate questions based on normalized instruction text."""
    seen_instructions: Set[str] = set()
    unique = []
    removed = 0
    
    for entry in entries:
        instruction = entry.get("instruction", "")
        # Normalize: lowercase, remove extra whitespace, remove punctuation
        normalized = re.sub(r'[^\w\s]', '', instruction.lower())
        normalized = ' '.join(normalized.split())
        
        if normalized and normalized not in seen_instructions:
            seen_instructions.add(normalized)
            unique.append(entry)
        else:
            removed += 1
    
    return unique, removed


def remove_source_metadata(entries: List[Dict]) -> List[Dict]:
    """Remove internal metadata fields from entries."""
    cleaned = []
    for entry in entries:
        clean_entry = {
            "instruction": entry.get("instruction", ""),
            "input": entry.get("input", ""),
            "output": entry.get("output", ""),
        }
        cleaned.append(clean_entry)
    return cleaned


def main():
    """Main quality control function."""
    print("=" * 60)
    print("NeuroNerd Dataset Pipeline - Step 4: Quality Control")
    print("=" * 60)
    
    # Setup
    os.makedirs(CLEANED_DIR, exist_ok=True)
    
    # Find generated file
    generated_file = os.path.join(GENERATED_DIR, "generated_qa.jsonl")
    
    if not os.path.exists(generated_file):
        print(f"\nNo generated data found at '{generated_file}'!")
        print("Please run 03_generate_qa.py first.")
        return
    
    print(f"\nLoading data from: {generated_file}")
    
    # Load data
    entries, json_errors = load_jsonl(generated_file)
    initial_count = len(entries)
    
    print(f"\nInitial entries: {initial_count}")
    print(f"JSON parse errors: {json_errors}")
    
    stats = {
        "initial": initial_count,
        "json_errors": json_errors,
    }
    
    # Apply filters
    print("\n" + "-" * 60)
    print("Applying quality filters...")
    
    # 1. Validate fields
    entries, removed = validate_fields(entries)
    stats["missing_fields_removed"] = removed
    print(f"  Missing fields removed: {removed}")
    
    # 2. Filter short answers
    entries, removed = filter_short_answers(entries)
    stats["short_answers_removed"] = removed
    print(f"  Short answers removed (<{MIN_ANSWER_WORDS} words): {removed}")
    
    # 3. Filter long answers
    entries, removed = filter_long_answers(entries)
    stats["long_answers_removed"] = removed
    print(f"  Long answers removed (>{MAX_ANSWER_WORDS} words): {removed}")
    
    # 4. Filter refusals
    entries, removed = filter_refusals(entries)
    stats["refusals_removed"] = removed
    print(f"  AI refusals removed: {removed}")
    
    # 5. Filter text references
    entries, removed = filter_text_references(entries)
    stats["text_references_removed"] = removed
    print(f"  Text references removed: {removed}")
    
    # 6. Deduplicate
    entries, removed = deduplicate(entries)
    stats["duplicates_removed"] = removed
    print(f"  Duplicates removed: {removed}")
    
    # Remove internal metadata
    entries = remove_source_metadata(entries)
    
    # Final stats
    stats["final_count"] = len(entries)
    stats["total_removed"] = initial_count - len(entries)
    stats["retention_rate"] = len(entries) / initial_count * 100 if initial_count > 0 else 0
    
    print("\n" + "-" * 60)
    print(f"Final entry count: {len(entries)}")
    print(f"Retention rate: {stats['retention_rate']:.1f}%")
    
    # Save cleaned data
    output_file = os.path.join(CLEANED_DIR, "cleaned_qa.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Save stats
    stats_file = os.path.join(CLEANED_DIR, "_quality_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Quality control complete!")
    print(f"Cleaned data saved to: {output_file}")
    print(f"Statistics saved to: {stats_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
