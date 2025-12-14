"""
02_preprocess.py - Text Cleaning and Semantic Chunking

Cleans extracted text (removes headers/footers, fixes hyphenation)
and splits into overlapping chunks for LLM processing.
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from config import (
    EXTRACTED_DIR, 
    CHUNKS_DIR, 
    CHUNK_SIZE_WORDS, 
    CHUNK_OVERLAP_WORDS
)


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing noise and fixing formatting issues.
    
    Operations:
    1. Remove page markers
    2. Remove headers/footers (repeated lines)
    3. Fix hyphenation
    4. Normalize whitespace
    5. Remove figure/table references artifacts
    """
    
    # Remove page markers we added during extraction
    text = re.sub(r'\n--- PAGE \d+ ---\n', '\n', text)
    
    # Fix hyphenation (words broken across lines)
    # Pattern: word- \n continuation
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove common header/footer patterns
    patterns_to_remove = [
        # Page numbers
        r'^\s*\d+\s*$',
        r'^\s*Page\s+\d+\s*$',
        r'^\s*-\s*\d+\s*-\s*$',
        
        # Chapter headers that repeat
        r'^Chapter\s+\d+\s*$',
        
        # Common footer text
        r'^\s*©.*$',
        r'^\s*Copyright.*$',
        
        # URLs and emails
        r'https?://\S+',
        
        # Figure/Table references artifacts
        r'^\s*Figure\s+\d+[\.\d]*\s*$',
        r'^\s*Table\s+\d+[\.\d]*\s*$',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Normalize multiple newlines to double newlines (paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove empty lines at start/end
    text = text.strip()
    
    return text


def find_sentence_boundary(text: str, position: int, search_range: int = 200) -> int:
    """
    Find the nearest sentence boundary near the given position.
    
    Tries to find a period, exclamation, or question mark followed by space.
    Returns the position after the sentence-ending punctuation.
    """
    # Search forward for sentence boundary
    end_search = min(position + search_range, len(text))
    
    for i in range(position, end_search):
        if i < len(text) - 1:
            if text[i] in '.!?' and text[i + 1] in ' \n':
                return i + 1
    
    # If no sentence boundary found, try searching backward
    start_search = max(position - search_range, 0)
    
    for i in range(position, start_search, -1):
        if i < len(text) - 1:
            if text[i] in '.!?' and text[i + 1] in ' \n':
                return i + 1
    
    # Fallback to original position
    return position


def create_chunks_with_overlap(
    text: str, 
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS
) -> List[Dict]:
    """
    Split text into overlapping chunks for LLM processing.
    
    Args:
        text: The cleaned text to chunk
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return []
    
    chunks = []
    current_pos = 0
    chunk_index = 0
    
    while current_pos < total_words:
        # Calculate end position
        end_pos = min(current_pos + chunk_size, total_words)
        
        # Get the chunk text
        chunk_words = words[current_pos:end_pos]
        chunk_text = ' '.join(chunk_words)
        
        # Try to find a good sentence boundary
        if end_pos < total_words:
            # Reconstruct to find character position
            text_so_far = ' '.join(words[:end_pos])
            char_pos = len(text_so_far)
            
            # Find sentence boundary in original text
            boundary = find_sentence_boundary(text, char_pos)
            
            # Recalculate chunk based on boundary
            adjusted_text = text[:boundary].split()
            if len(adjusted_text) > current_pos:
                actual_end = len(adjusted_text)
                if actual_end - current_pos >= chunk_size // 2:  # Don't make chunks too small
                    chunk_words = words[current_pos:actual_end]
                    chunk_text = ' '.join(chunk_words)
                    end_pos = actual_end
        
        chunks.append({
            'chunk_index': chunk_index,
            'text': chunk_text,
            'word_count': len(chunk_words),
            'start_word': current_pos,
            'end_word': end_pos,
        })
        
        chunk_index += 1
        
        # Move position forward, accounting for overlap
        current_pos = end_pos - overlap
        
        # Prevent infinite loop if overlap >= chunk_size
        if current_pos <= chunks[-1]['start_word']:
            current_pos = end_pos
    
    return chunks


def process_extracted_file(file_path: str, output_dir: str) -> Dict:
    """
    Process a single extracted text file into chunks.
    
    Returns metadata about the processing.
    """
    source_name = Path(file_path).stem
    
    # Read the extracted text
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Clean the text
    cleaned_text = clean_text(raw_text)
    
    # Create chunks
    chunks = create_chunks_with_overlap(cleaned_text)
    
    # Save each chunk as a separate JSON file
    chunk_files = []
    for chunk in chunks:
        chunk_data = {
            'source_file': source_name,
            'chunk_index': chunk['chunk_index'],
            'text': chunk['text'],
            'word_count': chunk['word_count'],
        }
        
        chunk_filename = f"{source_name}_chunk_{chunk['chunk_index']:04d}.json"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        chunk_files.append(chunk_filename)
    
    # Also save cleaned text for reference
    cleaned_path = os.path.join(output_dir, f"{source_name}_cleaned.txt")
    with open(cleaned_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    return {
        'source_file': source_name,
        'raw_words': len(raw_text.split()),
        'cleaned_words': len(cleaned_text.split()),
        'num_chunks': len(chunks),
        'chunk_files': chunk_files,
    }


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("NeuroNerd Dataset Pipeline - Step 2: Text Preprocessing")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    
    # Find all extracted text files
    extracted_files = list(Path(EXTRACTED_DIR).glob("*.txt"))
    
    if not extracted_files:
        print(f"\nNo extracted text files found in '{EXTRACTED_DIR}'!")
        print("Please run 01_extract_text.py first.")
        return
    
    print(f"\nFound {len(extracted_files)} extracted file(s)")
    print(f"Chunk size: {CHUNK_SIZE_WORDS} words")
    print(f"Overlap: {CHUNK_OVERLAP_WORDS} words (~{CHUNK_OVERLAP_WORDS/CHUNK_SIZE_WORDS*100:.0f}%)")
    print("\n" + "-" * 60)
    
    all_metadata = []
    total_chunks = 0
    
    for file_path in tqdm(extracted_files, desc="Processing files"):
        print(f"\nProcessing: {file_path.name}")
        
        metadata = process_extracted_file(str(file_path), CHUNKS_DIR)
        all_metadata.append(metadata)
        total_chunks += metadata['num_chunks']
        
        print(f"  Raw words: {metadata['raw_words']:,}")
        print(f"  Cleaned words: {metadata['cleaned_words']:,}")
        print(f"  Created {metadata['num_chunks']} chunks")
    
    # Save processing summary
    summary_path = os.path.join(CHUNKS_DIR, "_processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_chunks': total_chunks,
            'chunk_size_words': CHUNK_SIZE_WORDS,
            'overlap_words': CHUNK_OVERLAP_WORDS,
            'files_processed': all_metadata,
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Total chunks created: {total_chunks}")
    print(f"Output saved to: {CHUNKS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
