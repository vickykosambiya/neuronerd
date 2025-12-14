"""
03_generate_qa.py - Q&A Generation with Google Gemini (High Speed)

Processes text chunks through Gemini API using parallel threads
to maximize throughput on the paid tier (up to 1000 RPM).
"""

import os
import sys
import json
import time
import random
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

from config import (
    CHUNKS_DIR,
    GENERATED_DIR,
    GEMINI_MODEL,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    REQUESTS_PER_MINUTE,
    QA_PER_CHUNK,
)

# Load environment variables
load_dotenv()

# Parallel settings
MAX_WORKERS = 20  # Number of concurrent threads

# System prompt for Q&A generation
SYSTEM_PROMPT = """You are a neuroscience professor creating exam questions for graduate students.

Based ONLY on the provided text, generate {qa_count} high-quality Q&A pairs.

STRICT RULES:
1. Questions must be STANDALONE - NO phrases like "According to the text", "In this passage", "The author states", or any text references
2. Answers must be DETAILED (50-200 words) with scientific reasoning and explanations
3. Cover DIFFERENT question types: conceptual understanding, application, comparison, mechanism explanation
4. Use PROPER neuroscience terminology
5. Questions should test deep understanding, not just memorization
6. Answers should explain the 'why' and 'how', not just state facts

OUTPUT FORMAT (JSON array only, no other text):
[
  {{"instruction": "Clear, standalone question?", "input": "", "output": "Detailed answer with scientific reasoning..."}},
  ...
]

IMPORTANT: Return ONLY the JSON array. No markdown, no explanation, no additional text."""


def setup_gemini_client() -> genai.Client:
    """Initialize the Gemini client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or "your_" in api_key:
        print("\n" + "=" * 60)
        print("ERROR: Google API key not configured!")
        print("=" * 60)
        sys.exit(1)
    
    return genai.Client(api_key=api_key)


def generate_qa_for_chunk(
    client: genai.Client,
    chunk_text: str,
    qa_count: int = QA_PER_CHUNK
) -> Optional[List[Dict]]:
    """
    Generate Q&A pairs for a single chunk using Gemini.
    """
    prompt = SYSTEM_PROMPT.format(qa_count=qa_count)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"{prompt}\n\nTEXT TO PROCESS:\n{chunk_text}")]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=4096,
                )
            )
            
            # Extract the response text
            response_text = response.text.strip() if response.text else ""
            
            # Clean markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].strip() == "```": lines = lines[:-1]
                response_text = "\n".join(lines)
            
            # Parse JSON
            qa_pairs = json.loads(response_text)
            
            # Validate structure
            if isinstance(qa_pairs, list):
                valid_pairs = []
                for pair in qa_pairs:
                    if isinstance(pair, dict) and "instruction" in pair and "output" in pair:
                        if "input" not in pair: pair["input"] = ""
                        valid_pairs.append(pair)
                
                if valid_pairs:
                    return valid_pairs
            
        except Exception:
            # Silent retry
            pass
        
        # Exponential backoff
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY_SECONDS * (1.5 ** attempt) + random.uniform(0, 1))
    
    return None


def process_single_file(args: Tuple) -> Tuple[str, Optional[List[Dict]]]:
    """Worker function for parallel processing."""
    chunk_file, client_key = args
    
    # Create thread-local client (best practice)
    client = genai.Client(api_key=client_key)
    
    try:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
        
        chunk_text = chunk_data['text']
        qa_pairs = generate_qa_for_chunk(client, chunk_text)
        
        return (chunk_file.name, qa_pairs)
    except Exception:
        return (chunk_file.name, None)


def load_progress(progress_file: str) -> set:
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_progress(progress_file: str, processed: set):
    with open(progress_file, 'w') as f:
        json.dump(list(processed), f)


def main(dry_run: bool = False):
    """Main Q&A generation function."""
    print("=" * 60)
    print("NeuroNerd QC Pipeline - Step 3: High-Speed Generation")
    print(f"Model: {GEMINI_MODEL} | Workers: {MAX_WORKERS}")
    print("=" * 60)
    
    # Setup
    os.makedirs(GENERATED_DIR, exist_ok=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Find chunks
    all_chunks = sorted(Path(CHUNKS_DIR).glob("*_chunk_*.json"))
    
    if not all_chunks:
        print(f"No chunks found in {CHUNKS_DIR}")
        return
    
    if dry_run:
        print("*** DRY RUN: Processing 5 chunks only ***")
        all_chunks = random.sample(all_chunks, 5)
    
    # Load progress
    progress_file = os.path.join(GENERATED_DIR, "_progress.json")
    processed = load_progress(progress_file)
    remaining_chunks = [f for f in all_chunks if f.name not in processed]
    
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Remaining: {len(remaining_chunks)}")
    
    if not remaining_chunks:
        print("\nAll done!")
        return
        
    output_file = os.path.join(GENERATED_DIR, "generated_qa.jsonl")
    failed_chunks = []
    total_generated = 0
    
    # Rate limit interval
    # Even in parallel, we want to respect the 1000 RPM roughly
    # 20 workers * 3 sec/req = ~400 RPM (safe)
    
    print("\nStarting parallel generation...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Prepare arguments
        args = [(f, api_key) for f in remaining_chunks]
        
        # Submit tasks
        futures = {executor.submit(process_single_file, arg): arg[0] for arg in args}
        
        # Process results
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit="chunk"):
            chunk_name = futures[future]
            
            try:
                name, qa_pairs = future.result()
                
                if qa_pairs:
                    # Write immediately (thread-safe due to GIL + generic file I/O atomic nature usually enough for simple appends, 
                    # but simple append is fine here)
                    with open(output_file, 'a', encoding='utf-8') as f:
                        for pair in qa_pairs:
                            pair['_source_chunk'] = name
                            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                    
                    processed.add(chunk_name)
                    total_generated += len(qa_pairs)
                    
                    # Periodic save
                    if len(processed) % 50 == 0:
                        save_progress(progress_file, processed)
                else:
                    failed_chunks.append(chunk_name)
                    
            except Exception:
                failed_chunks.append(chunk_name)
    
    # Final save
    save_progress(progress_file, processed)
    
    # Report
    print("\n" + "=" * 60)
    print("Generation Completed!")
    print(f"Generated Q&A: {total_generated}")
    print(f"Failed chunks: {len(failed_chunks)}")
    print("=" * 60)
    
    if failed_chunks:
        with open(os.path.join(GENERATED_DIR, "_failed.json"), 'w') as f:
            json.dump(failed_chunks, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
