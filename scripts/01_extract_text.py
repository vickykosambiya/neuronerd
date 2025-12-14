"""
01_extract_text.py - PDF Text Extraction Script

Extracts text from PDF files using pdfplumber with automatic
detection and filtering of non-content pages (ToC, Index, etc.)
"""

import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pdfplumber
from tqdm import tqdm

from config import PDF_DIR, EXTRACTED_DIR


def detect_non_content_pages(pdf_path: str) -> dict:
    """
    Automatically detect pages that are likely non-content
    (Table of Contents, Index, Bibliography, etc.)
    
    Returns dict with 'skip_start' and 'skip_end' page counts
    """
    non_content_patterns = {
        'toc': [
            r'table\s+of\s+contents',
            r'contents\s*$',
            r'^\s*chapter\s+\d+\s*\.{3,}',  # Chapter 1 ..... 23
        ],
        'index': [
            r'^\s*index\s*$',
            r'subject\s+index',
            r'author\s+index',
        ],
        'bibliography': [
            r'^\s*bibliography\s*$',
            r'^\s*references\s*$',
            r'^\s*works\s+cited\s*$',
        ],
        'preface': [
            r'^\s*preface\s*$',
            r'^\s*foreword\s*$',
            r'^\s*acknowledgments?\s*$',
        ],
        'glossary': [
            r'^\s*glossary\s*$',
        ]
    }
    
    skip_start = 0
    skip_end = 0
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        # Check first 30 pages for start content (ToC, preface, etc.)
        start_check = min(30, total_pages)
        for i in range(start_check):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            text_lower = text.lower()[:500]  # Check first 500 chars
            
            is_non_content = False
            for category, patterns in non_content_patterns.items():
                if category in ['toc', 'preface']:
                    for pattern in patterns:
                        if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                            is_non_content = True
                            break
            
            # Also check for pages with mostly dots (ToC style)
            if text.count('.') > 50 and len(text) < 2000:
                is_non_content = True
            
            if is_non_content:
                skip_start = i + 1
            elif i > 5 and skip_start > 0:
                # Found content after initial non-content section
                break
        
        # Check last 50 pages for end content (Index, Bibliography, etc.)
        end_check = min(50, total_pages - skip_start)
        for i in range(total_pages - 1, total_pages - end_check - 1, -1):
            if i < skip_start:
                break
                
            page = pdf.pages[i]
            text = page.extract_text() or ""
            text_lower = text.lower()[:500]
            
            is_non_content = False
            for category, patterns in non_content_patterns.items():
                if category in ['index', 'bibliography', 'glossary']:
                    for pattern in patterns:
                        if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                            is_non_content = True
                            break
            
            if is_non_content:
                skip_end = total_pages - i
            elif skip_end > 0:
                break
    
    return {
        'skip_start': skip_start,
        'skip_end': skip_end,
        'total_pages': total_pages,
        'content_pages': total_pages - skip_start - skip_end
    }


def extract_text_from_pdf(pdf_path: str, skip_start: int = 0, skip_end: int = 0) -> str:
    """
    Extract text from a PDF file, skipping specified pages.
    
    Args:
        pdf_path: Path to the PDF file
        skip_start: Number of pages to skip from start
        skip_end: Number of pages to skip from end
    
    Returns:
        Extracted text as a string
    """
    all_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        end_page = total_pages - skip_end
        
        # Create progress bar for page extraction
        pages_to_process = range(skip_start, end_page)
        
        for page_num in tqdm(pages_to_process, desc=f"Extracting pages", leave=False):
            page = pdf.pages[page_num]
            text = page.extract_text()
            
            if text:
                # Add page marker for debugging (can be removed in preprocessing)
                all_text.append(f"\n--- PAGE {page_num + 1} ---\n")
                all_text.append(text)
    
    return "\n".join(all_text)


def save_extraction_metadata(pdf_name: str, metadata: dict, output_dir: str):
    """Save extraction metadata to JSON file."""
    metadata_path = os.path.join(output_dir, f"{pdf_name}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Main extraction function."""
    print("=" * 60)
    print("NeuroNerd Dataset Pipeline - Step 1: PDF Text Extraction")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    
    if not pdf_files:
        # Also check root directory for PDFs
        pdf_files = list(Path(".").glob("*.pdf"))
        if pdf_files:
            print(f"\nFound {len(pdf_files)} PDF(s) in root directory.")
            print("Moving them to the 'pdfs' folder...")
            os.makedirs(PDF_DIR, exist_ok=True)
            for pdf in pdf_files:
                new_path = Path(PDF_DIR) / pdf.name
                pdf.rename(new_path)
            pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print(f"\nNo PDF files found in '{PDF_DIR}' directory!")
        print("Please place your PDF files in the 'pdfs' folder and run again.")
        return
    
    print(f"\nFound {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    print("\n" + "-" * 60)
    
    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem
        print(f"\nProcessing: {pdf_path.name}")
        
        # Auto-detect non-content pages
        print("  Analyzing document structure...")
        detection = detect_non_content_pages(str(pdf_path))
        
        print(f"  Total pages: {detection['total_pages']}")
        print(f"  Skipping first {detection['skip_start']} pages (ToC/Preface)")
        print(f"  Skipping last {detection['skip_end']} pages (Index/Bibliography)")
        print(f"  Content pages to extract: {detection['content_pages']}")
        
        # Extract text
        print("  Extracting text...")
        text = extract_text_from_pdf(
            str(pdf_path),
            skip_start=detection['skip_start'],
            skip_end=detection['skip_end']
        )
        
        # Save extracted text
        output_path = os.path.join(EXTRACTED_DIR, f"{pdf_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Save metadata
        metadata = {
            'source_pdf': pdf_path.name,
            'extraction_date': datetime.now().isoformat(),
            'total_pages': detection['total_pages'],
            'pages_skipped_start': detection['skip_start'],
            'pages_skipped_end': detection['skip_end'],
            'content_pages': detection['content_pages'],
            'extracted_chars': len(text),
            'extracted_words': len(text.split()),
        }
        save_extraction_metadata(pdf_name, metadata, EXTRACTED_DIR)
        
        print(f"  ✓ Saved to: {output_path}")
        print(f"  ✓ Extracted {metadata['extracted_words']:,} words")
    
    print("\n" + "=" * 60)
    print("Text extraction complete!")
    print(f"Output saved to: {EXTRACTED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
