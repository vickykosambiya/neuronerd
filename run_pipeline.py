"""
run_pipeline.py - Main Orchestration Script

Runs the complete NeuroNerd dataset generation pipeline:
1. Extract text from PDFs
2. Preprocess and chunk text
3. Generate Q&A with Gemini
4. Quality control filtering
5. Finalize and split dataset
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_script(script_name: str, args: list = None) -> bool:
    """Run a pipeline script and return success status."""
    script_path = Path("scripts") / script_name
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Script failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="NeuroNerd Dataset Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all           Run complete pipeline
  python run_pipeline.py --step 1        Run only step 1 (extraction)
  python run_pipeline.py --step 3 --dry-run  Test Q&A generation with 5 chunks
  python run_pipeline.py --from 3        Run from step 3 onwards
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], help="Run specific step only")
    parser.add_argument("--from", dest="from_step", type=int, choices=[1, 2, 3, 4, 5], 
                       help="Run from this step onwards")
    parser.add_argument("--dry-run", action="store_true", help="For step 3: process only 5 chunks")
    
    args = parser.parse_args()
    
    # Determine which steps to run
    steps_to_run = []
    
    if args.all:
        steps_to_run = [1, 2, 3, 4, 5]
    elif args.step:
        steps_to_run = [args.step]
    elif args.from_step:
        steps_to_run = list(range(args.from_step, 6))
    else:
        parser.print_help()
        return
    
    # Script mapping
    scripts = {
        1: ("01_extract_text.py", "PDF Text Extraction", []),
        2: ("02_preprocess.py", "Text Preprocessing & Chunking", []),
        3: ("03_generate_qa.py", "Q&A Generation (Gemini High Speed)", ["--dry-run"] if args.dry_run else []),
        4: ("04_quality_control.py", "Quality Control Filtering", []),
        5: ("05_finalize.py", "Dataset Finalization", []),
    }
    
    print("\n" + "=" * 70)
    print("  NeuroNerd Dataset Generation Pipeline")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Steps to run: {steps_to_run}")
    if args.dry_run:
        print("  Mode: DRY RUN (step 3 will only process 5 chunks)")
    print("=" * 70 + "\n")
    
    # Move PDFs to pdfs/ folder if needed
    if 1 in steps_to_run:
        pdf_files = list(Path(".").glob("*.pdf"))
        if pdf_files:
            os.makedirs("pdfs", exist_ok=True)
            print("Moving PDF files to 'pdfs' folder...")
            for pdf in pdf_files:
                new_path = Path("pdfs") / pdf.name
                if not new_path.exists():
                    pdf.rename(new_path)
                    print(f"  Moved: {pdf.name}")
            print()
    
    # Run steps
    for step_num in steps_to_run:
        script, description, script_args = scripts[step_num]
        
        print(f"\n{'─' * 70}")
        print(f"  STEP {step_num}: {description}")
        print(f"{'─' * 70}\n")
        
        success = run_script(script, script_args)
        
        if not success:
            print(f"\n❌ Step {step_num} failed. Pipeline stopped.")
            return
        
        print(f"\n✓ Step {step_num} completed successfully")
    
    # Final summary
    print("\n" + "=" * 70)
    print("  Pipeline Complete!")
    print("=" * 70)
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if 5 in steps_to_run:
        train_file = Path("output/final/train.jsonl")
        test_file = Path("output/final/test.jsonl")
        
        if train_file.exists() and test_file.exists():
            train_count = sum(1 for _ in open(train_file, encoding='utf-8'))
            test_count = sum(1 for _ in open(test_file, encoding='utf-8'))
            
            print(f"\n  Output Files:")
            print(f"    Train: {train_file} ({train_count} entries)")
            print(f"    Test:  {test_file} ({test_count} entries)")
            print(f"\n  ✓ Dataset ready for Unsloth fine-tuning!")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
