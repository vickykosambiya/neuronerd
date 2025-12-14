# NeuroNerd Dataset Generation Pipeline

# Directories
PDF_DIR = "pdfs"
OUTPUT_DIR = "output"
EXTRACTED_DIR = f"{OUTPUT_DIR}/extracted"
CHUNKS_DIR = f"{OUTPUT_DIR}/chunks"
GENERATED_DIR = f"{OUTPUT_DIR}/generated"
CLEANED_DIR = f"{OUTPUT_DIR}/cleaned"
FINAL_DIR = f"{OUTPUT_DIR}/final"

# Chunking settings
CHUNK_SIZE_WORDS = 700  # Target words per chunk
CHUNK_OVERLAP_WORDS = 100  # Overlap between chunks (~15%)

# Gemini API settings (Paid Tier - High Performance)
GEMINI_MODEL = "gemini-2.5-flash-lite"  # Using 2.5-flash-lite as requested
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 2
REQUESTS_PER_MINUTE = 1000  # Paid tier limit (much higher than free tier's 15)

# Quality control thresholds
MIN_ANSWER_WORDS = 20  # Minimum words in answer
MAX_ANSWER_WORDS = 500  # Maximum words in answer

# Refusal phrases to filter out
REFUSAL_PHRASES = [
    "i cannot answer",
    "i can't answer",
    "i don't have information",
    "i do not have information",
    "as an ai language model",
    "as an ai",
    "based on the text provided",
    "according to the text",
    "in this passage",
    "the passage mentions",
    "the text states",
    "i'm unable to",
    "i am unable to",
    "i cannot provide",
    "i can't provide",
]

# Train/test split
TRAIN_SPLIT = 0.95  # 95% for training
TEST_SPLIT = 0.05   # 5% for testing

# Q&A generation settings
QA_PER_CHUNK = 4  # Target Q&A pairs per chunk
