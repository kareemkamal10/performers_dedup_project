"""
Central configuration for the performers image dedup pipeline.
Edit values here to tune behavior; nothing here should require code changes
in the other modules.
"""
import os

# ---- Paths ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(PROJECT_ROOT, "performers_data.json")

# Kaggle has two writable areas: /kaggle/working (persists as notebook output,
# small quota) and /kaggle/temp (much larger, wiped when the session ends).
# All downloaded images MUST live under KAGGLE_TEMP_DIR per project spec.
KAGGLE_TEMP_DIR = "/kaggle/temp/performers_images"
KAGGLE_WORKING_DIR = "/kaggle/working"

RESULT_OUTPUT_DIR = os.path.join(KAGGLE_WORKING_DIR, "result_output")
CHECKPOINT_PATH = os.path.join(KAGGLE_WORKING_DIR, "checkpoint.json")

# ---- Batching ----
BATCH_SIZE = 10_000  # fixed per project spec, not meant to be changed casually

# ---- Downloading ----
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_TIMEOUT_SECONDS = 20
DOWNLOAD_RETRY_BACKOFF_BASE = 1.5  # seconds; backoff = base * 2**(attempt-1)

# ---- Concurrency (auto-detected at runtime; capped here for safety) ----
# Downloading is I/O-bound -> many more workers than CPU cores is fine.
MAX_DOWNLOAD_WORKERS_CAP = 64
# Visual dedup / quality scoring is CPU-bound (image decode, hashing, Laplacian).
# Kept modest so a Kaggle session (few cores) never gets starved/killed.
MAX_PROCESS_WORKERS_CAP = 8

# ---- Visual near-duplicate detection ----
# Perceptual hash (pHash) Hamming-distance threshold below which two images are
# considered the same underlying photo (different size/compression/quality).
# This threshold decides WHETHER two images are duplicates; it is unrelated to
# the quality-ranking step, which always keeps the best of a group with no
# minimum margin required.
PHASH_HASH_SIZE = 8               # standard 8x8 -> 64-bit hash
PHASH_HAMMING_THRESHOLD = 10      # out of 64 bits; loosened from the textbook
                                   # default of 6 after testing - 6 misses some
                                   # same-photo pairs under aggressive downscale
                                   # + heavy recompression, which is exactly the
                                   # scenario this project needs to catch

# ---- Quality scoring weights (used only to RANK images already determined to
# be duplicates of each other; never used to decide duplicate-ness itself) ----
QUALITY_WEIGHT_RESOLUTION = 0.45
QUALITY_WEIGHT_SHARPNESS = 0.45
QUALITY_WEIGHT_FILESIZE = 0.10

# ---- HuggingFace dataset (private) ----
# Filled in from the Kaggle cell placeholders at runtime via env vars.
HF_DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ---- Logging ----
LOG_LEVEL = "INFO"
