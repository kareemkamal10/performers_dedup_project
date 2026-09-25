# Performers Image Dedup Pipeline

Merges each performer's `image` + `source_images` URLs, downloads them,
removes visual near-duplicates (keeping the best-quality copy of each),
excludes URLs that fail to download, and produces a final deduped JSON —
built to run on Kaggle against ~118k elements without blowing up the
session or the disk.

## What it does, stage by stage

1. **Merge & filter** (`merge_dedupe.py`) — for every element, combines
   `image` + `source_images` into one URL list, drops exact-duplicate URLs.
   Any element left with **fewer than 2 unique URLs is dropped entirely**,
   before anything is downloaded.

2. **Batching** — the kept elements are split into fixed batches of
   **10,000** each (`config.BATCH_SIZE`).

3. **Pipelined download + processing** (`pipeline.py`) — while batch *N* is
   being visually processed (CPU-bound), batch *N+1* is already downloading
   in the background, so it's ready the moment batch *N* finishes. Images
   are downloaded straight to `/kaggle/temp` (never `/kaggle/working`, which
   has a much smaller quota).

4. **Downloading** (`downloader.py`) — each URL gets up to 3 attempts with
   exponential backoff; after 3 failures it's recorded as failed and
   excluded. Concurrency auto-scales with `os.cpu_count()`.

5. **Visual near-duplicate detection** (`visual_dedup.py` + `quality.py`) —
   groups images per element via perceptual hash (pHash), then within each
   group keeps whichever copy scores highest on a resolution + sharpness
   (Laplacian variance) + file-size score. There's no minimum-margin rule —
   if one copy is even 1% better, it wins.

6. **WebP-candidate analysis** (`webp_analysis.py`) — for every surviving
   image, before it's deleted, checks whether it's already **WebP-Lossy**
   (by reading the RIFF container's VP8/VP8L sub-chunk, not just the file
   extension). Anything that isn't WebP-Lossy already — including
   WebP-Lossless files — counts as a conversion candidate. This is
   analysis only; no actual conversion is performed.

7. **Cleanup** — once a batch's final URLs and WebP stats are safely
   recorded, that batch's images are deleted from `/kaggle/temp` immediately
   (they are not needed/kept).

8. **Checkpointing** (`checkpoint.py` + `hf_sync.py`) — after every batch, a
   checkpoint (completed batch indices, results so far, failed downloads,
   running stats) is saved locally and — if HF credentials are set —
   mirrored to the HF dataset. If a Kaggle session dies mid-run, the next
   run picks the checkpoint back up and skips already-completed batches
   entirely (no re-downloading, no re-processing).

9. **Final outputs**, written to `result_output/` and (if HF credentials are
   set) uploaded to the HF dataset:
   - `performers_data_final.json` — `{"id": ..., "urls": [...]}` per kept
     element, final deduped URLs only.
   - `report.txt` — full run stats (merge/download/dedupe counts, WebP
     candidate count + size, total final size, run duration).
   - `failed_downloads.csv` — every URL that failed after 3 retries
     (`id, url, error`).

## Running it

### On Kaggle (recommended)
Open `kaggle_cell.py`, copy its **entire contents** into a single Kaggle
notebook cell, fill in the 3 placeholders at the top
(`GITHUB_REPO_URL`, `HF_TOKEN` — `HF_DATASET_REPO_ID` is already set to
`abdelwahabnabil500/datafile`), and run the cell. It clones this repo,
installs dependencies, and runs the whole pipeline — including fetching
`performers_data.json` from the dataset and uploading `result_output/`
back to it at the end.

### Manually / locally
Drop `performers_data.json` in the project root, then:

```bash
pip install -r requirements.txt
python main.py
```

HF credentials are optional in this mode — without them, the input file
must already be present locally, and results are only written locally
under `result_output/` (no auto-upload, no remote checkpoint resume).

## Tuning knobs (all in `config.py`)

| Setting | What it controls |
|---|---|
| `BATCH_SIZE` | Elements per batch (fixed at 10,000 per spec) |
| `DOWNLOAD_MAX_RETRIES` | Attempts per URL before marking it failed |
| `MAX_DOWNLOAD_WORKERS_CAP` / `MAX_PROCESS_WORKERS_CAP` | Hard ceilings for auto-detected concurrency |
| `PHASH_HAMMING_THRESHOLD` | How close two pHashes must be to count as the same photo |
| `QUALITY_WEIGHT_*` | How resolution / sharpness / file size are weighted when ranking duplicates |

## Project layout

```
config.py           - all tunable settings in one place
concurrency.py       - auto-detects safe worker counts
merge_dedupe.py      - stage 1: merge + dedupe URLs, drop single-image elements
downloader.py        - stage 2: concurrent retrying downloader
quality.py           - resolution/sharpness/size scoring
visual_dedup.py       - stage 3: pHash grouping + best-copy selection
webp_analysis.py      - WebP-Lossy vs Lossless/other detection for the report
checkpoint.py         - batch-level local checkpoint read/write
hf_sync.py            - HuggingFace dataset download/upload helpers
report.py             - builds report.txt and failed_downloads.csv
pipeline.py           - orchestrates batching + the download/process overlap
main.py               - entry point
kaggle_cell.py        - the single Kaggle notebook cell to paste and run
```
