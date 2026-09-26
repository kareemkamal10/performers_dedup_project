"""
Extra pass over the elements that the MAIN pipeline excluded at the merge
stage (0 or 1 unique URL after merging image+source_images - so nothing to
visually dedupe). Downloads each one's single image (if it has one) and
checks whether it's already WebP-Lossy, producing `single_report.txt`
scoped only to these elements. No near-duplicate detection here - each
element has at most 1 image, so there's nothing to compare against.

Usage: run this AFTER the main pipeline has already produced
result_output/performers_data_final.json (locally or on the HF dataset).
"""
from __future__ import annotations

import csv
import gc
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import hf_sync
from concurrency import get_download_workers
from downloader import _download_one
from merge_dedupe import collect_urls
from report import _human_size
from webp_analysis import analyze_webp_candidates

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("single_image_check")

SINGLE_TEMP_DIR = config.KAGGLE_TEMP_DIR + "_single"
SINGLE_CHECKPOINT_PATH = os.path.join(config.KAGGLE_WORKING_DIR, "single_checkpoint.json")
SINGLE_REPORT_PATH = os.path.join(config.RESULT_OUTPUT_DIR, "single_report.txt")
SINGLE_FAILED_CSV_PATH = os.path.join(config.RESULT_OUTPUT_DIR, "single_failed_downloads.csv")


def load_checkpoint() -> dict:
    if os.path.exists(SINGLE_CHECKPOINT_PATH):
        with open(SINGLE_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_batches": [], "stats": {}, "failed": []}


def save_checkpoint(state: dict) -> None:
    os.makedirs(os.path.dirname(SINGLE_CHECKPOINT_PATH), exist_ok=True)
    tmp = SINGLE_CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, SINGLE_CHECKPOINT_PATH)


def get_single_url_targets(raw_elements: list, final_ids: set) -> list:
    """Elements with exactly 1 unique URL whose id is NOT already present in
    the main pipeline's final output - i.e. exactly what the main run
    excluded at the merge stage. Elements with 0 urls have nothing to check
    and are skipped (they were also excluded, but there's no image at all)."""
    out = []
    for el in raw_elements:
        eid = el.get("id")
        if eid in final_ids:
            continue
        seen, unique = set(), []
        for u in collect_urls(el):
            if u not in seen:
                seen.add(u)
                unique.append(u)
        if len(unique) == 1:
            out.append({"id": eid, "url": unique[0]})
    return out


def download_batch(elements: list):
    os.makedirs(SINGLE_TEMP_DIR, exist_ok=True)
    workers = get_download_workers()
    downloaded_paths, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for el in elements:
            dest_dir = os.path.join(SINGLE_TEMP_DIR, str(el["id"]))
            os.makedirs(dest_dir, exist_ok=True)
            futures[pool.submit(_download_one, el["id"], el["url"], dest_dir, 0)] = el
        for fut in as_completed(futures):
            result = fut.result()
            if result["ok"]:
                downloaded_paths.append(result["path"])
            else:
                failed.append({"id": result["id"], "url": result["url"], "error": result["error"]})
    return downloaded_paths, failed


def run() -> None:
    started_at = time.time()
    os.makedirs(config.RESULT_OUTPUT_DIR, exist_ok=True)
    use_hf = bool(config.HF_DATASET_REPO_ID and config.HF_TOKEN)

    raw_path = config.INPUT_JSON
    if not os.path.exists(raw_path):
        if not use_hf:
            raise RuntimeError("performers_data.json not found and no HF credentials are set.")
        logger.info("performers_data.json not found locally - fetching from HF dataset...")
        raw_path = hf_sync.download_input_json(raw_path)

    final_path = os.path.join(config.RESULT_OUTPUT_DIR, "performers_data_final.json")
    if not os.path.exists(final_path):
        if not use_hf:
            raise RuntimeError(
                "result_output/performers_data_final.json not found and no HF credentials are set."
            )
        logger.info("performers_data_final.json not found locally - fetching from HF dataset...")
        final_path = hf_sync.download_final_json(final_path)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_elements = json.load(f)
    with open(final_path, "r", encoding="utf-8") as f:
        final_elements = json.load(f)
    final_ids = {item["id"] for item in final_elements}

    targets = get_single_url_targets(raw_elements, final_ids)
    logger.info("Single-url elements to check (excluded from the main run): %d", len(targets))

    batches = [targets[i : i + config.BATCH_SIZE] for i in range(0, len(targets), config.BATCH_SIZE)]

    state = load_checkpoint()
    completed = set(state["completed_batches"])
    failed_all = list(state["failed"])
    agg = {
        "total_targets": len(targets),
        "total_downloaded": 0,
        "total_failed": len(failed_all),
        "webp_candidate_count": 0,
        "webp_candidate_bytes": 0,
        "all_images_bytes": 0,
    }
    agg.update(state.get("stats", {}))

    for i, batch in enumerate(batches):
        if i in completed:
            continue
        logger.info("=== Single-image batch %d/%d (%d elements) ===", i + 1, len(batches), len(batch))

        downloaded_paths, batch_failed = download_batch(batch)
        failed_all.extend(batch_failed)
        agg["total_failed"] += len(batch_failed)
        agg["total_downloaded"] += len(downloaded_paths)

        webp_stats = analyze_webp_candidates(downloaded_paths)
        agg["webp_candidate_count"] += webp_stats["candidate_count"]
        agg["webp_candidate_bytes"] += webp_stats["candidate_total_bytes"]
        agg["all_images_bytes"] += webp_stats["all_images_total_bytes"]

        completed.add(i)
        for el in batch:
            shutil.rmtree(os.path.join(SINGLE_TEMP_DIR, str(el["id"])), ignore_errors=True)
        gc.collect()

        state = {"completed_batches": sorted(completed), "stats": agg, "failed": failed_all}
        save_checkpoint(state)
        if use_hf:
            try:
                hf_sync.upload_file_generic(SINGLE_CHECKPOINT_PATH, "single_checkpoint.json")
            except Exception as e:  # noqa: BLE001
                logger.warning("Checkpoint upload failed (will retry next batch): %s", e)

    os.makedirs(os.path.dirname(SINGLE_FAILED_CSV_PATH), exist_ok=True)
    with open(SINGLE_FAILED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "url", "error"])
        writer.writeheader()
        for row in failed_all:
            writer.writerow(row)

    duration_min = (time.time() - started_at) / 60
    lines = [
        "=" * 62,
        "Single-Image Elements - WebP Candidate Report",
        "(elements the main pipeline excluded: 0 or 1 unique url)",
        "=" * 62,
        f"Run duration:                          {duration_min:.1f} minutes",
        "",
        f"Total single-url elements considered:    {agg['total_targets']}",
        f"Successful downloads:                    {agg['total_downloaded']}",
        f"Failed downloads (after 3 retries):       {agg['total_failed']}",
        "",
        "-- WebP conversion candidates (analysis only, no conversion done) --",
        f"Images not already WebP-Lossy:            {agg['webp_candidate_count']}",
        f"Total size of those images:               {_human_size(agg['webp_candidate_bytes'])}",
        f"Total size of ALL checked images:         {_human_size(agg['all_images_bytes'])}",
        "=" * 62,
    ]
    with open(SINGLE_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Done. single_report.txt written to %s", SINGLE_REPORT_PATH)

    if use_hf:
        hf_sync.upload_file_generic(SINGLE_REPORT_PATH, "result_output/single_report.txt")
        hf_sync.upload_file_generic(SINGLE_FAILED_CSV_PATH, "result_output/single_failed_downloads.csv")


if __name__ == "__main__":
    run()
