"""
Orchestrates the whole run:
  - splits kept elements into fixed-size batches
  - while batch i is being visually processed, batch i+1 is already downloading
    in the background (so it's ready the moment batch i finishes)
  - deletes each batch's images from /kaggle/temp the moment everything needed
    from them (final URLs + webp stats) has been safely recorded
  - checkpoints (and optionally uploads the checkpoint to HF) after every batch
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import config
import hf_sync
from checkpoint import load_checkpoint, save_checkpoint
from concurrency import get_process_workers
from downloader import download_batch
from merge_dedupe import merge_and_filter
from visual_dedup import dedupe_element_images
from webp_analysis import analyze_webp_candidates

logger = logging.getLogger(__name__)


def _make_batches(elements: list, size: int):
    for i in range(0, len(elements), size):
        yield elements[i : i + size]


def run(input_json_path: str, use_hf: bool = True):
    started_at = time.time()
    os.makedirs(config.RESULT_OUTPUT_DIR, exist_ok=True)

    with open(input_json_path, "r", encoding="utf-8") as f:
        raw_elements = json.load(f)

    kept_elements, merge_stats = merge_and_filter(raw_elements)
    batches = list(_make_batches(kept_elements, config.BATCH_SIZE))
    logger.info("Split into %d batches of up to %d elements", len(batches), config.BATCH_SIZE)

    state = load_checkpoint()
    completed = set(state["completed_batches"])
    results_by_id = {r["id"]: r["urls"] for r in state["results"]}
    failed_downloads = list(state["failed_downloads"])

    agg = {
        "total_urls_attempted": 0,
        "total_urls_downloaded": 0,
        "total_urls_failed": len(failed_downloads),
        "duplicates_removed": 0,
        "unreadable_excluded": 0,
        "final_images_kept": 0,
        "webp_candidate_count": 0,
        "webp_candidate_bytes": 0,
        "all_images_bytes": 0,
    }
    agg.update(state.get("stats", {}))

    process_workers = get_process_workers()
    # A single-worker pool just for kicking off/awaiting whole-batch downloads,
    # so "download batch i+1" can run in the background while we process batch i.
    downloader_pool = ThreadPoolExecutor(max_workers=1)

    def kick_off_download(batch_idx):
        if batch_idx >= len(batches):
            return None
        return downloader_pool.submit(download_batch, batches[batch_idx])

    # Find the first not-yet-completed batch so we don't re-download batches
    # that a previous, interrupted run already finished.
    first_pending = next((i for i in range(len(batches)) if i not in completed), len(batches))
    prefetch_future = kick_off_download(first_pending)

    for i, batch in enumerate(batches):
        if i in completed:
            continue

        logger.info("=== Batch %d/%d (%d elements) ===", i + 1, len(batches), len(batch))

        downloaded_map, batch_failed = prefetch_future.result()
        failed_downloads.extend(batch_failed)
        agg["total_urls_failed"] += len(batch_failed)

        # Kick off the NEXT batch's download now, so it overlaps this batch's
        # (CPU-bound) visual processing below.
        prefetch_future = kick_off_download(i + 1)

        batch_urls_attempted = sum(len(el["urls"]) for el in batch)
        agg["total_urls_attempted"] += batch_urls_attempted
        agg["total_urls_downloaded"] += sum(len(v) for v in downloaded_map.values())

        final_paths_this_batch = []
        path_to_url_by_eid = {}
        futures = {}
        with ProcessPoolExecutor(max_workers=process_workers) as pool:
            for eid, url_path_list in downloaded_map.items():
                if not url_path_list:
                    continue
                paths = [p for _, p in url_path_list]
                path_to_url_by_eid[eid] = {p: u for u, p in url_path_list}
                futures[pool.submit(dedupe_element_images, paths)] = eid

            for fut in as_completed(futures):
                eid = futures[fut]
                kept_paths, dup_removed, unreadable = fut.result()
                agg["duplicates_removed"] += dup_removed
                agg["unreadable_excluded"] += unreadable
                agg["final_images_kept"] += len(kept_paths)
                if kept_paths:
                    kept_urls = [path_to_url_by_eid[eid][p] for p in kept_paths]
                    results_by_id[eid] = kept_urls
                    final_paths_this_batch.extend(kept_paths)

        # WebP candidate analysis on this batch's surviving images, BEFORE
        # deleting them - this is the only point they're inspected for it.
        webp_stats = analyze_webp_candidates(final_paths_this_batch)
        agg["webp_candidate_count"] += webp_stats["candidate_count"]
        agg["webp_candidate_bytes"] += webp_stats["candidate_total_bytes"]
        agg["all_images_bytes"] += webp_stats["all_images_total_bytes"]

        completed.add(i)

        # Everything needed from this batch's files (final URL lists + webp
        # stats) is now safely recorded -> delete the images to free space.
        for el in batch:
            shutil.rmtree(
                os.path.join(config.KAGGLE_TEMP_DIR, str(el["id"])), ignore_errors=True
            )
        gc.collect()

        state = {
            "completed_batches": sorted(completed),
            "results": [{"id": k, "urls": v} for k, v in results_by_id.items()],
            "failed_downloads": failed_downloads,
            "stats": agg,
        }
        save_checkpoint(state)
        if use_hf:
            try:
                hf_sync.upload_checkpoint()
            except Exception as e:  # noqa: BLE001 - don't kill the run over this
                logger.warning("Checkpoint upload failed (will retry next batch): %s", e)

    downloader_pool.shutdown(wait=True)

    return results_by_id, failed_downloads, agg, merge_stats, started_at
