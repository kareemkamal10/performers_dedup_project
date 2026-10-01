"""
Downloads scenes/scenes.json images and reports WebP-Lossy conversion candidates.
No visual deduplication or conversion is performed.
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
from report import _human_size
from webp_analysis import analyze_webp_candidates

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scenes_check")

SCENES_DIR = os.path.join(config.PROJECT_ROOT, "scenes")
SCENES_INPUT_PATH = os.path.join(SCENES_DIR, "scenes.json")
SCENES_TEMP_DIR = config.KAGGLE_TEMP_DIR + "_scenes"
SCENES_CHECKPOINT_PATH = os.path.join(SCENES_DIR, "scenes_checkpoint.json")
SCENES_REPORT_PATH = os.path.join(SCENES_DIR, "scenes_report.txt")
SCENES_FAILED_CSV_PATH = os.path.join(SCENES_DIR, "scenes_failed_downloads.csv")


def load_checkpoint() -> dict:
    if os.path.exists(SCENES_CHECKPOINT_PATH):
        with open(SCENES_CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_batches": [], "stats": {}, "failed": []}


def save_checkpoint(state: dict) -> None:
    os.makedirs(SCENES_DIR, exist_ok=True)
    tmp = SCENES_CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, SCENES_CHECKPOINT_PATH)


def get_scene_targets(scenes: list) -> list:
    targets = []
    seen_urls = set()
    for scene in scenes:
        scene_id = scene.get("id")
        for image in scene.get("images") or []:
            url = image.get("url") if isinstance(image, dict) else None
            if not url or (scene_id, url) in seen_urls:
                continue
            seen_urls.add((scene_id, url))
            targets.append({"id": scene_id, "url": url})
    return targets


def download_batch(targets: list) -> tuple[list, list]:
    os.makedirs(SCENES_TEMP_DIR, exist_ok=True)
    downloaded_paths, failed = [], []
    workers = get_download_workers()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for index, target in enumerate(targets):
            dest_dir = os.path.join(SCENES_TEMP_DIR, str(target["id"]))
            os.makedirs(dest_dir, exist_ok=True)
            futures[pool.submit(_download_one, target["id"], target["url"], dest_dir, index)] = target
        for future in as_completed(futures):
            result = future.result()
            if result["ok"]:
                downloaded_paths.append(result["path"])
            else:
                failed.append(
                    {"id": result["id"], "url": result["url"], "error": result["error"]}
                )
    return downloaded_paths, failed


def _write_report(stats: dict, started_at: float) -> None:
    image_count = stats["total_downloaded"]
    candidate_count = stats["webp_candidate_count"]
    total_bytes = stats["all_images_bytes"]
    candidate_bytes = stats["webp_candidate_bytes"]
    count_percent = (candidate_count / image_count * 100) if image_count else 0
    size_percent = (candidate_bytes / total_bytes * 100) if total_bytes else 0
    duration_min = (time.time() - started_at) / 60

    lines = [
        "=" * 62,
        "Scenes Images - WebP Candidate Report",
        "=" * 62,
        f"Run duration:                          {duration_min:.1f} minutes",
        "",
        f"Total scenes considered:                 {stats['total_scenes']}",
        f"Total image URLs found:                  {stats['total_targets']}",
        f"Successful downloads:                    {stats['total_downloaded']}",
        f"Failed downloads (after 3 retries):     {stats['total_failed']}",
        "",
        "-- WebP conversion candidates (analysis only, no conversion done) --",
        f"Images not already WebP-Lossy:           {candidate_count}",
        f"Candidate percentage of downloaded:     {count_percent:.2f}%",
        f"Total size of those images:              {_human_size(candidate_bytes)}",
        f"Candidate percentage of total size:     {size_percent:.2f}%",
        f"Total size of ALL checked images:        {_human_size(total_bytes)}",
        "=" * 62,
    ]
    with open(SCENES_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run() -> None:
    started_at = time.time()
    os.makedirs(SCENES_DIR, exist_ok=True)
    use_hf = bool(config.HF_DATASET_REPO_ID and config.HF_TOKEN)

    input_path = SCENES_INPUT_PATH
    if not os.path.exists(input_path):
        if not use_hf:
            raise RuntimeError("scenes/scenes.json not found and no HF credentials are set.")
        logger.info("scenes/scenes.json not found locally - fetching from HF dataset...")
        input_path = hf_sync.download_scenes_json(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    targets = get_scene_targets(scenes)
    batches = [targets[i : i + config.BATCH_SIZE] for i in range(0, len(targets), config.BATCH_SIZE)]

    state = load_checkpoint()
    completed = set(state["completed_batches"])
    failed_all = list(state["failed"])
    agg = {
        "total_scenes": len(scenes),
        "total_targets": len(targets),
        "total_downloaded": 0,
        "total_failed": len(failed_all),
        "webp_candidate_count": 0,
        "webp_candidate_bytes": 0,
        "all_images_bytes": 0,
    }
    agg.update(state.get("stats", {}))

    for index, batch in enumerate(batches):
        if index in completed:
            continue
        logger.info("=== Scenes batch %d/%d (%d images) ===", index + 1, len(batches), len(batch))
        downloaded_paths, batch_failed = download_batch(batch)
        failed_all.extend(batch_failed)
        agg["total_failed"] += len(batch_failed)
        agg["total_downloaded"] += len(downloaded_paths)

        webp_stats = analyze_webp_candidates(downloaded_paths)
        agg["webp_candidate_count"] += webp_stats["candidate_count"]
        agg["webp_candidate_bytes"] += webp_stats["candidate_total_bytes"]
        agg["all_images_bytes"] += webp_stats["all_images_total_bytes"]

        completed.add(index)
        for target in batch:
            shutil.rmtree(os.path.join(SCENES_TEMP_DIR, str(target["id"])), ignore_errors=True)
        gc.collect()
        save_checkpoint({"completed_batches": sorted(completed), "stats": agg, "failed": failed_all})
        if use_hf:
            try:
                hf_sync.upload_file_generic(
                    SCENES_CHECKPOINT_PATH, "scenes/scenes_checkpoint.json"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Checkpoint upload failed (will retry next batch): %s", e)

    with open(SCENES_FAILED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "url", "error"])
        writer.writeheader()
        writer.writerows(failed_all)
    _write_report(agg, started_at)
    logger.info("Done. scenes_report.txt written to %s", SCENES_REPORT_PATH)

    if use_hf:
        hf_sync.upload_file_generic(SCENES_REPORT_PATH, "scenes/scenes_report.txt")
        hf_sync.upload_file_generic(SCENES_FAILED_CSV_PATH, "scenes/scenes_failed_downloads.csv")


if __name__ == "__main__":
    run()