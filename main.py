"""
Entry point.

Usage on Kaggle: paste kaggle_cell.py into a notebook cell, fill in the 3
placeholders, and run it - it clones this repo and calls this file for you.

Usage locally/manually: drop performers_data.json in this project's root and
run `python main.py` (HF creds are optional in that case - only needed for
auto-fetching the input file, checkpoint resume, and auto-uploading results).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys

import config
import hf_sync
from pipeline import run as run_pipeline
from report import write_failed_csv, write_txt_report

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def main() -> None:
    use_hf = bool(config.HF_DATASET_REPO_ID and config.HF_TOKEN)
    input_path = config.INPUT_JSON

    if not os.path.exists(input_path):
        if use_hf:
            logger.info("performers_data.json not found locally - fetching from HF dataset...")
            input_path = hf_sync.download_input_json(input_path)
        else:
            logger.error(
                "performers_data.json not found at %s and no HF credentials are set. "
                "Either place the file there, or fill in the Kaggle cell placeholders.",
                input_path,
            )
            sys.exit(1)

    if use_hf:
        # Resume support: pull down any checkpoint already on the dataset
        # (e.g. from a previous session that got interrupted) before starting.
        remote_ckpt = hf_sync.download_checkpoint_if_exists()
        if remote_ckpt:
            os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
            shutil.copy(remote_ckpt, config.CHECKPOINT_PATH)
            logger.info("Resumed from an existing checkpoint found on the HF dataset.")

    results_by_id, failed_downloads, agg, merge_stats, started_at = run_pipeline(
        input_path, use_hf=use_hf
    )

    # ---- Final deduped JSON ----
    final_list = [{"id": eid, "urls": urls} for eid, urls in results_by_id.items() if urls]
    final_json_path = os.path.join(config.RESULT_OUTPUT_DIR, "performers_data_final.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    # ---- Failed downloads CSV ----
    failed_csv_path = os.path.join(config.RESULT_OUTPUT_DIR, "failed_downloads.csv")
    write_failed_csv(failed_downloads, failed_csv_path)

    # ---- TXT report ----
    stats = {**merge_stats, **agg, "final_elements": len(final_list)}
    report_path = os.path.join(config.RESULT_OUTPUT_DIR, "report.txt")
    write_txt_report(stats, report_path, started_at)

    logger.info("Done. Final outputs written to %s", config.RESULT_OUTPUT_DIR)

    if use_hf:
        hf_sync.upload_result_output()
        hf_sync.upload_checkpoint()


if __name__ == "__main__":
    main()
