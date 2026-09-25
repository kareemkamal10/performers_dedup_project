"""
Builds the final human-readable .txt report and the failed-downloads .csv.
"""
from __future__ import annotations

import csv
import os
import time


def _human_size(num_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def write_failed_csv(failed_downloads: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "url", "error"])
        writer.writeheader()
        for row in failed_downloads:
            writer.writerow(row)


def write_txt_report(stats: dict, path: str, started_at: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duration_min = (time.time() - started_at) / 60

    lines = [
        "=" * 62,
        "Performers Image Dedup Pipeline - Final Report",
        "=" * 62,
        f"Run duration:                       {duration_min:.1f} minutes",
        "",
        "-- Merge / dedupe stage (before download) --",
        f"Total input elements:                {stats['total_input_elements']}",
        f"Excluded (no urls at all):            {stats['excluded_no_url']}",
        f"Excluded (single url after dedupe):   {stats['excluded_single_url']}",
        f"Elements kept for download:           {stats['kept_elements']}",
        "",
        "-- Download stage --",
        f"Total URLs attempted:                 {stats['total_urls_attempted']}",
        f"Successful downloads:                 {stats['total_urls_downloaded']}",
        f"Failed downloads (after 3 retries):    {stats['total_urls_failed']}",
        "",
        "-- Visual near-duplicate stage --",
        f"Duplicate images removed:             {stats['duplicates_removed']}",
        f"Unreadable/corrupt images excluded:    {stats['unreadable_excluded']}",
        f"Final unique images kept:              {stats['final_images_kept']}",
        f"Elements present in final JSON:        {stats['final_elements']}",
        "",
        "-- WebP conversion candidates (analysis only, no conversion done) --",
        f"Images not already WebP-Lossy:         {stats['webp_candidate_count']}",
        f"Total size of those images:            {_human_size(stats['webp_candidate_bytes'])}",
        f"Total size of ALL final images:        {_human_size(stats['all_images_bytes'])}",
        "",
        "=" * 62,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
