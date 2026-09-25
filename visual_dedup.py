"""
Stage 3: within each element's downloaded images, group near-duplicates
(same underlying photo at different size/compression/quality) using a
perceptual hash, then keep only the highest-quality copy of each group.
"""
from __future__ import annotations

import logging

import imagehash
from PIL import Image

import config
from quality import pick_best

logger = logging.getLogger(__name__)


def _phash(path: str):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img, hash_size=config.PHASH_HASH_SIZE)
    except Exception:  # noqa: BLE001 - corrupt/truncated/non-image file
        return None


def dedupe_element_images(paths: list) -> tuple:
    """
    Returns (kept_paths, duplicates_removed, unreadable_excluded).

    Groups images whose pHash Hamming distance <= PHASH_HAMMING_THRESHOLD as
    the same underlying photo, keeps the best-quality one per group (via
    quality.pick_best - no minimum margin required), drops the rest.
    Images that can't even be opened are excluded and counted separately.
    """
    hashes = {}
    unreadable = 0
    for p in paths:
        h = _phash(p)
        if h is not None:
            hashes[p] = h
        else:
            unreadable += 1

    remaining = list(hashes.keys())
    groups = []
    while remaining:
        anchor = remaining.pop(0)
        group = [anchor]
        still_remaining = []
        for p in remaining:
            if hashes[anchor] - hashes[p] <= config.PHASH_HAMMING_THRESHOLD:
                group.append(p)
            else:
                still_remaining.append(p)
        remaining = still_remaining
        groups.append(group)

    kept = []
    duplicates_removed = 0
    for group in groups:
        if len(group) == 1:
            kept.append(group[0])
        else:
            best = pick_best(group)
            kept.append(best)
            duplicates_removed += len(group) - 1

    return kept, duplicates_removed, unreadable
