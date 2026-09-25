"""
Objective quality scoring, used ONLY to rank two-or-more images that
visual_dedup has already determined are the same underlying photo.
No similarity/duplicate-ness decisions are made here - just: given a group
that IS a duplicate group, which member is the best copy to keep.
"""
from __future__ import annotations

import math
import os

import cv2

import config


def _sharpness(gray_img) -> float:
    # Laplacian variance: higher = sharper / more in-focus / less pixelated.
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


def score_image(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = _sharpness(gray)
    file_size = os.path.getsize(path)

    # Log-scale each component since resolution/sharpness/size all vary over
    # orders of magnitude; this keeps the weighted sum meaningful.
    res_component = math.log1p(h * w)
    sharp_component = math.log1p(sharpness)
    size_component = math.log1p(file_size)

    return (
        res_component * config.QUALITY_WEIGHT_RESOLUTION
        + sharp_component * config.QUALITY_WEIGHT_SHARPNESS
        + size_component * config.QUALITY_WEIGHT_FILESIZE
    )


def pick_best(paths: list) -> str:
    """Given 2+ paths already known to be near-duplicates, return the
    highest-scoring one - even if the winning margin is tiny."""
    scored = [(p, score_image(p)) for p in paths]
    scored = [(p, s) for p, s in scored if s is not None]
    if not scored:
        # All copies unreadable/corrupt: fall back to largest file as a
        # last resort so the pipeline doesn't crash on bad data.
        return max(paths, key=lambda p: os.path.getsize(p))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]
