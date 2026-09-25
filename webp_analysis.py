"""
WebP lossy/lossless classification, purely for the final report - no actual
conversion happens here. A WebP file's extension alone doesn't tell you
whether it's Lossy (VP8 chunk) or Lossless (VP8L chunk) internally, so this
inspects the RIFF container header directly.
"""
from __future__ import annotations

import os


def is_webp_lossy(path: str):
    """
    Returns:
      True  -> file is WebP AND already lossy-encoded (not a conversion candidate)
      False -> file is WebP but lossless-encoded (IS a conversion candidate)
      None  -> file is not WebP at all (IS a conversion candidate)
    """
    try:
        with open(path, "rb") as f:
            header = f.read(30)
        if not (header[:4] == b"RIFF" and header[8:12] == b"WEBP"):
            return None
        fourcc = header[12:16]
        if fourcc == b"VP8 ":
            return True
        if fourcc == b"VP8L":
            return False
        if fourcc == b"VP8X":
            # Extended container: the actual lossy/lossless sub-chunk is
            # further in. Scan a small window for the marker.
            with open(path, "rb") as f:
                blob = f.read(4096)
            if b"VP8L" in blob:
                return False
            return True
        return True  # unrecognized sub-format: treat conservatively as lossy
    except Exception:  # noqa: BLE001
        return None


def analyze_webp_candidates(paths: list) -> dict:
    """
    Returns:
      candidate_count:        images that are NOT already WebP-Lossy
      candidate_total_bytes:  total size of those candidate images
      all_images_total_bytes: total size of every image passed in
    """
    candidate_count = 0
    candidate_bytes = 0
    total_bytes = 0

    for p in paths:
        try:
            size = os.path.getsize(p)
        except OSError:
            continue
        total_bytes += size

        already_webp_lossy = is_webp_lossy(p) is True
        if not already_webp_lossy:
            candidate_count += 1
            candidate_bytes += size

    return {
        "candidate_count": candidate_count,
        "candidate_total_bytes": candidate_bytes,
        "all_images_total_bytes": total_bytes,
    }
