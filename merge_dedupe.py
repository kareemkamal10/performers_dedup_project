"""
Stage 1: merge `image` + `source_images` into one deduped URL list per element,
and drop elements that end up with fewer than 2 unique URLs.
This happens BEFORE any downloading, per project spec.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def collect_urls(element: dict) -> list:
    urls = []
    single = element.get("image")
    if single:
        urls.append(single)
    extra = element.get("source_images") or []
    if isinstance(extra, str):
        extra = [extra]
    urls.extend(u for u in extra if u)
    return urls


def merge_and_filter(elements: list) -> tuple[list, dict]:
    """
    Returns (kept_elements, stats).
    Each kept element is: {"id": ..., "urls": [unique urls, order preserved]}.
    Elements whose deduped url list has fewer than 2 entries are dropped
    entirely (never downloaded, never appear anywhere downstream).
    """
    kept = []
    excluded_single = 0
    excluded_empty = 0

    for el in elements:
        raw_urls = collect_urls(el)
        seen = set()
        unique_urls = []
        for u in raw_urls:
            if u not in seen:
                seen.add(u)
                unique_urls.append(u)

        if len(unique_urls) == 0:
            excluded_empty += 1
            continue
        if len(unique_urls) == 1:
            excluded_single += 1
            continue

        kept.append({"id": el.get("id"), "urls": unique_urls})

    stats = {
        "total_input_elements": len(elements),
        "excluded_single_url": excluded_single,
        "excluded_no_url": excluded_empty,
        "kept_elements": len(kept),
    }
    logger.info("Merge/dedupe stage: %s", stats)
    return kept, stats
