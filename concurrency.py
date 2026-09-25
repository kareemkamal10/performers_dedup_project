"""
Auto-detects safe worker counts based on the machine's CPU, so the pipeline
adapts to whatever Kaggle session it happens to run on without manual tuning,
while never exceeding the hard caps in config.py (which protect against
starving the Kaggle session or overloading it).
"""
import os

import config


def get_download_workers() -> int:
    cpu = os.cpu_count() or 4
    # I/O-bound work: scale well past core count, but stay capped to avoid
    # overwhelming the network stack or tripping per-host rate limits.
    return min(cpu * 5, config.MAX_DOWNLOAD_WORKERS_CAP)


def get_process_workers() -> int:
    cpu = os.cpu_count() or 4
    # CPU-bound work: leave 1 core free for the main process / OS so the
    # Kaggle session stays responsive.
    return max(1, min(cpu - 1, config.MAX_PROCESS_WORKERS_CAP))
