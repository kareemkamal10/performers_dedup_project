"""
Thin wrapper around huggingface_hub for the 3 things this project needs from
the private HF dataset:
  - downloading performers_data.json (if not already dropped in project root)
  - mirroring the checkpoint after every batch (cheap resume insurance)
  - uploading the final result_output/ folder at the end of the run
"""
from __future__ import annotations

import logging
import os

from huggingface_hub import hf_hub_download, upload_file, upload_folder

import config

logger = logging.getLogger(__name__)


def _require_creds() -> None:
    if not config.HF_DATASET_REPO_ID or not config.HF_TOKEN:
        raise RuntimeError(
            "HF_DATASET_REPO_ID / HF_TOKEN are not set - fill them in the Kaggle cell."
        )


def download_input_json(local_path: str = None) -> str:
    _require_creds()
    local_path = local_path or config.INPUT_JSON
    path = hf_hub_download(
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename="performers_data.json",
        token=config.HF_TOKEN,
        local_dir=os.path.dirname(local_path),
    )
    return path


def download_scenes_json(local_path: str = None) -> str:
    """Fetch scenes/scenes.json while preserving its dataset subdirectory."""
    _require_creds()
    local_path = local_path or os.path.join(config.PROJECT_ROOT, "scenes", "scenes.json")
    return hf_hub_download(
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename="scenes/scenes.json",
        token=config.HF_TOKEN,
        local_dir=config.PROJECT_ROOT,
    )


def download_checkpoint_if_exists():
    _require_creds()
    try:
        return hf_hub_download(
            repo_id=config.HF_DATASET_REPO_ID,
            repo_type="dataset",
            filename="checkpoint.json",
            token=config.HF_TOKEN,
        )
    except Exception:  # noqa: BLE001 - no checkpoint on the dataset yet
        return None


def upload_checkpoint() -> None:
    _require_creds()
    if not os.path.exists(config.CHECKPOINT_PATH):
        return
    upload_file(
        path_or_fileobj=config.CHECKPOINT_PATH,
        path_in_repo="checkpoint.json",
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        token=config.HF_TOKEN,
        commit_message="Update pipeline checkpoint",
    )
    logger.info("Checkpoint uploaded to HF dataset")


def upload_result_output() -> None:
    _require_creds()
    upload_folder(
        folder_path=config.RESULT_OUTPUT_DIR,
        path_in_repo="result_output",
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        token=config.HF_TOKEN,
        commit_message="Upload final pipeline result_output",
    )
    logger.info("result_output uploaded to HF dataset")


def download_final_json(local_path: str = None) -> str:
    """Fetches the main run's already-uploaded result_output/performers_data_final.json."""
    _require_creds()
    local_path = local_path or os.path.join(
        config.RESULT_OUTPUT_DIR, "performers_data_final.json"
    )
    return hf_hub_download(
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        filename="result_output/performers_data_final.json",
        token=config.HF_TOKEN,
        local_dir=os.path.dirname(local_path),
    )


def upload_file_generic(local_path: str, path_in_repo: str) -> None:
    """Generic single-file uploader, used by side scripts (e.g. single_image_check.py)
    that don't fit the whole-folder upload_result_output() shape."""
    _require_creds()
    if not os.path.exists(local_path):
        return
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=config.HF_DATASET_REPO_ID,
        repo_type="dataset",
        token=config.HF_TOKEN,
        commit_message=f"Upload {path_in_repo}",
    )
    logger.info("%s uploaded to HF dataset", path_in_repo)
