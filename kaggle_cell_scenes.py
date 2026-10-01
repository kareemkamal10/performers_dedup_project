# PASTE THIS ENTIRE CELL INTO A KAGGLE NOTEBOOK (as ONE cell) AND RUN IT.

GITHUB_REPO_URL    = ""
HF_DATASET_REPO_ID = "abdelwahabnabil500/datafile"
HF_TOKEN           = ""

import os
import subprocess
import sys

assert GITHUB_REPO_URL, "Set GITHUB_REPO_URL above before running this cell."
assert HF_TOKEN, "Set HF_TOKEN above before running this cell."

PROJECT_DIR = "/kaggle/working/project"

if not os.path.exists(PROJECT_DIR):
    subprocess.run(
        ["git", "clone", "--branch", "main", "--single-branch", GITHUB_REPO_URL, PROJECT_DIR],
        check=True,
    )
else:
    subprocess.run(["git", "-C", PROJECT_DIR, "pull", "origin", "main"], check=True)

if not os.path.isfile(os.path.join(PROJECT_DIR, "scenes_check.py")):
    raise RuntimeError(
        "The selected GitHub repository/branch does not contain scenes_check.py. "
        "Use https://github.com/kareemkamal10/performers_dedup_project.git "
        "with the main branch, then rerun this cell."
    )

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     os.path.join(PROJECT_DIR, "requirements.txt")],
    check=True,
)

os.environ["HF_DATASET_REPO_ID"] = HF_DATASET_REPO_ID
os.environ["HF_TOKEN"] = HF_TOKEN

os.chdir(PROJECT_DIR)
subprocess.run([sys.executable, "scenes_check.py"], check=True)