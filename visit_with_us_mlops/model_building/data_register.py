
import os
from huggingface_hub import HfApi, login

# ---------------- CONFIG ----------------
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "Bash18/tourism-package-prediction"
RAW_FILE_PATH = "visit_with_us_mlops/data/tourism.csv"

# ---------------- CREATE DATASET REPO ----------------
api = HfApi(token=os.getenv("HF_TOKEN"))

api.create_repo(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    exist_ok=True
)

# ---------------- UPLOAD RAW DATA ----------------
api.upload_file(
    path_or_fileobj=RAW_FILE_PATH,
    path_in_repo="raw.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)

print("Data registration completed successfully.")
