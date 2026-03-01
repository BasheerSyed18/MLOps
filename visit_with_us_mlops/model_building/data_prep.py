
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, login

# ---------------- CONFIG ----------------
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "Bash18/tourism-package-prediction"

# ---------------- LOGIN ----------------
login(token=HF_TOKEN)

# ---------------- LOAD DATA ----------------
raw_url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/raw.csv"
df = pd.read_csv(raw_url)

print("Dataset loaded successfully.")

# ---------------- REMOVE IDENTIFIER ----------------
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# ---------------- HANDLE MISSING VALUES ----------------

# Fill numeric columns with median
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
for col in df.select_dtypes(include=["object"]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values handled using median and mode.")

# ---------------- ENCODE CATEGORICAL VARIABLES ----------------
label_encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Categorical encoding completed.")

# ---------------- SPLIT DATA ----------------
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# ---------------- SAVE LOCALLY ----------------
os.makedirs("visit_with_us_mlops/data", exist_ok=True)

train_path = "visit_with_us_mlops/data/train.csv"
test_path = "visit_with_us_mlops/data/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Train and test datasets saved locally.")

# ---------------- UPLOAD TO HF ----------------
api = HfApi()

api.upload_file(
    path_or_fileobj=train_path,
    path_in_repo="train.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)

api.upload_file(
    path_or_fileobj=test_path,
    path_in_repo="test.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)

print("Processed datasets uploaded successfully.")
