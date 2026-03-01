
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# ---------------- CONFIG ----------------
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "Bash18/tourism-package-prediction"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is not set.")

api = HfApi(token=HF_TOKEN)

# ---------------- LOAD DATA ----------------
raw_url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/raw.csv"
df = pd.read_csv(raw_url)

print("Dataset loaded successfully.")
print("Initial shape:", df.shape)

# ---------------- REMOVE IDENTIFIERS ----------------
df = df.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore")
print("Identifier columns removed.")

# ---------------- CLEAN GENDER COLUMN ----------------
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({
        "Fe Male": "Female"
    })

# ---------------- REMOVE NA ROWS ----------------
before_drop = df.shape[0]
df = df.dropna()
after_drop = df.shape[0]

print(f"Rows before dropping NA: {before_drop}")
print(f"Rows after dropping NA: {after_drop}")
print(f"Rows removed: {before_drop - after_drop}")

# ---------------- ENCODE CATEGORICAL VARIABLES ----------------
categorical_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}

print("\n🔎 Label Encoding Mapping:")

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\nColumn: {col}")
    print(mapping)

print("\nCategorical encoding completed.")

# ---------------- SPLIT DATA ----------------
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

print("Class Distribution (Counts):")
print(y.value_counts())

print("\nClass Distribution (Percentage):")
print(y.value_counts(normalize=True))

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

print("\nTrain and test datasets saved locally.")

# ---------------- UPLOAD TO HF ----------------
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
