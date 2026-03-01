
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from huggingface_hub import HfApi, login

# ---------------- CONFIG ----------------
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "Bash18/tourism-package-prediction"
MODEL_REPO = "Bash18/tourism-package-model"

# ---------------- TRACKING ----------------
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("MLOps_Visit_With_US_CICD_experiment")

# ---------------- HF LOGIN ----------------
login(token=HF_TOKEN)

with mlflow.start_run():

    # ---------------- LOAD DATA ----------------
    train_url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/train.csv"
    test_url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/test.csv"

    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)

    X_train = train_df.drop("ProdTaken", axis=1)
    y_train = train_df["ProdTaken"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print("Scale Pos Weight:", scale_pos_weight)

    X_test = test_df.drop("ProdTaken", axis=1)
    y_test = test_df["ProdTaken"]

    # ---------------- MODEL ----------------
    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1]
    }

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="roc_auc"
    )

    # ---------------- TRAIN ----------------
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # ---------------- LOG PARAMETERS ----------------
    mlflow.log_params(grid_search.best_params_)

    # ---------------- EVALUATION ----------------
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # ---------------- LOG MODEL ----------------
    mlflow.sklearn.log_model(
        best_model,
        name="model",
        input_example=X_train.iloc[:5]
    )

    # ---------------- SAVE LOCALLY ----------------
    os.makedirs("visit_with_us_mlops/model_building", exist_ok=True)
    model_path = "visit_with_us_mlops/model_building/model.pkl"
    joblib.dump(best_model, model_path)

# ---------------- UPLOAD TO HF MODEL HUB ----------------
api = HfApi()
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model.pkl",
    repo_id=MODEL_REPO,
    repo_type="model"
)

print("Training completed successfully.")
