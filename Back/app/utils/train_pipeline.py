import io
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score
)

# Imbalanced-Learn Imports
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from app.utils.storage.storage import data_storage, job_storage


# ─────────────────────────────────────────────
#  Shared Helpers
# ─────────────────────────────────────────────

def is_imbalanced(y: pd.Series, threshold: float = 1.5) -> bool:
    """Returns True when the majority class is > threshold × the minority class."""
    counts = y.value_counts()
    return len(counts) >= 2 and (counts.max() / counts.min()) > threshold


def build_preprocessor(df: pd.DataFrame, feature_cols: list) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that:
      - Imputes numeric columns with the median, then applies StandardScaler
        (matches Labs 3-6 preprocessing pattern).
      - Imputes categorical columns with the most-frequent value, then applies
        OneHotEncoder (handles unknown categories seen at inference time).
    """
    num_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
    )


def make_supervised_pipeline(
    df: pd.DataFrame,
    target: str,
    model_obj: Any,
    apply_smote: bool = False,
    k_neighbors: int = 5,
) -> ImbPipeline:
    """
    Assembles: preprocessor → (optional SMOTE) → model.
    Uses ImbPipeline so SMOTE is only applied during fit, not predict.
    """
    features = [c for c in df.columns if c != target]
    steps = [("preprocessor", build_preprocessor(df, features))]
    if apply_smote:
        steps.append(("resampler", SMOTE(random_state=42, k_neighbors=k_neighbors)))
    steps.append(("model", model_obj))
    return ImbPipeline(steps=steps)


# ─────────────────────────────────────────────
#  Regression
# ─────────────────────────────────────────────

async def train_regression(df: pd.DataFrame, target: str, job_id: str) -> Dict[str, Any]:
    # Validate the target is numeric — silently encoding a string target to integers
    # produces misleading regression results and was a silent bug in the original code.
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Regression requires a numeric target column. "
                f"'{target}' contains non-numeric values. "
                f"Consider using Classification instead."
            ),
        )

    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Two models: Linear Regression (baseline) + Random Forest Regressor (ensemble).
    # Matches Lab 7 which compared LinearRegression, SVR, and RandomForestRegressor.
    # Random Forest replaces DecisionTreeRegressor — trees overfit badly without pruning.
    pipe1 = make_supervised_pipeline(df, target, LinearRegression()).fit(X_train, y_train)
    pipe2 = make_supervised_pipeline(
        df, target, RandomForestRegressor(n_estimators=100, random_state=42)
    ).fit(X_train, y_train)

    p1, p2 = pipe1.predict(X_test), pipe2.predict(X_test)

    res1 = {
        "name": "Linear Regression",
        "mae": round(float(mean_absolute_error(y_test, p1)), 4),
        "mse": round(float(mean_squared_error(y_test, p1)), 4),
        "r2":  round(float(r2_score(y_test, p1)), 4),
    }
    res2 = {
        "name": "Random Forest Regressor",
        "mae": round(float(mean_absolute_error(y_test, p2)), 4),
        "mse": round(float(mean_squared_error(y_test, p2)), 4),
        "r2":  round(float(r2_score(y_test, p2)), 4),
    }

    # Select best by R² (higher is better); original used MSE which is equivalent
    # but R² is scale-independent and easier to interpret on the frontend.
    best_pipe = pipe1 if res1["r2"] >= res2["r2"] else pipe2
    best_name = res1["name"] if best_pipe is pipe1 else res2["name"]
    job_storage[job_id] = best_pipe

    return {
        "all_results": [res1, res2],
        "best_model": best_name,
        "message": f"Regression complete. Best model by R² Score: {best_name}",
    }


# ─────────────────────────────────────────────
#  Classification
# ─────────────────────────────────────────────

async def train_classification(df: pd.DataFrame, target: str, job_id: str) -> Dict[str, Any]:
    df[target] = df[target].astype(str)
    X, y = df.drop(columns=[target]), df[target]

    # Stratified split preserves class ratios in both train and test sets,
    # which is critical for imbalanced data (same approach as Lab 6 Oil Spill).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Only apply SMOTE when the training set is actually imbalanced (ratio > 1.5×).
    # The original code applied RandomUnderSampler unconditionally, which discards
    # valuable majority-class data even on balanced datasets.
    imbalanced = is_imbalanced(y_train)
    min_class_count = int(y_train.value_counts().min())
    k_neighbors = min(5, min_class_count - 1) if imbalanced else 5
    apply_smote = imbalanced and k_neighbors >= 1

    resampling_note = (
        " SMOTE oversampling applied (class imbalance detected)."
        if apply_smote
        else " Classes are balanced — no resampling needed."
    )

    # Two models: Logistic Regression (interpretable baseline) + Random Forest (ensemble).
    # Matches Lab 6 which used Logistic Regression and Random Forest, both wrapped in
    # a Pipeline. DecisionTreeClassifier was replaced because unpruned trees overfit.
    pipe1 = make_supervised_pipeline(
        df, target, LogisticRegression(max_iter=1000, random_state=42),
        apply_smote=apply_smote, k_neighbors=k_neighbors,
    ).fit(X_train, y_train)

    pipe2 = make_supervised_pipeline(
        df, target, RandomForestClassifier(n_estimators=100, random_state=42),
        apply_smote=apply_smote, k_neighbors=k_neighbors,
    ).fit(X_train, y_train)

    p1, p2 = pipe1.predict(X_test), pipe2.predict(X_test)

    # Class labels are returned so the frontend can render a labelled confusion matrix.
    labels = sorted(y_test.unique().tolist())

    res1 = {
        "name":      "Logistic Regression",
        "accuracy":  round(float(accuracy_score(y_test, p1)), 4),
        "precision": round(float(precision_score(y_test, p1, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, p1,    average="weighted", zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, p1,        average="weighted", zero_division=0)), 4),
        "cm":        confusion_matrix(y_test, p1, labels=labels).tolist(),
        "labels":    labels,
    }
    res2 = {
        "name":      "Random Forest Classifier",
        "accuracy":  round(float(accuracy_score(y_test, p2)), 4),
        "precision": round(float(precision_score(y_test, p2, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, p2,    average="weighted", zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, p2,        average="weighted", zero_division=0)), 4),
        "cm":        confusion_matrix(y_test, p2, labels=labels).tolist(),
        "labels":    labels,
    }

    # Select best by weighted F1: more robust than accuracy for multi-class/imbalanced tasks.
    best_pipe = pipe1 if res1["f1"] >= res2["f1"] else pipe2
    best_name = res1["name"] if best_pipe is pipe1 else res2["name"]
    job_storage[job_id] = best_pipe

    return {
        "all_results": [res1, res2],
        "best_model": best_name,
        "message": f"Classification complete.{resampling_note}",
    }


# ─────────────────────────────────────────────
#  Clustering
# ─────────────────────────────────────────────

async def train_clustering(df: pd.DataFrame, job_id: str) -> Dict[str, Any]:
    """
    Bug-fix: the original code used AgglomerativeClustering, which has NO predict()
    method and therefore makes the saved pipeline useless for inference on new data.
    Replaced with GaussianMixture (GMM), which supports predict() and fit_predict(),
    matching the spirit of Lab 8 which mentions GMM as the third clustering algorithm.

    The preprocessor is fitted separately here because clustering pipelines cannot
    use ImbPipeline (no y) and sklearn's Pipeline.fit() on unsupervised estimators
    would call fit_transform on the preprocessor and then fit on the model correctly,
    but getting labels back requires a workaround. By fitting separately we keep the
    silhouette calculation simple, then wrap the fitted objects for saving.
    """
    features = df.columns.tolist()
    preprocessor = build_preprocessor(df, features)
    X_scaled = preprocessor.fit_transform(df)

    # Model 1: K-Means — same as Lab 8, n_init=10 to avoid poor initializations.
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels1 = kmeans.fit_predict(X_scaled)
    s1 = float(silhouette_score(X_scaled, labels1)) if len(set(labels1)) > 1 else -1.0

    # Model 2: Gaussian Mixture Model — probabilistic soft clustering.
    # Supports .predict() on new data, unlike AgglomerativeClustering or DBSCAN.
    gmm = GaussianMixture(n_components=3, random_state=42)
    labels2 = gmm.fit_predict(X_scaled)
    s2 = float(silhouette_score(X_scaled, labels2)) if len(set(labels2)) > 1 else -1.0

    res1 = {"name": "K-Means",                 "silhouette": round(s1, 4)}
    res2 = {"name": "Gaussian Mixture Model",   "silhouette": round(s2, 4)}

    best_model = kmeans if s1 >= s2 else gmm
    best_name  = res1["name"] if best_model is kmeans else res2["name"]

    # Wrap the already-fitted preprocessor + model into a single sklearn Pipeline
    # so joblib/pickle can serialize the full inference pipeline.
    best_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        best_model),
    ])
    job_storage[job_id] = best_pipeline

    return {
        "all_results": [res1, res2],
        "best_model": best_name,
        "message": f"Clustering complete. Best model by Silhouette Score: {best_name}",
    }


# ─────────────────────────────────────────────
#  Main Entry Point
# ─────────────────────────────────────────────

async def train_pipeline(payload: dict):
    job_id    = payload.get("job_id")
    task_type = payload.get("task_type")
    target    = payload.get("target_column")

    if job_id not in data_storage:
        raise HTTPException(status_code=404, detail="Data not found. Please re-upload your file.")

    df = data_storage[job_id].copy()

    # Apply the same upfront cleaning steps used throughout Labs 2-5:
    #   1. Remove duplicate rows.
    #   2. Drop columns that are entirely null (zero information).
    df.drop_duplicates(inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    try:
        if task_type == "regression":
            return await train_regression(df, target, job_id)
        elif task_type == "classification":
            return await train_classification(df, target, job_id)
        elif task_type == "clustering":
            return await train_clustering(df, job_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid task_type. Must be 'classification', 'regression', or 'clustering'.")
    except HTTPException:
        raise  # Re-raise without wrapping so the client sees the correct status code.
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training Error: {str(e)}")
    finally:
        # Always clean up the raw data regardless of success or failure.
        if job_id in data_storage:
            del data_storage[job_id]


# ─────────────────────────────────────────────
#  Download
# ─────────────────────────────────────────────

async def download_model(job_id: str) -> StreamingResponse:
    """
    Serializes the best pipeline to a .pkl byte stream.
    NOTE: job_storage cleanup is intentionally left to the route handler
    so there is exactly ONE place responsible for cleanup (no double-delete).
    """
    if job_id not in job_storage:
        raise HTTPException(
            status_code=404,
            detail="Model not found. It may have already been downloaded or the session expired.",
        )

    buffer = io.BytesIO()
    pickle.dump(job_storage[job_id], buffer)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=model_pipeline_{job_id}.pkl"},
    )