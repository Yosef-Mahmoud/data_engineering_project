import io
import pickle
import pandas as pd
from typing import Dict, Any
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, silhouette_score
)

# Imbalanced-Learn Imports
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

# Assuming these are defined in your storage utility
from app.utils.storage.storage import data_storage, job_storage

# --- Internal Helpers ---

def build_pipeline(df: pd.DataFrame, target: str, model_obj: Any, task_type: str = None) -> ImbPipeline:
    """Creates an ImbPipeline handling Imputation, Scaling, Encoding, and Resampling."""
    features = [c for c in df.columns if c != target]
    num_cols = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df[features].select_dtypes(include=['object', 'category']).columns.tolist()

    # 1. Numerical: Fill missing with median -> Scale
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical: Fill missing with most frequent -> OneHot
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # 3. Assemble Steps
    steps = [('preprocessor', preprocessor)]
    
    # Apply UnderSampler only for classification to handle class imbalance
    if task_type == 'classification':
        steps.append(('resampler', RandomUnderSampler(random_state=42)))
    
    steps.append(('model', model_obj))

    return ImbPipeline(steps=steps)

# --- Core Training Functions ---

async def train_regression(df: pd.DataFrame, target: str, job_id: str) -> Dict[str, Any]:
    if not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = LabelEncoder().fit_transform(df[target].astype(str))
    
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipelines (task_type='regression' skips the resampler)
    pipe1 = build_pipeline(df, target, LinearRegression(), task_type='regression').fit(X_train, y_train)
    pipe2 = build_pipeline(df, target, DecisionTreeRegressor(), task_type='regression').fit(X_train, y_train)

    p1, p2 = pipe1.predict(X_test), pipe2.predict(X_test)

    res1 = {"name": "Linear Regression", "mae": mean_absolute_error(y_test, p1), "mse": mean_squared_error(y_test, p1), "r2": r2_score(y_test, p1)}
    res2 = {"name": "Decision Tree Regressor", "mae": mean_absolute_error(y_test, p2), "mse": mean_squared_error(y_test, p2), "r2": r2_score(y_test, p2)}
    
    best_pipe = pipe1 if res1["mse"] <= res2["mse"] else pipe2
    job_storage[job_id] = best_pipe
    
    return {
        "all_results": [res1, res2], 
        "best_model": res1["name"] if best_pipe == pipe1 else res2["name"],
        "message": f"Regression complete. Best: {res1['name'] if best_pipe == pipe1 else res2['name']}"
    }

async def train_classification(df: pd.DataFrame, target: str, job_id: str) -> Dict[str, Any]:
    df[target] = df[target].astype(str)
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipelines with task_type='classification' to trigger RandomUnderSampler
    pipe1 = build_pipeline(df, target, LogisticRegression(max_iter=1000), task_type='classification').fit(X_train, y_train)
    pipe2 = build_pipeline(df, target, DecisionTreeClassifier(), task_type='classification').fit(X_train, y_train)

    p1, p2 = pipe1.predict(X_test), pipe2.predict(X_test)

    res1 = {
        "name": "Logistic Regression", 
        "accuracy": accuracy_score(y_test, p1),
        "precision": precision_score(y_test, p1, average='weighted', zero_division=0),
        "recall": recall_score(y_test, p1, average='weighted', zero_division=0),
        "f1": f1_score(y_test, p1, average='weighted', zero_division=0),
        "cm": confusion_matrix(y_test, p1).tolist()
    }
    res2 = {
        "name": "Decision Tree Classifier", 
        "accuracy": accuracy_score(y_test, p2),
        "precision": precision_score(y_test, p2, average='weighted', zero_division=0),
        "recall": recall_score(y_test, p2, average='weighted', zero_division=0),
        "f1": f1_score(y_test, p2, average='weighted', zero_division=0),
        "cm": confusion_matrix(y_test, p2).tolist()
    }

    best_pipe = pipe1 if res1["f1"] >= res2["f1"] else pipe2 # F1 is better for balanced assessment
    job_storage[job_id] = best_pipe

    return {
        "all_results": [res1, res2], 
        "best_model": res1["name"] if best_pipe == pipe1 else res2["name"],
        "message": "Classification complete with class balancing (UnderSampling)."
    }

async def train_clustering(df: pd.DataFrame, job_id: str) -> Dict[str, Any]:
    # Clustering uses an empty string for target; resampler is skipped
    pipe1 = build_pipeline(df, "", KMeans(n_clusters=3, n_init=10)).fit(df)
    pipe2 = build_pipeline(df, "", AgglomerativeClustering(n_clusters=3)).fit(df)

    # Transform data for silhouette math using the pipeline's preprocessor
    X_transformed = pipe1.named_steps['preprocessor'].transform(df)
    
    s1 = silhouette_score(X_transformed, pipe1.named_steps['model'].labels_)
    s2 = silhouette_score(X_transformed, pipe2.named_steps['model'].labels_)

    res1 = {"name": "K-Means", "silhouette": s1}
    res2 = {"name": "Agglomerative Clustering", "silhouette": s2}

    best_pipe = pipe1 if s1 >= s2 else pipe2
    job_storage[job_id] = best_pipe

    return {
        "all_results": [res1, res2], 
        "best_model": res1["name"] if best_pipe == pipe1 else res2["name"],
        "message": "Clustering complete."
    }

# --- Main Task Handler ---

async def train_pipeline(payload: dict):
    job_id = payload.get('job_id')
    task_type = payload.get('task_type')
    target = payload.get('target_column')

    if job_id not in data_storage:
        raise HTTPException(status_code=404, detail="Data not found. Please upload again.")

    df = data_storage[job_id].copy()

    try:
        if task_type == 'regression':
            return await train_regression(df, target, job_id)
        elif task_type == 'classification':
            return await train_classification(df, target, job_id)
        elif task_type == 'clustering':
            return await train_clustering(df, job_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid task type.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Training Error: {str(e)}")
    finally:
        if job_id in data_storage:
            del data_storage[job_id]

async def download_model(job_id: str):
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Model not found in storage.")

    try:
        buffer = io.BytesIO()
        pickle.dump(job_storage[job_id], buffer)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=model_pipeline_{job_id}.pkl"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Download Error: {str(e)}")
    
    finally:
        if job_id in job_storage:
            del job_storage[job_id]   