"""
MODEL BENCHMARKING WITH MLFLOW TRACKING
This version tracks all experiments in MLflow UI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import time
import joblib
from pathlib import Path

# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost

print("=" * 80)
print("MODEL BENCHMARKING WITH MLFLOW")
print("=" * 80)

# ============================================================================
# STEP 1: Setup MLflow
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: MLflow Setup")
print("=" * 80)

# Set tracking URI (default is local)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment name
experiment_name = "Loan-Default-Prediction"
mlflow.set_experiment(experiment_name)

print(f"✓ MLflow tracking URI: http://127.0.0.1:5000")
print(f"✓ Experiment: {experiment_name}")
print(f"\n💡 View experiments at: http://127.0.0.1:5000")

# ============================================================================
# STEP 2: Load Data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Loading Data")
print("=" * 80)

df = pd.read_csv("data/processed/train_features.csv")

# Safety check
object_cols = df.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print(f"⚠️  Dropping {len(object_cols)} string columns")
    df = df.drop(columns=object_cols)

X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']

print(f"✓ Loaded: {df.shape}")
print(f"Features: {X.shape[1]}")
print(f"Default rate: {y.mean()*100:.2f}%")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate class weight
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# ============================================================================
# STEP 3: Evaluation Function with MLflow Logging
# ============================================================================

def evaluate_and_log_model(model, model_name, X_train, X_test, y_train, y_test, 
                           params, train_time):
    """
    Evaluate model and log everything to MLflow
    """
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'train_time': train_time
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Results:")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  Time:      {train_time:.2f}s")
    
    return metrics, y_pred_proba

# ============================================================================
# STEP 4: Model 1 - Logistic Regression with MLflow
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Logistic Regression")
print("=" * 80)

with mlflow.start_run(run_name="Logistic_Regression"):
    
    # Log parameters
    params = {
        'model_type': 'LogisticRegression',
        'class_weight': 'balanced',
        'max_iter': 1000,
        'random_state': 42
    }
    mlflow.log_params(params)
    
    # Train
    start_time = time.time()
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    lr_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate and log
    lr_metrics, lr_proba = evaluate_and_log_model(
        lr_model, "Logistic Regression",
        X_train, X_test, y_train, y_test,
        params, train_time
    )
    
    # Log metrics to MLflow
    mlflow.log_metrics(lr_metrics)
    
    # Log model
    mlflow.sklearn.log_model(lr_model, "model")
    
    # Log tags
    mlflow.set_tag("model_family", "linear")
    mlflow.set_tag("framework", "sklearn")
    
    print("✓ Logged to MLflow!")
    
    # End the run
    mlflow.end_run()

# ============================================================================
# STEP 5: Model 2 - Random Forest
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Random Forest")
print("=" * 80)

with mlflow.start_run(run_name="Random_Forest"):
    
    params = {
        'model_type': 'RandomForest',
        'n_estimators': 100,
        'max_depth': 10,
        'class_weight': 'balanced',
        'random_state': 42
    }
    mlflow.log_params(params)
    
    start_time = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    rf_metrics, rf_proba = evaluate_and_log_model(
        rf_model, "Random Forest",
        X_train, X_test, y_train, y_test,
        params, train_time
    )
    
    mlflow.log_metrics(rf_metrics)
    mlflow.sklearn.log_model(rf_model, "model")
    mlflow.set_tag("model_family", "tree_ensemble")
    mlflow.set_tag("framework", "sklearn")
    
    print("✓ Logged to MLflow!")

# ============================================================================
# STEP 6: Model 3 - XGBoost
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: XGBoost")
print("=" * 80)

with mlflow.start_run(run_name="XGBoost"):
    
    params = {
        'model_type': 'XGBoost',
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.01,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42
    }
    mlflow.log_params(params)
    
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.01,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_time = time.time() - start_time
    
    xgb_metrics, xgb_proba = evaluate_and_log_model(
        xgb_model, "XGBoost",
        X_train, X_test, y_train, y_test,
        params, train_time
    )
    
    mlflow.log_metrics(xgb_metrics)
    mlflow.xgboost.log_model(xgb_model, "model")
    mlflow.set_tag("model_family", "gradient_boosting")
    mlflow.set_tag("framework", "xgboost")
    
    print("✓ Logged to MLflow!")

# ============================================================================
# STEP 7: Model 4 - LightGBM
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: LightGBM")
print("=" * 80)

with mlflow.start_run(run_name="LightGBM"):
    
    params = {
        'model_type': 'LightGBM',
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.01,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42
    }
    mlflow.log_params(params)
    
    start_time = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.01,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc')
    train_time = time.time() - start_time
    
    lgb_metrics, lgb_proba = evaluate_and_log_model(
        lgb_model, "LightGBM",
        X_train, X_test, y_train, y_test,
        params, train_time
    )
    
    mlflow.log_metrics(lgb_metrics)
    mlflow.lightgbm.log_model(lgb_model, "model")
    mlflow.set_tag("model_family", "gradient_boosting")
    mlflow.set_tag("framework", "lightgbm")
    
    print("✓ Logged to MLflow!")

# ============================================================================
# STEP 8: Model 5 - CatBoost
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: CatBoost")
print("=" * 80)

with mlflow.start_run(run_name="CatBoost"):
    
    params = {
        'model_type': 'CatBoost',
        'iterations': 300,
        'depth': 6,
        'learning_rate': 0.01,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42
    }
    mlflow.log_params(params)
    
    start_time = time.time()
    cat_model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.01,
        scale_pos_weight=scale_pos_weight,
        eval_metric='AUC',
        random_state=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    train_time = time.time() - start_time
    
    cat_metrics, cat_proba = evaluate_and_log_model(
        cat_model, "CatBoost",
        X_train, X_test, y_train, y_test,
        params, train_time
    )
    
    mlflow.log_metrics(cat_metrics)
    mlflow.catboost.log_model(cat_model, "model")
    mlflow.set_tag("model_family", "gradient_boosting")
    mlflow.set_tag("framework", "catboost")
    
    print("✓ Logged to MLflow!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎉 ALL MODELS LOGGED TO MLFLOW!")
print("=" * 80)

print(f"""
✓ Tracked 5 models in MLflow
✓ Logged all parameters
✓ Logged all metrics
✓ Saved all models

📊 VIEW YOUR EXPERIMENTS:
   http://127.0.0.1:5000

WHAT YOU CAN DO IN MLFLOW UI:
1. Compare all models side-by-side
2. See parameter impact on performance
3. Download any model
4. Register best model to Model Registry
5. Track experiment history

NEXT STEPS:
1. Open MLflow UI (http://127.0.0.1:5000)
2. Click "Loan-Default-Prediction" experiment
3. Compare runs
4. Register best model!
""")

print("=" * 80)