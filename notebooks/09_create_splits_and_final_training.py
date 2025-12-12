"""
Create Persistent Train/Test Splits and Train Final Production Model
Project: Loan Default Prediction MLOps
Author: Mahmo
Date: November 2025

This script:
1. Creates reproducible 80/20 stratified train/test splits
2. Saves splits as CSV files for DVC tracking
3. Trains final production model with Optuna-optimized hyperparameters
4. Evaluates on held-out test set
5. Logs results to MLflow for comparison
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import joblib
from pathlib import Path
import time
from datetime import datetime

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_TRACKING_URI = "./mlruns"  # Use local file-based tracking
EXPERIMENT_NAME = "Loan-Default-Prediction"

print("=" * 80)
print("FINAL PRODUCTION MODEL TRAINING WITH PROPER TRAIN/TEST SPLIT")
print("=" * 80)
print(f"Configuration:")
print(f"  - Train/Test Split: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
print(f"  - Random State: {RANDOM_STATE}")
print(f"  - Stratified: Yes")
print("=" * 80 + "\n")

# ============================================================================
# STEP 1: Load Full Dataset
# ============================================================================
print("[1/7] Loading full dataset...")
df = pd.read_csv("data/processed/train_features.csv")

# Remove string columns if any
object_cols = df.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print(f"  Dropping {len(object_cols)} string columns: {object_cols}")
    df = df.drop(columns=object_cols)

print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Separate features and target
if 'SK_ID_CURR' in df.columns:
    X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
    ids = df['SK_ID_CURR']
else:
    X = df.drop(columns=['TARGET'])
    ids = None

y = df['TARGET']

print(f"  Features: {X.shape[1]}")
print(f"  Samples: {len(X):,}")
print(f"  Target distribution:")
print(f"    - No Default (0): {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
print(f"    - Default (1): {(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")
print(f"    - Imbalance ratio: {(y==0).sum()/(y==1).sum():.2f}:1")

# ============================================================================
# STEP 2: Create Stratified Train/Test Split
# ============================================================================
print(f"\n[2/7] Creating {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} stratified train/test split...")
print(f"  Random state: {RANDOM_STATE}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"  Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ============================================================================
# STEP 3: Verify Class Balance Preservation
# ============================================================================
print(f"\n[3/7] Verifying class balance preservation...")

train_default_rate = y_train.mean()
test_default_rate = y_test.mean()
original_default_rate = y.mean()

print(f"  Original default rate: {original_default_rate*100:.2f}%")
print(f"  Train default rate: {train_default_rate*100:.2f}%")
print(f"  Test default rate: {test_default_rate*100:.2f}%")
print(f"  Difference (train-test): {abs(train_default_rate - test_default_rate)*100:.4f}%")

# Check if stratification worked
if abs(train_default_rate - test_default_rate) < 0.01:
    print("  Status: PASS - Class balance preserved")
else:
    print("  Status: WARNING - Significant imbalance detected")

print(f"\n  Train set distribution:")
print(f"    - No Default (0): {(y_train==0).sum():,} ({(y_train==0).mean()*100:.2f}%)")
print(f"    - Default (1): {(y_train==1).sum():,} ({(y_train==1).mean()*100:.2f}%)")

print(f"\n  Test set distribution:")
print(f"    - No Default (0): {(y_test==0).sum():,} ({(y_test==0).mean()*100:.2f}%)")
print(f"    - Default (1): {(y_test==1).sum():,} ({(y_test==1).mean()*100:.2f}%)")

# ============================================================================
# STEP 4: Save Train/Test Splits as CSV Files
# ============================================================================
print(f"\n[4/7] Saving train/test splits to CSV files...")

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Save splits
X_train.to_csv(output_dir / "X_train.csv", index=False)
X_test.to_csv(output_dir / "X_test.csv", index=False)
y_train.to_csv(output_dir / "y_train.csv", index=False, header=['TARGET'])
y_test.to_csv(output_dir / "y_test.csv", index=False, header=['TARGET'])

print(f"  Saved: data/processed/X_train.csv ({X_train.shape[0]:,} x {X_train.shape[1]})")
print(f"  Saved: data/processed/X_test.csv ({X_test.shape[0]:,} x {X_test.shape[1]})")
print(f"  Saved: data/processed/y_train.csv ({len(y_train):,} samples)")
print(f"  Saved: data/processed/y_test.csv ({len(y_test):,} samples)")

# ============================================================================
# STEP 5: Train Final Production Model with Optuna-Optimized Hyperparameters
# ============================================================================
print(f"\n[5/7] Training final production model...")
print("=" * 80)

# Optuna-optimized hyperparameters from study
optuna_best_params = {
    'learning_rate': 0.0198,
    'n_estimators': 650,
    'max_depth': 10,
    'num_leaves': 94,
    'min_child_samples': 40,
    'min_child_weight': 1.219,
    'subsample': 0.8515,
    'subsample_freq': 5,
    'colsample_bytree': 0.5489,
    'reg_alpha': 2.826e-08,
    'reg_lambda': 3.058e-05,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
    'class_weight': 'balanced'
}

print("Hyperparameters (from Optuna optimization):")
for param, value in optuna_best_params.items():
    if param not in ['random_state', 'n_jobs', 'verbose', 'class_weight']:
        print(f"  {param}: {value}")

print("\nStarting training...")
print("  This will take 5-15 minutes...")
print("-" * 80)

# Start MLflow run
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="LightGBM_Final_Production_Proper_Split"):

    # Log parameters
    mlflow.log_params(optuna_best_params)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("features", X_train.shape[1])
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("stratified", True)

    # Create and train model
    model = lgb.LGBMClassifier(**optuna_best_params)

    start_time = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.log_evaluation(period=50)]
    )

    train_time = time.time() - start_time

    print("-" * 80)
    print(f"Training completed in {train_time:.1f} seconds ({train_time/60:.2f} minutes)")

    mlflow.log_metric("train_time_seconds", train_time)

    # ========================================================================
    # STEP 6: Evaluate on Test Set
    # ========================================================================
    print(f"\n[6/7] Evaluating on held-out test set...")
    print("=" * 80)

    # Training set predictions
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]

    # Test set predictions
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics for training set
    train_metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_pred_proba)
    }

    # Calculate metrics for test set
    test_metrics = {
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_pred_proba)
    }

    # Log all metrics to MLflow
    mlflow.log_metrics(train_metrics)
    mlflow.log_metrics(test_metrics)

    # Calculate overfitting gap
    overfit_gap = train_metrics['train_auc'] - test_metrics['test_auc']
    mlflow.log_metric("overfit_gap_auc", overfit_gap)

    # Print results
    print("\nTraining Set Performance:")
    print(f"  Accuracy:  {train_metrics['train_accuracy']:.4f}")
    print(f"  Precision: {train_metrics['train_precision']:.4f}")
    print(f"  Recall:    {train_metrics['train_recall']:.4f}")
    print(f"  F1-Score:  {train_metrics['train_f1']:.4f}")
    print(f"  AUC-ROC:   {train_metrics['train_auc']:.4f}")

    print("\nTest Set Performance (FINAL HONEST METRICS):")
    print(f"  Accuracy:  {test_metrics['test_accuracy']:.4f}")
    print(f"  Precision: {test_metrics['test_precision']:.4f}")
    print(f"  Recall:    {test_metrics['test_recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['test_f1']:.4f}")
    print(f"  AUC-ROC:   {test_metrics['test_auc']:.4f}")

    print(f"\nOverfitting Analysis:")
    print(f"  Train AUC: {train_metrics['train_auc']:.4f}")
    print(f"  Test AUC:  {test_metrics['test_auc']:.4f}")
    print(f"  Gap:       {overfit_gap:.4f} ({overfit_gap/train_metrics['train_auc']*100:.2f}%)")

    if overfit_gap < 0.02:
        print("  Status: Excellent - Minimal overfitting")
    elif overfit_gap < 0.05:
        print("  Status: Good - Acceptable overfitting")
    else:
        print("  Status: Warning - Significant overfitting detected")

    # Confusion Matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix (Test Set):")
    print(f"                Predicted")
    print(f"                No      Yes")
    print(f"  Actual No   {cm_test[0,0]:6d}  {cm_test[0,1]:6d}")
    print(f"  Actual Yes  {cm_test[1,0]:6d}  {cm_test[1,1]:6d}")

    # Business interpretation
    total_defaults = cm_test[1,0] + cm_test[1,1]
    caught_defaults = cm_test[1,1]
    missed_defaults = cm_test[1,0]
    false_alarms = cm_test[0,1]

    print(f"\nBusiness Impact Analysis:")
    print(f"  Total defaults in test set: {total_defaults:,}")
    print(f"  Caught defaults: {caught_defaults:,} ({caught_defaults/total_defaults*100:.1f}%)")
    print(f"  Missed defaults: {missed_defaults:,} ({missed_defaults/total_defaults*100:.1f}%) - Lost money")
    print(f"  False alarms: {false_alarms:,} - Missed opportunities")

    # Save model
    print(f"\n[7/7] Saving final production model...")

    output_model_dir = Path("models/production")
    output_model_dir.mkdir(parents=True, exist_ok=True)

    # Save with joblib
    model_path = output_model_dir / "lightgbm_final_proper_split.pkl"
    joblib.dump(model, model_path)
    print(f"  Saved: {model_path}")

    # Log model to MLflow (skip if fails)
    try:
        mlflow.lightgbm.log_model(model, "model")
        print(f"  Logged to MLflow")
    except Exception as e:
        print(f"  MLflow model logging skipped (error: {str(e)[:100]})")

    # Add tags
    mlflow.set_tag("model_type", "production")
    mlflow.set_tag("training_strategy", "train_test_split")
    mlflow.set_tag("hyperparameter_source", "optuna_optimized")
    mlflow.set_tag("train_samples", str(len(X_train)))
    mlflow.set_tag("test_samples", str(len(X_test)))
    mlflow.set_tag("test_auc", f"{test_metrics['test_auc']:.4f}")

    run_id = mlflow.active_run().info.run_id
    print(f"  MLflow Run ID: {run_id}")

# End MLflow run
mlflow.end_run()

# ============================================================================
# Save Feature Importance
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Save to CSV
feature_importance_path = Path("outputs/feature_importance_final_proper_split.csv")
feature_importance.to_csv(feature_importance_path, index=False)
print(f"\nSaved: {feature_importance_path}")

# ============================================================================
# Generate Comparison Report
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON REPORT")
print("=" * 80)

comparison_data = {
    'Model': [
        'Baseline LightGBM',
        'Optuna-Optimized LightGBM',
        'Final Production Model (Optuna Hyperparams)'
    ],
    'Training Strategy': [
        '80/20 split, default params',
        '80/20 split + 5-fold CV, optimized params',
        '80/20 split, Optuna best params'
    ],
    'Test AUC-ROC': [
        0.7558,  # From benchmarking
        0.7709,  # From Optuna
        test_metrics['test_auc']  # Current run
    ],
    'Test Recall': [
        0.6540,  # From benchmarking
        'N/A',   # Optuna didn't save this
        test_metrics['test_recall']
    ],
    'Test Precision': [
        0.1718,  # From benchmarking
        'N/A',
        test_metrics['test_precision']
    ],
    'Test F1-Score': [
        0.2721,  # From benchmarking
        'N/A',
        test_metrics['test_f1']
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Save comparison
comparison_path = Path("outputs/final_model_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)
print(f"\nSaved: {comparison_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 80)

print(f"""
Data Splits:
  Train samples: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)
  Test samples: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)
  Features: {X_train.shape[1]}

Model Configuration:
  Algorithm: LightGBM
  Hyperparameters: Optuna-optimized (50 trials)
  Training time: {train_time/60:.2f} minutes

Performance (Test Set):
  AUC-ROC: {test_metrics['test_auc']:.4f}
  Recall: {test_metrics['test_recall']:.4f} (catches {test_metrics['test_recall']*100:.1f}% of defaults)
  Precision: {test_metrics['test_precision']:.4f}
  F1-Score: {test_metrics['test_f1']:.4f}

Comparison vs Previous Models:
  Baseline (default params): {0.7558:.4f} AUC
  Optuna-optimized: {0.7709:.4f} AUC
  Final Production: {test_metrics['test_auc']:.4f} AUC
  Improvement vs Baseline: {(test_metrics['test_auc'] - 0.7558)*100:.2f}%

Files Created:
  - data/processed/X_train.csv
  - data/processed/X_test.csv
  - data/processed/y_train.csv
  - data/processed/y_test.csv
  - models/production/lightgbm_final_proper_split.pkl
  - outputs/feature_importance_final_proper_split.csv
  - outputs/final_model_comparison.csv

Next Steps:
  1. Review test set performance metrics
  2. Add DVC tracking for train/test split files
  3. Commit to Git with proper message
  4. Ready for Week 4: API Development
""")

print("=" * 80)
print("ALL DONE!")
print("=" * 80)
