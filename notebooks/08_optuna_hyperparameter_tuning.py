"""
Hyperparameter Optimization using Optuna for LightGBM Classifier
Project: Loan Default Prediction
Author: Mahmo
Date: November 2025
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import joblib
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_TRIALS = 50
TIMEOUT = 3600
CV_FOLDS = 5

print("="*80)
print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
print("="*80)
print(f"Configuration: {N_TRIALS} trials, {CV_FOLDS}-fold CV, timeout: {TIMEOUT/60:.0f}min")
print("="*80 + "\n")

# Load data
print("[1/8] Loading data...")
train_data = pd.read_csv('data/processed/train_features.csv')

if 'TARGET' in train_data.columns:
    X = train_data.drop('TARGET', axis=1)
    y = train_data['TARGET'].values
elif 'target' in train_data.columns:
    X = train_data.drop('target', axis=1)
    y = train_data['target'].values
else:
    raise ValueError("Target column not found in data")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Class distribution - 0: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print(f"Class distribution - 1: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
print(f"Imbalance ratio: {(y_train==0).sum()/(y_train==1).sum():.2f}:1\n")

# Define objective function
print("[2/8] Configuring objective function...")

def objective(trial):
    """
    Objective function for Optuna optimization.
    Uses stratified k-fold cross-validation with pruning.
    """
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced',
        
        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        pruning_callback = LightGBMPruningCallback(trial, 'auc')
        
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            callbacks=[pruning_callback]
        )
        
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        fold_score = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(fold_score)
        
        trial.report(fold_score, fold_idx)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    mean_score = np.mean(cv_scores)
    trial.set_user_attr('cv_std', np.std(cv_scores))
    
    return mean_score

print("Objective function configured\n")

# Create Optuna study
print("[3/8] Creating Optuna study...")

mlflow.set_experiment("Optuna-Hyperparameter-Tuning")

study = optuna.create_study(
    direction='maximize',
    study_name=f'lightgbm_opt_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=3,
        interval_steps=1
    )
)

print("Study created: TPESampler + MedianPruner\n")

# Run optimization
print("[4/8] Running optimization...")
print("This may take 30-45 minutes...\n")

start_time = datetime.now()

with mlflow.start_run(run_name="Optuna_Study_Main"):
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True,
        n_jobs=1
    )
    
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_cv_auc_roc", study.best_value)
    mlflow.log_metric("n_trials", len(study.trials))
    mlflow.log_metric("n_completed", len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
    mlflow.log_metric("n_pruned", len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

print(f"\nOptimization completed in {duration/60:.1f} minutes")
print(f"Trials completed: {n_completed}, pruned: {n_pruned}")
print(f"Best CV AUC-ROC: {study.best_value:.4f}")
print(f"Best trial: #{study.best_trial.number}\n")

# Display best hyperparameters
print("[5/8] Best hyperparameters:")
print("="*80)
for param, value in study.best_params.items():
    print(f"{param:25s}: {value}")
print("="*80 + "\n")

# Train final model
print("[6/8] Training final model with best hyperparameters...")

best_params = study.best_params.copy()
best_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE
})

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train, y_train)

# Evaluate final model
y_train_pred = final_model.predict(X_train)
y_train_pred_proba = final_model.predict_proba(X_train)[:, 1]
y_test_pred = final_model.predict(X_test)
y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]

train_metrics = {
    'auc_roc': roc_auc_score(y_train, y_train_pred_proba),
    'accuracy': accuracy_score(y_train, y_train_pred),
    'precision': precision_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred),
    'f1_score': f1_score(y_train, y_train_pred)
}

test_metrics = {
    'auc_roc': roc_auc_score(y_test, y_test_pred_proba),
    'accuracy': accuracy_score(y_test, y_test_pred),
    'precision': precision_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),
    'f1_score': f1_score(y_test, y_test_pred)
}

print("\nFinal model performance:")
print("="*80)
print(f"{'Metric':<20} {'Train':<15} {'Test':<15} {'Difference':<15}")
print("-"*80)
for metric in train_metrics.keys():
    diff = test_metrics[metric] - train_metrics[metric]
    print(f"{metric.upper():<20} {train_metrics[metric]:<15.4f} {test_metrics[metric]:<15.4f} {diff:+.4f}")
print("="*80 + "\n")

# Compare with baseline
print("[7/8] Comparing with baseline model...")

try:
    baseline_model = joblib.load('models/trained/best_model.pkl')
    y_test_pred_baseline = baseline_model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, y_test_pred_baseline)
except (FileNotFoundError, Exception) as e:
    print(f"Cannot load baseline model: {e}")
    print("Using reported baseline value from previous work")
    baseline_auc = 0.7558

improvement_abs = test_metrics['auc_roc'] - baseline_auc
improvement_rel = (improvement_abs / baseline_auc) * 100

print("\nBaseline vs Optimized:")
print("="*80)
print(f"Baseline AUC-ROC:      {baseline_auc:.4f}")
print(f"Optimized AUC-ROC:     {test_metrics['auc_roc']:.4f}")
print(f"Absolute improvement:  {improvement_abs:+.4f}")
print(f"Relative improvement:  {improvement_rel:+.2f}%")
print("="*80 + "\n")

# Save results
print("[8/8] Saving results...")

os.makedirs('models/trained', exist_ok=True)
os.makedirs('models/optuna', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

joblib.dump(final_model, 'models/trained/lightgbm_optimized.pkl')
joblib.dump(study, 'models/optuna/optuna_study.pkl')

print("Saved: models/trained/lightgbm_optimized.pkl")
print("Saved: models/optuna/optuna_study.pkl")

# Log to MLflow
with mlflow.start_run(run_name="LightGBM_Optimized_Final"):
    mlflow.log_params(best_params)
    
    for metric, value in train_metrics.items():
        mlflow.log_metric(f"train_{metric}", value)
    for metric, value in test_metrics.items():
        mlflow.log_metric(f"test_{metric}", value)
    
    mlflow.log_metric("improvement_vs_baseline_pct", improvement_rel)
    mlflow.lightgbm.log_model(final_model, "model")

print("Logged to MLflow")

# Save summary
summary_file = 'outputs/optuna_optimization_summary.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("OPTUNA OPTIMIZATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Duration: {duration/60:.1f} minutes\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write(f"  Trials: {N_TRIALS}\n")
    f.write(f"  Completed: {n_completed}\n")
    f.write(f"  Pruned: {n_pruned}\n")
    f.write(f"  CV folds: {CV_FOLDS}\n\n")
    
    f.write("RESULTS:\n")
    f.write(f"  Best CV AUC-ROC: {study.best_value:.4f}\n")
    f.write(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f}\n")
    f.write(f"  Improvement vs baseline: {improvement_rel:+.2f}%\n\n")
    
    f.write("BEST HYPERPARAMETERS:\n")
    for param, value in study.best_params.items():
        f.write(f"  {param}: {value}\n")

print(f"Saved: {summary_file}")

# Create basic visualizations
print("\nCreating visualizations...")
os.makedirs('outputs/optuna_plots', exist_ok=True)

try:
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    plt.tight_layout()
    plt.savefig('outputs/optuna_plots/optimization_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/optuna_plots/optimization_history.png")
except Exception as e:
    print(f"Could not create optimization_history.png: {e}")

try:
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Hyperparameter Importances')
    plt.tight_layout()
    plt.savefig('outputs/optuna_plots/param_importances.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/optuna_plots/param_importances.png")
except Exception as e:
    print(f"Could not create param_importances.png: {e}")

# Final summary
print("\n" + "="*80)
print("OPTIMIZATION COMPLETED")
print("="*80)
print(f"Best CV AUC-ROC:        {study.best_value:.4f}")
print(f"Test AUC-ROC:           {test_metrics['auc_roc']:.4f}")
print(f"Improvement:            {improvement_rel:+.2f}%")
print(f"Duration:               {duration/60:.1f} minutes")
print(f"Trials executed:        {len(study.trials)}")
print(f"Pruning efficiency:     {n_pruned} trials stopped early")
print("\nFiles generated:")
print("  models/trained/lightgbm_optimized.pkl")
print("  models/optuna/optuna_study.pkl")
print("  outputs/optuna_optimization_summary.txt")
print("  outputs/optuna_plots/")
print("\nNext steps:")
print("  1. Review visualizations in outputs/optuna_plots/")
print("  2. Check MLflow UI: mlflow ui")
print("  3. Run detailed analysis: python notebooks/06_optuna_analysis.py")
print("  4. Proceed to imbalance handling with SMOTE")
print("="*80)