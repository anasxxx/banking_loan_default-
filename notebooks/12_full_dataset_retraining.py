import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import optuna
import joblib
from datetime import datetime
import json

print("="*80)
print("FULL DATASET RETRAINING WITH ENHANCED FEATURES")
print("="*80)

print("\nLoading enhanced dataset...")
df = pd.read_csv('data/processed/train_features_enhanced.csv')
print(f"Initial shape: {df.shape}")

# Keep only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[numeric_cols]

# Remove ID column if present
if 'SK_ID_CURR' in df.columns:
    df = df.drop('SK_ID_CURR', axis=1)

# Verify no object columns remain
object_cols = df.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"\nWARNING: Found {len(object_cols)} object columns, dropping them")
    df = df.drop(columns=object_cols)

print(f"After preprocessing: {df.shape}")
print(f"Features (excluding TARGET): {df.shape[1] - 1}")

# Verify TARGET exists
if 'TARGET' not in df.columns:
    raise ValueError("TARGET column not found")

# Separate features and target
X = df.drop('TARGET', axis=1)
y = df['TARGET']

print(f"\nDataset summary:")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]:,}")
print(f"Data types: {X.dtypes.value_counts().to_dict()}")

# Train/test split
print(f"\nCreating train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Default rate (train): {y_train.mean():.2%}")
print(f"Default rate (test): {y_test.mean():.2%}")


def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)


print("\n" + "="*80)
print("STARTING OPTUNA OPTIMIZATION")
print("="*80)
print(f"Target: 500 trials (max 10 hours)")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)

study.optimize(objective, n_trials=500, timeout=36000, show_progress_bar=True)

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Best AUC: {study.best_value:.4f}")
print(f"Total trials: {len(study.trials)}")
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")


print("\n" + "="*80)
print("TRAINING FINAL MODEL")
print("="*80)

final_params = study.best_params.copy()
final_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
})

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc'
)

y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

metrics = {
    'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
    'recall': float(recall_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'f1_score': float(f1_score(y_test, y_pred)),
    'n_features': int(X_train.shape[1]),
    'n_samples_train': int(len(X_train)),
    'n_samples_test': int(len(X_test)),
    'optuna_trials': len(study.trials),
    'trained_at': datetime.now().isoformat(),
    'best_params': study.best_params
}

print("\n" + "="*80)
print("FINAL PRODUCTION MODEL METRICS")
print("="*80)
print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"F1-Score:  {metrics['f1_score']:.4f}")
print(f"\nFeatures:  {metrics['n_features']}")
print(f"Train samples: {metrics['n_samples_train']:,}")
print(f"Test samples:  {metrics['n_samples_test']:,}")
print(f"Optuna trials: {metrics['optuna_trials']}")


print("\n" + "="*80)
print("SAVING MODEL AND METRICS")
print("="*80)

model_path = 'models/production/lightgbm_full_optimized.pkl'
joblib.dump(final_model, model_path)
print(f"Model saved: {model_path}")

metrics_path = 'models/production/metrics_final.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved: {metrics_path}")

study_path = 'models/production/optuna_study_final.pkl'
joblib.dump(study, study_path)
print(f"Study saved: {study_path}")

print("\n" + "="*80)
print("RETRAINING COMPLETE")
print("="*80)
print(f"Previous AUC: 0.7732")
print(f"New AUC:      {metrics['auc_roc']:.4f}")
print(f"Improvement:  {metrics['auc_roc'] - 0.7732:+.4f}")
print("="*80)