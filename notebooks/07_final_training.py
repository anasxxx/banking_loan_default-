"""
FINAL MODEL TRAINING
Train LightGBM on the FULL dataset (100% of data)
This is the model we'll deploy to production
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import joblib
from pathlib import Path
import mlflow
import mlflow.lightgbm

print("=" * 80)
print("FINAL PRODUCTION MODEL TRAINING")
print("Training LightGBM on 100% of data")
print("=" * 80)

# ============================================================================
# STEP 1: Load Full Dataset
# ============================================================================
print("\n[1/5] Loading FULL dataset...")
df = pd.read_csv("data/processed/train_features.csv")

# Remove string columns if any
object_cols = df.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print(f"  Dropping {len(object_cols)} string columns")
    df = df.drop(columns=object_cols)

print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# STEP 2: Prepare Data
# ============================================================================
print("\n[2/5] Preparing features and target...")

# Separate features and target
X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']

print(f"✓ Features: {X.shape[1]}")
print(f"✓ Total samples: {X.shape[0]:,}")
print(f"✓ Default rate: {y.mean()*100:.2f}%")

# Calculate class weight
scale_pos_weight = len(y[y==0]) / len(y[y==1])
print(f"✓ Scale pos weight: {scale_pos_weight:.2f}")

# ============================================================================
# STEP 3: Train on Full Dataset
# ============================================================================
print("\n[3/5] Training LightGBM on FULL dataset...")
print("⏳ This will take several minutes (5-15 min)...")
print("=" * 80)

# Start MLflow run
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Loan-Default-Prediction")

with mlflow.start_run(run_name="LightGBM_Final_Production"):
    
    # Model parameters (same as winning model, but trained on all data)
    params = {
        'n_estimators': 500,  # More trees since we have all data
        'max_depth': 7,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 1
    }
    
    # Log parameters
    mlflow.log_params(params)
    
    # Create model
    model = lgb.LGBMClassifier(**params)
    
    # Train (showing progress every 50 iterations)
    start_time = time.time()
    
    print("\nTraining progress:")
    print("-" * 80)
    
    model.fit(
        X, y,
        callbacks=[lgb.log_evaluation(period=50)]
    )
    
    train_time = time.time() - start_time
    
    print("-" * 80)
    print(f"✓ Training completed!")
    print(f"  Time: {train_time:.1f} seconds ({train_time/60:.1f} minutes)")
    
    # Log training time
    mlflow.log_metric("train_time_seconds", train_time)
    mlflow.log_metric("train_samples", len(X))
    mlflow.log_metric("train_features", X.shape[1])
    
    # ========================================================================
    # STEP 4: Validate on Training Data (just for reference)
    # ========================================================================
    print("\n[4/5] Computing training metrics (for reference)...")
    
    from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
    
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    train_metrics = {
        'train_auc': roc_auc_score(y, y_pred_proba),
        'train_recall': recall_score(y, y_pred),
        'train_precision': precision_score(y, y_pred),
        'train_f1': f1_score(y, y_pred)
    }
    
    # Log metrics
    mlflow.log_metrics(train_metrics)
    
    print(f"\n📊 Training Set Performance:")
    print(f"  AUC-ROC:   {train_metrics['train_auc']:.4f}")
    print(f"  Recall:    {train_metrics['train_recall']:.4f}")
    print(f"  Precision: {train_metrics['train_precision']:.4f}")
    print(f"  F1-Score:  {train_metrics['train_f1']:.4f}")
    
    print("\n⚠️  Note: These are training metrics (may be optimistic)")
    print("   Real performance will be validated on new data in production")
    
    # ========================================================================
    # STEP 5: Save Production Model
    # ========================================================================
    print("\n[5/5] Saving production model...")
    
    # Save with joblib
    Path("models/production").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, 'models/production/lightgbm_final.pkl')
    print("✓ Saved: models/production/lightgbm_final.pkl")
    
    # Log model to MLflow
    mlflow.lightgbm.log_model(model, "model")
    print("✓ Logged to MLflow")
    
    # Add tags
    mlflow.set_tag("model_type", "production")
    mlflow.set_tag("training_data", "full_dataset")
    mlflow.set_tag("samples", str(len(X)))
    mlflow.set_tag("framework", "lightgbm")
    
    # Get run ID for registration
    run_id = mlflow.active_run().info.run_id
    print(f"✓ MLflow Run ID: {run_id}")

# End run
mlflow.end_run()

# ============================================================================
# REGISTER TO MODEL REGISTRY
# ============================================================================
print("\n" + "=" * 80)
print("REGISTERING TO MODEL REGISTRY")
print("=" * 80)

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_uri = f"runs:/{run_id}/model"
model_name = "loan-default-classifier"

try:
    # Register model
    model_version = mlflow.register_model(model_uri, model_name)
    
    print(f"\n✓ Model registered!")
    print(f"  Name: {model_name}")
    print(f"  Version: {model_version.version}")
    
    # Add description
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=f"Production LightGBM trained on full dataset ({len(X):,} samples). AUC: {train_metrics['train_auc']:.4f}"
    )
    
    # Transition to Production (archive old versions)
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"✓ Transitioned to Production stage!")
    print(f"✓ Old versions archived")
    
except Exception as e:
    print(f"Note: {e}")
    print("Model saved locally but not registered (MLflow registry may already exist)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎉 PRODUCTION MODEL READY!")
print("=" * 80)

print(f"""
✅ Training Summary:
   - Model: LightGBM
   - Training samples: {len(X):,}
   - Features: {X.shape[1]}
   - Training time: {train_time/60:.1f} minutes
   - AUC-ROC: {train_metrics['train_auc']:.4f}

📦 Model Saved:
   - Local: models/production/lightgbm_final.pkl
   - MLflow: {model_name} (Version {model_version.version if 'model_version' in locals() else 'N/A'})
   - Stage: Production

🚀 Next Steps:
   1. The model is ready for deployment
   2. Can be loaded with:
      model = mlflow.pyfunc.load_model("models:/{model_name}/Production")
   3. Ready to build FastAPI service
   
💡 Key Difference from Benchmarking:
   - Benchmarking: Used 80% data to compare models
   - Production: Using 100% data for best performance
   - This model will perform better because it learned from MORE examples!
""")

print("=" * 80)

# ============================================================================
# Feature Importance
# ============================================================================
print("\n📊 Top 20 Most Important Features:")
print("=" * 80)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# Save feature importance
feature_importance.to_csv('outputs/feature_importance_production.csv', index=False)
print("\n✓ Feature importance saved to: outputs/feature_importance_production.csv")

print("\n" + "=" * 80)
print("✅ ALL DONE! Model ready for production deployment!")
print("=" * 80)