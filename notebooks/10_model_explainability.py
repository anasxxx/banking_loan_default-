"""
Week 5: Model Explainability Analysis using SHAP
Author: Anas Mahmoudi
Date: November 2025

This script analyzes the trained LightGBM model using SHAP values to understand:
- Which features are most important globally
- How features contribute to individual predictions
- Feature interactions and dependencies
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*80)
print("MODEL EXPLAINABILITY ANALYSIS - SHAP")
print("="*80)

# Configuration
MODEL_PATH = 'models/production/lightgbm_final_proper_split.pkl'
X_TEST_PATH = 'data/processed/X_test.csv'
Y_TEST_PATH = 'data/processed/y_test.csv'
OUTPUT_DIR = 'outputs/explainability/'
N_SAMPLES = 1000  # Use subset for faster computation

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n[1/6] Loading model and test data...")
model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

print(f"  Model loaded: {MODEL_PATH}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Features: {len(X_test.columns)}")

# Use subset for faster computation
X_test_sample = X_test.head(N_SAMPLES)
y_test_sample = y_test[:N_SAMPLES]

print(f"\n[2/6] Creating SHAP explainer...")
print(f"  Using {N_SAMPLES} samples for analysis")

# Create SHAP explainer for tree-based models
explainer = shap.TreeExplainer(model)

print(f"\n[3/6] Computing SHAP values...")
print(f"  This may take a few minutes...")

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_sample)

# For binary classification, SHAP returns values for class 1 (default)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Get values for positive class

print(f"  SHAP values computed successfully")
print(f"  Shape: {shap_values.shape}")

print(f"\n[4/6] Generating SHAP visualizations...")

# Visualization 1: Summary Plot (Bar) - Global Feature Importance
print(f"  Creating summary plot (bar)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.title('Global Feature Importance (SHAP)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}shap_importance_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}shap_importance_bar.png")

# Visualization 2: Summary Plot (Beeswarm) - Feature Impact Distribution
print(f"  Creating summary plot (beeswarm)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title('Feature Impact Distribution (SHAP)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}shap_summary_beeswarm.png")

# Visualization 3: Feature Importance (Top 20)
print(f"  Creating feature importance plot...")
feature_importance = pd.DataFrame({
    'feature': X_test_sample.columns,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance.head(20)['feature'][::-1], 
         feature_importance.head(20)['importance'][::-1])
plt.xlabel('Mean |SHAP value|', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Features by SHAP Importance', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}shap_top20_features.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {OUTPUT_DIR}shap_top20_features.png")

# Save feature importance to CSV
feature_importance.to_csv(f'{OUTPUT_DIR}shap_feature_importance.csv', index=False)
print(f"    Saved: {OUTPUT_DIR}shap_feature_importance.csv")

print(f"\n[5/6] Analyzing top features...")
print(f"\nTop 10 Most Important Features (SHAP):")
print("-" * 60)
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

print(f"\n[6/6] Generating individual prediction explanations...")

# Get 5 random predictions to explain
sample_indices = np.random.choice(len(X_test_sample), 5, replace=False)

for i, idx in enumerate(sample_indices):
    print(f"\n  Sample {i+1}/5 (Index {idx}):")
    
    # Get prediction
    pred_proba = model.predict_proba(X_test_sample.iloc[idx:idx+1])[0][1]
    actual = y_test_sample[idx]
    
    print(f"    Actual: {'Default' if actual == 1 else 'No Default'}")
    print(f"    Predicted probability: {pred_proba:.2%}")
    
    # Create waterfall plot
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test_sample.iloc[idx],
            feature_names=X_test_sample.columns.tolist()
        ),
        show=False
    )
    plt.title(f'Prediction Explanation - Sample {i+1}\n'
              f'Actual: {"Default" if actual == 1 else "No Default"} | '
              f'Predicted: {pred_proba:.1%}',
              fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}shap_waterfall_sample_{i+1}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n    Saved 5 waterfall plots to {OUTPUT_DIR}")

print(f"\n{'='*80}")
print(f"SHAP ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"\nGenerated files:")
print(f"  - {OUTPUT_DIR}shap_importance_bar.png")
print(f"  - {OUTPUT_DIR}shap_summary_beeswarm.png")
print(f"  - {OUTPUT_DIR}shap_top20_features.png")
print(f"  - {OUTPUT_DIR}shap_feature_importance.csv")
print(f"  - {OUTPUT_DIR}shap_waterfall_sample_*.png (5 files)")
print(f"\nNext step: Run robustness validation testing")
print("="*80)