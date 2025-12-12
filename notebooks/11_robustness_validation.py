"""
Week 5: Model Robustness Validation Testing
Author: Anas Mahmoudi
Date: November 2025

This script validates model robustness by testing performance across:
- Different demographic segments
- Different income groups
- Different credit amount ranges
- Different age groups
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*80)
print("MODEL ROBUSTNESS VALIDATION TESTING")
print("="*80)

# Configuration
MODEL_PATH = 'models/production/lightgbm_final_proper_split.pkl'
X_TEST_PATH = 'data/processed/X_test.csv'
Y_TEST_PATH = 'data/processed/y_test.csv'
OUTPUT_DIR = 'outputs/robustness/'

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n[1/5] Loading model and test data...")
model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

print(f"  Model loaded: {MODEL_PATH}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Test default rate: {y_test.mean():.2%}")

# Function to evaluate segment
def evaluate_segment(X, y, segment_name):
    """Evaluate model performance on a data segment"""
    if len(X) == 0:
        return None
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    metrics = {
        'segment': segment_name,
        'n_samples': len(X),
        'default_rate': y.mean(),
        'auc_roc': roc_auc_score(y, y_pred_proba),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0)
    }
    
    return metrics

# Store all results
all_results = []

print(f"\n[2/5] Testing robustness across INCOME GROUPS...")
print("-" * 80)

# Create income groups
income_percentiles = [0, 33, 67, 100]
income_labels = ['Low Income', 'Medium Income', 'High Income']
X_test['INCOME_GROUP'] = pd.qcut(
    X_test['AMT_INCOME_TOTAL'], 
    q=3, 
    labels=income_labels,
    duplicates='drop'
)

for group in income_labels:
    mask = X_test['INCOME_GROUP'] == group
    X_subset = X_test[mask].drop('INCOME_GROUP', axis=1)
    y_subset = y_test[mask]
    
    metrics = evaluate_segment(X_subset, y_subset, group)
    if metrics:
        all_results.append(metrics)
        print(f"\n  {group}:")
        print(f"    Samples: {metrics['n_samples']:,}")
        print(f"    Default rate: {metrics['default_rate']:.2%}")
        print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")

# Remove temporary column
X_test = X_test.drop('INCOME_GROUP', axis=1)

print(f"\n[3/5] Testing robustness across CREDIT AMOUNT GROUPS...")
print("-" * 80)

# Create credit amount groups
credit_labels = ['Small Loan', 'Medium Loan', 'Large Loan']
X_test['CREDIT_GROUP'] = pd.qcut(
    X_test['AMT_CREDIT'], 
    q=3, 
    labels=credit_labels,
    duplicates='drop'
)

for group in credit_labels:
    mask = X_test['CREDIT_GROUP'] == group
    X_subset = X_test[mask].drop('CREDIT_GROUP', axis=1)
    y_subset = y_test[mask]
    
    metrics = evaluate_segment(X_subset, y_subset, group)
    if metrics:
        all_results.append(metrics)
        print(f"\n  {group}:")
        print(f"    Samples: {metrics['n_samples']:,}")
        print(f"    Default rate: {metrics['default_rate']:.2%}")
        print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")

# Remove temporary column
X_test = X_test.drop('CREDIT_GROUP', axis=1)

print(f"\n[4/5] Testing robustness across AGE GROUPS...")
print("-" * 80)

# Create age groups
X_test['AGE'] = -X_test['DAYS_BIRTH'] / 365
age_bins = [0, 30, 40, 50, 100]
age_labels = ['Young (< 30)', 'Adult (30-40)', 'Middle-aged (40-50)', 'Senior (50+)']
X_test['AGE_GROUP'] = pd.cut(
    X_test['AGE'], 
    bins=age_bins, 
    labels=age_labels
)

for group in age_labels:
    mask = X_test['AGE_GROUP'] == group
    X_subset = X_test[mask].drop(['AGE_GROUP', 'AGE'], axis=1)
    y_subset = y_test[mask]
    
    metrics = evaluate_segment(X_subset, y_subset, group)
    if metrics:
        all_results.append(metrics)
        print(f"\n  {group}:")
        print(f"    Samples: {metrics['n_samples']:,}")
        print(f"    Default rate: {metrics['default_rate']:.2%}")
        print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")

# Remove temporary columns
X_test = X_test.drop(['AGE_GROUP', 'AGE'], axis=1)

print(f"\n[5/5] Generating robustness report and visualizations...")

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Save results to CSV
results_df.to_csv(f'{OUTPUT_DIR}robustness_results.csv', index=False)
print(f"\n  Saved: {OUTPUT_DIR}robustness_results.csv")

# Visualization 1: AUC-ROC comparison across segments
plt.figure(figsize=(12, 6))
segments = results_df['segment']
x_pos = np.arange(len(segments))

plt.bar(x_pos, results_df['auc_roc'], color='steelblue', alpha=0.7)
plt.axhline(y=0.7712, color='red', linestyle='--', label='Overall Test AUC (0.7712)')
plt.xticks(x_pos, segments, rotation=45, ha='right')
plt.ylabel('AUC-ROC', fontsize=12)
plt.title('Model Performance Across Different Segments', fontsize=14, pad=20)
plt.ylim(0.7, 0.8)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}robustness_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}robustness_auc_comparison.png")

# Visualization 2: Recall comparison
plt.figure(figsize=(12, 6))
plt.bar(x_pos, results_df['recall'], color='green', alpha=0.7)
plt.axhline(y=0.6359, color='red', linestyle='--', label='Overall Test Recall (0.6359)')
plt.xticks(x_pos, segments, rotation=45, ha='right')
plt.ylabel('Recall', fontsize=12)
plt.title('Default Detection Rate (Recall) Across Segments', fontsize=14, pad=20)
plt.ylim(0.5, 0.8)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}robustness_recall_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}robustness_recall_comparison.png")

# Visualization 3: Heatmap of all metrics
plt.figure(figsize=(10, 8))
metrics_for_heatmap = results_df[['segment', 'auc_roc', 'precision', 'recall', 'f1_score']].set_index('segment')
sns.heatmap(metrics_for_heatmap.T, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0.0, vmax=1.0, cbar_kws={'label': 'Score'})
plt.title('Performance Metrics Heatmap Across Segments', fontsize=14, pad=20)
plt.ylabel('Metrics', fontsize=12)
plt.xlabel('Segments', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}robustness_metrics_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUTPUT_DIR}robustness_metrics_heatmap.png")

# Calculate stability metrics
print(f"\n{'='*80}")
print(f"ROBUSTNESS ANALYSIS SUMMARY")
print(f"{'='*80}")

auc_std = results_df['auc_roc'].std()
auc_min = results_df['auc_roc'].min()
auc_max = results_df['auc_roc'].max()
auc_range = auc_max - auc_min

print(f"\nPerformance Stability (AUC-ROC):")
print(f"  Standard Deviation: {auc_std:.4f}")
print(f"  Min AUC: {auc_min:.4f} ({results_df.loc[results_df['auc_roc'].idxmin(), 'segment']})")
print(f"  Max AUC: {auc_max:.4f} ({results_df.loc[results_df['auc_roc'].idxmax(), 'segment']})")
print(f"  Range: {auc_range:.4f}")

if auc_std < 0.02:
    stability = "EXCELLENT - Very stable across segments"
elif auc_std < 0.05:
    stability = "GOOD - Reasonably stable across segments"
else:
    stability = "POOR - High variance across segments"

print(f"\n  Overall Stability: {stability}")

# Identify problematic segments
print(f"\nSegments to Monitor:")
overall_auc = 0.7712
threshold = 0.05

for _, row in results_df.iterrows():
    diff = overall_auc - row['auc_roc']
    if abs(diff) > threshold:
        status = "underperforming" if diff > 0 else "overperforming"
        print(f"  - {row['segment']}: AUC {row['auc_roc']:.4f} ({status} by {abs(diff):.4f})")

print(f"\n{'='*80}")
print(f"ROBUSTNESS VALIDATION COMPLETE")
print(f"{'='*80}")
print(f"\nGenerated files:")
print(f"  - {OUTPUT_DIR}robustness_results.csv")
print(f"  - {OUTPUT_DIR}robustness_auc_comparison.png")
print(f"  - {OUTPUT_DIR}robustness_recall_comparison.png")
print(f"  - {OUTPUT_DIR}robustness_metrics_heatmap.png")
print(f"\nWeek 5 Complete! Ready to proceed to Week 6: Apache Airflow")
print("="*80)