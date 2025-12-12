"""
Detailed Analysis of Optuna Optimization Results
Project: Loan Default Prediction
Author: Mahmo
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
print("Loading Optuna study and models...")

study = joblib.load('models/optuna/optuna_study.pkl')
optimized_model = joblib.load('models/trained/lightgbm_optimized.pkl')

try:
    baseline_model = joblib.load('models/trained/best_model.pkl')
    has_baseline = True
except FileNotFoundError:
    has_baseline = False
    print("Baseline model not found")

print(f"Study loaded: {len(study.trials)} trials")
print(f"Best score: {study.best_value:.4f}\n")

# Trial statistics
trials_df = study.trials_dataframe()

print("="*70)
print("TRIAL STATISTICS")
print("="*70)
print(f"Total trials:        {len(trials_df)}")
print(f"Completed:           {len(trials_df[trials_df['state'] == 'COMPLETE'])}")
print(f"Pruned:              {len(trials_df[trials_df['state'] == 'PRUNED'])}")
print(f"Failed:              {len(trials_df[trials_df['state'] == 'FAIL'])}")
print(f"Best score:          {trials_df['value'].max():.4f}")
print(f"Mean score:          {trials_df['value'].mean():.4f}")
print(f"Std deviation:       {trials_df['value'].std():.4f}")

pruning_rate = len(trials_df[trials_df['state'] == 'PRUNED']) / len(trials_df) * 100
print(f"Pruning rate:        {pruning_rate:.1f}%")
print("="*70)

# Top 5 trials
print("\nTop 5 trials:")
top_5_cols = ['number', 'value', 'params_learning_rate', 'params_n_estimators', 'params_max_depth']
available_cols = [col for col in top_5_cols if col in trials_df.columns]
print(trials_df.nlargest(5, 'value')[available_cols].to_string(index=False))

# Visualization 1: Optimization overview
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Score progression
ax = axes[0, 0]
ax.plot(trials_df['number'], trials_df['value'], marker='o', alpha=0.6, linewidth=1)
ax.axhline(y=study.best_value, color='r', linestyle='--', linewidth=2, label=f'Best: {study.best_value:.4f}')
ax.fill_between(trials_df['number'], trials_df['value'], study.best_value,
                 where=(trials_df['value'] <= study.best_value), alpha=0.3, color='gray')
ax.set_xlabel('Trial Number')
ax.set_ylabel('AUC-ROC Score')
ax.set_title('Optimization Progress')
ax.legend()
ax.grid(True, alpha=0.3)

# Score distribution
ax = axes[0, 1]
completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
ax.hist(completed_trials['value'].dropna(), bins=25, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=study.best_value, color='r', linestyle='--', linewidth=2, label='Best')
ax.axvline(x=completed_trials['value'].mean(), color='orange', linestyle='--', linewidth=2, label='Mean')
ax.set_xlabel('AUC-ROC Score')
ax.set_ylabel('Frequency')
ax.set_title('Score Distribution')
ax.legend()

# Execution time
ax = axes[1, 0]
durations = trials_df['duration'].dt.total_seconds()
colors = ['green' if state == 'COMPLETE' else 'red' for state in trials_df['state']]
ax.scatter(trials_df['number'], durations, alpha=0.6, c=colors, s=50)
ax.set_xlabel('Trial Number')
ax.set_ylabel('Duration (seconds)')
ax.set_title('Execution Time per Trial')
ax.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='Completed'),
                   Patch(facecolor='red', label='Pruned')]
ax.legend(handles=legend_elements)

# Trial states
ax = axes[1, 1]
state_counts = trials_df['state'].value_counts()
colors_pie = ['#2ecc71' if state == 'COMPLETE' else '#e74c3c' for state in state_counts.index]
wedges, texts, autotexts = ax.pie(state_counts, labels=state_counts.index, autopct='%1.1f%%',
                                    startangle=90, colors=colors_pie)
ax.set_title('Trial Status Distribution')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('outputs/optuna_plots/trials_overview.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSaved: outputs/optuna_plots/trials_overview.png")

# Visualization 2: Hyperparameter impact
param_cols = [col for col in trials_df.columns if col.startswith('params_')]

if len(param_cols) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    completed = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    
    for idx, param_col in enumerate(param_cols[:6]):
        if idx >= 6:
            break
            
        ax = axes[idx]
        param_name = param_col.replace('params_', '')
        
        ax.scatter(completed[param_col], completed['value'], alpha=0.5, s=50)
        
        # Trend line
        mask = completed[param_col].notna()
        if mask.sum() > 1:
            z = np.polyfit(completed.loc[mask, param_col],
                          completed.loc[mask, 'value'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(completed[param_col].min(), completed[param_col].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('AUC-ROC Score')
        ax.set_title(f'Impact of {param_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/optuna_plots/hyperparameter_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved: outputs/optuna_plots/hyperparameter_impact.png")

# Load test data for comparison
train_data = pd.read_csv('data/processed/train_features.csv')

if 'TARGET' in train_data.columns:
    X = train_data.drop('TARGET', axis=1)
    y = train_data['TARGET'].values
elif 'target' in train_data.columns:
    X = train_data.drop('target', axis=1)
    y = train_data['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predictions
y_pred_optimized = optimized_model.predict_proba(X_test)[:, 1]
auc_optimized = roc_auc_score(y_test, y_pred_optimized)
fpr_opt, tpr_opt, _ = roc_curve(y_test, y_pred_optimized)

# Try to load baseline model
if has_baseline:
    try:
        y_pred_baseline = baseline_model.predict_proba(X_test)[:, 1]
        auc_baseline = roc_auc_score(y_test, y_pred_baseline)
        fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_baseline)
    except Exception as e:
        print(f"Cannot use baseline model for prediction: {e}")
        print("Using reported baseline AUC value")
        has_baseline = False
        auc_baseline = 0.7558
else:
    auc_baseline = 0.7558

# Visualization 3: ROC comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC curves
ax = axes[0]
ax.plot(fpr_opt, tpr_opt, linewidth=2, label=f'Optimized (AUC = {auc_optimized:.4f})')

if has_baseline:
    ax.plot(fpr_base, tpr_base, linewidth=2, label=f'Baseline (AUC = {auc_baseline:.4f})')
else:
    ax.axhline(y=auc_baseline, color='orange', linestyle='--', linewidth=2, 
               label=f'Baseline (AUC = {auc_baseline:.4f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Score comparison
ax = axes[1]
models = ['Baseline', 'Optimized']
scores = [auc_baseline, auc_optimized]
colors = ['steelblue', 'darkgreen']

bars = ax.bar(models, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('AUC-ROC Score')
improvement = ((auc_optimized - auc_baseline) / auc_baseline) * 100
ax.set_title(f'AUC-ROC Comparison\n(Improvement: {improvement:+.2f}%)')
ax.set_ylim([0.7, max(scores) + 0.02])
ax.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{score:.4f}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/optuna_plots/baseline_vs_optimized.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: outputs/optuna_plots/baseline_vs_optimized.png")

print(f"\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
print(f"Baseline AUC-ROC:        {auc_baseline:.4f}")
print(f"Optimized AUC-ROC:       {auc_optimized:.4f}")
print(f"Absolute improvement:    {auc_optimized - auc_baseline:+.4f}")
print(f"Relative improvement:    {improvement:+.2f}%")
print(f"{'='*70}")

# Best hyperparameters
best_params = study.best_params

print("\nBest hyperparameters:")
print("="*70)
params_df = pd.DataFrame({
    'Hyperparameter': list(best_params.keys()),
    'Value': list(best_params.values())
})
print(params_df.to_string(index=False))
print("="*70)

# Visualization 4: Best hyperparameters
fig, ax = plt.subplots(figsize=(12, 8))

params_to_plot = list(best_params.keys())[:8]
values = [best_params[p] for p in params_to_plot]

bars = ax.barh(params_to_plot, values, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Value')
ax.set_title('Best Hyperparameters Found by Optuna')
ax.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, values):
    width = bar.get_width()
    label = f'{val:.4f}' if val < 1 else f'{int(val)}'
    ax.text(width, bar.get_y() + bar.get_height()/2., f'  {label}',
            ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/optuna_plots/best_hyperparameters.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSaved: outputs/optuna_plots/best_hyperparameters.png")

# Summary
print("\n" + "="*70)
print("OPTIMIZATION SUMMARY")
print("="*70)

print(f"\nPerformance:")
print(f"  Best CV AUC-ROC:        {study.best_value:.4f}")
print(f"  Test AUC-ROC:           {auc_optimized:.4f}")
print(f"  Improvement:            {improvement:+.2f}%")

print(f"\nOptimization:")
print(f"  Trials completed:       {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"  Trials pruned:          {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
print(f"  Pruning rate:           {pruning_rate:.1f}%")
print(f"  Best trial:             #{study.best_trial.number}")

print(f"\nConclusions:")
print(f"  1. Optuna improved the model by {improvement:.2f}%")
print(f"  2. Pruning saved computation time ({pruning_rate:.0f}% trials stopped early)")
print(f"  3. Optimal hyperparameters identified")
print(f"  4. Model ready for next steps")

print("="*70)

print("\nGenerated files:")
print("  outputs/optuna_plots/trials_overview.png")
print("  outputs/optuna_plots/hyperparameter_impact.png")
print("  outputs/optuna_plots/baseline_vs_optimized.png")
print("  outputs/optuna_plots/best_hyperparameters.png")