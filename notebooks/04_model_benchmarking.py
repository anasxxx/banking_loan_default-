import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import time
import joblib
from pathlib import Path

pd.set_option('display.max_columns', 100)

print("=" * 80)
print("MODEL BENCHMARKING")
print("=" * 80)

df = pd.read_csv("data/processed/train_features.csv")
print(f"✓ Loaded: {df.shape}")

# Separate features and target
X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']

print(f"\nFeatures (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"\nTarget distribution:")
print(y.value_counts())
print(f"Default rate: {y.mean()*100:.2f}%")


df = pd.read_csv("data/processed/train_features.csv")
print(f"✓ Loaded: {df.shape}")

# Separate features and target
X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']

print(f"\nFeatures (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"\nTarget distribution:")
print(y.value_counts())
print(f"Default rate: {y.mean()*100:.2f}%")

# Split: 80% train, 20% test
# stratify=y ensures same class distribution in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # IMPORTANT for imbalanced data!
)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

print(f"\nTrain target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print(f"\nTest target distribution:")
print(y_test.value_counts(normalize=True) * 100)

# Calculate scale_pos_weight for XGBoost/LightGBM
# This tells the model: "pay more attention to minority class"
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

print(f"\nClass imbalance:")
print(f"  No default (0): {len(y_train[y_train==0]):,}")
print(f"  Default (1): {len(y_train[y_train==1]):,}")
print(f"  Imbalance ratio: {scale_pos_weight:.2f}:1")
print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")
print("(We'll use this in tree-based models)")


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Calculate and display all important metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print('='*60)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Print metrics
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (of predicted defaults, {precision*100:.1f}% are correct)")
    print(f"  Recall:    {recall:.4f} (catches {recall*100:.1f}% of actual defaults)")
    print(f"  F1-Score:  {f1:.4f} (balance between precision/recall)")
    print(f"  AUC-ROC:   {auc:.4f} (overall classification performance)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                No    Yes")
    print(f"  Actual No   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"  Actual Yes  {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Important metrics for loan default
    print(f"\nInterpretation for Loan Default:")
    print(f"  • False Negatives (missed defaults): {cm[1,0]} ← BAD! Lost money")
    print(f"  • False Positives (wrongly rejected): {cm[0,1]} ← Missed opportunity")
    print(f"  • True Positives (caught defaults): {cm[1,1]} ← GOOD! Saved money")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc
    }

print("\n✓ Evaluation function ready")
print("\nKey metrics explained:")
print("  • Accuracy: Overall correctness (can be misleading with imbalanced data)")
print("  • Precision: Of predicted defaults, how many are actually defaults")
print("  • Recall: Of actual defaults, how many did we catch (MOST IMPORTANT!)")
print("  • F1-Score: Balance between precision and recall")
print("  • AUC-ROC: Ability to distinguish between classes (higher = better)")

print("\nTraining Logistic Regression...")
print("Why start with this? Simple, fast, interpretable baseline")

start_time = time.time()

# Train model with class_weight='balanced' to handle imbalance
lr_model = LogisticRegression(
    class_weight='balanced',  # Handles imbalance
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

lr_model.fit(X_train, y_train)

train_time = time.time() - start_time

# Make predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

print(f"✓ Training complete in {train_time:.2f} seconds")

# Evaluate
results = []
lr_results = evaluate_model(y_test, y_pred_lr, y_pred_proba_lr, "Logistic Regression")
lr_results['Training Time (s)'] = train_time
results.append(lr_results)

print("\nTraining Random Forest...")
print("Why Random Forest? Handles non-linear relationships, robust")

start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,  # 100 trees
    max_depth=10,      # Prevent overfitting
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

train_time = time.time() - start_time

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print(f"✓ Training complete in {train_time:.2f} seconds")

# Evaluate
rf_results = evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, "Random Forest")
rf_results['Training Time (s)'] = train_time
results.append(rf_results)


print("\nTraining XGBoost...")
print("Why XGBoost? State-of-the-art for structured data, fast")

start_time = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.01,
    scale_pos_weight=scale_pos_weight,  # Handle imbalance
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

train_time = time.time() - start_time

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print(f"✓ Training complete in {train_time:.2f} seconds")

# Evaluate
xgb_results = evaluate_model(y_test, y_pred_xgb, y_pred_proba_xgb, "XGBoost")
xgb_results['Training Time (s)'] = train_time
results.append(xgb_results)


print("\nTraining LightGBM...")
print("Why LightGBM? Even faster than XGBoost, handles large datasets")

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

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc'
)

train_time = time.time() - start_time

# Predictions
y_pred_lgb = lgb_model.predict(X_test)
y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

print(f"✓ Training complete in {train_time:.2f} seconds")

# Evaluate
lgb_results = evaluate_model(y_test, y_pred_lgb, y_pred_proba_lgb, "LightGBM")
lgb_results['Training Time (s)'] = train_time
results.append(lgb_results)


print("\nTraining CatBoost...")
print("Why CatBoost? Excellent with categorical features, robust")

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

cat_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=False
)

train_time = time.time() - start_time

# Predictions
y_pred_cat = cat_model.predict(X_test)
y_pred_proba_cat = cat_model.predict_proba(X_test)[:, 1]

print(f"✓ Training complete in {train_time:.2f} seconds")

# Evaluate
cat_results = evaluate_model(y_test, y_pred_cat, y_pred_proba_cat, "CatBoost")
cat_results['Training Time (s)'] = train_time
results.append(cat_results)


# Create comparison dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AUC-ROC', ascending=False)

print("\n📊 BENCHMARKING RESULTS:")
print("=" * 100)
print(results_df.to_string(index=False))
print("=" * 100)

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_auc = results_df.iloc[0]['AUC-ROC']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   AUC-ROC: {best_auc:.4f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: AUC-ROC comparison
axes[0, 0].barh(results_df['Model'], results_df['AUC-ROC'], color='skyblue')
axes[0, 0].set_xlabel('AUC-ROC Score')
axes[0, 0].set_title('Model Comparison: AUC-ROC', fontsize=14, fontweight='bold')
axes[0, 0].set_xlim(0, 1)
for i, v in enumerate(results_df['AUC-ROC']):
    axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center')

# Plot 2: Recall comparison (most important for default detection!)
axes[0, 1].barh(results_df['Model'], results_df['Recall'], color='lightcoral')
axes[0, 1].set_xlabel('Recall Score')
axes[0, 1].set_title('Model Comparison: Recall (Catching Defaults)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlim(0, 1)
for i, v in enumerate(results_df['Recall']):
    axes[0, 1].text(v + 0.01, i, f'{v:.4f}', va='center')

# Plot 3: F1-Score comparison
axes[1, 0].barh(results_df['Model'], results_df['F1-Score'], color='lightgreen')
axes[1, 0].set_xlabel('F1-Score')
axes[1, 0].set_title('Model Comparison: F1-Score', fontsize=14, fontweight='bold')
axes[1, 0].set_xlim(0, 1)
for i, v in enumerate(results_df['F1-Score']):
    axes[1, 0].text(v + 0.01, i, f'{v:.4f}', va='center')

# Plot 4: Training Time comparison
axes[1, 1].barh(results_df['Model'], results_df['Training Time (s)'], color='plum')
axes[1, 1].set_xlabel('Training Time (seconds)')
axes[1, 1].set_title('Model Comparison: Training Time', fontsize=14, fontweight='bold')
for i, v in enumerate(results_df['Training Time (s)']):
    axes[1, 1].text(v + 0.5, i, f'{v:.1f}s', va='center')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_comparison.png")
plt.show()

# ROC Curves comparison
plt.figure(figsize=(10, 8))

models_data = [
    ('Logistic Regression', y_pred_proba_lr, lr_results['AUC-ROC']),
    ('Random Forest', y_pred_proba_rf, rf_results['AUC-ROC']),
    ('XGBoost', y_pred_proba_xgb, xgb_results['AUC-ROC']),
    ('LightGBM', y_pred_proba_lgb, lgb_results['AUC-ROC']),
    ('CatBoost', y_pred_proba_cat, cat_results['AUC-ROC'])
]

for model_name, y_proba, auc_score in models_data:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curves_comparison.png")
plt.show()


#Determine which model is best
model_map = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'CatBoost': cat_model
}

best_model = model_map[best_model_name]

# Save model
Path("models/trained").mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, 'models/trained/best_model.pkl')

print(f"\n✓ Saved best model: {best_model_name}")
print(f"  Location: models/trained/best_model.pkl")

# Save comparison results
results_df.to_csv('outputs/model_comparison_results.csv', index=False)
print(f"✓ Saved comparison results: outputs/model_comparison_results.csv")