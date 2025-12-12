"""
Deep verification to check if model actually learned
Tests:
1. Model complexity (number of trees, leaves)
2. Prediction variance (not all same prediction)
3. Feature importance (are features being used?)
4. Compare with dummy baseline
5. Manual prediction test
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.dummy import DummyClassifier
import lightgbm as lgb

print("=" * 80)
print("DEEP MODEL VERIFICATION")
print("Checking if the model ACTUALLY learned from data")
print("=" * 80)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================
print("\n[1/7] Loading model and data...")

model = joblib.load('models/production/lightgbm_final.pkl')
df = pd.read_csv("data/processed/train_features.csv")

# Clean
object_cols = df.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    df = df.drop(columns=object_cols)

X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']

print(f"✓ Model loaded")
print(f"✓ Data: {X.shape[0]:,} rows")

# ============================================================================
# TEST 1: Check Model Complexity
# ============================================================================
print("\n" + "=" * 80)
print("[2/7] TEST 1: Model Complexity Check")
print("=" * 80)

# Check if it's actually a trained model
print(f"\nModel Type: {type(model).__name__}")
print(f"Number of estimators (trees): {model.n_estimators}")

# Get the actual booster
booster = model.booster_

# Count trees
num_trees = booster.num_trees()
print(f"Actual trees built: {num_trees}")

# Check tree depth
tree_info = booster.dump_model()
num_leaves_list = [len(tree['tree_structure']) for tree in tree_info['tree_info']]
avg_leaves = np.mean(num_leaves_list)
max_leaves = np.max(num_leaves_list)

print(f"Average leaves per tree: {avg_leaves:.1f}")
print(f"Max leaves in a tree: {max_leaves}")
print(f"Total leaves across all trees: {sum(num_leaves_list):,}")

# Verdict
if num_trees < 10:
    print("\n❌ PROBLEM: Too few trees! Model barely trained")
elif num_trees < 500:
    print(f"\n⚠️  WARNING: Only {num_trees} trees built (expected 500)")
else:
    print(f"\n✅ GOOD: All {num_trees} trees built successfully")

if avg_leaves < 5:
    print("❌ PROBLEM: Trees too shallow! Not learning complexity")
elif avg_leaves < 20:
    print("⚠️  Trees are simple (avg {avg_leaves:.1f} leaves)")
else:
    print(f"✅ GOOD: Trees have good complexity ({avg_leaves:.1f} leaves avg)")

# ============================================================================
# TEST 2: Prediction Variance Check
# ============================================================================
print("\n" + "=" * 80)
print("[3/7] TEST 2: Prediction Variance Check")
print("=" * 80)
print("If model didn't learn, all predictions would be the same...")

y_pred_proba = model.predict_proba(X)[:, 1]

print(f"\nPrediction Statistics:")
print(f"  Min probability:  {y_pred_proba.min():.6f}")
print(f"  Max probability:  {y_pred_proba.max():.6f}")
print(f"  Mean probability: {y_pred_proba.mean():.6f}")
print(f"  Std deviation:    {y_pred_proba.std():.6f}")

# Check unique predictions
unique_preds = len(np.unique(y_pred_proba))
print(f"  Unique predictions: {unique_preds:,} out of {len(y_pred_proba):,}")

# Show distribution
print(f"\nPrediction Distribution:")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(y_pred_proba, bins=bins)
for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
    pct = (hist[i] / len(y_pred_proba)) * 100
    bar = "█" * int(pct / 2)
    print(f"  {low:.1f}-{high:.1f}: {pct:5.1f}% {bar}")

# Verdict
if y_pred_proba.std() < 0.01:
    print("\n❌ PROBLEM: All predictions are nearly identical!")
    print("   Model didn't learn anything!")
elif unique_preds < 100:
    print(f"\n⚠️  WARNING: Only {unique_preds} unique predictions")
    print("   Model may not be using features properly")
else:
    print(f"\n✅ GOOD: Model produces {unique_preds:,} diverse predictions")

# ============================================================================
# TEST 3: Feature Importance Check
# ============================================================================
print("\n" + "=" * 80)
print("[4/7] TEST 3: Feature Usage Check")
print("=" * 80)
print("Checking if model actually uses features...")

feature_importance = model.feature_importances_
non_zero_features = np.sum(feature_importance > 0)
total_features = len(feature_importance)

print(f"\nFeatures used: {non_zero_features} out of {total_features}")
print(f"Total importance: {feature_importance.sum():.0f}")
print(f"Top 10 most important features:")

feature_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(feature_df.head(10).to_string(index=False))

# Check if any feature dominates
top_feature_pct = (feature_df.iloc[0]['importance'] / feature_importance.sum()) * 100
print(f"\nTop feature accounts for: {top_feature_pct:.1f}% of importance")

# Verdict
if non_zero_features < 10:
    print("\n❌ PROBLEM: Model uses very few features!")
elif non_zero_features < 50:
    print(f"\n⚠️  Model uses only {non_zero_features} features")
else:
    print(f"\n✅ GOOD: Model uses {non_zero_features} features")

if top_feature_pct > 50:
    print("⚠️  WARNING: One feature dominates")
else:
    print("✅ GOOD: Features well distributed")

# ============================================================================
# TEST 4: Compare with Dummy Baseline
# ============================================================================
print("\n" + "=" * 80)
print("[5/7] TEST 4: Compare with Dummy Baseline")
print("=" * 80)
print("If model is just guessing, it won't beat random baseline...")

# Train dummy classifier (always predicts majority class)
dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy.fit(X, y)

# Predictions
dummy_pred_proba = dummy.predict_proba(X)[:, 1]

# Calculate AUC
lgbm_auc = roc_auc_score(y, y_pred_proba)
dummy_auc = roc_auc_score(y, dummy_pred_proba)

print(f"\nAUC Comparison:")
print(f"  LightGBM AUC:  {lgbm_auc:.4f}")
print(f"  Dummy AUC:     {dummy_auc:.4f}")
print(f"  Improvement:   {lgbm_auc - dummy_auc:.4f} ({((lgbm_auc - dummy_auc)/dummy_auc)*100:.1f}%)")

# Verdict
if lgbm_auc < 0.55:
    print("\n❌ CRITICAL: Model performs like random guessing!")
elif lgbm_auc < 0.65:
    print("\n⚠️  WARNING: Model barely better than random")
elif lgbm_auc < 0.75:
    print("\n✅ GOOD: Model learned something useful")
else:
    print("\n✅ EXCELLENT: Model learned very well!")

# ============================================================================
# TEST 5: Manual Prediction Test
# ============================================================================
print("\n" + "=" * 80)
print("[6/7] TEST 5: Manual Prediction Test")
print("=" * 80)
print("Testing predictions on specific examples...")

# Get high-risk and low-risk examples
high_risk_idx = y_pred_proba.argsort()[-5:][::-1]  # Top 5 riskiest
low_risk_idx = y_pred_proba.argsort()[:5]          # Top 5 safest

print("\n🔴 TOP 5 HIGHEST RISK Predictions:")
for i, idx in enumerate(high_risk_idx, 1):
    actual = "DEFAULT" if y.iloc[idx] == 1 else "NO DEFAULT"
    print(f"  {i}. Predicted: {y_pred_proba[idx]:.4f} | Actual: {actual}")

print("\n🟢 TOP 5 LOWEST RISK Predictions:")
for i, idx in enumerate(low_risk_idx, 1):
    actual = "DEFAULT" if y.iloc[idx] == 1 else "NO DEFAULT"
    print(f"  {i}. Predicted: {y_pred_proba[idx]:.4f} | Actual: {actual}")

# Check if high risk predictions actually default more
high_risk_default_rate = y.iloc[high_risk_idx].mean()
low_risk_default_rate = y.iloc[low_risk_idx].mean()
overall_default_rate = y.mean()

print(f"\nDefault Rates:")
print(f"  High-risk group:  {high_risk_default_rate:.1%}")
print(f"  Low-risk group:   {low_risk_default_rate:.1%}")
print(f"  Overall dataset:  {overall_default_rate:.1%}")

if high_risk_default_rate > overall_default_rate:
    print("\n✅ GOOD: Model correctly identifies high-risk applicants")
else:
    print("\n❌ PROBLEM: Model not identifying risk properly")

# ============================================================================
# TEST 6: Check Training was Real
# ============================================================================
print("\n" + "=" * 80)
print("[7/7] TEST 6: Training Verification")
print("=" * 80)

# Check model attributes that only exist after training
print(f"\nModel Training Indicators:")
print(f"  Model has been fitted: {hasattr(model, 'fitted_')}")
print(f"  Number of features seen: {model.n_features_in_}")
print(f"  Number of classes: {model.n_classes_}")
print(f"  Best iteration: {model.best_iteration_}")
print(f"  Best score: {model.best_score_}")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "=" * 80)
print("🎯 FINAL VERDICT")
print("=" * 80)

issues_found = 0
checks_passed = 0

print("\nTest Results Summary:")

# Test 1
if num_trees >= 500 and avg_leaves > 20:
    print("  ✅ Model Complexity: PASS")
    checks_passed += 1
else:
    print("  ❌ Model Complexity: FAIL")
    issues_found += 1

# Test 2
if y_pred_proba.std() > 0.05 and unique_preds > 1000:
    print("  ✅ Prediction Variance: PASS")
    checks_passed += 1
else:
    print("  ❌ Prediction Variance: FAIL")
    issues_found += 1

# Test 3
if non_zero_features > 50:
    print("  ✅ Feature Usage: PASS")
    checks_passed += 1
else:
    print("  ❌ Feature Usage: FAIL")
    issues_found += 1

# Test 4
if lgbm_auc > 0.75:
    print("  ✅ Performance vs Baseline: PASS")
    checks_passed += 1
else:
    print("  ❌ Performance vs Baseline: FAIL")
    issues_found += 1

# Test 5
if high_risk_default_rate > overall_default_rate:
    print("  ✅ Risk Identification: PASS")
    checks_passed += 1
else:
    print("  ❌ Risk Identification: FAIL")
    issues_found += 1

print("\n" + "=" * 80)
print(f"Tests Passed: {checks_passed}/5")
print(f"Issues Found: {issues_found}")
print("=" * 80)

if issues_found == 0:
    print("\n🎉 EXCELLENT: Model is properly trained and working!")
    print("   Training time is fast because LightGBM is optimized.")
    print("   Your model is READY for production!")
elif issues_found <= 2:
    print("\n✅ GOOD: Model is mostly working, minor issues")
    print("   Model did learn but could be improved")
else:
    print("\n❌ PROBLEM: Model has serious issues!")
    print("   Training may not have completed properly")

print("\n" + "=" * 80)