"""
Test script to verify realistic training time
Shows detailed progress and timing
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import time
from datetime import datetime

print("=" * 80)
print("TRAINING TIME VERIFICATION TEST")
print("=" * 80)

# Load data
print("\n📂 Loading data...")
start_load = time.time()
df = pd.read_csv("data/processed/train_features.csv")
load_time = time.time() - start_load

# Clean data
object_cols = df.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    df = df.drop(columns=object_cols)

X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']

print(f"✓ Data loaded in {load_time:.2f} seconds")
print(f"✓ Dataset: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"✓ Memory: {X.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Calculate scale
scale_pos_weight = len(y[y==0]) / len(y[y==1])

# ============================================================================
# TEST 1: Small Model (Fast Baseline)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Quick Model (10 trees) - Baseline")
print("=" * 80)

start = time.time()
quick_model = lgb.LGBMClassifier(
    n_estimators=10,
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    verbose=-1
)
quick_model.fit(X, y)
quick_time = time.time() - start

print(f"⏱️  Time: {quick_time:.2f} seconds")
print(f"📊 Expected: ~5-15 seconds")

# ============================================================================
# TEST 2: Medium Model (Realistic)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Medium Model (100 trees) - More Realistic")
print("=" * 80)
print("⏳ Training... (watch the progress)")

start = time.time()

def print_callback(env):
    """Custom callback to show detailed progress"""
    iteration = env.iteration
    elapsed = time.time() - start
    if iteration % 10 == 0:
        avg_time_per_iter = elapsed / (iteration + 1)
        remaining_iters = 100 - (iteration + 1)
        eta = avg_time_per_iter * remaining_iters
        print(f"  [{iteration + 1:3d}/100] Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s")

medium_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    verbose=-1
)

medium_model.fit(
    X, y,
    callbacks=[print_callback]
)

medium_time = time.time() - start

print(f"\n⏱️  Total time: {medium_time:.2f} seconds ({medium_time/60:.1f} minutes)")
print(f"📊 Expected: ~1-3 minutes")

# ============================================================================
# TEST 3: Full Production Model (What we'll actually use)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Full Production Model (500 trees)")
print("=" * 80)
print("⏳ This is the real training time you'll see...")
print("=" * 80)

start_production = time.time()
iteration_times = []

def detailed_callback(env):
    """Track each iteration timing"""
    iteration = env.iteration
    elapsed = time.time() - start_production
    
    if iteration > 0:
        avg_time_per_iter = elapsed / (iteration + 1)
        iteration_times.append(avg_time_per_iter)
    
    # Print every 50 iterations
    if (iteration + 1) % 50 == 0:
        avg_time = elapsed / (iteration + 1)
        remaining = 500 - (iteration + 1)
        eta_seconds = avg_time * remaining
        eta_minutes = eta_seconds / 60
        
        print(f"  [{iteration + 1:3d}/500] "
              f"Elapsed: {elapsed/60:5.1f}m | "
              f"Avg: {avg_time:.3f}s/tree | "
              f"ETA: {eta_minutes:5.1f}m")

production_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=7,
    num_leaves=31,
    learning_rate=0.03,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print(f"\n🚀 Starting training at: {datetime.now().strftime('%H:%M:%S')}")
print("-" * 80)

production_model.fit(
    X, y,
    callbacks=[detailed_callback]
)

production_time = time.time() - start_production

print("-" * 80)
print(f"✅ Finished training at: {datetime.now().strftime('%H:%M:%S')}")
print(f"⏱️  Total time: {production_time:.2f} seconds ({production_time/60:.1f} minutes)")

# ============================================================================
# ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("📊 TRAINING TIME ANALYSIS")
print("=" * 80)

print(f"""
Dataset:
  Rows: {X.shape[0]:,}
  Features: {X.shape[1]}
  Size in memory: {X.memory_usage(deep=True).sum() / 1024**2:.1f} MB

Training Times:
  Quick model (10 trees):     {quick_time:6.1f}s
  Medium model (100 trees):   {medium_time:6.1f}s ({medium_time/60:.1f}m)
  Production (500 trees):     {production_time:6.1f}s ({production_time/60:.1f}m)

Average Time per Tree:
  {np.mean(iteration_times):.4f} seconds/tree
  
Total Trees per Minute:
  {60 / np.mean(iteration_times):.1f} trees/minute
""")

# ============================================================================
# VERDICT
# ============================================================================
print("\n" + "=" * 80)
print("🎯 VERDICT")
print("=" * 80)

if production_time < 60:
    print("⚠️  WARNING: Training suspiciously fast (< 1 minute)")
    print("   This might indicate:")
    print("   - Data not fully loaded")
    print("   - Early stopping triggered")
    print("   - Model not actually training on full data")
    print("\n   ❌ Something is WRONG - need to investigate!")
    
elif production_time < 300:  # Less than 5 minutes
    print("✅ NORMAL: Training time is reasonable for this dataset size")
    print("   LightGBM is VERY fast compared to deep learning")
    print("   For 300K rows with tree models, 2-5 minutes is expected")
    print("\n   ✅ Everything looks CORRECT!")
    
elif production_time < 900:  # Less than 15 minutes
    print("✅ NORMAL: Slightly slower but still reasonable")
    print("   Could be due to:")
    print("   - Deeper trees (more complex)")
    print("   - Less powerful CPU")
    print("   - Background processes")
    print("\n   ✅ Everything looks CORRECT!")
    
else:
    print("⏰ SLOW: Training taking longer than expected")
    print("   This is fine for production but slower than typical")
    print("\n   ✅ Still CORRECT, just slower machine")

print("=" * 80)

# Compare to deep learning
print("\n💡 For comparison:")
print("   - LightGBM on 300K rows: 2-10 minutes")
print("   - Neural Network on same data: 30-60 minutes")
print("   - ResNet50 on ImageNet: 12-24 hours")
print("   - GPT-3: Several weeks on supercomputers!")

print("\n" + "=" * 80)
print("✅ TEST COMPLETE")
print("=" * 80)