import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

pd.set_option('display.max_columns', 100)

print("=" * 80)
print("DATA CLEANING PIPELINE")
print("=" * 80)

DATA_PATH = "data/raw"

train_df = pd.read_csv(f"{DATA_PATH}/application_train.csv")
print(f"\n✓ Loaded: {train_df.shape}")

df = train_df.copy()

missing_pct = (df.isnull().sum() / len(df)) * 100
cols_to_drop = missing_pct[missing_pct > 70].index.tolist()

print(f"\nColumns with >70% missing data: {len(cols_to_drop)}")
print(cols_to_drop)

print(f"\nBefore dropping: {df.shape[1]} columns")
df = df.drop(columns=cols_to_drop)
print(f"After dropping: {df.shape[1]} columns")
print(f"Dropped: {len(cols_to_drop)} columns")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if 'SK_ID_CURR' in numeric_cols:
    numeric_cols.remove('SK_ID_CURR')
if 'TARGET' in numeric_cols:
    numeric_cols.remove('TARGET')

print(f"\nNumerical columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

numeric_missing = df[numeric_cols].isnull().sum()
numeric_missing = numeric_missing[numeric_missing > 0]

print(f"\nNumerical columns with missing values: {len(numeric_missing)}")
print(numeric_missing)

numeric_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

Path("models/preprocessors").mkdir(parents=True, exist_ok=True)
joblib.dump(numeric_imputer, 'models/preprocessors/numeric_imputer.pkl')

print("\n✓ Numerical imputation complete")
print(f"✓ Saved: models/preprocessors/numeric_imputer.pkl")

print(f"\nMissing values after imputation: {df[numeric_cols].isnull().sum().sum()}")

cat_missing = df[categorical_cols].isnull().sum()
cat_missing = cat_missing[cat_missing > 0]

print(f"\nCategorical columns with missing values: {len(cat_missing)}")
print(cat_missing)

for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

print("\n✓ Categorical imputation complete")
print(f"Missing values after imputation: {df[categorical_cols].isnull().sum().sum()}")

key_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']

print("\nBefore outlier handling:")
for feat in key_features:
    if feat in df.columns:
        print(f"\n{feat}:")
        print(f"  Min: {df[feat].min():,.0f}")
        print(f"  Max: {df[feat].max():,.0f}")
        print(f"  Mean: {df[feat].mean():,.0f}")
        print(f"  Median: {df[feat].median():,.0f}")
        print(f"  99th percentile: {df[feat].quantile(0.99):,.0f}")

for feat in key_features:
    if feat in df.columns:
        percentile_99 = df[feat].quantile(0.99)
        outlier_count = (df[feat] > percentile_99).sum()
        print(f"\n{feat}: Capping {outlier_count} outliers at {percentile_99:,.0f}")
        df[feat] = df[feat].clip(upper=percentile_99)

if 'DAYS_BIRTH' in df.columns:
    df['AGE_YEARS'] = -(df['DAYS_BIRTH'] / 365)
    print(f"✓ Created AGE_YEARS from DAYS_BIRTH")
    print(f"  Age range: {df['AGE_YEARS'].min():.1f} to {df['AGE_YEARS'].max():.1f} years")

if 'DAYS_EMPLOYED' in df.columns:
    df['EMPLOYMENT_YEARS'] = -(df['DAYS_EMPLOYED'] / 365)
    df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].replace({1000.665: 0})
    df.loc[df['EMPLOYMENT_YEARS'] < 0, 'EMPLOYMENT_YEARS'] = 0
    print(f"✓ Created EMPLOYMENT_YEARS from DAYS_EMPLOYED")
    print(f"  Employment range: {df['EMPLOYMENT_YEARS'].min():.1f} to {df['EMPLOYMENT_YEARS'].max():.1f} years")

print("\n" + "=" * 80)
print("STEP 6: Checking for Duplicates")
print("=" * 80)

duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

if duplicates > 0:
    print(f"Removing {duplicates} duplicates...")
    df = df.drop_duplicates()
    print(f"✓ Duplicates removed")

duplicate_ids = df['SK_ID_CURR'].duplicated().sum()
print(f"Duplicate IDs: {duplicate_ids}")

print(f"\nFinal dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\nData type distribution:")
print(df.dtypes.value_counts())

memory_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"\nMemory usage: {memory_mb:.2f} MB")

Path("data/processed").mkdir(parents=True, exist_ok=True)

output_path = "data/processed/train_cleaned.csv"
df.to_csv(output_path, index=False)

print(f"\n✓ Saved cleaned data: {output_path}")
print(f"  Shape: {df.shape}")
print(f"  Size: {Path(output_path).stat().st_size / 1024**2:.2f} MB")