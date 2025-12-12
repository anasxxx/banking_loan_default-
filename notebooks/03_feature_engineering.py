import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path

pd.set_option('display.max_columns', 100)

print("=" * 80)
print("FEATURE ENGINEERING PIPELINE")
print("=" * 80)

# Load cleaned data
df = pd.read_csv("data/processed/train_cleaned.csv")
print(f"\n✓ Loaded cleaned data: {df.shape}")

# High ratio = borrowing a lot relative to income = risky
df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
print(f"✓ CREDIT_INCOME_RATIO - Average: {df['CREDIT_INCOME_RATIO'].mean():.2f}")


# Feature 2: Annuity-to-Income Ratio  
# What it means: Monthly payment burden relative to income
# High ratio = large payment burden = risky
df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
print(f"✓ ANNUITY_INCOME_RATIO - Average: {df['ANNUITY_INCOME_RATIO'].mean():.2f}")

# Feature 3: Loan Term (in months)
# What it means: How long to repay the loan
# Longer term might indicate difficulty paying
df['LOAN_TERM_MONTHS'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
print(f"✓ LOAN_TERM_MONTHS - Average: {df['LOAN_TERM_MONTHS'].mean():.1f} months")

# Feature 4: Goods Price to Credit Ratio
# What it means: Is the loan amount reasonable for the goods?
# Ratio far from 1 might be suspicious
df['GOODS_PRICE_TO_CREDIT'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
print(f"✓ GOODS_PRICE_TO_CREDIT - Average: {df['GOODS_PRICE_TO_CREDIT'].mean():.2f}")

# Feature 5: Days Employed to Age Ratio
# What it means: What portion of life has been employed
# Lower ratio = less stable employment history
df['EMPLOYMENT_TO_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / df['AGE_YEARS']
print(f"✓ EMPLOYMENT_TO_AGE_RATIO - Average: {df['EMPLOYMENT_TO_AGE_RATIO'].mean():.2f}")

# Feature 6: Income per Family Member
# What it means: How much income per person in household
# Lower value = more financial pressure
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'])
print(f"✓ INCOME_PER_PERSON - Average: {df['INCOME_PER_PERSON'].mean():,.0f}")


# Sometimes combining features creates powerful predictors
# Example: Age + Income might be less important than Age * Income

# Income * Credit (total financial exposure)
df['INCOME_CREDIT_INTERACTION'] = df['AMT_INCOME_TOTAL'] * df['AMT_CREDIT'] / 1e9
print(f"✓ INCOME_CREDIT_INTERACTION created")

# Age * Employment (career stability indicator)
df['AGE_EMPLOYMENT_INTERACTION'] = df['AGE_YEARS'] * df['EMPLOYMENT_YEARS']
print(f"✓ AGE_EMPLOYMENT_INTERACTION created")

# Sometimes categories work better than continuous values
# Example: Instead of exact age, use age groups

# Age groups
df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], 
                          bins=[0, 25, 35, 45, 55, 100],
                          labels=['18-25', '26-35', '36-45', '46-55', '55+'])
print("\n✓ AGE_GROUP created")
print(df['AGE_GROUP'].value_counts().sort_index())

# Income groups
df['INCOME_GROUP'] = pd.cut(df['AMT_INCOME_TOTAL'],
                             bins=[0, 100000, 200000, 300000, 1000000],
                             labels=['Low', 'Medium', 'High', 'Very High'])
print("\n✓ INCOME_GROUP created")
print(df['INCOME_GROUP'].value_counts().sort_index())

# Credit amount groups
df['CREDIT_GROUP'] = pd.qcut(df['AMT_CREDIT'], 
                               q=4, 
                               labels=['Small', 'Medium', 'Large', 'Very Large'])
print("\n✓ CREDIT_GROUP created")
print(df['CREDIT_GROUP'].value_counts().sort_index())


# Create yes/no features for important conditions

# Is person a car owner?
df['HAS_CAR'] = (df['FLAG_OWN_CAR'] == 'Y').astype(int)
print(f"✓ HAS_CAR - {df['HAS_CAR'].sum()} people have cars ({df['HAS_CAR'].mean()*100:.1f}%)")

# Is person a property owner?
df['HAS_PROPERTY'] = (df['FLAG_OWN_REALTY'] == 'Y').astype(int)
print(f"✓ HAS_PROPERTY - {df['HAS_PROPERTY'].sum()} people have property ({df['HAS_PROPERTY'].mean()*100:.1f}%)")

# Is high credit burden? (annuity > 30% of income is risky)
df['HIGH_CREDIT_BURDEN'] = (df['ANNUITY_INCOME_RATIO'] > 0.3).astype(int)
print(f"✓ HIGH_CREDIT_BURDEN - {df['HIGH_CREDIT_BURDEN'].sum()} people ({df['HIGH_CREDIT_BURDEN'].mean()*100:.1f}%)")

# Is young borrower? (< 30 years old)
df['IS_YOUNG'] = (df['AGE_YEARS'] < 30).astype(int)
print(f"✓ IS_YOUNG - {df['IS_YOUNG'].sum()} young borrowers ({df['IS_YOUNG'].mean()*100:.1f}%)")

# Has children?
df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
print(f"✓ HAS_CHILDREN - {df['HAS_CHILDREN'].sum()} people have children ({df['HAS_CHILDREN'].mean()*100:.1f}%)")

# Get categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns to encode: {len(categorical_cols)}")

# We'll use Label Encoding for now (simple approach)
# Later you can try One-Hot Encoding or Target Encoding

label_encoders = {}

for col in categorical_cols:
    print(f"\nEncoding {col}...")
    print(f"  Unique values: {df[col].nunique()}")
    
    # Create encoder
    le = LabelEncoder()
    
    # Fit and transform
    df[f'{col}_ENCODED'] = le.fit_transform(df[col].astype(str))
    
    # Save encoder
    label_encoders[col] = le
    
    print(f"  ✓ Created {col}_ENCODED")
    print(f"  Encoding: {dict(list(zip(le.classes_, le.transform(le.classes_)))[:5])}...")

# Save all encoders
joblib.dump(label_encoders, 'models/preprocessors/label_encoders.pkl')
print("\n✓ Saved all label encoders")

# Why scale? Features have different ranges:
# - Income: 0 to 1,000,000
# - Age: 20 to 70
# Scaling puts them on same scale

# Get numerical columns (including our new features)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove columns we don't want to scale
cols_to_exclude = ['SK_ID_CURR', 'TARGET'] + [col for col in numeric_cols if col.endswith('_ENCODED')]
numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]

print(f"\nScaling {len(numeric_cols)} numerical features...")

# Create scaler
scaler = StandardScaler()

# Fit and transform
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save scaler
joblib.dump(scaler, 'models/preprocessors/scaler.pkl')

print("✓ Scaling complete")
print("✓ Saved scaler")

# Show example of scaling
print("\nExample - AMT_INCOME_TOTAL after scaling:")
print(f"  Mean: {df['AMT_INCOME_TOTAL'].mean():.10f} (should be ~0)")
print(f"  Std: {df['AMT_INCOME_TOTAL'].std():.10f} (should be ~1)")


# Remove highly correlated features (redundant)
# If two features are 95%+ correlated, we only need one

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr().abs()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print(f"\nHighly correlated feature pairs (>0.95): {len(high_corr_pairs)}")
if high_corr_pairs:
    print("\nTop 10 highly correlated pairs:")
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:10]:
        print(f"  {feat1} <-> {feat2}: {corr:.3f}")

# Save the feature-engineered dataset
output_path = "data/processed/train_features.csv"
df.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")
print(f"  Shape: {df.shape}")
print(f"  Size: {Path(output_path).stat().st_size / 1024**2:.2f} MB")

# Also save feature names for later
feature_names = [col for col in df.columns if col not in ['SK_ID_CURR', 'TARGET']]
joblib.dump(feature_names, 'models/preprocessors/feature_names.pkl')
print(f"✓ Saved {len(feature_names)} feature names")
