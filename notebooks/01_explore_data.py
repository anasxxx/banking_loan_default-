import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

print("=" * 80)
print("STEP 1: LOAD THE DATA")
print("=" * 80)

DATA_PATH = r"C:\Users\mahmo\Downloads\home-credit-default-risk (1)"

train_df = pd.read_csv(f"{DATA_PATH}/application_train.csv")
print(f"\n✓ Data loaded successfully!")
print(f"Shape: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")


print("\nColumn names:")
print(train_df.columns.tolist())
train_df.info()


print("\nTarget distribution:")
print(train_df['TARGET'].value_counts())
print("\nTarget percentage:")
print(train_df['TARGET'].value_counts(normalize=True) * 100)

count_0 = train_df['TARGET'].value_counts()[0]
count_1 = train_df['TARGET'].value_counts()[1]
imbalance_ratio = count_0 / count_1

print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
print(f"This means for every 1 default, there are {imbalance_ratio:.0f} non-defaults")


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
train_df['TARGET'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Target Distribution (Count)', fontsize=14, fontweight='bold')
plt.xlabel('Target (0=No Default, 1=Default)')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
train_df['TARGET'].value_counts(normalize=True).plot(kind='bar', color=['green', 'red'])
plt.title('Target Distribution (Percentage)', fontsize=14, fontweight='bold')
plt.xlabel('Target (0=No Default, 1=Default)')
plt.ylabel('Percentage')
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: target_distribution.png")
plt.show()


# Calculate missing values
missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df)) * 100

# Combine into a dataframe
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Percentage': missing_pct.values
})

# Only show columns with missing values
missing_df = missing_df[missing_df['Missing_Count'] > 0]
missing_df = missing_df.sort_values('Percentage', ascending=False)

print(f"\nColumns with missing values: {len(missing_df)} out of {len(train_df.columns)}")
print("\nTop 20 columns with most missing data:")
print(missing_df.head(20))
high_missing = missing_df[missing_df['Percentage'] > 50]

plt.figure(figsize=(12, 8))
top_20_missing = missing_df.head(20)
plt.barh(range(len(top_20_missing)), top_20_missing['Percentage'])
plt.yticks(range(len(top_20_missing)), top_20_missing['Column'])
plt.xlabel('Missing Percentage (%)')
plt.title('Top 20 Features with Missing Values', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: missing_values.png")
plt.show()




# Select numerical columns
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove ID and TARGET
numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'TARGET']]

print(f"\nNumber of numerical features: {len(numeric_cols)}")
print("\nFirst 10 numerical features:")
print(numeric_cols[:10])

# Basic statistics
print("\nBasic statistics for first 5 numerical features:")
print(train_df[numeric_cols[:5]].describe())

# IMPORTANT FEATURES - Let's look at key ones
important_features = [
    'AMT_INCOME_TOTAL',    # Client income
    'AMT_CREDIT',          # Credit amount of the loan
    'AMT_ANNUITY',         # Loan annuity (yearly payment)
    'AMT_GOODS_PRICE',     # Price of goods for which loan is given
    'DAYS_BIRTH',          # Age (in days, negative)
    'DAYS_EMPLOYED'        # Employment length (in days, negative)
]

print("\n" + "=" * 80)
print("Key Features to Understand:")
print("=" * 80)

for feat in important_features:
    if feat in train_df.columns:
        print(f"\n{feat}:")
        print(f"  Mean: {train_df[feat].mean():,.2f}")
        print(f"  Median: {train_df[feat].median():,.2f}")
        print(f"  Min: {train_df[feat].min():,.2f}")
        print(f"  Max: {train_df[feat].max():,.2f}")
        print(f"  Missing: {train_df[feat].isnull().sum()} ({train_df[feat].isnull().sum()/len(train_df)*100:.1f}%)")


#Select categorical columns
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumber of categorical features: {len(categorical_cols)}")
print("\nCategorical features:")
for col in categorical_cols:
    unique_count = train_df[col].nunique()
    print(f"  {col}: {unique_count} unique values")

# Look at some important categorical features
print("\n" + "-" * 80)
print("NAME_CONTRACT_TYPE distribution:")
print(train_df['NAME_CONTRACT_TYPE'].value_counts())

print("\n" + "-" * 80)
print("CODE_GENDER distribution:")
print(train_df['CODE_GENDER'].value_counts())

print("\n" + "-" * 80)
print("NAME_EDUCATION_TYPE distribution:")
print(train_df['NAME_EDUCATION_TYPE'].value_counts())

print("\n" + "-" * 80)
print("NAME_INCOME_TYPE distribution:")
print(train_df['NAME_INCOME_TYPE'].value_counts())


# Let's look at income vs default
print("\nAverage income by target:")
print(train_df.groupby('TARGET')['AMT_INCOME_TOTAL'].mean())

print("\nAverage credit amount by target:")
print(train_df.groupby('TARGET')['AMT_CREDIT'].mean())

# Age analysis (convert DAYS_BIRTH to years)
train_df['AGE_YEARS'] = -(train_df['DAYS_BIRTH'] / 365)

print("\nAverage age by target:")
print(train_df.groupby('TARGET')['AGE_YEARS'].mean())

# Visualize some relationships
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Income distribution by target
axes[0, 0].hist([train_df[train_df['TARGET']==0]['AMT_INCOME_TOTAL'], 
                 train_df[train_df['TARGET']==1]['AMT_INCOME_TOTAL']], 
                bins=50, label=['No Default', 'Default'], alpha=0.7)
axes[0, 0].set_xlabel('Income')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Income Distribution by Target')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 500000)

# Age distribution by target
axes[0, 1].hist([train_df[train_df['TARGET']==0]['AGE_YEARS'], 
                 train_df[train_df['TARGET']==1]['AGE_YEARS']], 
                bins=50, label=['No Default', 'Default'], alpha=0.7)
axes[0, 1].set_xlabel('Age (years)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Age Distribution by Target')
axes[0, 1].legend()

# Credit amount distribution
axes[1, 0].hist([train_df[train_df['TARGET']==0]['AMT_CREDIT'], 
                 train_df[train_df['TARGET']==1]['AMT_CREDIT']], 
                bins=50, label=['No Default', 'Default'], alpha=0.7)
axes[1, 0].set_xlabel('Credit Amount')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Credit Amount Distribution by Target')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 2000000)

# Gender vs Target
gender_target = pd.crosstab(train_df['CODE_GENDER'], train_df['TARGET'], normalize='index') * 100
gender_target.plot(kind='bar', ax=axes[1, 1], color=['green', 'red'])
axes[1, 1].set_xlabel('Gender')
axes[1, 1].set_ylabel('Percentage')
axes[1, 1].set_title('Default Rate by Gender')
axes[1, 1].set_xticklabels(['Female', 'Male', 'Other'], rotation=0)
axes[1, 1].legend(['No Default', 'Default'])

plt.tight_layout()
plt.savefig('target_vs_features.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: target_vs_features.png")
plt.show()

# Calculate correlations with TARGET
correlations = train_df[numeric_cols + ['TARGET']].corr()['TARGET'].drop('TARGET')
correlations = correlations.sort_values(ascending=False)

print("\nTop 10 positively correlated features with TARGET:")
print(correlations.head(10))

print("\nTop 10 negatively correlated features with TARGET:")
print(correlations.tail(10))

# Visualize top correlations
plt.figure(figsize=(10, 8))
top_corr = pd.concat([correlations.head(15), correlations.tail(15)])
plt.barh(range(len(top_corr)), top_corr.values)
plt.yticks(range(len(top_corr)), top_corr.index)
plt.xlabel('Correlation with TARGET')
plt.title('Top Features Correlated with TARGET', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: feature_correlations.png")
plt.show()
