import pandas as pd
import numpy as np
import gc

print("="*80)
print("ADVANCED FEATURE ENGINEERING WITH PREPROCESSING")
print("="*80)

print("\nLoading all data tables...")
app_train = pd.read_csv('data/raw/application_train.csv')
bureau = pd.read_csv('data/raw/bureau.csv')
bureau_balance = pd.read_csv('data/raw/bureau_balance.csv')
prev_app = pd.read_csv('data/raw/previous_application.csv')
pos_cash = pd.read_csv('data/raw/POS_CASH_balance.csv')
installments = pd.read_csv('data/raw/installments_payments.csv')
credit_card = pd.read_csv('data/raw/credit_card_balance.csv')

print(f"Application: {app_train.shape}")
print(f"Bureau: {bureau.shape}")
print(f"Bureau balance: {bureau_balance.shape}")
print(f"Previous app: {prev_app.shape}")
print(f"POS CASH: {pos_cash.shape}")
print(f"Installments: {installments.shape}")
print(f"Credit card: {credit_card.shape}")


def preprocess_table(df, table_name):
    print(f"\nPreprocessing {table_name}...")
    initial_shape = df.shape
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns with >90% missing
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.9].index.tolist()
    if cols_to_drop:
        print(f"  Dropping {len(cols_to_drop)} columns with >90% missing")
        df = df.drop(columns=cols_to_drop)
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"  Removing {duplicates} duplicate rows")
        df = df.drop_duplicates()
    
    # Handle outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q99)
    
    print(f"  {initial_shape} -> {df.shape}")
    return df


print("\n" + "="*80)
print("PREPROCESSING ALL TABLES")
print("="*80)

app_train = preprocess_table(app_train, "Application")
bureau = preprocess_table(bureau, "Bureau")
prev_app = preprocess_table(prev_app, "Previous Application")
pos_cash = preprocess_table(pos_cash, "POS CASH")
installments = preprocess_table(installments, "Installments")
credit_card = preprocess_table(credit_card, "Credit Card")


print("\n" + "="*80)
print("CREATING AGGREGATED FEATURES")
print("="*80)


def bureau_features(bureau):
    print("\nBureau aggregations...")
    
    numeric_cols = bureau.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'SK_ID_BUREAU']]
    
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'max', 'min', 'sum']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_dict)
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
    
    bureau_agg['BUREAU_COUNT'] = bureau.groupby('SK_ID_CURR').size()
    
    if 'CREDIT_ACTIVE' in bureau.columns:
        active = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size()
        bureau_agg['BUREAU_ACTIVE_COUNT'] = active
        
        closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed'].groupby('SK_ID_CURR').size()
        bureau_agg['BUREAU_CLOSED_COUNT'] = closed
    
    bureau_agg = bureau_agg.reset_index()
    print(f"  Created {bureau_agg.shape[1]-1} features")
    return bureau_agg


def prev_app_features(prev):
    print("\nPrevious application aggregations...")
    
    numeric_cols = prev.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'SK_ID_PREV']]
    
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'max', 'min', 'sum']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg(agg_dict)
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
    
    prev_agg['PREV_COUNT'] = prev.groupby('SK_ID_CURR').size()
    
    if 'NAME_CONTRACT_STATUS' in prev.columns:
        approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').size()
        prev_agg['PREV_APPROVED_COUNT'] = approved
        
        refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').size()
        prev_agg['PREV_REFUSED_COUNT'] = refused
    
    prev_agg = prev_agg.reset_index()
    print(f"  Created {prev_agg.shape[1]-1} features")
    return prev_agg


def pos_cash_features(pos):
    print("\nPOS CASH aggregations...")
    
    numeric_cols = pos.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'SK_ID_PREV']]
    
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'max', 'min', 'sum']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(agg_dict)
    pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    
    pos_agg = pos_agg.reset_index()
    print(f"  Created {pos_agg.shape[1]-1} features")
    return pos_agg


def installments_features(ins):
    print("\nInstallments aggregations...")
    
    numeric_cols = ins.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'SK_ID_PREV']]
    
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'max', 'min', 'sum']
    
    ins_agg = ins.groupby('SK_ID_CURR').agg(agg_dict)
    ins_agg.columns = ['INS_' + '_'.join(col).upper() for col in ins_agg.columns]
    ins_agg['INS_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    if 'AMT_PAYMENT' in ins.columns and 'AMT_INSTALMENT' in ins.columns:
        ins['PAYMENT_DIFF'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
        payment_diff = ins.groupby('SK_ID_CURR')['PAYMENT_DIFF'].mean()
        ins_agg['INS_PAYMENT_DIFF_MEAN'] = payment_diff
    
    ins_agg = ins_agg.reset_index()
    print(f"  Created {ins_agg.shape[1]-1} features")
    return ins_agg


def credit_card_features(cc):
    print("\nCredit card aggregations...")
    
    numeric_cols = cc.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'SK_ID_PREV']]
    
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'max', 'min', 'sum']
    
    cc_agg = cc.groupby('SK_ID_CURR').agg(agg_dict)
    cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    
    cc_agg = cc_agg.reset_index()
    print(f"  Created {cc_agg.shape[1]-1} features")
    return cc_agg


bureau_agg = bureau_features(bureau)
prev_agg = prev_app_features(prev_app)
pos_agg = pos_cash_features(pos_cash)
ins_agg = installments_features(installments)
cc_agg = credit_card_features(credit_card)

del bureau, bureau_balance, prev_app, pos_cash, installments, credit_card
gc.collect()


print("\n" + "="*80)
print("MERGING FEATURES WITH MAIN APPLICATION DATA")
print("="*80)

df = app_train.copy()
print(f"Starting shape: {df.shape}")

df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
print(f"After bureau: {df.shape}")

df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
print(f"After prev_app: {df.shape}")

df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
print(f"After pos: {df.shape}")

df = df.merge(ins_agg, on='SK_ID_CURR', how='left')
print(f"After installments: {df.shape}")

df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
print(f"After credit_card: {df.shape}")

print(f"\nOriginal features: {app_train.shape[1]}")
print(f"New features: {df.shape[1] - app_train.shape[1]}")
print(f"Total features: {df.shape[1]}")


print("\n" + "="*80)
print("CREATING INTERACTION FEATURES")
print("="*80)

df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
if all(col in df.columns for col in ext_sources):
    df['EXT_SOURCE_MEAN'] = df[ext_sources].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_sources].std(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

print(f"Added interaction features: {df.shape}")


print("\n" + "="*80)
print("FINAL PREPROCESSING")
print("="*80)

# Fill missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(-999)

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna('MISSING')

print(f"Missing values handled")

# Remove low variance features
from sklearn.feature_selection import VarianceThreshold
numeric_features = df.select_dtypes(include=[np.number]).columns
numeric_features = [col for col in numeric_features if col not in ['SK_ID_CURR', 'TARGET']]

selector = VarianceThreshold(threshold=0.01)
X_temp = df[numeric_features]
selector.fit(X_temp)

low_var_cols = [col for col, selected in zip(numeric_features, selector.get_support()) if not selected]
print(f"\nRemoving {len(low_var_cols)} low variance features")

df = df.drop(columns=low_var_cols)

print(f"\nFinal shape: {df.shape}")


print("\n" + "="*80)
print("SAVING ENHANCED DATASET")
print("="*80)

output_path = 'data/processed/train_features_enhanced.csv'
df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Size: {df.shape}")

feature_names = [col for col in df.columns if col not in ['SK_ID_CURR', 'TARGET']]
print(f"Total features: {len(feature_names)}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)