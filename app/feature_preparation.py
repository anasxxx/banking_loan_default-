"""
Feature Preparation Module
Matches the feature engineering pipeline used during training
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features matching training pipeline"""
    # Create interaction features (from 13_advanced_feature_engineering.py)
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'].replace(0, np.nan)
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'].replace(0, np.nan)
    
    # Handle division by zero for INCOME_PER_PERSON
    if 'CNT_FAM_MEMBERS' in df.columns:
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, np.nan)
    else:
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL']
    
    # External source features
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    if all(col in df.columns for col in ext_sources):
        df['EXT_SOURCE_MEAN'] = df[ext_sources].mean(axis=1)
        df['EXT_SOURCE_STD'] = df[ext_sources].std(axis=1)
        df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    return df


def get_default_feature_values() -> Dict[str, float]:
    """Get default values for features that might not be available in real-time predictions"""
    defaults = {}
    
    # For aggregated features from other tables (BUREAU, PREV, INS, CC, POS)
    # Use -999 as default (same as training pipeline uses for missing values)
    defaults['BUREAU_MISSING'] = -999
    defaults['PREV_MISSING'] = -999
    defaults['INS_MISSING'] = -999
    defaults['CC_MISSING'] = -999
    defaults['POS_MISSING'] = -999
    
    return defaults


def prepare_features_for_prediction(
    application_data: Dict,
    expected_feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare features from application data matching the training pipeline.
    
    Args:
        application_data: Dictionary containing application fields
        expected_feature_names: List of expected feature names in order (if available)
    
    Returns:
        DataFrame with features matching training pipeline
    """
    # Start with application-level features
    features = {}
    
    # Map application fields
    field_mapping = {
        'AMT_INCOME_TOTAL': 'AMT_INCOME_TOTAL',
        'AMT_CREDIT': 'AMT_CREDIT',
        'AMT_ANNUITY': 'AMT_ANNUITY',
        'AMT_GOODS_PRICE': 'AMT_GOODS_PRICE',
        'DAYS_BIRTH': 'DAYS_BIRTH',
        'DAYS_EMPLOYED': 'DAYS_EMPLOYED',
        'EXT_SOURCE_1': 'EXT_SOURCE_1',
        'EXT_SOURCE_2': 'EXT_SOURCE_2',
        'EXT_SOURCE_3': 'EXT_SOURCE_3',
    }
    
    # Extract basic features from application data
    for api_field, feature_name in field_mapping.items():
        if api_field in application_data:
            features[feature_name] = application_data[api_field]
        else:
            features[feature_name] = None
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([features])
    
    # Add other application features with defaults if not provided
    # These might be in the training data but not in simplified API input
    default_app_features = {
        'CNT_CHILDREN': 0,
        'DAYS_REGISTRATION': -1800,  # Default ~5 years
        'DAYS_ID_PUBLISH': -1800,
        'OWN_CAR_AGE': 0,
        'FLAG_EMP_PHONE': 1,
        'FLAG_WORK_PHONE': 1,
        'FLAG_PHONE': 1,
        'FLAG_EMAIL': 0,
        'CNT_FAM_MEMBERS': 2,
        'REGION_RATING_CLIENT': 2,
        'REGION_RATING_CLIENT_W_CITY': 2,
        'HOUR_APPR_PROCESS_START': 12,
        'REG_REGION_NOT_LIVE_REGION': 0,
        'REG_REGION_NOT_WORK_REGION': 0,
        'LIVE_REGION_NOT_WORK_REGION': 0,
        'REG_CITY_NOT_LIVE_CITY': 0,
        'REG_CITY_NOT_WORK_CITY': 0,
        'LIVE_CITY_NOT_WORK_CITY': 0,
    }
    
    for feature, default_value in default_app_features.items():
        if feature not in df.columns:
            df[feature] = default_value
    
    # Create interaction features
    df = create_interaction_features(df)
    
    # Handle missing values - use -999 like training pipeline
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(-999)
    
    # If we have expected feature names, align the DataFrame
    if expected_feature_names:
        # Create a DataFrame with all expected features
        aligned_df = pd.DataFrame(index=df.index)
        
        for feature_name in expected_feature_names:
            if feature_name in df.columns:
                aligned_df[feature_name] = df[feature_name]
            elif feature_name.startswith('BUREAU_'):
                # Aggregated bureau features - use -999
                aligned_df[feature_name] = -999
            elif feature_name.startswith('PREV_'):
                # Aggregated previous application features - use -999
                aligned_df[feature_name] = -999
            elif feature_name.startswith('INS_'):
                # Aggregated installments features - use -999
                aligned_df[feature_name] = -999
            elif feature_name.startswith('CC_'):
                # Aggregated credit card features - use -999
                aligned_df[feature_name] = -999
            elif feature_name.startswith('POS_'):
                # Aggregated POS features - use -999
                aligned_df[feature_name] = -999
            else:
                # Unknown feature - use -999
                aligned_df[feature_name] = -999
        
        # Ensure columns are in exact order
        aligned_df = aligned_df[expected_feature_names]
        return aligned_df
    
    return df


def load_expected_feature_names(model_path: str = None) -> Optional[List[str]]:
    """
    Load expected feature names from training data or model.
    
    Returns:
        List of feature names in expected order, or None if not available
    """
    try:
        # Try to load from saved feature names file
        import json
        import os
        
        feature_names_path = 'models/production/feature_names.json'
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                data = json.load(f)
                return data.get('feature_names')
        
        # If not available, try to extract from training data
        training_data_path = 'data/processed/train_features_enhanced.csv'
        if os.path.exists(training_data_path):
            df = pd.read_csv(training_data_path, nrows=1)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'TARGET']]
            return numeric_cols
        
    except Exception as e:
        print(f"Warning: Could not load feature names: {e}")
    
    return None
