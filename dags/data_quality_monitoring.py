
from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
import pandas as pd
import os

default_args = {
    'owner': 'mahmoudi',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

PROJECT_ROOT = '/usr/local/airflow'

with DAG(
    dag_id='data_quality_monitoring',
    default_args=default_args,
    description='Daily data quality checks and monitoring',
    schedule='@daily',  
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['monitoring', 'data-quality', 'daily'],
) as dag:
    
    @task
    def check_data_completeness():
        
        print("="*80)
        print("CHECKING DATA COMPLETENESS")
        print("="*80)
        
        required_files = [
            f'{PROJECT_ROOT}/data/processed/train_features.csv',
            f'{PROJECT_ROOT}/data/processed/X_train.csv',
            f'{PROJECT_ROOT}/data/processed/X_test.csv',
        ]
        
        issues = []
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                issues.append(f"Missing file: {file_path}")
            else:
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    issues.append(f"Empty file: {file_path}")
                else:
                    print(f"{file_path} - {file_size:,} bytes")
        
        if issues:
            raise Exception(f"Data completeness issues:\n" + "\n".join(issues))
        
        print("\n Data completeness: PASSED")
        return True
    
    @task
    
    def check_data_quality():
      
        print("="*80)
        print("CHECKING DATA QUALITY")
        print("="*80)
        
        df = pd.read_csv(f'{PROJECT_ROOT}/data/processed/train_features.csv')
        
        # Calculate quality metrics
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = (duplicate_rows / len(df)) * 100
        
        print(f"\nData Quality Metrics:")
        print(f"  Total samples: {len(df):,}")
        print(f"  Total features: {len(df.columns)}")
        print(f"  Missing values: {missing_pct:.2f}%")
        print(f"  Duplicate rows: {duplicate_pct:.2f}%")
        
        # Check target distribution
        if 'TARGET' in df.columns:
            target_dist = df['TARGET'].value_counts(normalize=True)
            default_rate = target_dist.get(1, 0)
            print(f"  Default rate: {default_rate:.2%}")
            
            # Alert if default rate is unusual
            if default_rate < 0.05 or default_rate > 0.15:
                print(f"WARNING: Unusual default rate ({default_rate:.2%})")
        
        # Quality thresholds
        issues = []
        
        if missing_pct > 5.0:
            issues.append(f"High missing values: {missing_pct:.2f}%")
        
        if duplicate_pct > 1.0:
            issues.append(f"High duplicate rate: {duplicate_pct:.2f}%")
        
        if issues:
            print("\n Quality issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nData quality: EXCELLENT")
        
        return True
    
    @task
    
    def check_model_exists():
        print("="*80)
        print("CHECKING MODEL AVAILABILITY")
        print("="*80)
        
        model_path = f'{PROJECT_ROOT}/models/production/lightgbm_final_proper_split.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Production model not found: {model_path}")
        
        # Try loading model
        import joblib
        try:
            model = joblib.load(model_path)
            print(f" Model loaded successfully: {model_path}")
            print(f" Model type: {type(model).__name__}")
            print(f" Features: {model.n_features_}")
            return True
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
   
    completeness_check = check_data_completeness()
    
    quality_check = check_data_quality()
    
    model_check = check_model_exists()
    
    [completeness_check, quality_check, model_check]
   
   