from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mahmoudi',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

PROJECT_ROOT = '/usr/local/airflow'
NOTEBOOKS_DIR = f'{PROJECT_ROOT}/notebooks'

with DAG(
    dag_id='manual_model_retrain',
    default_args=default_args,
    description='Full retraining pipeline with enhanced features',
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'manual', 'retrain'],
) as dag:
    
    create_enhanced_features = BashOperator(
        task_id='create_enhanced_features',
        bash_command=f'cd {PROJECT_ROOT} && python {NOTEBOOKS_DIR}/13_advanced_feature_engineering.py',
    )
    
    train_optimized_model = BashOperator(
        task_id='train_optimized_model',
        bash_command=f'cd {PROJECT_ROOT} && python {NOTEBOOKS_DIR}/12_full_dataset_retraining.py',
    )
    
    validate_robustness = BashOperator(
        task_id='robustness_validation',
        bash_command=f'cd {PROJECT_ROOT} && python {NOTEBOOKS_DIR}/11_robustness_validation.py',
    )
    
    create_enhanced_features >> train_optimized_model >> validate_robustness