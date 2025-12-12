from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'mahmoudi',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

PROJECT_ROOT = '/usr/local/airflow'
NOTEBOOKS_DIR = f'{PROJECT_ROOT}/notebooks'

with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='Complete ML pipeline with enhanced features and optimized training',
    schedule='@weekly',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'production', 'weekly'],
) as dag:
    
    data_preparation = BashOperator(
        task_id='data_preparation',
        bash_command=f'cd {PROJECT_ROOT} && python {NOTEBOOKS_DIR}/02_clean_data.py',
    )
    
    feature_engineering = BashOperator(
        task_id='feature_engineering',
        bash_command=f'cd {PROJECT_ROOT} && python {NOTEBOOKS_DIR}/03_feature_engineering.py',
    )
    
    enhanced_feature_engineering = BashOperator(
        task_id='enhanced_feature_engineering',
        bash_command=f'cd {PROJECT_ROOT} && python {NOTEBOOKS_DIR}/13_advanced_feature_engineering.py',
    )
    
    model_training = BashOperator(
        task_id='model_training',
        bash_command=f'cd {PROJECT_ROOT} && python {NOTEBOOKS_DIR}/12_full_dataset_retraining.py',
    )
    
    robustness_testing = BashOperator(
        task_id='robustness_validation',
        bash_command=f'cd {PROJECT_ROOT} && python {NOTEBOOKS_DIR}/11_robustness_validation.py',
    )
    
    @task
    def promote_model_to_production():
        import shutil
        from datetime import datetime
        
        source_model = f'{PROJECT_ROOT}/models/production/lightgbm_full_optimized.pkl'
        prod_model = f'{PROJECT_ROOT}/models/production/current_production_model.pkl'
        backup_model = f'{PROJECT_ROOT}/models/production/backup_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        
        if os.path.exists(prod_model):
            shutil.copy(prod_model, backup_model)
            print(f"Backed up current model: {backup_model}")
        
        if os.path.exists(source_model):
            shutil.copy(source_model, prod_model)
            print(f"Model promoted to production: {prod_model}")
            
            os.makedirs(f'{PROJECT_ROOT}/outputs', exist_ok=True)
            with open(f'{PROJECT_ROOT}/outputs/promotion_log.txt', 'a') as f:
                f.write(f"{datetime.now().isoformat()} - Model promoted to production\n")
            
            return True
        else:
            raise FileNotFoundError(f"Model not found: {source_model}")
    
    model_promotion = promote_model_to_production()
    
    data_preparation >> feature_engineering >> enhanced_feature_engineering >> model_training >> robustness_testing >> model_promotion