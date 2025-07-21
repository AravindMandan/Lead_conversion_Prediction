from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
 
with DAG("ml_pipeline_dag",
         start_date=datetime(2025, 7, 1),
         schedule_interval=None,
         catchup=False) as dag:
 
    ingest = BashOperator(
        task_id="data_ingestion",
        bash_command="python /opt/airflow/data_load.py"
    )
 
    
 
    train_model = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/main.py"
    )
 
    predict = BashOperator(
        task_id="predict",
        bash_command="python /opt/airflow/predict.py"
    )
 
    run_flask = BashOperator(
        task_id="run_flask_app",
        bash_command="python /opt/airflow/flaskapp/app_bulk.py"
    )
 
    ingest >> train_model >> predict >> run_flask