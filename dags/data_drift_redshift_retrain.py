
# dags/data_drift_redshift_retrain.py

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.amazon.aws.hooks.redshift_sql import RedshiftSQLHook
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from datetime import timedelta
import pandas as pd
import tempfile
import os

from evidently.report import Report
from evidently.metrics import DataDriftPreset

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "redshift_drift_and_retrain",
    default_args=default_args,
    description="Detect drift on Redshift data and retrain if needed",
    schedule_interval="0 2 * * *",  # Once daily at 2am, or adjust as you like
    start_date=days_ago(1),
    catchup=False,
)

# Helper: Run query and load as pandas DataFrame
def load_redshift_to_df(sql, redshift_conn_id):
    hook = RedshiftSQLHook(redshift_conn_id=redshift_conn_id)
    conn = hook.get_conn()
    return pd.read_sql(sql, con=conn)

def load_data(**context):
    # Adjust SQL queries as per your schema
    HISTORICAL_SQL = "SELECT * FROM leadstable WHERE label IS NOT NULL"
    NEWDATA_SQL = "SELECT * FROM leadstable WHERE label IS NULL AND ingestion_time > dateadd(day,-1, getdate())"

    redshift_conn_id = "redshift_default"
    historical = load_redshift_to_df(HISTORICAL_SQL, redshift_conn_id)
    newdata = load_redshift_to_df(NEWDATA_SQL, redshift_conn_id)

    # Store in temp files for XCom sharing
    temp_hist = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    temp_new = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    historical.to_csv(temp_hist.name, index=False)
    newdata.to_csv(temp_new.name, index=False)

    context['ti'].xcom_push(key='historical_path', value=temp_hist.name)
    context['ti'].xcom_push(key='newdata_path', value=temp_new.name)

def detect_drift(**context):
    hist_path = context['ti'].xcom_pull(key='historical_path')
    new_path = context['ti'].xcom_pull(key='newdata_path')
    hist = pd.read_csv(hist_path)
    new = pd.read_csv(new_path)

    # Customize column_mapping if needed
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=hist, current_data=new)
    drift = report.as_dict()["metrics"][0]["result"]["dataset_drift"]

    flag = "retrain_task" if drift else "no_drift_notify"
    context['ti'].xcom_push(key='drift_flag', value=flag)
    return flag

def notify_no_drift(**context):
    print("No significant data drift detected. No retraining needed.")

def retrain_model(**context):
    hist_path = context['ti'].xcom_pull(key='historical_path')
    new_path = context['ti'].xcom_pull(key='newdata_path')
    hist = pd.read_csv(hist_path)
    new = pd.read_csv(new_path)
    all_data = pd.concat([hist, new])
    print("Drift detected. Retraining with full data size:", all_data.shape)

    # Option 1: Save to S3 and kick off SageMaker job
    # Option 2: Initiate your train.py pipeline here (locally, via SageMaker, or on EC2)
    # Example placeholder for S3 upload, actual implementation needed
    # s3 = boto3.client('s3')
    # all_data.to_csv('/tmp/train_data.csv', index=False)
    # s3.upload_file('/tmp/train_data.csv', 'your-bucket', 'ml/train/train_data.csv')

    # Optionally trigger downstream task with SageMakerTrainingOperator or boto3 here

with dag:
    load = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        provide_context=True,
    )

    drift = BranchPythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift,
        provide_context=True,
    )

    no_drift = PythonOperator(
        task_id="no_drift_notify",
        python_callable=notify_no_drift,
        trigger_rule=TriggerRule.NONE_FAILED,
        provide_context=True,
    )

    retrain = PythonOperator(
        task_id="retrain_task",
        python_callable=retrain_model,
        trigger_rule=TriggerRule.NONE_FAILED,
        provide_context=True,
    )

    load >> drift
    drift >> [no_drift, retrain]
