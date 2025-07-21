import os
import mlflow
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_and_log_drift_reports(historical_data, new_data, output_dir, feature_names=None):
    def ensure_df(data, feature_names):
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data, columns=feature_names)

    # Ensure proper DataFrame format
    historical_data = ensure_df(historical_data, feature_names)
    new_data = ensure_df(new_data, feature_names)

    os.makedirs(output_dir, exist_ok=True)

    comparison_name = "historical_vs_new"

    # Set MLflow experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Drift Monitoring")

    with mlflow.start_run(run_name=comparison_name):
        print(f"🚀 Checking drift: {comparison_name}")

        # Run Evidently report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=historical_data, current_data=new_data)

        # Save HTML report
        html_path = os.path.join(output_dir, f"{comparison_name}.html")
        report.save_html(html_path)
        mlflow.log_artifact(html_path, artifact_path="evidently_html_reports")

        # Extract drift metrics from dict
        report_dict = report.as_dict()
        drift_result = next(
            (m["result"] for m in report_dict["metrics"] if m.get("metric") == "DataDriftTable"),
            None
        )

        if drift_result:
            # Log overall drift ratio
            mlflow.log_metric(f"{comparison_name}_drift_ratio", round(drift_result["share_of_drifted_columns"], 4))

            # Log per-column drift scores
            for feature, vals in drift_result["drift_by_columns"].items():
                score = vals.get("drift_score")
                if score is not None:
                    clean_name = feature.replace(" ", "_").replace("(", "").replace(")", "")
                    mlflow.log_metric(f"{comparison_name}_{clean_name}", round(score, 4))

        print(f"Drift report logged to MLflow (run_id: {mlflow.active_run().info.run_id})")
        print(f"Report saved: {html_path}")
generate_and_log_drift_reports(
    historical_data=historical_df,
    new_data=new_leads_df,
    output_dir="drift_reports",
    feature_names=preprocessor.get_feature_names_out()
)
