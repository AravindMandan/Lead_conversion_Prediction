
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import joblib
import time
def save_and_register_best_model_pipeline(results_df, best_models,
                                          X_train_val, y_train_val,
                                          preprocessor,
                                          save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    #  Step 1: Pick best model (based on highest F1-score)
    best_model_name = results_df.sort_values(by="f1_score", ascending=False).iloc[0]["model"]
    best_model = best_models[best_model_name]
    print(f"\n Best model selected: {best_model_name}")

    #  Step 2: Retrain on full data
    best_model.fit(X_train_val, y_train_val)

    #  Step 3: Build pipeline
    full_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", best_model)
    ])

    #  Step 4: Save pipeline locally
    model_path = os.path.join(save_dir, f"final_{best_model_name}_pipeline.pkl")
    joblib.dump(full_pipeline, model_path)
    print(f" Final pipeline saved at: {model_path}")

    #  Step 5: Log & register to MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Lead Conversion Classification")
    client = MlflowClient()

    with mlflow.start_run(run_name=f"Final_{best_model_name}") as run:
        run_id = run.info.run_id

        #  Log the full pipeline with sklearn flavor
        mlflow.sklearn.log_model(full_pipeline, artifact_path="model")

        print(f" Registering model to MLflow Model Registry: {best_model_name}")
        model_uri = f"runs:/{run_id}/model"

        #  Register the model under the given name
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=best_model_name
        )

        #  Wait for model to fully register (prevents race conditions)
        time.sleep(10)

        #  Transition model version to "Production" stage
        client.transition_model_version_stage(
            name=best_model_name,
            version=registered_model.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f" Model '{best_model_name}' version {registered_model.version} moved to 'Production'.")

        #  Optionally assign alias "champion" to this production version
        try:
            client.set_model_version_alias(
                name=best_model_name,
                version=registered_model.version,
                alias="champion"
            )
            print(f"🏷️ Alias 'champion' assigned to version {registered_model.version}.")
        except Exception as e:
            print(f"⚠️ Unable to set alias 'champion': {e}")

        #  Provide direct link to MLflow run
        print(f"🏃 View run: http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run_id}")

    return full_pipeline, best_model_name, model_path


