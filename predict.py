import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

def get_latest_production_model_name(stage="Production", alias=None):
    client = MlflowClient()
    registered = client.search_registered_models()
    
    if not registered:
        raise RuntimeError("No models registered in MLflow!")

    candidates = []
    for m in registered:
        for lv in m.latest_versions:
            if alias:
                aliases = getattr(lv, 'aliases', [])
                if alias in aliases:
                    candidates.append((m.name, lv.version, lv.creation_timestamp))
            else:
                if lv.current_stage == stage:
                    candidates.append((m.name, lv.version, lv.creation_timestamp))

    if not candidates:
        raise ValueError(f"No model found in MLflow registry for stage='{stage}' alias='{alias}'")

    candidates.sort(key=lambda t: t[2], reverse=True)
    chosen_model = candidates[0][0]
    print(f" Will load {chosen_model} version {candidates[0][1]} (stage/alias: '{alias or stage}')")
    return chosen_model

def load_and_predict_from_registry_auto(df, stage="Production", alias=None):
    model_name = get_latest_production_model_name(stage=stage, alias=alias)
    model_uri = f"models:/{model_name}/{alias or stage}"
    print(f" Loading model from {model_uri}")

    try:
        loaded_pipeline = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f" Failed to load model: {e}")

    # Ensure input schema is valid
    expected_columns = getattr(loaded_pipeline, "feature_names_in_", None)

    if expected_columns is None:
        # Try fetching from inner pipeline step
        for step in getattr(loaded_pipeline, "steps", []):
            if hasattr(step[1], "feature_names_in_"):
                expected_columns = step[1].feature_names_in_
                break

    if expected_columns is None:
        raise ValueError(" Cannot determine expected input columns. Ensure model is a pipeline with feature_names_in_.")

    # Align DataFrame
    try:
        X_test = df[list(expected_columns)].copy()
    except KeyError as e:
        raise ValueError(f" DataFrame missing required columns: {e}")

    # Predict
    try:
        predictions = loaded_pipeline.predict(X_test)
    except Exception as e:
        raise ValueError(f" Prediction failed: {e}")

    print(f" Predictions complete. Sample: {predictions[:4]}")
    return predictions



