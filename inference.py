import joblib
import os
import numpy as np
import pandas as pd
import json
import io

def model_fn(model_dir):
    """Load the model and scaler from the directory."""
    clf = joblib.load(os.path.join(model_dir, "LogisticRegression_best_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "preprocess.pkl"))
    return clf, scaler

def input_fn(request_body, content_type="text/csv"):
    """Parse CSV input into a pandas DataFrame or NumPy array."""
    if content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body.decode("utf-8")), header=None)
        return df.values
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Apply scaler and make prediction."""
    clf, scaler = model
    input_scaled = scaler.transform(input_data)
    prediction = clf.predict(input_scaled)
    return prediction

def output_fn(prediction, accept="application/json"):
    """Format output as JSON."""
    if accept == "application/json":
        return json.dumps(prediction.tolist()), accept
    elif accept == "text/csv":
        return ",".join(map(str, prediction)), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")