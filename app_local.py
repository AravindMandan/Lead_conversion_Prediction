from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import os
import logging
from werkzeug.utils import secure_filename
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from sqlalchemy import create_engine

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ✅ Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # ⚠️ Replace with actual tracking URI if needed

# ✅ PostgreSQL DB URI
DB_URI = 'postgresql+psycopg2://postgres:Aravind45@localhost:5432/mydb45'
engine = create_engine(DB_URI)

# 🔁 Get latest production model name from MLflow registry
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
    print(f"✅ Will load model: {chosen_model} (version {candidates[0][1]})")
    return chosen_model

# 🔁 Load model from MLflow registry
def load_model_from_registry(stage="Production", alias=None):
    try:
        model_name = get_latest_production_model_name(stage=stage, alias=alias)
        model_uri = f"models:/{model_name}/{alias or stage}"
        logger.info(f"📦 Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded successfully from MLflow")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model from MLflow: {str(e)}")
        return None

# 🔁 Load model once
pipeline = load_model_from_registry(stage="Production", alias=None)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return "❌ No file part in request"

    file = request.files['file']
    if file.filename == '':
        return "❌ No selected file"

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            df = pd.read_csv(file_path)

            # ✅ Predict
            preds = pipeline.predict(df)
            df['prediction'] = preds
            df['prediction_label'] = df['prediction'].map({0: 'Not Converted', 1: 'Converted'})

# Save to DB
            df.to_sql("outputs_table", engine, if_exists="append", index=False)

            
           

            # ✅ Save to PostgreSQL
            #df.to_sql("LeadScoring", engine, if_exists="append", index=False)

            # ✅ Save locally
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"scored_{filename}")
            df.to_csv(output_path, index=False)

            return render_template('predict.html',
                                   tables=[df.head(10).to_html(classes='table table-bordered')],
                                   csv_file=output_path)

        except Exception as e:
            logger.exception("Prediction failed")
            return f"❌ Prediction failed: {str(e)}"

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4545)
