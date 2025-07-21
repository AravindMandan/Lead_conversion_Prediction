from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# Load final pipeline (with preprocessing)
pipeline = joblib.load("saved_models/final_LightGBM_pipeline.pkl")  # update with your pipeline path

# PostgreSQL connection
#DB_URI = 'postgresql+psycopg2://postgres:Aravind45@localhost:5432/mydb45'
#engine = create_engine(DB_URI)

# Define required features (as per your final selection)
features = [
    'Lead Origin',
    'Lead Source',
    'Do Not Email',
    'Do Not Call',
    'TotalVisits',
    'Total Time Spent on Website',
    'Page Views Per Visit',
    'Last Activity',
    'Country',
    'Specialization',
    'How did you hear about X Education',
    'What is your current occupation',
    'What matters most to you in choosing a course',  # ✅ correct spelling
    'Search',
    'Newspaper Article',
    'X Education Forums',
    'Newspaper',
    'Digital Advertisement',
    'Through Recommendations',
    'Tags',
    'Lead Quality',
    'Lead Profile',
    'City',
    'Asymmetrique Activity Index',
    'Asymmetrique Profile Index',
    'Asymmetrique Activity Score',
    'Asymmetrique Profile Score',
    'A free copy of Mastering The Interview',  # ✅ correct casing
    'Last Notable Activity'
]


@app.route("/")
def home():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_text = None

    if request.method == "POST":
        try:
            # Capture form input
            input_data = {feat: request.form.get(feat, '') for feat in features}

            # Numeric conversions
            numeric_cols = [
                'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit',
                'Asymmetrique Activity Index', 'Asymmetrique Profile Index',
                'Asymmetrique Activity Score', 'Asymmetrique Profile Score'
            ]
            for col in numeric_cols:
                input_data[col] = float(input_data[col]) if input_data[col] else 0.0

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Predict using pipeline
            prediction = pipeline.predict(input_df)[0]
            confidence = pipeline.predict_proba(input_df).max()

            prediction_text = f"📊 Lead Conversion Prediction: {int(prediction)} (Confidence: {confidence:.2%})"

            # Log prediction
           # input_data['prediction'] = int(prediction)
            #input_data['confidence'] = round(confidence, 4)
           # pd.DataFrame([input_data]).to_sql("leadstable", engine, if_exists="append", index=False)

        except Exception as e:
            prediction_text = f"❌ Error: {str(e)}"

    return render_template("predict.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
