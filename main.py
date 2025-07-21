# -------------------- Imports --------------------
import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# DB
from sqlalchemy import create_engine
import psycopg2

# ML & Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Flask (for later use)
from flask import Flask, request, jsonify

# -------------------- Custom Imports --------------------
import data_load
from mapping import apply_custom_mappings
from processing import build_preprocessing_pipeline
from data_transform import transform_data
from evaluate_metrics import evaluate_classification_model
from train import train_log_and_shap_classification
from savemodel import save_and_register_best_model_pipeline
from predict import load_and_predict_from_registry_auto

# -------------------- Pipeline Start --------------------
# Load data from PostgreSQL or any source
df = data_load.data_ingestion()
print(df.head())

# Drop unwanted features
features_to_drop = [
    'Prospect ID',
    'Lead Number',
    'Get updates on DM Content',
    'Receive More Updates About Our Courses',
    'I agree to pay the amount through cheque',
    'Magazine',
    'Update me on Supply Chain Content',
]

df.drop(columns=features_to_drop, inplace=True)
print(df.shape)

# Apply mappings
df = apply_custom_mappings(df)

# Feature transformation for skew reduction
df['TotalVisits'] = np.log1p(df['TotalVisits'])
df['Page Views Per Visit'] = np.log1p(df['Page Views Per Visit'])
df['Total Time Spent on Website'] = np.sqrt(df['Total Time Spent on Website'])

# Print skewness
print(df.skew(numeric_only=True))

# Split features and target
X = df.drop(columns=["Converted"])
y = df["Converted"]

# Build preprocessing pipeline
preprocessor, numeric_features, ordinal_features, nominal_features = build_preprocessing_pipeline(X)
all_categorical = ordinal_features + nominal_features

# Split before transformation 
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Apply transformation on both train and test
X_train_transformed = transform_data(X_train, preprocessor, numeric_features, all_categorical, fit=True)
X_test_transformed = transform_data(X_test, preprocessor, numeric_features, all_categorical, fit=False)
X_transformed_df = transform_data(X, preprocessor, numeric_features, all_categorical)

#Applying SMOTE to handle class imbalance for Target Variable
smote = SMOTE(random_state=42)
X_train_transformed, y_train = smote.fit_resample(X_train_transformed, y_train)

# Train models + SHAP
results, best_models = train_log_and_shap_classification(
    X_train=X_train_transformed,
    y_train=y_train,
    X_test=X_test_transformed,
    y_test=y_test,
    preprocessor=preprocessor
)
for result in results:
    print("\n Model:", result["model"])
    for k, v in result.items():
        if k != "model":
            print(f"   {k}: {v}")

# Save best model + pipeline
results_df = pd.DataFrame(results)

pipeline, best_model_name, model_path = save_and_register_best_model_pipeline(
    results_df=results_df,
    best_models=best_models,
    X_train_val=X_transformed_df,
    y_train_val=y,
    preprocessor=preprocessor
    
)

print(f"\n Best Model: {best_model_name}")
print(f" Model saved at: {model_path}")

#calling Predict.py
y_pred = load_and_predict_from_registry_auto(df, stage="Production")





