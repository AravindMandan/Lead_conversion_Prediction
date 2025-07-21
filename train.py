from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from evaluate_metrics import evaluate_classification_model
import joblib
import os

def train_log_and_shap_classification(X_train, y_train, X_test, y_test,
                                      preprocessor, save_dir="saved_models", shap_dir="shap_outputs"):

    #X_train, X_test, y_train, y_test = train_test_split(X_transformed_df, y, stratify=y, test_size=0.2, random_state=42)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Lead Conversion Classification")

    #  Define models here
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
            'params': {
                'C': [0.1, 1.0, 10.0]
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(class_weight='balanced', random_state=42),
            'params': {
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(scale_pos_weight=1, use_label_encoder=False, eval_metric='logloss', random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6]
            }
        },
        'LightGBM': {
        'model': LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6]
    }
}

    }

    results = []
    best_models = {}

    for name, model_info in models.items():
        print(f"\n Training: {name}")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(model_info['model'], model_info['params'],
                            cv=skf, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_test_pred = grid.predict(X_test)
        y_test_proba = grid.predict_proba(X_test)[:, 1] if hasattr(grid, "predict_proba") else None

        metrics = evaluate_classification_model(y_test, y_test_pred, y_test_proba)
        results.append({"model": name, "best_params": grid.best_params_, **metrics})
        best_models[name] = grid.best_estimator_

        # Save model
        model_path = os.path.join(save_dir, f"{name}_best_model.pkl")
        joblib.dump(grid.best_estimator_, model_path)
        
       
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    
        #  MLflow logging
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(scalar_metrics)
            mlflow.sklearn.log_model(grid.best_estimator_, "model")

           

                
    return results, best_models
#results, best_models = train_log_and_shap_classification(
   # X_train=X_train,
    #y_train=y_train,
    #X_test=X_test,
   # y_test=y_test,
    #preprocessor=preprocessor
#)

