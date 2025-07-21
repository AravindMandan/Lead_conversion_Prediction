from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  MinMaxScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import joblib
import pandas as pd


def build_preprocessing_pipeline(X):
    
    # Manual ordinal features (with meaningful order)
    ordinal_features = [
        "Lead Quality", "Asymmetrique Activity Index", "Asymmetrique Profile Index"
    ]
    
    
    # Detect features
    categorical_all = X.select_dtypes(include=['object']).columns.tolist()
    ordinal_features = [col for col in ordinal_features if col in categorical_all]
    nominal_features = [col for col in categorical_all if col not in ordinal_features]
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()

    # Pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('ord', ordinal_pipeline, ordinal_features),
        ('nom', nominal_pipeline, nominal_features)
    ])
    joblib.dump(preprocessor, 'preprocess.pkl')
    print(f" Preprocessing pipeline saved at:preprocess.pkl")

    return preprocessor, numeric_features, ordinal_features, nominal_features
#preprocessor, numeric_features, ordinal_features, nominal_features = build_preprocessing_pipeline(X)

