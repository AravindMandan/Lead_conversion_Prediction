import pandas as pd
from sklearn.pipeline import Pipeline

import pandas as pd
from sklearn.pipeline import Pipeline

def transform_data(X, preprocessor, numeric_features, categorical_features, fit=False):
    """
    Transforms the input features using the provided preprocessor pipeline.
    
    Parameters:
    - X: pd.DataFrame - Raw input features
    - preprocessor: sklearn ColumnTransformer or Pipeline
    - numeric_features: list of numeric column names
    - categorical_features: list of all categorical column names
    - fit: bool - Whether to fit the preprocessor (True for training data only)
    
    Returns:
    - pd.DataFrame - Transformed data with proper feature names
    """
    if fit:
        processed = preprocessor.fit_transform(X)
    else:
        processed = preprocessor.transform(X)

    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        all_columns = numeric_features + list(cat_feature_names)
        return pd.DataFrame(processed, columns=all_columns, index=X.index)
    except Exception as e:
        print("Warning: Could not extract feature names. Reason:", e)
        return pd.DataFrame(processed, index=X.index)


# Apply transformation
#X_transformed_df = transform_data(X, preprocessor, numeric_features, all_categorical)
#X_transformed_df.head()
