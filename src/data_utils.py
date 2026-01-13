import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from src.target_detection import detect_columns_to_drop

def load_dataset(path):
    return pd.read_csv(path)

def preprocess_features(df, target_column):
    df.columns = [
        col.strip()                          
           .replace(' ', '_')               
           .replace('(', '')
           .replace(')', '')
           .replace('[', '')
           .replace(']', '')
           .replace(',', '_')
        for col in df.columns
    ]

    # Also clean the target_column string to match
    target_column = (
        target_column.strip()
        .replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('[', '')
        .replace(']', '')
        .replace(',', '_')
    )

    # Auto-drop irrelevant columns (LLM first, heuristics fallback)
    drops = detect_columns_to_drop(df, target_column)
    if drops:
        print(f"Dropping columns: {drops}")
        df = df.drop(columns=drops)
    else:
        print("No columns dropped.")

    df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '') for col in df.columns]


    #Auto Drop columns with >70% missing values
    missing_pct = df.isna().mean() *100
    high_missing_cols = missing_pct[(missing_pct >70) & (missing_pct.index != target_column)].index.tolist()
    if high_missing_cols:
        print(f"Dropping columns with >70% missing values: {high_missing_cols}")
        df = df.drop(columns=high_missing_cols)

    #Drop rows with missing values in target column
    initial_rows = len(df)
    if df[target_column].isna().any():
        missing_target_count = df[target_column].isna().sum()
        print(f"Dropping {missing_target_count} rows with missing target values.")
        df = df.dropna(subset=[target_column])
        print(f"Row reduced: {initial_rows} -> {len(df)} after dropping missing target rows.")
        
#Split X and y

    X = df.drop(columns=[target_column])
    y = df[target_column]


    

    #Separate numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = [col for col in X.select_dtypes(include=['object']).columns if X[col].nunique() < 20]


    high_card_cats = [col for col in X.select_dtypes(include=['object']).columns if X[col].nunique() >= 20]
    if high_card_cats:
        print(f"Dropping high-cardinality categorical columns: {high_card_cats}")
        X = X.drop(columns=high_card_cats)


    #Transformers

    numeric_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
   
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    X_processed = preprocessor.fit_transform(X)

    #Get feature name safely

    try:
        feature_names = preprocessor.get_feature_names_out()
    except ValueError:
        #Fallback: use original column names
        feature_names = X.columns

    if isinstance(X_processed, np.ndarray):
        X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    else:
        X_processed = pd.DataFrame(X_processed.toarray(), columns=feature_names, index=X.index)

    #Handling missing values for x
    nan_count = X_processed.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} remaining NaN in features after imputation. Filling with 0 (safe for one-hot encoded data)...")
        X_processed = X_processed.fillna(0)
    X_processed = X_processed.astype('float64')
    return X_processed, y