import pandas as pd


def load_dataset(path):
    return pd.read_csv(path)

def preprocess_features(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]


     # One-hot encode categorical, fill missing
    X = pd.get_dummies(X)
    X = X.fillna(0)

   
    return X, y




