import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
def detect_problem_type(y):
    unique_values = y.nunique()
    if pd.api.types.is_numeric_dtype(y):
        return 'regression' if unique_values > 20 else 'classification'
    return 'classification'

def train_models(X, y):
    problem_type = detect_problem_type(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if problem_type == 'classification':
        models = {
            'LightGBM': LGBMClassifier(
                random_state=42,
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                verbose=-1  # Suppress warnings
            ),
            'XGBoost': XGBClassifier(
                random_state=42,
                n_estimators=200,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss'
            ),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000),


        }
    else:
        models = {
            'LightGBM': LGBMRegressor(
                random_state=42,
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                verbose=-1
            ),
            'XGBoost': XGBRegressor(
                random_state=42,
                n_estimators=200,
                max_depth=3,
                learning_rate=0.1
            ),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        if problem_type == 'classification':
            score = accuracy_score(y_test, predictions)
            score_percentage = score * 100  # Convert to percentage
            results[name] = score  # Keep original for comparison
            print(f"{name} {problem_type} score: {score_percentage:.2f}%")
        else:
            score = root_mean_squared_error(y_test, predictions)
            results[name] = score
            print(f"{name} {problem_type} score (RMSE): {score:.4f}")

    best_model = max(results, key=results.get) if problem_type == 'classification' else min(results, key=results.get)
    print(f"Best model: {best_model} with score: {results[best_model]}")
    return best_model