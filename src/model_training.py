import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

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
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000)
        }
    else:
        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions) if problem_type == 'classification' else mean_squared_error(y_test, predictions)
        results[name] = score
        print(f"{name} {problem_type} score: {score}")

    best_model = max(results, key=results.get) if problem_type == 'classification' else min(results, key=results.get)
    print(f"Best model: {best_model} with score: {results[best_model]}")
    return best_model