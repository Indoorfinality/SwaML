import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import LabelEncoder

# #Output directories
# os.makedirs('models', exist_ok=True)
# os.makedirs('plot', exist_ok=True)

def detect_problem_type(y):
    unique_values = y.nunique()
    if pd.api.types.is_numeric_dtype(y):
        return 'regression' if unique_values > 20 else 'classification'
    return 'classification'

def train_models(X, y):
    problem_type = detect_problem_type(y)
    print(f"Detected problem type: {problem_type.capitalize()}")
    
    #-------Label Encoding for Classification for XGBoost---------
    label_encoder = None
    if problem_type == 'classification' and y.dtype == 'object':
        print("String target labels detected. Encoding to integers for model compatibility...") 
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y)) 
        mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
        print("Label encoding mapping:", mapping)


    #---------------------------------------------------------------

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

    trained_models = {}

    print(f"\nTraining {len(models)} models...\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        trained_models[name] = model
        if problem_type == 'classification':
            score = accuracy_score(y_test, predictions)
            score_percentage = score * 100  # Convert to percentage
            results[name] = score  # Keep original for comparison
            print(f"{name} {problem_type} score: {score_percentage:.2f}%")
        else:
            rmse = root_mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            results[name] = rmse
            print(f"✓ {name}: RMSE = {rmse:.4f}, R² = {r2:.4f}")

    print("\n" + "=" * 70)


    if problem_type == 'classification':
        best_name = max(results, key=results.get)
        best_score = results[best_name] * 100
        print(f"BEST MODEL: {best_name} with {best_score:.2f}% accuracy")
    else:
        best_name = min(results, key=results.get)
        best_rmse = results[best_name]
        best_r2 = r2_score(y_test, trained_models[best_name].predict(X_test))
        print(f"BEST MODEL: {best_name} with RMSE = {best_rmse:.4f}, R² = {best_r2:.4f}")

    best_model = trained_models[best_name]

    # ------------------ Stratified/KFold Cross-Validation ------------------


    print("\nRunning 5-fold cross-validation ....")
    if problem_type == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'accuracy'
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'r2'

    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring=scoring, n_jobs = -1)
    print(f"5-fold CV {scoring.upper()}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return best_name, best_model, results