from src.data_utils import load_dataset, preprocess_features
from src.target_detection import detect_target
from src.model_training import train_models


def main():
    #Load dataset
    df = load_dataset('data/StudentPerformance.csv')
    print("Dataset loaded with shape: ", df.shape)

    #Detect target column
    target_column = detect_target(df)
    print("Detected target column: ", target_column)

    #Preprocess features and target
    X, y = preprocess_features(df, target_column)
    print(f"Features and target preprocessed. Feature shape: {X.shape}, Target shape: {y.shape}")

    #train models automatically

    best_model_name = train_models(X, y)

    print("\n SwaML pipeline finished successfully!")
    print("Best model selected:", best_model_name)

if __name__ == "__main__":
    main()
