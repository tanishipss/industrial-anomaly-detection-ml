from src.data_loader import load_data
from src.feature_engineering import create_features
from src.model import train_base_model
from src.inference import pseudo_label_and_predict
import pandas as pd

def main():
    print("Loading data...")
    train, test = load_data()

    print("Creating features...")
    train = create_features(train)
    test = create_features(test)

    X = train.drop(columns=["target", "Date"])
    y = train["target"]

    X_test = test.drop(columns=["Date"])
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    print("Training model...")
    model, threshold = train_base_model(X, y)

    print("Running inference...")
    preds = pseudo_label_and_predict(model, X, X_test, y)

    output = pd.DataFrame({"prediction": preds})
    output.to_csv("artifacts/predictions.csv", index=False)

    print("Pipeline completed successfully.")
    print("Predictions saved to artifacts/predictions.csv")

if __name__ == "__main__":
    main()
