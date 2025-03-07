import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

# Load data
X_test = pd.read_csv("data/normalized/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Load trained model
model = joblib.load("models/trained_model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Save predictions
pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).to_csv("data/processed/predictions.csv", index=False)

# Compute metrics
metrics = {
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
}

# Save metrics
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)