import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load data
X_train = pd.read_csv("data/normalized/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Load best parameters
best_params = joblib.load("models/best_params.pkl")

# Train model
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "models/trained_model.pkl")