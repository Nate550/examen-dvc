import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import yaml

# Load parameters from params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Load datasets
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Define model and parameters from YAML
param_grid = params['grid_search']

grid_search = GridSearchCV(RandomForestRegressor(random_state=params['model']['random_state']), 
                           param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Save best parameters
joblib.dump(grid_search.best_params_, "models/best_params.pkl")