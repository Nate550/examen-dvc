import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/raw/raw.csv")

# Drop 'date' column as it's not numerical
df = df.drop(columns=["date"])

# Split into features and target
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Data splitting completed successfully.")