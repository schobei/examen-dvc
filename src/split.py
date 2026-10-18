import os
import pandas as pd
from sklearn.model_selection import train_test_split

# load raw data
df = pd.read_csv("data/raw_data/raw.csv")

# target variable silica_concentrate
# only include numeric variables
X = df.iloc[:, :-1].select_dtypes(include=["number"])
y = df.iloc[:, -1]

# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save to processed directory
#os.makedirs("data/processed_data", exist_ok=True)
X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)
print("Data splitting complete.")