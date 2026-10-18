import json
import os
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X_test_scaled)

# save predictions dataset
os.makedirs("data", exist_ok=True)
pd.DataFrame({"predictions": predictions}).to_csv("data/predictions.csv", index=False)

# calculate metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

os.makedirs("metrics", exist_ok=True)
metrics = {"mse": mse, "r2": r2}
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Model evaluation complete.")