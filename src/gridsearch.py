import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# define model and parameter grid
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

# save best parameters
os.makedirs("models", exist_ok=True)
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)
print("GridSearch complete. Best params saved.")