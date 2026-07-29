from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
import xgboost as xgb

import pandas as pd
from sklearn.model_selection import train_test_split

## Load data
df = pd.read_csv("../combined_50k_filtered_prepped_regression.csv")

## Specify the target column
target = "neg_log10_affinity_M"

## Prepare the features and target variable
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1337,
)

## Create the preprocessing + model pipeline
model = Pipeline([
    (
        "scaler",
        MaxAbsScaler(),
    ),
    (
        "regressor",
        xgb.XGBRegressor(
            tree_method="auto",
            n_estimators=10000,
        ),
    ),
])

## Train the model
model.fit(X_train, y_train)

## Save the trained model
import joblib
with open("mango_sponge_66k38p1d.pkl", "wb") as f:
    joblib.dump(model, f)

## Predictions
y_pred = model.predict(X_test)


## Evaluate
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    root_mean_squared_error
)

train_r2 = model.score(X_train, y_train)
print(f"Training R^2: {train_r2:.4f}")

test_r2 = model.score(X_test, y_test)
print(f"Test R^2: {test_r2:.4f}")

mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAPE:                            {mape:.4f}")
print(f"RMSE:                            {rmse:.4f}")
print(f"R^2:                             {r2:.4f}")

