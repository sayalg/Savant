import pandas as pd
from sklearn.model_selection import train_test_split
from tabfm import TabFMRegressor, TabFMClassifier, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

working_dir = "/scratch/cford38/savant_tests/"

## Load in data
reg_df = pd.read_csv(f"{working_dir}/combined_50k_filtered_prepped_regression.csv")

## Create the training and testing datasets
X = reg_df.drop(columns=['neg_log10_affinity_M'])
y = reg_df['neg_log10_affinity_M']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1337
)

## Train the TabFM regression model
model = tabfm_v1_0_0.load(model_type="regression", device="cuda")
reg = TabFMRegressor(model=model)
reg.fit(X_train, y_train)

## Evaluate the regression model
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

preds = reg.predict(X_test)
y_pred = preds['tabfm_prediction']

rmse = math.sqrt(mean_squared_error(y_test, preds))
mape = mean_absolute_percentage_error(y_test, preds)
r2 = r2_score(y_test, preds)


## Save the model as a pkl
with open(f"{working_dir}/tabfm_regression_model.pkl", "wb") as f:
    import pickle
    pickle.dump(reg, f)