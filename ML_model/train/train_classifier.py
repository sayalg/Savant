from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from sklearn.model_selection import train_test_split

## Load data
df = pd.read_csv("../combined_50k_filtered_prepped_binary_classification.csv")

## Specify the target column
target = "binding_level"

## Prepare the features and target variable
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1337,
    stratify=y
)

## Create the preprocessing + model pipeline
model = Pipeline([
    (
        "scaler",
        RobustScaler(
            quantile_range=(25, 75),
            with_centering=False,
            with_scaling=True,
        ),
    ),
    (
        "classifier",
        RandomForestClassifier(
            bootstrap=False,
            class_weight="balanced",
            criterion="entropy",
            max_features=0.8,
            min_samples_leaf=0.01,
            min_samples_split=0.056842105263157895,
            n_estimators=10,
            oob_score=False,
            random_state=1337,
        ),
    ),
])

## Train the model
model.fit(X_train, y_train)

## Save the trained model
import joblib
with open("tender_comb_63w6mzz3.pkl", "wb") as f:
    joblib.dump(model, f)

## Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)


## Evaluate
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

train_accuracy = model.score(X_train, y_train)
print(f"Training accuracy: {train_accuracy:.4f}")

test_accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
)

precision_weighted = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall_weighted = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

print(f"Accuracy:                        {accuracy:.4f}")
print(f"Balanced Accuracy:               {balanced_accuracy:.4f}")
print(f"F1 Score (weighted):             {f1_weighted:.4f}")
print(f"Precision (weighted):            {precision_weighted:.4f}")
print(f"Recall (weighted):               {recall_weighted:.4f}")

