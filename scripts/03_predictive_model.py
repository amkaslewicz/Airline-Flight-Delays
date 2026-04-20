#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:21:30 2026

@author: alaina
"""

# --------------------------------------------------
# Flight Delay Modeling
# This script builds predictive models for:
# 1. delay probability
# 2. delay severity
# 3. dominant delay cause
# Outputs: model performance summary + example predictions
# --------------------------------------------------

#Import Libraries
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#Setup folders
os.makedirs("output/modeling", exist_ok=True)

#Load data
delays = pd.read_csv("Airline_Delay_Cause.csv")

# --------------------------------------------------
#Data Cleaning

#Keep rows with required values
required_cols = ["year", "month", "airport", "carrier",
                 "arr_flights", "arr_del15", "arr_delay"]
delays = delays.dropna(subset=required_cols)

#Avoid divide-by-zero issues
delays = delays[delays["arr_flights"] > 0].copy()

#Ensure correct types
delays["year"] = delays["year"].astype(int)
delays["month"] = delays["month"].astype(int)

# --------------------------------------------------
#Target variables

#Delay probability target: proportion of flights delayed in each airport-carrier-month observation
delays["delay_rate"] = delays["arr_del15"] / delays["arr_flights"]

#Delay severity target: average delay length among delayed flights only
delays["delay_severity"] = (
    delays["arr_delay"] / delays["arr_del15"].replace(0, pd.NA)
)
delays["delay_severity"] = delays["delay_severity"].fillna(0)

#Cap extreme severity values to reduce influence of outliers
severity_cap = delays["delay_severity"].quantile(0.99)
delays["delay_severity"] = delays["delay_severity"].clip(upper=severity_cap)

#Dominant cause target: assign the delay cause with the highest delay minutes in each row
delays["dominant_cause"] = delays[[
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay"
]].idxmax(axis=1)

# --------------------------------------------------
#Feature matrix

#Basic predictors available in the dataset
features = ["airport", "carrier", "month", "year"]

#One-hot encode categorical variables
X = pd.get_dummies(delays[features], drop_first=True)

#Targets
y_prob = delays["delay_rate"]
y_sev = delays["delay_severity"]
y_cause = delays["dominant_cause"]

# --------------------------------------------------
#Train/test split

#Time-based split: train on all years before the most recent year, test on the most recent year
latest_year = delays["year"].max()

train_mask = delays["year"] < latest_year
test_mask = delays["year"] == latest_year

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_prob_train = y_prob.loc[train_mask]
y_prob_test = y_prob.loc[test_mask]

y_sev_train = y_sev.loc[train_mask]
y_sev_test = y_sev.loc[test_mask]

y_cause_train = y_cause.loc[train_mask]
y_cause_test = y_cause.loc[test_mask]

# --------------------------------------------------

#Model 1: predict probability of delay
model_prob = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=5,
    n_jobs=-1
)
model_prob.fit(X_train, y_prob_train)

#Model 2: predict delay severity
model_sev = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=5,
    n_jobs=-1
)
model_sev.fit(X_train, y_sev_train)

#Model 3: predict most likely cause of delay
model_cause = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=5,
    n_jobs=-1
)
model_cause.fit(X_train, y_cause_train)

# --------------------------------------------------

#Predictions
y_prob_pred = model_prob.predict(X_test)
y_sev_pred = model_sev.predict(X_test)
y_cause_pred = model_cause.predict(X_test)

#Keep regression outputs in sensible ranges
y_prob_pred = pd.Series(y_prob_pred).clip(lower=0, upper=1)
y_sev_pred = pd.Series(y_sev_pred).clip(lower=0)

#Performance metrics
prob_rmse = mean_squared_error(y_prob_test, y_prob_pred) ** 0.5
prob_r2 = r2_score(y_prob_test, y_prob_pred)

sev_rmse = mean_squared_error(y_sev_test, y_sev_pred) ** 0.5
sev_r2 = r2_score(y_sev_test, y_sev_pred)

cause_acc = accuracy_score(y_cause_test, y_cause_pred)

print("\nModel Performance")
print("-----------------")
print("Test year:", latest_year)
print("Delay probability RMSE:", round(prob_rmse, 4))
print("Delay probability R^2:", round(prob_r2, 4))
print("Delay severity RMSE:", round(sev_rmse, 4))
print("Delay severity R^2:", round(sev_r2, 4))
print("Cause classification accuracy:", round(cause_acc, 4))

# --------------------------------------------------
#Prediction function

def predict_delay(airport: str, carrier: str, month: int, year: int) -> dict:
    """
    Predict delay probability, delay severity, and dominant cause
    for a given airport-carrier-month-year combination.
    """
    input_df = pd.DataFrame({
        "airport": [airport],
        "carrier": [carrier],
        "month": [month],
        "year": [year]
    })

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    prob = model_prob.predict(input_encoded)[0]
    sev = model_sev.predict(input_encoded)[0]
    cause = model_cause.predict(input_encoded)[0]

    # Constrain outputs
    prob = max(0, min(1, float(prob)))
    sev = max(0, float(sev))

    return {
        "predicted_delay_probability": round(prob, 3),
        "predicted_delay_if_delayed_minutes": round(sev, 1),
        "predicted_most_likely_cause": cause
    }

# --------------------------------------------------
#Example prediction

example_prediction = predict_delay("PIT", "AA", 6, latest_year)

print("\nExample Prediction")
print("------------------")
print(f"Delay Probability: {example_prediction['predicted_delay_probability']*100:.1f}%")
print(f"Expected Delay (if delayed): {example_prediction['predicted_delay_if_delayed_minutes']:.1f} minutes")
print(f"Most Likely Cause: {example_prediction['predicted_most_likely_cause']}")

# --------------------------------------------------
#Save outputs

#Save model performance summary
with open("output/modeling/model_performance.txt", "w") as f:
    f.write("Model Performance\n")
    f.write("-----------------\n")
    f.write(f"Test year: {latest_year}\n")
    f.write(f"Delay probability RMSE: {prob_rmse:.4f}\n")
    f.write(f"Delay probability R^2: {prob_r2:.4f}\n")
    f.write(f"Delay severity RMSE: {sev_rmse:.4f}\n")
    f.write(f"Delay severity R^2: {sev_r2:.4f}\n")
    f.write(f"Cause classification accuracy: {cause_acc:.4f}\n")

#Save example predictions
examples = [
    {"airport": "JFK", "carrier": "F9", "month": 1, "year": latest_year},
    {"airport": "MIA", "carrier": "UA", "month": 7, "year": latest_year},
    {"airport": "PIT", "carrier": "AA", "month": 9, "year": latest_year},
    {"airport": "ATL", "carrier": "DL", "month": 12, "year": latest_year}
]

results = []

for ex in examples:
    prediction = predict_delay(
        airport=ex["airport"],
        carrier=ex["carrier"],
        month=ex["month"],
        year=ex["year"]
    )

    row = {
        "airport": ex["airport"],
        "carrier": ex["carrier"],
        "month": ex["month"],
        "predicted_delay_probability": prediction["predicted_delay_probability"],
        "predicted_delay_if_delayed_minutes": prediction["predicted_delay_if_delayed_minutes"],
        "predicted_most_likely_cause": prediction["predicted_most_likely_cause"]
    }
    results.append(row)

example_df = pd.DataFrame(results)
example_df.to_csv("output/modeling/example_predictions.csv", index=False)

print("\nExample Predictions")
print("-------------------")
print(example_df)
