#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:21:30 2026

@author: alaina
"""

#Import libraries
import pandas as pd
import numpy as np
import os

#Modeling libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

#Evaluation
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#Create folder for modeling outputs
os.makedirs("output/modeling", exist_ok=True)

#Load airline delay dataset
delays = pd.read_csv("Airline_Delay_Cause.csv")

#Clean data
#Drop rows with missing values in key columns needed for modeling
cols_to_check = ['arr_flights', 'arr_del15', 'arr_delay']
delays = delays.dropna(subset=cols_to_check)

#Avoid divide-by-zero issues
delays = delays[delays['arr_flights'] > 0].copy()

#Create delay severity
delays['delay_severity'] = delays['arr_delay'] / delays['arr_del15'].replace(0, pd.NA)
#Fill NaNs (cases where arr_del15 = 0)
delays['delay_severity'] = delays['delay_severity'].fillna(0)
#Cap extreme values
cap = delays['delay_severity'].quantile(0.99)
delays['delay_severity'] = delays['delay_severity'].clip(upper=cap)

#Create target variables

#Delay probability target:
#Proportion of flights delayed in each airport-carrier-month observation
delays['delay_rate'] = delays['arr_del15'] / delays['arr_flights']

#Delay severity target:
#Average delay length among delayed flights only
delays['delay_severity'] = delays['arr_delay'] / delays['arr_del15'].replace(0, pd.NA)
delays['delay_severity'] = delays['delay_severity'].fillna(0)

#Most likely cause of delay target:
#Assign the delay cause with the highest delay minutes in each row
delays['dominant_cause'] = delays[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].idxmax(axis=1)



# Build feature matrix and target variables

#Select input features
features = ['airport', 'carrier', 'month']

#Convert categorical features into model-usable binary columns
X = pd.get_dummies(delays[features], drop_first=True)

#Create targets for the three models

#Probability of delay
y_prob = delays['delay_rate']

#Delay severity (if delayed)
y_sev = delays['delay_severity']

#Most likely cause of delay
y_cause = delays['dominant_cause']


#Train/test split

#Split once so all three targets use the exact same train/test rows
X_train, X_test, y_prob_train, y_prob_test, y_sev_train, y_sev_test, y_cause_train, y_cause_test = train_test_split(
    X, y_prob, y_sev, y_cause, test_size=0.2, random_state=42
)


#Train models

#Model 1: Predict probability of delay
model_prob = LinearRegression()
model_prob.fit(X_train, y_prob_train)

#Model 2: Predict delay severity
model_sev = LinearRegression()
model_sev.fit(X_train, y_sev_train)

#Model 3: Predict most likely cause of delay
model_cause = RandomForestClassifier(n_estimators=100, random_state=42)
model_cause.fit(X_train, y_cause_train)


# Evaluate models

#Predict on test set
y_prob_pred = model_prob.predict(X_test)
y_sev_pred = model_sev.predict(X_test)
y_cause_pred = model_cause.predict(X_test)

#Probability model performance
prob_rmse = mean_squared_error(y_prob_test, y_prob_pred) ** 0.5
prob_r2 = r2_score(y_prob_test, y_prob_pred)

#Severity model performance
sev_rmse = mean_squared_error(y_sev_test, y_sev_pred) ** 0.5
sev_r2 = r2_score(y_sev_test, y_sev_pred)

#Cause model performance
cause_acc = accuracy_score(y_cause_test, y_cause_pred)

print("\nModel Performance")
print("-----------------")
print("Delay probability RMSE:", round(prob_rmse, 4))
print("Delay probability R^2:", round(prob_r2, 4))
print("Delay severity RMSE:", round(sev_rmse, 4))
print("Delay severity R^2:", round(sev_r2, 4))
print("Cause classification accuracy:", round(cause_acc, 4))


#Prediction function

#This function takes a flight context and returns:
#- predicted probability of delay, predicted delay length if delayed, and most likely cause of delay
def predict_delay(airport, carrier, month):
    #Create one-row input dataframe
    input_df = pd.DataFrame({
        'airport': [airport],
        'carrier': [carrier],
        'month': [month]
    })

    #Encode input
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    #Generate predictions
    prob = model_prob.predict(input_encoded)[0]
    sev = model_sev.predict(input_encoded)[0]
    cause = model_cause.predict(input_encoded)[0]

    #Keep outputs in sensible ranges
    prob = max(0, min(1, prob))
    sev = max(0, sev)

    return {
        "predicted_delay_probability": round(float(prob), 3),
        "predicted_delay_if_delayed_minutes": round(float(sev), 1),
        "predicted_most_likely_cause": cause
    }


#Prediction

prediction = predict_delay('PIT', 'AA', 6)

prob = prediction['predicted_delay_probability']
sev = prediction['predicted_delay_if_delayed_minutes']
cause = prediction['predicted_most_likely_cause']

print(f"Delay Probability: {prob*100:.1f}%")
print(f"Expected Delay (if delayed): {sev:.1f} minutes")
print(f"Most Likely Cause: {cause}")




#Output

#Save model performance summary
with open("output/modeling/model_performance.txt", "w") as f:
    f.write("Model Performance\n")
    f.write("-----------------\n")
    f.write(f"Delay probability RMSE: {prob_rmse:.4f}\n")
    f.write(f"Delay probability R^2: {prob_r2:.4f}\n")
    f.write(f"Delay severity RMSE: {sev_rmse:.4f}\n")
    f.write(f"Delay severity R^2: {sev_r2:.4f}\n")
    f.write(f"Cause classification accuracy: {cause_acc:.4f}\n")
    
#Save example predictions
examples = [
    {'airport': 'JFK', 'carrier': 'F9', 'month': 1},
    {'airport': 'MIA', 'carrier': 'AA', 'month': 7},
    {'airport': 'ATL', 'carrier': 'DL', 'month': 12}
]

results = []

for ex in examples:
    prediction = predict_delay(ex['airport'], ex['carrier'], ex['month'])
    row = {
        'airport': ex['airport'],
        'carrier': ex['carrier'],
        'month': ex['month'],
        'predicted_delay_probability': prediction['predicted_delay_probability'],
        'predicted_delay_if_delayed_minutes': prediction['predicted_delay_if_delayed_minutes'],
        'predicted_most_likely_cause': prediction['predicted_most_likely_cause']
    }
    
    results.append(row)

#Convert to DataFrame
example_df = pd.DataFrame(results)

#Save to CSV
example_df.to_csv("output/modeling/example_predictions.csv", index=False)

#Print
print(example_df)
