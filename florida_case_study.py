#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:17:30 2026

@author: alaina
"""

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

#Set up output folder
os.makedirs("output/florida_case_study", exist_ok=True)

#Load airport datasets
fll = pd.read_csv("Airline_Delay_Cause_fll.csv")
mco = pd.read_csv("Airline_Delay_Cause_mco.csv")
mia = pd.read_csv("Airline_Delay_Cause_mia.csv")

#Combine into one NYC dataset
florida = pd.concat([fll, mco, mia], ignore_index=True)

#Drop rows with missing values in key columns
cols_to_check = ['arr_flights', 'arr_delay', 'weather_delay', 'nas_delay', 'weather_ct', 'nas_ct']
florida = florida.dropna(subset=cols_to_check)

#Remove rows where there are zero flights
florida = florida[florida['arr_flights'] > 0].copy()


#IMPORTANT: BTS weather_delay only captures direct/extreme weather. Some weather-related disruption is embedded in NAS delays.
#Based on DOT/BTS documentation, we approximate true weather impact by assigning 45.8% of NAS delay and NAS counts to weather.

#Adjusted weather delay minutes
florida['weather_adjusted'] = (
    florida['weather_delay'] +
    0.458 * florida['nas_delay']
)

#Adjusted weather delay counts
florida['weather_ct_adjusted'] = (
    florida['weather_ct'] +
    0.458 * florida['nas_ct']
)


#Keep each airport separate and summarize yearly totals
florida_yearly = florida.groupby(['airport', 'year']).agg({
    'arr_flights': 'sum',
    'weather_adjusted': 'sum',
    'weather_ct_adjusted': 'sum'
}).reset_index()


#Probability of weather delay - the share of flights affected by weather-related delay
florida_yearly['weather_probability'] = (
    florida_yearly['weather_ct_adjusted'] /
    florida_yearly['arr_flights']
)

#Severity of weather delay - average minutes of delay when weather-related delay occurs
florida_yearly['weather_severity'] = (
    florida_yearly['weather_adjusted'] /
    florida_yearly['weather_ct_adjusted'].replace(0, pd.NA)
)


#Fill division-related missing values with 0
florida_yearly[['weather_probability', 'weather_severity']] = (
    florida_yearly[['weather_probability', 'weather_severity']].fillna(0)
)


#Standardize probability and severity so they are comparable
prob_std = (
    (florida_yearly['weather_probability'] - florida_yearly['weather_probability'].mean()) /
    florida_yearly['weather_probability'].std()
)

sev_std = (
    (florida_yearly['weather_severity'] - florida_yearly['weather_severity'].mean()) /
    florida_yearly['weather_severity'].std()
)

#Combined weather score = equal weight on frequency + severity
florida_yearly['weather_score'] = 0.5 * prob_std + 0.5 * sev_std


#Visualization 1: Weather Delay Probability Over Time

plt.figure(figsize=(10, 6))
for airport in ['FLL', 'MCO', 'MIA']:
    subset = florida_yearly[florida_yearly['airport'] == airport]
    plt.plot(subset['year'], subset['weather_probability'], marker='o', label=airport)

plt.title("Weather Delay Probability Over Time (Florida Airports)")
plt.xlabel("Year")
plt.ylabel("Probability of Weather Delay")
plt.xticks(subset['year'])
plt.ylim(0, 0.2)
plt.legend()
plt.tight_layout()
plt.savefig("output/florida_case_study/florida_weather_probability.png")
plt.close()


#Visualization 2: Weather Delay Severity Over Time

plt.figure(figsize=(10, 6))
for airport in ['FLL', 'MCO', 'MIA']:
    subset = florida_yearly[florida_yearly['airport'] == airport]
    plt.plot(subset['year'], subset['weather_severity'], marker='o', label=airport)

plt.title("Weather Delay Severity Over Time (Florida Airports)")
plt.xlabel("Year")
plt.ylabel("Weather Delay Minutes (If Delayed)")
plt.xticks(subset['year'])
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.savefig("output/florida_case_study/florida_weather_severity.png")
plt.close()


#Visualization 3: Weather Score Over Time

plt.figure(figsize=(10, 6))
for airport in ['FLL', 'MCO', 'MIA']:
    subset = florida_yearly[florida_yearly['airport'] == airport]
    plt.plot(subset['year'], subset['weather_score'], marker='o', label=airport)

plt.title("Weather Impact Score Over Time (Florida Airports)")
plt.xlabel("Year")
plt.ylabel("Weather Score (Standardized)")
plt.xticks(subset['year'])
plt.ylim(-3, 3)
plt.legend()
plt.tight_layout()
plt.savefig("output/florida_case_study/florida_weather_score.png")
plt.close()


#Compare Total Delay vs Weather Delay
plt.figure(figsize=(10, 6))

for airport in ['FLL', 'MCO', 'MIA']:
    
    # Aggregate yearly totals for EACH airport
    subset = florida[florida['airport'] == airport].groupby('year').agg({
        'arr_flights': 'sum',
        'arr_delay': 'sum',
        'weather_adjusted': 'sum'
    }).reset_index()
    
    # Create metrics
    subset['total_delay_per_flight'] = (
        subset['arr_delay'] / subset['arr_flights']
    )
    
    subset['weather_delay_per_flight'] = (
        subset['weather_adjusted'] / subset['arr_flights']
    )
    
    # Plot BOTH lines
    plt.plot(
        subset['year'],
        subset['total_delay_per_flight'],
        linestyle='--',
        label=f"{airport} Total Delay"
    )
    
    plt.plot(
        subset['year'],
        subset['weather_delay_per_flight'],
        label=f"{airport} Weather Delay"
    )

# Labels + formatting
plt.title("Total vs Weather Delay Per Flight (Florida Airports)")
plt.xlabel("Year")
plt.ylabel("Delay Minutes per Flight")

# Clean x-axis (no decimals)
import matplotlib.ticker as ticker
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig("output/florida_case_study/florida_total_vs_weather.png")
plt.close()

# ============================================
# Weather Share of Total Delay
# ============================================

plt.figure(figsize=(10, 6))

for airport in ['FLL', 'MCO', 'MIA']:
    
    subset = florida[florida['airport'] == airport].groupby('year').agg({
        'arr_delay': 'sum',
        'weather_adjusted': 'sum'
    }).reset_index()
    
    subset['weather_share'] = (
        subset['weather_adjusted'] / subset['arr_delay']
    )
    
    # Convert to percent
    plt.plot(
        subset['year'],
        subset['weather_share'] * 100,
        marker='o',
        label=airport
    )

plt.title("Weather Share of Total Delay (Florida Airports)")
plt.xlabel("Year")
plt.ylabel("Weather Share (%)")

import matplotlib.ticker as ticker
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig("output/florida_case_study/florida_weather_share.png")
plt.close()

#Export yearly dataset for later use
florida_yearly.to_csv("output/florida_case_study/florida_yearly_weather_metrics.csv", index=False)
