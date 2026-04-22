#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:18:33 2026

@author: alaina
"""

# NYC Weather Delay Case Study (2005–2025)
# This script analyzes airline delay data for NYC airports
# to evaluate the role of weather in delay frequency and severity.
# Outputs: cleaned datasets + 6 visualization figures


#Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

#Setup folder
os.makedirs("output/nyc_case_study", exist_ok=True)

#Load airport datasets
jfk = pd.read_csv("Airline_Delay_Cause_jfk.csv")
lga = pd.read_csv("Airline_Delay_Cause_lga.csv")
ewr = pd.read_csv("Airline_Delay_Cause_ewr.csv")

#Combine into one NYC dataset
nyc = pd.concat([jfk, lga, ewr], ignore_index=True)

#Keep only rows with required values
required_cols = [
    "arr_flights", "arr_delay",
    "weather_delay", "nas_delay",
    "weather_ct", "nas_ct"
]
nyc = nyc.dropna(subset=required_cols)

#Remove rows with zero flights
nyc = nyc[nyc["arr_flights"] > 0].copy()

#Weather adjustment
#Weather_delay captures direct/extreme weather.
#Some weather-related disruption is embedded in NAS delays.
#Approximate total weather impact by assigning 45.8% of NAS delay/counts to weather.

nyc["weather_adjusted"] = nyc["weather_delay"] + 0.458 * nyc["nas_delay"]
nyc["weather_ct_adjusted"] = nyc["weather_ct"] + 0.458 * nyc["nas_ct"]


#Aggregate yearly totals by airport to compute probability and severity metrics

nyc_yearly = (
    nyc.groupby(["airport", "year"], as_index=False)
    .agg({
        "arr_flights": "sum",
        "arr_delay": "sum",
        "weather_adjusted": "sum",
        "weather_ct_adjusted": "sum"
    })
)

#Probability: share of flights affected by weather-related delay
nyc_yearly["weather_probability"] = (
    nyc_yearly["weather_ct_adjusted"] / nyc_yearly["arr_flights"]
)

#Severity: average minutes of delay when weather-related delay occurs
nyc_yearly["weather_severity"] = (
    nyc_yearly["weather_adjusted"] /
    nyc_yearly["weather_ct_adjusted"].replace(0, pd.NA)
)


#Aggregate monthly data to analyze distribution and extreme events
nyc_monthly = (
    nyc.groupby(["airport", "year", "month"], as_index=False)
    .agg({
        "arr_flights": "sum",
        "weather_adjusted": "sum",
        "weather_ct_adjusted": "sum"
    })
)

nyc_monthly["weather_probability"] = (
    nyc_monthly["weather_ct_adjusted"] / nyc_monthly["arr_flights"]
)

nyc_monthly["weather_severity"] = (
    nyc_monthly["weather_adjusted"] /
    nyc_monthly["weather_ct_adjusted"].replace(0, pd.NA)
)

#Define severe events: a month is considered "severe" if average weather delay exceeds 75 minutes
#This captures extreme disruption rather than typical delays
nyc_monthly["severe_event"] = nyc_monthly["weather_severity"] > 75


#Visualization 1: NYC-wide weather delay probability over time
nyc_prob = (
    nyc.groupby("year", as_index=False)
    .agg({
        "arr_flights": "sum",
        "weather_ct_adjusted": "sum"
    })
)

nyc_prob["weather_probability"] = (
    nyc_prob["weather_ct_adjusted"] / nyc_prob["arr_flights"]
)

plt.figure(figsize=(10, 6))
plt.plot(
    nyc_prob["year"],
    nyc_prob["weather_probability"],
    marker="o",
    linewidth=2,
    markersize=7
)

plt.title("NYC Weather Delay Probability Over Time", fontsize=18)
plt.xlabel("Year", fontsize=13)
plt.ylabel("Probability of Weather Delay", fontsize=13)

years_to_show = nyc_prob["year"][::2]
plt.xticks(years_to_show, rotation=45)
plt.ylim(0, 0.14)

plt.tight_layout()
plt.savefig("output/nyc_case_study/nyc_weather_probability.png")
plt.close()

#Visualization 2: NYC-wide total vs weather delay per flight

nyc_delay_compare = (
    nyc.groupby("year", as_index=False)
    .agg({
        "arr_flights": "sum",
        "arr_delay": "sum",
        "weather_adjusted": "sum"
    })
)

nyc_delay_compare["total_delay_per_flight"] = (
    nyc_delay_compare["arr_delay"] / nyc_delay_compare["arr_flights"]
)

nyc_delay_compare["weather_delay_per_flight"] = (
    nyc_delay_compare["weather_adjusted"] / nyc_delay_compare["arr_flights"]
)

plt.figure(figsize=(10, 6))

plt.plot(
    nyc_delay_compare["year"],
    nyc_delay_compare["total_delay_per_flight"],
    linestyle="--",
    linewidth=2,
    marker="o",
    markersize=6,
    label="Total Delay per Flight"
)

plt.plot(
    nyc_delay_compare["year"],
    nyc_delay_compare["weather_delay_per_flight"],
    linewidth=2,
    marker="o",
    markersize=6,
    label="Weather Delay per Flight"
)

plt.title("NYC Delay Burden: Total vs Weather", fontsize=18)
plt.xlabel("Year", fontsize=13)
plt.ylabel("Delay Minutes per Flight", fontsize=13)

years_to_show = nyc_delay_compare["year"][::2]
plt.xticks(years_to_show, rotation=45)

plt.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig("output/nyc_case_study/nyc_total_vs_weather.png")
plt.close()


#Visualization 3: Monthly severity distribution over time

severity_75 = nyc_monthly.groupby("year")["weather_severity"].quantile(0.75)

plt.figure(figsize=(10, 6))

plt.plot(
    severity_75.index,
    severity_75.values,
    marker="o",
    linewidth=2,
    markersize=7
)

plt.title("Weather Delay Severity Over Time (75th Percentile)", fontsize=18)
plt.xlabel("Year", fontsize=13)
plt.ylabel("Delay Minutes (75th Percentile)", fontsize=13)

years_to_show = severity_75.index[::2]
plt.xticks(years_to_show, rotation=45)
plt.ylim(0, 100)

plt.tight_layout()
plt.savefig("output/nyc_case_study/nyc_weather_severity_distribution.png")
plt.close()


#Visualization 4: Share of extreme weather delay months

severe_trend = nyc_monthly.groupby("year")["severe_event"].mean()

plt.figure(figsize=(10, 6))
plt.plot(
    severe_trend.index,
    severe_trend.values,
    marker="o",
    linewidth=2,
    markersize=7
)
plt.title("Frequency of Extreme Weather Delay Months", fontsize=18)
plt.xlabel("Year", fontsize=13)
plt.ylabel("Share of Airport-Months", fontsize=13)
years_to_show = severe_trend.index[::2]
plt.xticks(years_to_show, rotation=45)
plt.ylim(0, 0.6)

plt.tight_layout()
plt.savefig("output/nyc_case_study/nyc_severe_weather_frequency_monthly.png")
plt.close()


#Visualization 5: Weather share of total delay minutes

nyc_share = (
    nyc.groupby("year", as_index=False)
    .agg({
        "arr_delay": "sum",
        "weather_adjusted": "sum"
    })
)

nyc_share["weather_delay_share"] = (
    nyc_share["weather_adjusted"] / nyc_share["arr_delay"]
)

plt.figure(figsize=(10, 6))
plt.plot(
    nyc_share["year"],
    nyc_share["weather_delay_share"],
    marker="o",
    linewidth=2,
    markersize=7
)

plt.title("NYC Weather Share of Total Delay Minutes", fontsize=18)
plt.xlabel("Year", fontsize=13)
plt.ylabel("Weather Share of Total Delay Minutes", fontsize=13)

years_to_show = nyc_share["year"][::2]
plt.xticks(years_to_show, rotation=45)
plt.ylim(0, nyc_share["weather_delay_share"].max() * 1.1)

plt.tight_layout()
plt.savefig("output/nyc_case_study/nyc_weather_share_delay_minutes.png")
plt.close()


#Visualization 6: Early vs recent comparison of frequency and severity

#Build yearly comparison dataset
combined = (
    nyc.groupby("year", as_index=False)
    .agg({
        "arr_flights": "sum",
        "weather_ct_adjusted": "sum"
    })
)

combined["weather_probability"] = (
    combined["weather_ct_adjusted"] / combined["arr_flights"]
)

#Add yearly severity measure from monthly 75th percentile values
combined["severity_75"] = combined["year"].map(severity_75)

#Define comparison periods
early = combined[combined["year"] <= 2010]
late = combined[combined["year"] >= 2016]

#Compute average frequency and severity for each period
freq_early = early["weather_probability"].mean()
freq_late = late["weather_probability"].mean()

sev_early = early["severity_75"].mean()
sev_late = late["severity_75"].mean()

#Plot side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#Frequency panel
axes[0].bar(
    ["Early", "Recent"],
    [freq_early, freq_late]
)
axes[0].set_title("Frequency ↓", fontsize=16)
axes[0].set_ylabel("Weather Delay Probability", fontsize=12)

#Add labels to bars
for i, v in enumerate([freq_early, freq_late]):
    axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")

#Severity panel
axes[1].bar(
    ["Early", "Recent"],
    [sev_early, sev_late]
)
axes[1].set_title("Severity ↑", fontsize=16)
axes[1].set_ylabel("Delay Minutes", fontsize=12)

#Add labels to bars
for i, v in enumerate([sev_early, sev_late]):
    axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom")

plt.suptitle("Shift in Weather Delay Patterns", fontsize=18)
plt.tight_layout()
plt.savefig("output/nyc_case_study/nyc_shift_comparison.png")
plt.close()

#Print summary tables for quick interpretation

print("\nYearly airport-level summary:")
print(nyc_yearly.head())

print("\nMonthly severe-event trend:")
print(severe_trend)

print("\nWeather share of total delay minutes:")
print(nyc_share[["year", "weather_delay_share"]])