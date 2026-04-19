#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:18:33 2026

@author: alaina
"""
#Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


#Yearly airport-level summary

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


#Monthly airport-level summary
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

#Severe event threshold: average weather delay > 75 minutes
nyc_monthly["severe_event"] = nyc_monthly["weather_severity"] > 75


#Visualization 1: Weather delay probability over time

plt.figure(figsize=(10, 6))

for airport in ["JFK", "LGA", "EWR"]:
    subset = nyc_yearly[nyc_yearly["airport"] == airport]
    plt.plot(
        subset["year"],
        subset["weather_probability"],
        marker="o",
        label=airport
    )

plt.title("Weather Delay Probability Over Time (NYC Airports)")
plt.xlabel("Year")
plt.ylabel("Probability of Weather Delay")
plt.xticks(sorted(nyc_yearly["year"].unique()), rotation=45)
plt.ylim(0, 0.2)
plt.legend()
plt.tight_layout()
plt.savefig("output/nyc_case_study/nyc_weather_probability.png")
plt.close()


#Visualization 2: Total vs weather delay per flight

plt.figure(figsize=(10, 6))

for airport in ["JFK", "LGA", "EWR"]:
    subset = (
        nyc[nyc["airport"] == airport]
        .groupby("year", as_index=False)
        .agg({
            "arr_flights": "sum",
            "arr_delay": "sum",
            "weather_adjusted": "sum"
        })
    )

    subset["total_delay_per_flight"] = subset["arr_delay"] / subset["arr_flights"]
    subset["weather_delay_per_flight"] = subset["weather_adjusted"] / subset["arr_flights"]

    plt.plot(
        subset["year"],
        subset["total_delay_per_flight"],
        linestyle="--",
        label=f"{airport} Total Delay"
    )
    plt.plot(
        subset["year"],
        subset["weather_delay_per_flight"],
        label=f"{airport} Weather Delay"
    )

plt.title("Total vs Weather Delay Per Flight (NYC Airports)")
plt.xlabel("Year")
plt.ylabel("Delay Minutes per Flight")
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("output/nyc_case_study/nyc_total_vs_weather.png")
plt.close()


#Visualization 3: Monthly severity distribution over time

severity_75 = nyc_monthly.groupby("year")["weather_severity"].quantile(0.75)
severity_median = nyc_monthly.groupby("year")["weather_severity"].median()

plt.figure(figsize=(10, 6))
plt.plot(severity_75.index, severity_75.values, marker="o", label="75th Percentile")
plt.plot(severity_median.index, severity_median.values, marker="o", label="Median")

plt.title("Distribution of Monthly Weather Delay Severity Over Time (NYC Airports)")
plt.xlabel("Year")
plt.ylabel("Weather Delay Minutes (If Delayed)")
plt.xticks(severity_75.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("output/nyc_case_study/nyc_weather_severity_distribution.png")
plt.close()


#Visualization 4: Share of severe weather delay months

severe_trend = nyc_monthly.groupby("year")["severe_event"].mean()

plt.figure(figsize=(10, 6))
plt.plot(severe_trend.index, severe_trend.values, marker="o")

plt.title("Share of Severe Weather Delay Months Over Time (NYC Airports)")
plt.xlabel("Year")
plt.ylabel("Share of Airport-Months with Severe Weather Delays")
plt.xticks(severe_trend.index, rotation=45)
plt.ylim(0, 1)
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
plt.plot(nyc_share["year"], nyc_share["weather_delay_share"], marker="o")

plt.title("Share of Total Delay Minutes Attributable to Weather (NYC Airports)")
plt.xlabel("Year")
plt.ylabel("Weather Share of Total Delay Minutes")
plt.xticks(nyc_share["year"], rotation=45)
plt.ylim(0, nyc_share["weather_delay_share"].max() * 1.1)
plt.tight_layout()
plt.savefig("output/nyc_case_study/nyc_weather_share_delay_minutes.png")
plt.close()


#Print summary tables for quick interpretation

print("\nYearly airport-level summary:")
print(nyc_yearly.head())

print("\nMonthly severe-event trend:")
print(severe_trend)

print("\nWeather share of total delay minutes:")
print(nyc_share[["year", "weather_delay_share"]])