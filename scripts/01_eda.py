#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:16:30 2026

@author: alaina
"""

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#Load data
delays = pd.read_csv("Airline_Delay_Cause.csv")

#Create output folders
os.makedirs("output/eda", exist_ok=True)

base_output = "output/eda"

folders = [
    "01_overview",
    "02_distributions",
    "03_airlines",
    "04_airports",
    "05_seasonality",
    "06_delay_causes"
]

for folder in folders:
    os.makedirs(os.path.join(base_output, folder), exist_ok=True)

#Basic dataset inspection
print("Shape:", delays.shape)
print("\nColumns:\n", delays.columns)
print("\nInfo:")
print(delays.info())

#Preview data
print("\nHead:")
print(delays.head())
print("\nMissing values:")
print(delays.isnull().sum())

#Drop missing values in key numeric columns
cols_to_check = ['arr_flights', 'arr_del15', 'arr_delay']
delays = delays.dropna(subset=cols_to_check)

#Create Core Values

#Delay rate (row-level)
delays['delay_rate'] = delays['arr_del15'] / delays['arr_flights']

#Avg delay minutes per flight (row-level)
delays['avg_delay_minutes'] = delays['arr_delay'] / delays['arr_flights']


# =========================
#EDA: 01 Overview

#Calculate total flights and delay minutes
total_flights = delays['arr_flights'].sum()
total_delay_minutes = delays['arr_delay'].sum()

print("Total flights:", total_flights)
print("Total delay minutes:", total_delay_minutes)

#System-wide weighted metrics
system_delay_rate = delays['arr_del15'].sum() / delays['arr_flights'].sum()
system_avg_delay = delays['arr_delay'].sum() / delays['arr_flights'].sum()
avg_delay_given_delay = delays['arr_delay'].sum() / delays['arr_del15'].sum()

print("System-wide delay rate:", system_delay_rate)
print("System-wide avg delay minutes:", system_avg_delay)
print("Average delay (only delayed flights):", avg_delay_given_delay)

#Create a 'date' column by combining year and month
delays['date'] = pd.to_datetime(delays[['year', 'month']].assign(day=1))

#Group data by month and sum across all airlines and airports
monthly = delays.groupby('date').agg({
    'arr_flights': 'sum',
    'arr_del15': 'sum',
    'arr_delay': 'sum'
})

#Delay rate is proportion of flights delayed in each month
monthly['delay_rate'] = monthly['arr_del15'] / monthly['arr_flights']
#Average delay per flight (including non-delayed flights)
monthly['avg_delay_minutes'] = monthly['arr_delay'] / monthly['arr_flights']

#Visualization 1.1: Total Number of Flights Per Month Over Time (2021-2025)"
#Shows overall air traffic volume trends over time
plt.figure()
monthly['arr_flights'].plot()
plt.title("Total Number of Flights Per Month Over Time (2021-2025)")
plt.xlabel("Date")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.savefig("output/eda/01_overview/flights_per_month_over_time.png")
plt.close()

#Visualization 1.2: Delay Rate By Month Over Time
#Shows how flight reliability changes over time
plt.figure()
monthly['delay_rate'].plot()
plt.title("Flight Delay Rate By Month Over Time (Weighted)")
plt.xlabel("Date")
plt.ylabel("Delay Rate")
plt.tight_layout()
plt.savefig("output/eda/01_overview/delay_rate_over_time.png")
plt.close()

#Visualization 1.3: Average Delay Per Flight By Month Over Time
#Shows the average number of delay minutes contributed per flight each month
plt.figure()
monthly['avg_delay_minutes'].plot()
plt.title("Average Delay Per Flight By Month Over Time (2021–2025)")
plt.xlabel("Date")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("output/eda/01_overview/avg_delay_minutes.png")
plt.close()

# =========================
#EDA: 02 Distributions 

#Visualization 2.1: Distribution of Monthly Total Flights
#Shows how total air traffic volume varies across months (system-level aggregation)
plt.figure()
monthly['arr_flights'].plot(kind='hist', bins=20)
plt.title("Distribution of Monthly Total Flights")
plt.xlabel("Number of Flights per Month")
plt.ylabel("Frequency (Months)")
plt.tight_layout()
plt.savefig("output/eda/02_distributions/monthly_flights_distribution.png")
plt.close()

#Visualization 2.2: Distribution of Monthly Delay Rate
#Shows how system-wide delay performance varies across months
plt.figure()
monthly['delay_rate'].plot(kind='hist', bins=20)
plt.title("Distribution of Monthly Delay Rate")
plt.xlabel("Delay Rate")
plt.ylabel("Frequency (Months)")
plt.tight_layout()
plt.savefig("output/eda/02_distributions/monthly_delay_rate_distribution.png")
plt.close()

#Visualization 2.3: Distribution of Monthly Average Delay per Flight
#Shows how delay burden per flight varies across months (includes on-time flights)
plt.figure()
monthly['avg_delay_minutes'].plot(kind='hist', bins=20)
plt.title("Distribution of Monthly Average Delay per Flight")
plt.xlabel("Average Delay Minutes per Flight")
plt.ylabel("Frequency (Months)")
plt.tight_layout()
plt.savefig("output/eda/02_distributions/monthly_avg_delay_distribution.png")
plt.close()

#Visualization 2.4: Boxplot of Monthly Delay Rate
#Highlights variability and outlier months in system reliability
plt.figure()
monthly['delay_rate'].plot(kind='box')
plt.title("Boxplot of Monthly Delay Rate")
plt.ylabel("Delay Rate")
plt.tight_layout()
plt.savefig("output/eda/02_distributions/monthly_delay_rate_boxplot.png")
plt.close()

# =========================
#EDA: 03 Airlines

#Aggregate data by airline
airline = delays.groupby('carrier').agg({
    'arr_flights': 'sum',   #total flights operated by airline
    'arr_del15': 'sum',     #total delayed flights
    'arr_delay': 'sum'      #total delay minutes
})

#Create key performance indicators

#Delay rate
airline['delay_rate'] = airline['arr_del15'] / airline['arr_flights']

#Average delay per flight (includes on-time flights)
airline['avg_delay_per_flight'] = airline['arr_delay'] / airline['arr_flights']

# Average delay severity (only for delayed flights)
airline['avg_delay_if_delayed'] = airline['arr_delay'] / airline['arr_del15']

#Visualization 3.1: Delay Rate by Airline
#Shows which airlines have the highest proportion of delayed flights
plt.figure()
airline['delay_rate'].sort_values().plot(kind='bar')
plt.title("Delay Rate by Airline (System-Wide)")
plt.xlabel("Airline")
plt.ylabel("Delay Rate")
plt.tight_layout()
plt.savefig("output/eda/03_airlines/airline_delay_rate.png")
plt.close()

#Visualization 3.2: Average Delay Per Flight by Airline
#Shows overall delay per flight across airlines including on-time flights
plt.figure()
airline['avg_delay_per_flight'].sort_values().plot(kind='bar')
plt.title("Average Delay Minutes per Flight by Airline")
plt.xlabel("Airline")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("output/eda/03_airlines/airline_avg_delay_per_flight.png")
plt.close()

#Visualization 3.3: Average Delay (Delayed Flights Only)
#Shows severity of delays when they occur for each airline
plt.figure()
airline['avg_delay_if_delayed'].sort_values().plot(kind='bar')
plt.title("Average Delay (Only Delayed Flights) by Airline")
plt.xlabel("Airline")
plt.ylabel("Minutes (Delayed Flights Only)")
plt.tight_layout()
plt.savefig("output/eda/03_airlines/airline_delay_severity.png")
plt.close()

#Visualization 3.4: Total Flights by Airline
#Shows airline size and operational scale for context
plt.figure()
airline['arr_flights'].sort_values().plot(kind='bar')
plt.title("Total Flights by Airline")
plt.xlabel("Airline")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.savefig("output/eda/03_airlines/airline_flight_volume.png")
plt.close()

# =========================
#EDA: 04 Airports

#Aggregate data by airport
#This creates one row per airport with total flights, delays, and delay minutes
airport_full = delays.groupby('airport').agg({
    'arr_flights': 'sum',   #total flights handled by airport
    'arr_del15': 'sum',     #total delayed flights
    'arr_delay': 'sum'      #total delay minutes
})

#Create key performance metrics

#Delay rate
airport_full['delay_rate'] = airport_full['arr_del15'] / airport_full['arr_flights']

#Average delay per flight (includes on-time flights)
airport_full['avg_delay_per_flight'] = airport_full['arr_delay'] / airport_full['arr_flights']

#Average delay severity (only delayed flights)
airport_full['avg_delay_if_delayed'] = airport_full['arr_delay'] / airport_full['arr_del15']


#Create airport groups for comparison

# Filter out airports with too few flights to produce reliable statistics
airport_filtered = airport_full[airport_full['arr_flights'] >= 5000]

#Top 20 busiest airports (major hubs)
top_airports = airport_filtered['arr_flights'].nlargest(20).index
airport_top = airport_filtered.loc[top_airports]

#Bottom 20 smallest airports (low-traffic airports)
small_airports = airport_filtered['arr_flights'].nsmallest(20).index
airport_small = airport_filtered.loc[small_airports]

#Random sample of 20 airports
airport_random = airport_filtered.sample(20, random_state=42)

#Top 20 WORST airports (highest delay rate)
worst_airports = airport_filtered['delay_rate'].nlargest(20).index
airport_worst = airport_filtered.loc[worst_airports]

#Top 20 BEST airports (lowest delay rate)
best_airports = airport_filtered['delay_rate'].nsmallest(20).index
airport_best = airport_filtered.loc[best_airports]

#Function to generate all airport visualizations
def plot_airport_group(df, label):
    
    #Visualization 4.1: Delay Rate
    #Shows proportion of flights delayed at each airport
    plt.figure(figsize=(10, 8))
    df['delay_rate'].sort_values().plot(kind='barh')
    plt.title(f"Delay Rate by Airport ({label})")
    plt.xlabel("Delay Rate")
    plt.ylabel("Airport")
    plt.tight_layout()
    plt.savefig(f"output/eda/04_airports/{label}_delay_rate.png")
    plt.close()

    #Visualization 4.2: Average Delay per Flight
    #Shows total delay burden per flight (including on-time flights)
    plt.figure(figsize=(10, 8))
    df['avg_delay_per_flight'].sort_values().plot(kind='barh')
    plt.title(f"Average Delay Minutes per Flight ({label})")
    plt.xlabel("Minutes")
    plt.ylabel("Airport")
    plt.tight_layout()
    plt.savefig(f"output/eda/04_airports/{label}_avg_delay_per_flight.png")
    plt.close()

    #Visualization 4.3: Delay Severity (Delayed Flights Only)
    #Shows how severe delays are when they occur
    plt.figure(figsize=(10, 8))
    df['avg_delay_if_delayed'].sort_values().plot(kind='barh')
    plt.title(f"Average Delay (Only Delayed Flights) ({label})")
    plt.xlabel("Minutes")
    plt.ylabel("Airport")
    plt.tight_layout()
    plt.savefig(f"output/eda/04_airports/{label}_severity.png")
    plt.close()

    #Visualization 4.4: Total Flights
    #Shows airport size and traffic volume for context
    plt.figure(figsize=(10, 8))
    df['arr_flights'].sort_values().plot(kind='barh')
    plt.title(f"Total Flights by Airport ({label})")
    plt.xlabel("Number of Flights")
    plt.ylabel("Airport")
    plt.tight_layout()
    plt.savefig(f"output/eda/04_airports/{label}_volume.png")
    plt.close()

#Generate plots for each airport group

#Major hub airports
plot_airport_group(airport_top, "top_20")

#Small regional airports
plot_airport_group(airport_small, "small_20")

#Random sample of airports
plot_airport_group(airport_random, "random_20")

#Top 20 WORST airports (highest delay rate)
plot_airport_group(airport_worst, "worst_20")

#Top 20 BEST airports (lowest delay rate)
plot_airport_group(airport_best, "best_20")

#Visualization 4.5: Delay Rate vs Airport Flight Volume
#Shows relationship between airport size and delay rate
plt.figure()
plt.scatter(airport_full['arr_flights'], airport_full['delay_rate'])
x = airport_full['arr_flights']
y = airport_full['delay_rate']
#Fit linear regression (degree 1)
coef = np.polyfit(x, y, 1)
poly_fn = np.poly1d(coef)
x_sorted = np.sort(x) #Sort x values for a smooth regression line
plt.plot(x_sorted, poly_fn(x_sorted))
plt.xscale('log')
plt.title("Delay Rate vs Airport Size (with Trend Line)")
plt.xlabel("Total Flights (Log Scale)")
plt.ylabel("Delay Rate")
plt.tight_layout()
plt.savefig("output/eda/04_airports/delay_rate_vs_volume.png")
plt.close()

# =========================
#EDA: 05 Seasonality

#Extract month number from the monthly datetime index
monthly['month'] = monthly.index.month

#Group across all years by month and sum totals
seasonality = monthly.groupby('month').agg({
    'arr_flights': 'sum',   # total flights across all years for each month
    'arr_del15': 'sum',     # total delayed flights
    'arr_delay': 'sum'      # total delay minutes
})

#Create seasonal performance metrics

#Delay rate
seasonality['delay_rate'] = seasonality['arr_del15'] / seasonality['arr_flights']

# Average delay per flight (includes on-time flights)
seasonality['avg_delay_per_flight'] = seasonality['arr_delay'] / seasonality['arr_flights']

# Average delay severity (only delayed flights)
seasonality['avg_delay_if_delayed'] = seasonality['arr_delay'] / seasonality['arr_del15']

#Replace numeric month index with month abbreviations for readability
seasonality.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#Visualization 5.1: Delay Rate by Month
#Shows how the probability of delays changes throughout the year
plt.figure()
seasonality['delay_rate'].plot(marker='o')
plt.title("Seasonal Pattern of Flight Delay Rate")
plt.xlabel("Month")
plt.ylabel("Delay Rate")
plt.xticks(range(len(seasonality.index)), seasonality.index)
plt.tight_layout()
plt.savefig("output/eda/05_seasonality/seasonal_delay_rate.png")
plt.close()

#Visualization 5.2: Average Delay per Flight by Month
#Shows the total delay per flight across the year, including on-time flights
plt.figure()
seasonality['avg_delay_per_flight'].plot(marker='o')
plt.title("Seasonal Pattern of Average Delay per Flight")
plt.xlabel("Month")
plt.ylabel("Minutes")
plt.xticks(range(len(seasonality.index)), seasonality.index)
plt.tight_layout()
plt.savefig("output/eda/05_seasonality/seasonal_avg_delay_per_flight.png")
plt.close()

#Visualization 5.3: Average Delay When Delays Occur
#Shows how severe delays are in each month, considering only delayed flights
plt.figure()
seasonality['avg_delay_if_delayed'].plot(marker='o')
plt.title("Seasonal Pattern of Delay Severity (Delayed Flights Only)")
plt.xlabel("Month")
plt.ylabel("Minutes")
plt.xticks(range(len(seasonality.index)), seasonality.index)
plt.tight_layout()
plt.savefig("output/eda/05_seasonality/seasonal_delay_severity.png")
plt.close()

#Visualization 5.4: Flight Volume by Month
#Shows seasonal demand patterns in air travel
plt.figure()
seasonality['arr_flights'].plot(marker='o')
plt.title("Seasonal Pattern of Flight Volume")
plt.xlabel("Month")
plt.ylabel("Total Flights")
plt.xticks(range(len(seasonality.index)), seasonality.index)
plt.tight_layout()
plt.savefig("output/eda/05_seasonality/seasonal_flight_volume.png")
plt.close()


# =========================
#EDA: 06 Delay Causes

#IMPORTANT NOTE: This section uses TWO different ways of measuring delay causes
# 1. Count-based measures (*_ct columns):
# 2. Minutes-based measures (*_delay columns):

#The weather variable is extreme weather only, weather is also partially captured in NAS delays

#Sum total counts of delays by cause across the full dataset
cause_counts = delays[[
    'carrier_ct',
    'weather_ct',
    'nas_ct',
    'security_ct',
    'late_aircraft_ct'
]].sum()

#Rename for readability
cause_counts.index = [
    'Carrier Delay',
    'Weather Delay',
    'NAS Delay',
    'Security Delay',
    'Late Aircraft Delay'
]

#Convert to shares of total counted delays
cause_count_share = cause_counts / cause_counts.sum()


#Visualization 6.1A: Share of Delay Counts by Cause
#Shows the proportion of all delay counts attributable to each cause
plt.figure(figsize=(10, 6))
cause_count_share.sort_values().plot(kind='barh')
plt.title("Share of Delay Counts by Cause")
plt.xlabel("Proportion of Total Delay Counts")
plt.ylabel("Delay Cause")
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/delay_count_share_by_cause.png")
plt.close()

#Sum total delay minutes by cause across the full dataset
delay_minutes = delays[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].sum()

#Rename for readability
delay_minutes.index = [
    'Carrier Delay',
    'Weather Delay',
    'NAS Delay',
    'Security Delay',
    'Late Aircraft Delay'
]

#Convert to shares of total delay minutes
delay_minute_share = delay_minutes / delay_minutes.sum()

#Visualization 6.1B: Share of Delay Minutes by Cause
#Shows the proportion of overall delay minutes attributable to each cause
plt.figure(figsize=(10, 6))
delay_minute_share.sort_values().plot(kind='barh')
plt.title("Share of Total Delay Minutes by Cause")
plt.xlabel("Proportion of Total Delay Minutes")
plt.ylabel("Delay Cause")
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/delay_minute_share_by_cause.png")
plt.close()


#SEASONAL CAUSE

#Group by month and sum delay counts
seasonal_cause_counts = delays.groupby('month')[[
    'carrier_ct',
    'weather_ct',
    'nas_ct',
    'security_ct',
    'late_aircraft_ct'
]].sum()

#Rename columns
seasonal_cause_counts.columns = [
    'Carrier',
    'Weather',
    'NAS',
    'Security',
    'Late Aircraft'
]

#Rename month index
seasonal_cause_counts.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#Convert to within-month proportions
seasonal_cause_count_share = seasonal_cause_counts.div(
    seasonal_cause_counts.sum(axis=1),
    axis=0
)

#Visualization 6.2A: Seasonal Composition of Delay Counts
#Shows how the mix of causes changes throughout the year based on counts
plt.figure(figsize=(10, 6))
for col in seasonal_cause_count_share.columns:
    plt.plot(seasonal_cause_count_share.index,
             seasonal_cause_count_share[col],
             marker='o',
             label=col)

plt.title("Seasonal Composition of Delay Causes (Based on Counts)")
plt.xlabel("Month")
plt.ylabel("Share of Total Delay Counts")
plt.legend()
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/seasonal_delay_count_composition.png")
plt.close()


#Group by month and sum delay minutes
seasonal_cause_minutes = delays.groupby('month')[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].sum()

#Rename columns
seasonal_cause_minutes.columns = [
    'Carrier',
    'Weather',
    'NAS',
    'Security',
    'Late Aircraft'
]

#Rename month index
seasonal_cause_minutes.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#Convert to within-month proportions
seasonal_cause_minute_share = seasonal_cause_minutes.div(
    seasonal_cause_minutes.sum(axis=1),
    axis=0
)

#Visualization 6.2B: Seasonal Composition of Delay Causes
#Shows how the mix of causes changes throughout the year based on minutes
plt.figure(figsize=(10, 6))
for col in seasonal_cause_minute_share.columns:
    plt.plot(seasonal_cause_minute_share.index,
             seasonal_cause_minute_share[col],
             marker='o',
             label=col)

plt.title("Seasonal Composition of Delay Causes (Based on Minutes)")
plt.xlabel("Month")
plt.ylabel("Share of Total Delay Minutes")
plt.legend()
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/seasonal_delay_minute_composition.png")
plt.close()


#AIRLINE COMPOSITION

#Aggregate delay counts by airline
airline_cause_counts = delays.groupby('carrier').agg({
    'carrier_ct': 'sum',
    'weather_ct': 'sum',
    'nas_ct': 'sum',
    'security_ct': 'sum',
    'late_aircraft_ct': 'sum',
    'arr_flights': 'sum'
})

#Keep top 10 airlines by flight volume
top_airlines = airline_cause_counts['arr_flights'].nlargest(10).index
airline_cause_counts = airline_cause_counts.loc[top_airlines]

#Convert to proportions
airline_cause_count_share = airline_cause_counts[[
    'carrier_ct',
    'weather_ct',
    'nas_ct',
    'security_ct',
    'late_aircraft_ct'
]].div(
    airline_cause_counts[[
        'carrier_ct',
        'weather_ct',
        'nas_ct',
        'security_ct',
        'late_aircraft_ct'
    ]].sum(axis=1),
    axis=0
)

#Visualization 6.3A: Delay Cause Composition by Airline (Counts)
plt.figure(figsize=(10, 6))
airline_cause_count_share.plot(kind='bar', stacked=True)
plt.title("Delay Cause Composition by Airline (Based on Counts)")
plt.xlabel("Airline")
plt.ylabel("Share of Total Delay Counts")
plt.legend(title="Cause", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/airline_delay_count_composition.png")
plt.close()


#Aggregate delay minutes by airline
airline_cause_minutes = delays.groupby('carrier').agg({
    'carrier_delay': 'sum',
    'weather_delay': 'sum',
    'nas_delay': 'sum',
    'security_delay': 'sum',
    'late_aircraft_delay': 'sum',
    'arr_flights': 'sum'
})

#Keep top 10 airlines by flight volume
top_airlines = airline_cause_minutes['arr_flights'].nlargest(10).index
airline_cause_minutes = airline_cause_minutes.loc[top_airlines]

#Convert to proportions
airline_cause_minute_share = airline_cause_minutes[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].div(
    airline_cause_minutes[[
        'carrier_delay',
        'weather_delay',
        'nas_delay',
        'security_delay',
        'late_aircraft_delay'
    ]].sum(axis=1),
    axis=0
)

#Visualization 6.3B: Delay Cause Composition by Airline (Minutes)
plt.figure(figsize=(10, 6))
airline_cause_minute_share.plot(kind='bar', stacked=True)
plt.title("Delay Cause Composition by Airline (Minutes)")
plt.xlabel("Airline")
plt.ylabel("Share of Total Delay Minutes")
plt.legend(title="Cause", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/airline_delay_minute_composition.png")
plt.close()

# ============================================
#WEATHER SCORE CONSTRUCTION

#IMPORTANT CONTEXT:
#The dataset separates delay causes into categories:
#- weather_delay = extreme weather (direct)
#- nas_delay = system delays (includes weather-related congestion)
#According to DOT: ~45.8% of NAS delays are actually caused by weather
#Therefore, to estimate TRUE weather impact, we adjust by incorporating part of NAS delays.


#Adjusted weather delay (minutes)
delays['weather_adjusted'] = (
    delays['weather_delay'] +
    0.458 * delays['nas_delay']
)

#Adjusted weather count (number of delayed flights)
delays['weather_ct_adjusted'] = (
    delays['weather_ct'] +
    0.458 * delays['nas_ct']
)


#Create Core Weather Metric

#Weather probability - how often weather causes delays
delays['weather_prob'] = (
    delays['weather_ct_adjusted'] /
    delays['arr_flights'].replace(0, pd.NA)
)

#Weather severity - how much delay weather causes per flight
delays['weather_per_flight'] = (
    delays['weather_adjusted'] /
    delays['arr_flights'].replace(0, pd.NA)
)

#Weather share - percent of total delay caused by weather
delays['weather_share'] = (
    delays['weather_adjusted'] /
    delays['arr_delay'].replace(0, pd.NA)
)


#Clean division issues
delays['weather_prob'] = delays['weather_prob'].fillna(0)
delays['weather_per_flight'] = delays['weather_per_flight'].fillna(0)
delays['weather_share'] = delays['weather_share'].fillna(0)


#Create combined weather score

#Standardize variables so they are comparable
weather_prob_std = (
    (delays['weather_prob'] - delays['weather_prob'].mean()) /
    delays['weather_prob'].std()
)

weather_severity_std = (
    (delays['weather_per_flight'] - delays['weather_per_flight'].mean()) /
    delays['weather_per_flight'].std()
)

#Combine into one metric (equal weighting)
delays['weather_score'] = (
    0.5 * weather_prob_std +
    0.5 * weather_severity_std
)


#Raw vs Adjuster Weather Share
raw_weather = delays['weather_delay'].sum()
adjusted_weather = delays['weather_adjusted'].sum()
total_delay = delays['arr_delay'].sum()

print("\nWeather Share Comparison:")
print("Raw weather share:", round(raw_weather / total_delay, 3))
print("Adjusted weather share:", round(adjusted_weather / total_delay, 3))



#Visualization 6.4: Weather Score By Month
#Which months are most impacted by weather
weather_monthly = delays.groupby('month')['weather_score'].mean()

weather_monthly.index = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure()
weather_monthly.plot(marker='o')
plt.title("Weather Disruption Score by Month (Frequency + Severity)")
plt.xlabel("Month")
plt.ylabel("Weather Score (Standardized)")
plt.xticks(range(len(weather_monthly.index)), weather_monthly.index)
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/weather_score_monthly.png")
plt.close()


#Visualization 6.5: Weather Hotsports (Airports)

weather_airport = delays.groupby('airport').agg({
    'weather_adjusted': 'sum',
    'weather_ct_adjusted': 'sum',
    'arr_flights': 'sum'
})

#Normalize
weather_airport['weather_per_flight'] = (
    weather_airport['weather_adjusted'] /
    weather_airport['arr_flights']
)

weather_airport['weather_prob'] = (
    weather_airport['weather_ct_adjusted'] /
    weather_airport['arr_flights']
)

#Filter small airports to avoid noise
weather_airport = weather_airport[weather_airport['arr_flights'] >= 5000]

#Top 20 weather-heavy airports
top_weather = weather_airport['weather_per_flight'].nlargest(20)
plt.figure(figsize=(10,8))
top_weather.sort_values().plot(kind='barh')
plt.title("Airports with Highest Weather Delay per Flight")
plt.xlabel("Minutes per Flight")
plt.ylabel("Airport")
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/weather_airports.png")
plt.close()


#Visualization 6.6: Stacked Bar With Adjusted Weather

#Remaining NAS after removing weather portion
delays['nas_remaining'] = (
    delays['nas_delay'] * (1 - 0.458)
)

#Aggregate by month
adjusted_causes = delays.groupby('month').agg({
    'carrier_delay': 'sum',
    'weather_adjusted': 'sum',
    'nas_remaining': 'sum',
    'security_delay': 'sum',
    'late_aircraft_delay': 'sum'
})

#Rename columns for readability
adjusted_causes.columns = [
    'Carrier',
    'Weather (Adjusted)',
    'NAS (Non-Weather)',
    'Security',
    'Late Aircraft'
]


#Replace month index
adjusted_causes.index = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']

#Convert to proportions (STACKED SHARE)
adjusted_share = adjusted_causes.div(
    adjusted_causes.sum(axis=1),
    axis=0
)

#Plot stacked bar
plt.figure(figsize=(12, 6))
adjusted_share.plot(
    kind='bar',
    stacked=True
)
plt.title("Delay Cause Composition (Weather Adjusted)")
plt.xlabel("Month")
plt.ylabel("Share of Total Delay Minutes")
plt.legend(title="Cause", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/adjusted_weather_stacked.png")
plt.close()