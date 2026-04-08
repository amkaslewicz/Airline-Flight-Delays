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

#Sum total delay minutes by cause across the full dataset
delay_causes = delays[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].sum()

#Convert cause names into more readable labels for plotting
delay_causes.index = [
    'Carrier Delay',
    'Weather Delay',
    'NAS Delay',
    'Security Delay',
    'Late Aircraft Delay'
]

#Calculate each cause's share of total delay minutes
delay_share = delay_causes / delay_causes.sum()


#Visualization 6.1: Total Delay Minutes by Cause
#Shows which causes contribute the most total delay minutes
plt.figure(figsize=(10, 6))
delay_causes.sort_values().plot(kind='barh')
plt.title("Total Delay Minutes by Cause")
plt.xlabel("Total Delay Minutes")
plt.ylabel("Delay Cause")
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/total_delay_by_cause.png")
plt.close()

#Visualization 6.2: Share of Total Delay by Cause
#Shows the proportion of overall delay attributable to each cause
plt.figure(figsize=(10, 6))
delay_share.sort_values().plot(kind='barh')
plt.title("Share of Total Delay by Cause")
plt.xlabel("Proportion of Total Delay")
plt.ylabel("Delay Cause")
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/delay_share_by_cause.png")
plt.close()

#Visualization 6.3: Seasonal Pattern of Delay Causes
#Groups total delay minutes by month across all years
seasonal_causes = delays.groupby('month')[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].sum()

#Rename columns for readability
seasonal_causes.columns = [
    'Carrier Delay',
    'Weather Delay',
    'NAS Delay',
    'Security Delay',
    'Late Aircraft Delay'
]

#Replace month numbers with month abbreviations
seasonal_causes.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#Visualization 6.3: Seasonal Pattern of Delay Causes
#Shows how different delay causes fluctuate across the year
plt.figure(figsize=(10, 6))
for cause in seasonal_causes.columns:
    plt.plot(seasonal_causes.index, seasonal_causes[cause], marker='o', label=cause)
plt.title("Seasonal Pattern of Delay Causes")
plt.xlabel("Month")
plt.ylabel("Total Delay Minutes")
plt.legend()
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/seasonal_delay_causes.png")
plt.close()


#Visualization 6.35: Seasonal Composition of Delay Causes (Normalized Within Month)

#Group by month and sum delay minutes
seasonal_causes = delays.groupby('month')[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].sum()

#Convert to proportions (each month sums to 1)
seasonal_share = seasonal_causes.div(seasonal_causes.sum(axis=1), axis=0)

#Rename index for readability
seasonal_share.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#Visualization 6.35: Seasonal Composition of Delay Causes
plt.figure(figsize=(10, 6))
for col, label in zip(
    seasonal_share.columns,
    ['Carrier','Weather','NAS','Security','Late Aircraft']
):
    plt.plot(seasonal_share.index, seasonal_share[col], marker='o', label=label)
plt.title("Seasonal Composition of Delay Causes (Proportion of Total Delays)")
plt.xlabel("Month")
plt.ylabel("Share of Total Delay")
plt.legend()
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/seasonal_delay_composition.png")
plt.close()

#Visualization 6.4: Causes by Airlines

#Aggregate delay causes by airline
airline_causes = delays.groupby('carrier').agg({
    'carrier_delay': 'sum',
    'weather_delay': 'sum',
    'nas_delay': 'sum',
    'security_delay': 'sum',
    'late_aircraft_delay': 'sum',
    'arr_flights': 'sum'
})

#Keep only top 10 airlines by volume
top_airlines = airline_causes['arr_flights'].nlargest(10).index
airline_causes = airline_causes.loc[top_airlines]

#Convert each airline's delay causes into proportions
airline_cause_share = airline_causes.div(
    airline_causes[['carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay']].sum(axis=1),
    axis=0
)

#Visualization 6.4: Causes by Airlines (Stacked Bar)
plt.figure(figsize=(10, 6))
airline_cause_share[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].plot(kind='bar', stacked=True)
plt.title("Delay Cause Composition by Airline")
plt.xlabel("Airline")
plt.ylabel("Proportion of Delay")
plt.legend(title="Cause", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/airline_delay_composition.png")
plt.close()

#Visualization 6.5: Causes by Airport

#Aggregate delay causes by airport
airport_causes = delays.groupby('airport').agg({
    'carrier_delay': 'sum',
    'weather_delay': 'sum',
    'nas_delay': 'sum',
    'security_delay': 'sum',
    'late_aircraft_delay': 'sum',
    'arr_flights': 'sum'
})

#Keep top 25 airports
top_airports = airport_causes['arr_flights'].nlargest(25).index
airport_causes = airport_causes.loc[top_airports]

#Convert each airport's delay causes into proportions
airport_cause_share = airport_causes.div(
    airport_causes[['carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay']].sum(axis=1),
    axis=0
)

#Visualization 6.5: Causes by Airport (Stacked Bar)
plt.figure(figsize=(10, 6))
airport_cause_share[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].plot(kind='bar', stacked=True)
plt.title("Delay Cause Composition by Airport")
plt.xlabel("Airport")
plt.ylabel("Proportion of Delay")
plt.legend(title="Cause", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("output/eda/06_delay_causes/airport_delay_composition.png")
plt.close()
