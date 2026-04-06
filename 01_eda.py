#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:16:30 2026

@author: alaina
"""

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
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
airport = delays.groupby('airport').agg({
    'arr_flights': 'sum',   #total flights handled by airport
    'arr_del15': 'sum',     #total delayed flights
    'arr_delay': 'sum'      #total delay minutes
})

#Filter to top 20 busiest airports by flight volume
top_airports = airport['arr_flights'].nlargest(20).index
airport = airport.loc[top_airports]


#Create key performance metrics

#Delay rate
airport['delay_rate'] = airport['arr_del15'] / airport['arr_flights']

#Average delay per flight
airport['avg_delay_per_flight'] = airport['arr_delay'] / airport['arr_flights']

#Average delay severity (only delayed flights)
airport['avg_delay_if_delayed'] = airport['arr_delay'] / airport['arr_del15']

#Visualization 4.1: Delay Rate by Airport
#Shows which major airports have the highest proportion of delayed flights
plt.figure(figsize=(10, 8))
airport['delay_rate'].sort_values().plot(kind='barh')
plt.title("Delay Rate by Airport (Top 20 by Volume)")
plt.xlabel("Delay Rate")
plt.ylabel("Airport")
plt.tight_layout()
plt.savefig("output/eda/04_airports/airport_delay_rate.png")
plt.close()

#Visualization 4.2: Average Delay per Flight by Airport
#Shows overall delay per flight at each airport including on-time flights
plt.figure(figsize=(10, 8))
airport['avg_delay_per_flight'].sort_values().plot(kind='barh')
plt.title("Average Delay Minutes per Flight by Airport")
plt.xlabel("Minutes")
plt.ylabel("Airport")
plt.tight_layout()
plt.savefig("output/eda/04_airports/airport_avg_delay_per_flight.png")
plt.close()

#Visualization 4.3: Average Delay (Only Delayed Flights)
#Shows severity of delays when they occur at each airport
plt.figure(figsize=(10, 8))
airport['avg_delay_if_delayed'].sort_values().plot(kind='barh')
plt.title("Average Delay (Only Delayed Flights) by Airport")
plt.xlabel("Minutes")
plt.ylabel("Airport")
plt.tight_layout()
plt.savefig("output/eda/04_airports/airport_delay_severity.png")
plt.close()

#Visualization 4.4: Total Flights by Airport
#Shows airport size and traffic volume for context
plt.figure(figsize=(10, 8))
airport['arr_flights'].sort_values().plot(kind='barh')
plt.title("Total Flights by Airport (Top 20)")
plt.xlabel("Number of Flights")
plt.ylabel("Airport")
plt.tight_layout()
plt.savefig("output/eda/04_airports/airport_volume.png")
plt.close()

