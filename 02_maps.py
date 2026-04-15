#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:08:17 2026

@author: alaina
"""

#Import libraries
import pandas as pd
import os

#Set up output folders

os.makedirs("output/mapping", exist_ok=True)

#Load delay dataset
delays = pd.read_csv("Airline_Delay_Cause.csv")

#Drop missing values in key columns
cols_to_check = ['arr_flights', 'arr_del15', 'arr_delay']
delays = delays.dropna(subset=cols_to_check)

#Avoid divide-by-zero problems
delays = delays[delays['arr_flights'] > 0].copy()

#Load airport coordinate dataset
airports = pd.read_csv("us-airports.csv")

#Keep only U.S. airports with valid IATA codes and coordinates
airports = airports[
    (airports['iso_country'] == 'US') &
    (airports['iata_code'].notna()) &
    (airports['latitude_deg'].notna()) &
    (airports['longitude_deg'].notna())
]

#Keep only the columns needed for mapping
airports = airports[['iata_code', 'name', 'type', 'municipality', 'latitude_deg', 'longitude_deg']]


#Adjusted weather delay minutes
delays['weather_adjusted'] = (
    delays['weather_delay'] +
    0.458 * delays['nas_delay']
)

#Adjusted weather delay counts
delays['weather_ct_adjusted'] = (
    delays['weather_ct'] +
    0.458 * delays['nas_ct']
)

#Aggregate airport-level totals

airport_full = delays.groupby('airport').agg({
    'arr_flights': 'sum',
    'arr_del15': 'sum',
    'arr_delay': 'sum',

    #raw cause minutes
    'carrier_delay': 'sum',
    'weather_delay': 'sum',
    'nas_delay': 'sum',
    'security_delay': 'sum',
    'late_aircraft_delay': 'sum',

    #raw cause counts
    'carrier_ct': 'sum',
    'weather_ct': 'sum',
    'nas_ct': 'sum',
    'security_ct': 'sum',
    'late_aircraft_ct': 'sum',

    #adjusted weather
    'weather_adjusted': 'sum',
    'weather_ct_adjusted': 'sum'
}).reset_index()


#Core airport performance metrics

#General delay metrics
airport_full['delay_rate'] = (
    airport_full['arr_del15'] /
    airport_full['arr_flights'].replace(0, pd.NA)
)

airport_full['avg_delay_per_flight'] = (
    airport_full['arr_delay'] /
    airport_full['arr_flights'].replace(0, pd.NA)
)

airport_full['avg_delay_if_delayed'] = (
    airport_full['arr_delay'] /
    airport_full['arr_del15'].replace(0, pd.NA)
)


#Weather-specific mapping metrics

#Raw weather share of total delay minutes
airport_full['weather_share_raw'] = (
    airport_full['weather_delay'] /
    airport_full['arr_delay'].replace(0, pd.NA)
)

#Adjusted weather share of total delay minutes
airport_full['weather_share_adjusted'] = (
    airport_full['weather_adjusted'] /
    airport_full['arr_delay'].replace(0, pd.NA)
)

#Weather delay minutes per flight (raw)
airport_full['weather_per_flight_raw'] = (
    airport_full['weather_delay'] /
    airport_full['arr_flights'].replace(0, pd.NA)
)

#Weather delay minutes per flight (adjusted)
airport_full['weather_per_flight_adjusted'] = (
    airport_full['weather_adjusted'] /
    airport_full['arr_flights'].replace(0, pd.NA)
)

#Weather probability / frequency per flight (adjusted counts)
airport_full['weather_prob_adjusted'] = (
    airport_full['weather_ct_adjusted'] /
    airport_full['arr_flights'].replace(0, pd.NA)
)

#Weather severity when weather-related delays occur
airport_full['weather_severity_adjusted'] = (
    airport_full['weather_adjusted'] /
    airport_full['weather_ct_adjusted'].replace(0, pd.NA)
)

#Standardize weather frequency and severity
weather_prob_std = (
    (airport_full['weather_prob_adjusted'] - airport_full['weather_prob_adjusted'].mean()) /
    airport_full['weather_prob_adjusted'].std()
)

weather_severity_std = (
    (airport_full['weather_per_flight_adjusted'] - airport_full['weather_per_flight_adjusted'].mean()) /
    airport_full['weather_per_flight_adjusted'].std()
)

#Combined score: equal weight on frequency + severity
airport_full['weather_score'] = (
    0.5 * weather_prob_std +
    0.5 * weather_severity_std
)

#Clean missing values from derived columns
derived_cols = [
    'delay_rate',
    'avg_delay_per_flight',
    'avg_delay_if_delayed',
    'weather_share_raw',
    'weather_share_adjusted',
    'weather_per_flight_raw',
    'weather_per_flight_adjusted',
    'weather_prob_adjusted',
    'weather_severity_adjusted',
    'weather_score'
]

for col in derived_cols:
    airport_full[col] = airport_full[col].fillna(0)


#Dominant cause by delay MINUTES
airport_full['dominant_cause_minutes'] = airport_full[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].idxmax(axis=1)

#Dominant cause by delay COUNTS
airport_full['dominant_cause_counts'] = airport_full[[
    'carrier_ct',
    'weather_ct',
    'nas_ct',
    'security_ct',
    'late_aircraft_ct'
]].idxmax(axis=1)


#Merge airport metrics with coordinates
airport_map = airport_full.merge(
    airports,
    left_on='airport',
    right_on='iata_code',
    how='inner'
)


#Export map-ready file
airport_map.to_csv("output/mapping/airport_map_data.csv", index=False)


# --------------------------------------------
#Seasonal Maps
#Create a season variable from month

#Function to assign seasons based on month
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"   # Dec–Feb
    elif month in [3, 4, 5]:
        return "Spring"   # Mar–May
    elif month in [6, 7, 8]:
        return "Summer"   # Jun–Aug
    else:
        return "Fall"     # Sep–Nov

#Apply function to create new column
delays['season'] = delays['month'].apply(get_season)

#Group by both airport and season
seasonal_airport = delays.groupby(['airport', 'season']).agg({
    'arr_flights': 'sum',        # total flights in that season
    'arr_delay': 'sum',          # total delay minutes
    'weather_adjusted': 'sum'    # adjusted weather delay minutes
}).reset_index()

#Weather share = proportion of delay minutes caused by weather
seasonal_airport['weather_share_adjusted'] = (
    seasonal_airport['weather_adjusted'] /
    seasonal_airport['arr_delay'].replace(0, pd.NA)
)

#Replace missing values
seasonal_airport['weather_share_adjusted'] = seasonal_airport['weather_share_adjusted'].fillna(0)

#Merge with airport coordinates
seasonal_map = seasonal_airport.merge(
    airports,
    left_on='airport',     # from delays dataset
    right_on='iata_code',  # from airport dataset
    how='inner'
)

#Export for QGIS
seasonal_map.to_csv("output/mapping/seasonal_weather_map.csv", index=False)


