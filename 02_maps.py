#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:08:17 2026

@author: alaina
"""

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

#reate output folders
os.makedirs("output/mapping", exist_ok=True)

#Load delay dataset
delays = pd.read_csv("Airline_Delay_Cause.csv")

#Drop missing values in key columns
cols_to_check = ['arr_flights', 'arr_del15', 'arr_delay']
delays = delays.dropna(subset=cols_to_check)

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

#Aggregate delay data by airport
airport_full = delays.groupby('airport').agg({
    'arr_flights': 'sum',
    'arr_del15': 'sum',
    'arr_delay': 'sum'
}).reset_index()

#Create performance metrics
airport_full['delay_rate'] = airport_full['arr_del15'] / airport_full['arr_flights'].replace(0, pd.NA)
airport_full['avg_delay_per_flight'] = airport_full['arr_delay'] / airport_full['arr_flights'].replace(0, pd.NA)
airport_full['avg_delay_if_delayed'] = airport_full['arr_delay'] / airport_full['arr_del15'].replace(0, pd.NA)

#Merge delay data with airport coordinates
airport_map = airport_full.merge(
    airports,
    left_on='airport',
    right_on='iata_code',
    how='inner'
)

#Aggregate delay causes by airport
airport_causes = delays.groupby('airport')[[
    'carrier_delay',
    'weather_delay',
    'nas_delay',
    'security_delay',
    'late_aircraft_delay'
]].sum()

#Identify dominant cause
airport_causes['dominant_cause'] = airport_causes.idxmax(axis=1)

#Merge dominant cause into main dataset
airport_map = airport_map.merge(
    airport_causes[['dominant_cause']],
    left_on='airport',
    right_index=True,
    how='left'
)

#Quick check
print(airport_map.shape)
print(airport_map.head())

#Export map-ready file
airport_map.to_csv("output/mapping/airport_map_data.csv", index=False)
