import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('ML_Assignment_2\weather_forecast_data.csv') #Read data from file
print(df.head()) #Show a sample of the Data
print(df.info())#show info for the data


# Dropping rows with any missing values
df_dropped = df.dropna()

# Select all columns except the last one
columns_except_last = df.columns[:-1]

# Replace missing values in all columns except the last one with the column mean
df_filled = df.copy()
df_filled[columns_except_last] = df_filled[columns_except_last].fillna(df_filled[columns_except_last].mean())
# print(df_filled)
print(df_filled.info())