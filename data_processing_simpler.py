# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:15:47 2020

@author: Guillermo Arce
"""

#%% Imports

# Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
from ta_functions import add_all_ta
import joblib

#%% Constants definition
TIME = "time"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"

#%% Basic data preprocessing

# Loading in the df
df = pd.read_csv("..\..\Data\dataset.csv")

# Convert "time" column to DateTime dftype
df[TIME]= pd.to_datetime(df[TIME], errors='coerce') 

# Dropping any NaNs
df.dropna(inplace=True)

print(df)
#%% Adding Technical Indicators to dataset
    
df = add_all_ta(df)
    
# Dropping everything but the Indicators
df.drop([TIME, OPEN, HIGH, LOW, CLOSE, VOLUME], axis=1, inplace=True)

#%% Plotting

# plt.plot(df["close"].head(300), label='Close')
plt.plot(df["close_sma"].head(300), label='Close SMA')
plt.title('PRICE')
plt.legend()
plt.show()

#%% Scaling 

# Scale fitting the close prices separately for inverse_transformations purposes later
close_scaler = RobustScaler()

close_scaler.fit(df[["close_sma"]])

joblib.dump(close_scaler, 'scalers/close_scaler.pkl') 

# Normalizing/Scaling the DF
scaler = RobustScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

joblib.dump(scaler, 'scalers/scaler.pkl') 

#%% Export dataset 

print(df)

(df.drop(df.index[:50])).to_csv("..\..\Data\dataset_pytorch.csv")
