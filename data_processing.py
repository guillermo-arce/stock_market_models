# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:15:47 2020

@author: Guillermo Arce
"""

#%% Imports
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from ta_functions import add_all_ta, add_reduced_ta

#%% Constant definition
TIME = "time"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"

#Please, change accordingly to the desired model (True for Pytorch, False for Tensorflow)
PYTORCH_MODEL=True

DATA_SOURCE = "dataset.csv"

#%% Basic data preprocessing

print("Preprocessing data ("+DATA_SOURCE+"), please wait...")

if(PYTORCH_MODEL):
    directory = "pytorch"
else:
    directory = "tensorflow"

# Loading in the df
df = pd.read_csv(DATA_SOURCE)

# Convert "time" column to DateTime dftype
df[TIME]= pd.to_datetime(df[TIME], errors='coerce') 

# Dropping any NaNs
df.dropna(inplace=True)

print("Data loading...✓")

#%% Adding Technical Indicators to dataset (also adding the smoothed data)
    
if (PYTORCH_MODEL):
    df = add_reduced_ta(df)
else:
    df = add_all_ta(df)
    
# Dropping everything but the Indicators
df.drop([TIME, OPEN, HIGH, LOW, CLOSE, VOLUME], axis=1, inplace=True)

print("Technical indicators and data smoothing...✓")

#%% Scaling 

# Scale fitting the close prices separately for inverse_transformations purposes later
close_scaler = RobustScaler()
close_scaler.fit(df[["close_sma"]])
joblib.dump(close_scaler, directory+'/close_scaler.pkl') 

# Normalizing/Scaling the DF
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

print("Scaling...✓")

#%% Export dataset 

(df.drop(df.index[:50])).to_csv(directory+"/data_preprocessed.csv")

print("Converting to csv...✓")  

print("Done! You can find it in '" + directory + "' directory.")


