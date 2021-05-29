""""@author: Guillermo Arce"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import joblib

#%% Constants definition

TIME = "time"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close_sma"
VOLUME = "volume"

#Already trained models support these combinations:
#    (1) TIME_STEPS=100 and NUMBER_PREDICTIONS=10
#    (2) TIME_STEPS=100 and NUMBER_PREDICTIONS=1

TIME_STEPS = 600        
NUMBER_PREDICTIONS=60
CLOSE_POSITION = 0

#%% Load pre-processed data

df = pd.read_csv("tensorflow/data_preprocessed.csv")

df.drop(df.columns[0], axis=1, inplace=True)

#%% Convert data to time-series format, this is specific to the requirements of the model

# Splitting dataset into train and test data
n = len(df)
train_df = df[0:int(n*0.7)]
test_df = df[int(n*0.7):]

# It is not efficient to loop through the dataset while training the model. 
# So we want to transform the dataset with each row representing the historical data and the target.
# 1 - Split into X(input) and y(output)
# 2 - Transform data into time series format

def build_timeseries(mat, y_col_index):
    
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]

    x = np.zeros((dim_0-NUMBER_PREDICTIONS, TIME_STEPS, dim_1))
    y = np.zeros((dim_0-NUMBER_PREDICTIONS,NUMBER_PREDICTIONS))

    for i in range(dim_0-NUMBER_PREDICTIONS):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i:TIME_STEPS+i+NUMBER_PREDICTIONS,y_col_index]

    return x, y
    
X_train, y_train = build_timeseries(train_df.values,CLOSE_POSITION)
X_test, y_test = build_timeseries(test_df.values,CLOSE_POSITION)

print("X_train shape: ", X_train.shape)
print("y_train shape: ",y_train.shape)
print("X_test shape: ",X_test.shape)
print("y_test shape: ",y_test.shape)

#%% Create the Stacked LSTM model and start the training

model = Sequential()
model.add(LSTM(200,return_sequences=True,input_shape=(TIME_STEPS,X_train.shape[2])))
model.add(LSTM(200,return_sequences=True))
model.add(LSTM(100))
model.add(Dense(NUMBER_PREDICTIONS))

model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

res = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64,verbose=1)

#%% Load a model, if desired

model= tf.keras.models.load_model("tensorflow/trained_models/"+str(TIME_STEPS)+"_"+str(NUMBER_PREDICTIONS))
model.summary()

#%% Function to make predictions

def make_predictions(plot_extension, data):
    all_predictions = []
    
    for i in range(0,plot_extension,NUMBER_PREDICTIONS):
       
        input_data = data[i].reshape(-1,data.shape[1],data.shape[2]) 
        
        prediction=model.predict(input_data)
        
        for j in range(NUMBER_PREDICTIONS):
            all_predictions.append(prediction[0,j])
    
    return all_predictions
    
#%% Function to calculate direction prediction accuraccy

#   The direction accuraccy is calculated by 
#   comparing the last price value predicted with the last price value of the **previous prediction**. 
#   So, in order to calculate the direction of a prediction, we would need to calculate the previous 
#   prediction and compare those values. Otherwise, if we compare the predicted price with the real price
#   in order to get the direction of the prediction, we would be harming the capacity of the model of 
#   predicting the direction; as the error of the exact price value prediction would be affecting it.
#   To sum up, the idea of working in this way is not to mix the (1) capacity of the model of predicting 
#   the direction (that may be easier to predict and with less error) and the (2) capacity of the model 
#   of predicting the exact price (more difficult to calculate and with more error).
    
#Example:
# Predicted prices in batches of NUMBER_PREDICTIONS = 3:
# 54.6 54.3 54.2(A) | 57.8 57.5 57.4(B) | 57.9 60.1 60.2(C) | 57.8 57.5 57.4(D)
# Number A (54.2) is compared to number B (57.4) in order to get if the prediction direction has been
# that the price will increase (UP) or decrease (DOWN); in this case the price has raised (UP)
# In the same way, (B) would be compared with (C), (C) with (D)...
def direction_accuraccy(predictions,real):    
    results_pred = []
    results_real = []
    for i in range(NUMBER_PREDICTIONS-1, len(predictions)-1, NUMBER_PREDICTIONS):
        pred_first_price = predictions[i]
        pred_last_price = predictions[i+NUMBER_PREDICTIONS]
        up = pred_first_price<pred_last_price
        results_pred.append(up)
        
        real_first_price = real[i]
        real_last_price= real[i+NUMBER_PREDICTIONS]
        up = real_first_price<real_last_price
        results_real.append(up)
    
    #Count the errors in direction prediction
    direction_errors = 0
    for i in range (len(results_pred)):
        if(results_pred[i]!=results_real[i]):
            direction_errors+=1  
            
    print("Direction Errors: ", direction_errors,"/",len(results_pred))
    print("Direction Accuraccy: ", (len(results_pred)-direction_errors)*100/len(results_pred), "%")
        
#%% Execution of predictions until the selected extension, also if train or test data is used should be specified

#With current pre-processed data, plot_extension value could be until 49300 
plot_extension = 3000
test_data = True

if(test_data):
    predictions = make_predictions(plot_extension, X_test)
else:
    predictions = make_predictions(plot_extension, X_train)

#%% Re-scale prediction
close_scaler = joblib.load("tensorflow/close_scaler.pkl") 
predictions = close_scaler.inverse_transform(np.asarray(predictions).reshape(1, -1))[0]
#%% Calculate RMSE and direction accuracy; also plot

#(1) Getting original prices
if(test_data):
    aux = (test_df[CLOSE]).to_numpy()
    real_prices = aux[TIME_STEPS:plot_extension+TIME_STEPS]
else:
    aux = (train_df[CLOSE]).to_numpy()
    real_prices = aux[TIME_STEPS:plot_extension+TIME_STEPS]
    
#(2) Re-scale original prices
real_prices = close_scaler.inverse_transform(np.asarray(real_prices).reshape(1, -1))[0]
    
#(3) Calculate RMSE
error = math.sqrt(mean_squared_error(real_prices,predictions))
print ("RMSE: ",error)

#(4) Plot
plt.plot(predictions)
plt.plot(real_prices)
plt.legend(['Prediction', 'Real'], loc='best')
plt.xlabel("Minutes")
plt.ylabel("Price")
plt.show()

#(5) Calculate direction accuraccy of predictions
direction_accuraccy(predictions, real_prices)
