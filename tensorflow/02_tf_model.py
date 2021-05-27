# -*- coding: utf-8 -*-

# Importing Libraries
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

#%% Load data

df = pd.read_csv("..\..\Data\dataset_ta_indicators.csv")

df.drop(df.columns[0], axis=1, inplace=True)

#%% Convert data to time series

# Splitting dataset into train and test data

n = len(df)
train_df = df[0:int(n*0.7)]
test_df = df[int(n*0.7):]

print(train_df.shape[0])
print(train_df.shape[1])

# It is not efficient to loop through the dataset while training the model. 
# So we want to transform the dataset with each row representing the historical data and the target.
# 1 - Split into X(input) and y(output)
# 2 - Transform data into time series format

TIME_STEPS = 100
NUMBER_PREDICTIONS=1
CLOSE_POSITION = 0

def build_timeseries(mat, y_col_index):
    
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]


    x = np.zeros((dim_0-NUMBER_PREDICTIONS, TIME_STEPS, dim_1))
    y = np.zeros((dim_0-NUMBER_PREDICTIONS,NUMBER_PREDICTIONS))
    
    print(y.shape)

    print("Length of inputs", dim_0)

    for i in range(dim_0-NUMBER_PREDICTIONS):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i:TIME_STEPS+i+NUMBER_PREDICTIONS,y_col_index]

    print("length of time-series - inputs", x.shape)
    print("length of time-series - outputs", y.shape)

    return x, y
    
X_train, y_train = build_timeseries(train_df.values,CLOSE_POSITION)
X_test, y_test = build_timeseries(test_df.values,CLOSE_POSITION)


print("X_train shape: ", X_train.shape)
print("y_train shape: ",y_train.shape)
print("X_test shape: ",X_test.shape)
print("y_test shape: ",y_test.shape)

#%% We create the Stacked LSTM model and start the training

model = Sequential()
model.add(LSTM(200,return_sequences=True,input_shape=(TIME_STEPS,X_train.shape[2])))
model.add(LSTM(200,return_sequences=True))
model.add(LSTM(100))
model.add(Dense(NUMBER_PREDICTIONS))

model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

res = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64,verbose=1)

#%% We can load a model if necessary

model= tf.keras.models.load_model('models_comparison/models_tested_tf/in100_out1_full')
model.summary()

#%% Definition of methods to make predictions and to format predictions for plotting
def make_predictions(plot_extension, test):
    all_predictions = []
    for i in range(0,plot_extension,NUMBER_PREDICTIONS):
        if(test):
            input_data = X_test[i].reshape(-1,X_test.shape[1],X_test.shape[2]) 
        else:
            input_data = X_train[i].reshape(-1,X_train.shape[1],X_train.shape[2]) 
        
        prediction=model.predict(input_data)
        
        for j in range(NUMBER_PREDICTIONS):
            all_predictions.append(prediction[0,j])
    return all_predictions

def adapt_to_plot(predictions):
    plot_data=[]
    for i in range(TIME_STEPS):
        plot_data.append(np.nan)
    for i in range(len(predictions)):
        plot_data.append(predictions[i])
    return plot_data
    
#%% Method to calculate direction error of predictions
    
def accuraccy_up_down(predictions,real):
    results_pred = []
    results_real = []
    
    current_price_pred = predictions[0]
    current_price_real= real[0]
    
    for i in range(NUMBER_PREDICTIONS-1,len(predictions),NUMBER_PREDICTIONS):
        up= current_price_pred<=predictions[i]
        results_pred.append(up)
        
        up= current_price_real<=real[i]
        results_real.append(up)
        
        if(i+1<len(predictions)):
            current_price_pred=predictions[i+1]
            current_price_real=real[i+1]
        
    counter_errors = 0
    for i in range (len(results_pred)):
        if(results_pred[i]!=results_real[i]):
            counter_errors+=1           
    
    print("Direction Errors: ", counter_errors,"/",len(results_pred))
    
    percentageAccuracy = (len(results_pred)-counter_errors)*100/len(results_pred)
    print("Direction Accuraccy: ", percentageAccuracy, "%")
    
def accuraccy_up_down_one(predictions,real):
    results_pred = []
    results_real = []
    
    current_price_pred = predictions[0]
    current_price_real= real[0]
    
    for i in range(1,len(predictions),NUMBER_PREDICTIONS):
        up= current_price_pred<=predictions[i]
        results_pred.append(up)
        
        up= current_price_real<=real[i]
        results_real.append(up)
        
        if(i+1<len(predictions)):
            current_price_pred=predictions[i]
            current_price_real=real[i]
        
    counter_errors = 0
    for i in range (len(results_pred)):
        if(results_pred[i]!=results_real[i]):
            counter_errors+=1           
    
    print("Direction Errors: ", counter_errors,"/",len(results_pred))
    
    percentageAccuracy = (len(results_pred)-counter_errors)*100/len(results_pred)
    print("Direction Accuraccy: ", percentageAccuracy, "%")
#%% Execution of predictions until the selected extension, also if train or test data is used should be specified

plot_extension = 10000
test_data = True

predictions = make_predictions(plot_extension, test_data)

#Re-scale prediction
close_scaler = joblib.load("models_comparison/models_tested_tf/in100_out1_full/close_scaler_complex.pkl") 
predictions = close_scaler.inverse_transform(np.asarray(predictions).reshape(1, -1))[0]

#%% Plot and calculate RMSE error, for training or testing data

#Prepare predictions and real data for plotting
if(test_data):
    testD = (test_df[CLOSE]).to_numpy()
    real_prices = testD[:plot_extension+TIME_STEPS]
else:
    real_prices = (df[CLOSE].head(plot_extension+TIME_STEPS)).to_numpy() 
    
#Re-scale data
real_prices = close_scaler.inverse_transform(np.asarray(real_prices).reshape(1, -1))[0]
    
#Error
error = math.sqrt(mean_squared_error(real_prices[TIME_STEPS:],predictions))
print ("ERROR: ",error)

#Plot
plot_predictions = adapt_to_plot(predictions)
plt.plot(plot_predictions)
plt.plot(real_prices)
plt.legend(['Prediction', 'Real'], loc='best')
plt.xlabel("Minutes")
plt.ylabel("Price")
plt.show()

#Calculate direction error
if(NUMBER_PREDICTIONS==1):
    accuraccy_up_down_one(predictions, real_prices[TIME_STEPS:])
else:
    accuraccy_up_down(predictions, real_prices[TIME_STEPS:])   


#%%
accuraccy_up_down_one(predictions, real_prices[TIME_STEPS:])