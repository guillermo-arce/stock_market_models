# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 11:15:47 2020

@author: Guillermo Arce
"""

# %% Imports

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch as torch
import torch.nn as nn

import math
from sklearn.metrics import mean_squared_error

import time

import joblib

# %% Constants definition

TIME = "time"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close_sma"
VOLUME = "volume"

# %%

# Loading in the df
df = pd.read_csv("../../Data/pytorch_datasets/sept_oct/dataset_ta_indicators_simple_pytorch.csv")

df.drop(df.columns[0], axis=1, inplace=True)

#%% Convert data to time series

# Splitting dataset into train and test data

n = len(df)
train_df = df[0:int(n*0.8)]
test_df = df[int(n*0.8):]

# It is not efficient to loop through the dataset while training the model. 
# So we want to transform the dataset with each row representing the historical data and the target.
# 1 - Split into X(input) and y(output)
# 2 - Transform data into time series format

TIME_STEPS = 100
NUMBER_PREDICTIONS = 5  # Output
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

# We need to transform our data into tensors, which is the basic structure for building a Pytorch model
X_train = torch.from_numpy(X_train).type(torch.Tensor)
X_test = torch.from_numpy(X_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

print("X_train shape: ", X_train.shape)
print("y_train shape: ",y_train.shape)
print("X_test shape: ",X_test.shape)
print("y_test shape: ",y_test.shape)

#%% We create the Stacked LSTM model and define its parameters

input_dim = 7
hidden_dim = 100
num_layers = 3
output_dim = NUMBER_PREDICTIONS

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden layer dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers
        # Stacked LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Linear layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):        
        # Initialize hidden state with zeros
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        # Initialize cell state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        #Last time steps 
        output = self.fc(output[:, -1, :]) 
        
        return output
    

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim,
             num_layers=num_layers,output_dim=output_dim)
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
#%% Start the training of the model

epoch_time = time.time()
start_time = time.time()

num_epochs=2
hist = np.zeros(num_epochs)
val = np.zeros(num_epochs)

for t in range(num_epochs):
    
    # Zero out gradient, else they will accumulate between epochs
    optimizer.zero_grad()
    
    # Forward pass
    y_train_pred = model(X_train)
    
    # Loss function calculation
    loss = loss_fn(y_train_pred, y_train)
    hist[t] = loss.item()/X_train.shape[0]

    # Backward pass (gradient calculation)
    loss.backward()

    # Update parameters (optimization)
    optimizer.step()          

    #Validation
    val_pred = model(X_test)
    val_loss = loss_fn(val_pred,y_test)
    val[t]=val_loss.item()/X_test.shape[0]
    
    if(t%5==0):
        print("TRAIN ERROR: ",hist[t])
        print("VAL ERROR: ",val[t])
        print("EPOCH ",t, " TIME: ", time.time()-epoch_time)
        epoch_time = time.time()
 	
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

torch.save(model.state_dict(), "model.pth") 
np.savetxt("validation.csv", val, delimiter=",")
np.savetxt("loss.csv", hist, delimiter=",")

#%% Load model if necessary

model.load_state_dict(torch.load('models_comparison/models_tested_torch/sept_oct_5preds/model.pth'))
model.eval()

#%% Definition of methods to make predictions and to format predictions for plotting
def make_predictions(plot_extension, test):
    all_predictions = []
    for i in range(0,plot_extension,NUMBER_PREDICTIONS):
        if(test):
            input_data = X_test[i].reshape(-1,X_test.shape[1],X_test.shape[2]) 
        else:
            input_data = X_train[i].reshape(-1,X_train.shape[1],X_train.shape[2]) 
        
        prediction=model(input_data)
        
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

plot_extension = 3432
test_data = True

predictions = make_predictions(plot_extension, test_data)

#Re-scale predictions
close_scaler = joblib.load("models_comparison/models_tested_torch/sept_oct_2/close_scaler.pkl") 
predictions = close_scaler.inverse_transform(np.asarray(predictions).reshape(1, -1))[0]

#%% Plot and calculate RMSE error, for training or testing data

#Prepare predictions and real data for plotting
if(test_data):
    testD = (test_df[CLOSE]).to_numpy()
    real_prices = testD[:plot_extension+TIME_STEPS+3]
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
accuraccy_up_down(predictions, real_prices[TIME_STEPS-1:])   

