# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:03:41 2023

@author: Emmanuel Maseruka
"""

import numpy as np
import pandas as pd
#from keras.models import Sequential
#from keras.layers import LSTM, Dense,Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import talib


desired_start_date = pd.to_datetime('2017-05-03')
desired_end_date = pd.to_datetime('2022-12-31')

data=pd.read_excel("C:/Users/Kevin Meng/OneDrive/Desktop/Dissertation/price_data.xlsx",sheet_name="ethereum")
data['14_sma'] = data['Close**'].rolling(14).mean()
ema_values = talib.EMA(data['Close**'], timeperiod=14)
data['ema']=ema_values
macd_fast_period = 5
macd_slow_period = 10
macd_signal_period = 5
macd, macd_signal, _ = talib.MACD(data['Close**'], fastperiod=macd_fast_period,slowperiod=macd_slow_period, signalperiod=macd_signal_period)

data['macd']= macd
data['macd_signal'] = macd_signal

data=data[['Date','Volume','14_sma','ema','macd_signal','macd','Close**']]
data=data.dropna()
data['Date'] = pd.to_datetime(data['Date'])  #convert to xts
data['Date'] = data['Date'].dt.date #strip off min/hrs/sec from date
data.sort_values(by='Date', inplace=True)
data = data[(data['Date'] >= desired_start_date) & (data['Date'] <= desired_end_date)]




btc=pd.read_excel("C:/Users/Kevin Meng/OneDrive/Desktop/Dissertation/price_data.xlsx",sheet_name="Sheet1")
btc=btc[['Date','Close**']]
btc.rename(columns={'Close**': 'Close**btc'}, inplace=True)
btc['Date'] = pd.to_datetime(btc['Date'])  #convert to xts
btc['Date'] = btc['Date'].dt.date #strip off min/hrs/sec from date
btc.sort_values(by='Date', inplace=True)
btc = btc[(btc['Date'] >= desired_start_date) & (btc['Date'] <= desired_end_date)]



transaction_growth = pd.read_csv('C:/Users/Kevin Meng/OneDrive/Desktop/Dissertation/export-TxGrowth.csv')
transaction_growth = transaction_growth[['Date(UTC)','transaction_growth']]
transaction_growth['Date(UTC)'] = pd.to_datetime(transaction_growth['Date(UTC)'])  #convert to xts
transaction_growth['Date(UTC)'] = transaction_growth['Date(UTC)'].dt.date#trip off min/hrs/sec from date
transaction_growth.sort_values(by='Date(UTC)', inplace=True)
transaction_growth = transaction_growth[(transaction_growth['Date(UTC)'] >= desired_start_date) & (transaction_growth['Date(UTC)'] <= desired_end_date)]
transaction_growth.rename(columns={'Date(UTC)': 'Date'}, inplace=True)

block_size = pd.read_csv('C:/Users/Kevin Meng/OneDrive/Desktop/Dissertation/export-BlockSize.csv')
block_size = block_size[['Date(UTC)','block_size_bytes']]
block_size['Date(UTC)'] = pd.to_datetime(block_size['Date(UTC)'])  #convert to xts
block_size['Date(UTC)'] = block_size['Date(UTC)'].dt.date#trip off min/hrs/sec from date
block_size.sort_values(by='Date(UTC)', inplace=True)
block_size = block_size[(block_size['Date(UTC)'] >= desired_start_date) & (block_size['Date(UTC)'] <= desired_end_date)]
block_size.rename(columns={'Date(UTC)': 'Date'}, inplace=True)

################################################################
gas = pd.read_csv('C:/Users/Kevin Meng/OneDrive/Desktop/Dissertation/export-AvgGasPrice.csv')
gas = gas[['Date(UTC)','gas_price_wei']]
gas['Date(UTC)'] = pd.to_datetime(gas['Date(UTC)'])  #convert to xts
gas['Date(UTC)'] = gas['Date(UTC)'].dt.date#trip off min/hrs/sec from date
gas.sort_values(by='Date(UTC)', inplace=True)
gas = gas[(gas['Date(UTC)'] >= desired_start_date) & (gas['Date(UTC)'] <= desired_end_date)]
gas.rename(columns={'Date(UTC)': 'Date'}, inplace=True)



token_tr = pd.read_csv('C:/Users/Kevin Meng/OneDrive/Desktop/Dissertation/export-AvgGasPrice.csv')
token_tr = token_tr[['Date(UTC)','gas_price_wei']]
token_tr['Date(UTC)'] = pd.to_datetime(token_tr['Date(UTC)'])  #convert to xts
token_tr['Date(UTC)'] = token_tr['Date(UTC)'].dt.date#trip off min/hrs/sec from date
token_tr.sort_values(by='Date(UTC)', inplace=True)
token_tr = token_tr[(token_tr['Date(UTC)'] >= desired_start_date) & (token_tr['Date(UTC)'] <= desired_end_date)]
token_tr.rename(columns={'Date(UTC)': 'Date'}, inplace=True)


vader=pd.read_csv("C:/Users/Kevin Meng/OneDrive/Desktop/Dissertation/vader_sentiment_variable.csv")
vader=vader[['Date','compound_score']]
vader['Date'] = pd.to_datetime(vader['Date'])  #convert to xts
vader['Date'] = vader['Date'].dt.date #strip off min/hrs/sec from date
vader.sort_values(by='Date', inplace=True)
vader = vader[(vader['Date'] >= desired_start_date) & (vader['Date'] <= desired_end_date)]






gt=pd.read_csv('E:/blockchain and crypto/blockchain&crypto_dataset.csv')
gt=gt[['Date','google_trend']]
gt['Date'] = pd.to_datetime(gt['Date'])  #convert to xts
gt['Date'] = gt['Date'].dt.date#trip off min/hrs/sec from date
gt.sort_values(by='Date', inplace=True)
gt = gt[(gt['Date'] >= desired_start_date) & (gt['Date'] <= desired_end_date)]


data=pd.merge(data, btc, on='Date', how='outer')
data=pd.merge(data, transaction_growth, on='Date', how='outer')
data=pd.merge(data, block_size, on='Date', how='outer')
data=pd.merge(data, gt, on='Date', how='outer')
data=pd.merge(data, gas, on='Date', how='outer')
data=pd.merge(data, token_tr, on='Date', how='outer')
data=pd.merge(data, vader, on='Date', how='outer')

column_mean = data['compound_score'].mean()

# Replace NA values with the column mean
data['compound_score'].fillna(column_mean, inplace=True)
################################################################
data=data.drop('Date', axis=1)

data_for_test = data.iloc[int(0.8 * len(data)):len(data)]

data_for_train = data.iloc[0:int(0.8 * len(data))]

scaler = StandardScaler()
scaler = scaler.fit(data_for_train)
data_scaled = scaler.transform(data_for_train)

#LSTM
# Prepare data for LSTM

trainX = []
trainY = []


n_future = 1
n_past = 7

for i in range(n_past, len(data_scaled) - n_future +1):
    trainX.append(data_scaled[i-n_past : i, 0:data_for_train.shape[1]])  #take 7 rows for 11 columns of data scaled
    trainY.append(data_scaled[i + n_future - 1:i + n_future, 5]) #take  column 5 with close prices
    
trainX, trainY = np.array(trainX), np.array(trainY)


from kerastuner.tuners import RandomSearch


# Define the LSTM model function
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers import legacy

#from tensorflow.keras.layers import Input, Dense
#from tensorflow_addons.layers import tcn,Model

from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

# Assume you have trainX and trainY prepared

def build_tcn_model(hp):
    input_layer = Input(shape=(trainX.shape[1], trainX.shape[2]))
    
    num_filters = hp.Int('num_filters', min_value=32, max_value=128, step=32)
    kernel_size = hp.Int('kernel_size', min_value=2, max_value=5, step=1)
    
    conv_blocks = []
    for dilation_rate in [2, 4, 8]:
        x = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(input_layer)
        conv_blocks.append(x)
    
    concat = tf.keras.layers.Concatenate()(conv_blocks)
    flatten = Flatten()(concat)
    output_layer = Dense(trainY.shape[1])(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    # Custom Adam optimizer with the specified parameters
    learning_rate = 0.01
    decay_rate = 0.0005
    beta_1 = 0.91
    beta_2 = 0.98
    
    learning_rate_schedule = schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        decay_rate=decay_rate
    )
    
    optimizer = legacy.Adam(
        learning_rate=learning_rate_schedule,
        beta_1=beta_1,
        beta_2=beta_2
    )
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Initialize the RandomSearch tuner
tuner = RandomSearch(
    build_tcn_model,
    objective='val_loss',
    max_trials=5,  # Number of hyperparameter combinations to try
    executions_per_trial=1,  # Number of models to train per trial (set to 1 for faster tuning)
    directory='tuner_logs',  # Optional, if you want to save logs and checkpoints
    project_name='my_lstm_model'  # Optional, a name for the project
)

# Perform hyperparameter tuning
tuner.search(trainX, trainY, epochs=10, batch_size=32, validation_split=0.1)

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:", best_hyperparameters)

# Build the best LSTM model
best_model = tuner.hypermodel.build(best_hyperparameters)

# Fit the best model on the entire training data
best_model.fit(trainX, trainY, epochs=10, batch_size=16, verbose=1)


from sklearn.metrics import mean_absolute_error
n_splits = 10

# Initialize the KFold cross-validator
kf = KFold(n_splits=n_splits, shuffle=True)

mae_scores = []  # To store the MAE scores for each fold

# Iterate through the folds
for train_idx, val_idx in kf.split(trainX):
    # Split the data into training and validation sets for this fold
    fold_trainX, fold_valX = trainX[train_idx], trainX[val_idx]
    fold_trainY, fold_valY = trainY[train_idx], trainY[val_idx]

    # Train the best model on the current fold's training data
    best_model.fit(fold_trainX, fold_trainY, epochs=10, batch_size=16, verbose=0)
    
    # Make predictions using the trained model on the validation data
    predictions = best_model.predict(fold_valX)
    
    # Calculate the MAE for this fold and store it in the list
    fold_mae = mean_absolute_error(fold_valY, predictions)
    mae_scores.append(fold_mae)

# Print the MAE scores for each fold
for fold_num, mae in enumerate(mae_scores, start=1):
    print(f"Fold {fold_num}: MAE = {mae}")
    
# Calculate and print the average MAE across all folds
average_mae = sum(mae_scores) / n_splits
print("Average MAE:", average_mae)


from sklearn.metrics import mean_squared_error, r2_score

for fold_num, mae in enumerate(mae_scores, start=1):
    mse = mean_squared_error(fold_valY, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(fold_valY, predictions)
    
    print(f"{mae:.4f}")




#test stage

data_for_test_scaled = scaler.transform(data_for_test)
testX = []
testY = []

for i in range(n_past, len(data_for_test_scaled) - n_future + 1):
     testX.append(data_for_test_scaled[i - n_past : i, 0:data_for_test.shape[1]])
     testY.append(data_for_test_scaled[i + n_future - 1:i + n_future, 5])

testX, testY = np.array(testX), np.array(testY)



y_pred_test = best_model.predict(testX)

y_pred_unscaled = scaler.inverse_transform(np.repeat(y_pred_test, data_for_train.shape[1], axis=-1))[:, 5]
y_val_unscaled = scaler.inverse_transform(np.repeat(testY, data_for_train.shape[1], axis=-1))[:, 5]


import matplotlib.pyplot as plt

# Plot the unscaled predicted and true values
plt.figure(figsize=(10, 6))
plt.plot(y_pred_unscaled, label='Predicted')
plt.plot(y_val_unscaled, label='True')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Unscaled Predicted vs. True Values')
plt.legend()
plt.show()


