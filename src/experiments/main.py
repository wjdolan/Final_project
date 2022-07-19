import pandas as pd
import numpy as np
import requests
import os
import json
import psycopg2 as ps
from modules.funct_library import *
from tensorflow.keras.callbacks import EarlyStopping


# import time series data

df_gas = pd.DataFrame(columns=["Date", "Volume_kbbld_gas"])


root_URL = "https://api.eia.gov/series/?api_key=" 

gasoline_series = "&series_id=PET.MGFUPUS2.M"


request_to_df(root_URL, gasoline_series, "gasoline")


# Data preprocessing

df_gas['Date'] = pd.to_datetime(df['Date'], format='%Y%m')

df_gas.sort_values(by=['Date'], ascending=True, inplace=True)


# Load database

host_name = 'database-fp.cw6wnzmbhtcw.us-west-2.rds.amazonaws.com'
dbname = 'FinalProject_db'
port = '5432'
username = USERNAME = str(os.environ.get('AWS_RDS_USERNAME'))
password = PWORD= str(os.environ.get('AWS_RDS_PWD'))
conn = None


# Pull from database


# Models:

train_length = 800 # train / test split

train, test = tts(df,train_length, 'Volume_kbbld_gas')

eval_df = pd.DataFrame(columns=['Model', 'MAPE', 'RMSE'])

# Baseline model (6-step moving average)

df['MA6'] = df['Volume_kbbld_gas'].rolling(6).mean()
df = df.fillna(0)

df['Base'] = np.where(df.index < length, 0, 8897) # use last MA value for prediction

base_mape, base_RMSE = metric_evals(test[:128], df.loc[train_length:, 'base'])

eval_df.loc[0] = ['Base'] + [base_mape] + [base_RMSE]

# ARIMA model

modelAR = pm.auto_arima(train, start_p=1, start_q=1,
                      test='adf',                       # use adftest to find optimal 'd'
                      max_p=3, max_q=3,                 # maximum p and q
                      m=12,                             # monthly frequency
                      d=None,                           # let model determine 'd'
                      seasonal=True,                    # Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(modelAR.summary())

modelAR.plot_diagnostics(figsize=(7,5))

ARpreds, ARconf_int = modelAR.predict(n_periods=test.shape[0], return_conf_int=True)

AR_mape, AR_rmse = metric_evals(test, ARpreds)

eval_df.loc[1] = ['ARIMA'] + [AR_mape] + [AR_rmse]

# SARIMA model

modelSAR = pm.auto_arima(train, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

print(modelSAR.summary())

SARpreds, SARconf_int = modelSAR.predict(n_periods=test.shape[0], return_conf_int=True)

SAR_mape, SAR_rmse = metric_evals(test, SARpreds)

eval_df.loc[2] = ['SARIMA'] + [SAR_mape] + [SAR_rmse]


# LSTM model

time_step = 20

X_train, y_train = create_dataset(train, time_step)
X_test, ytest = create_dataset(test, time_step)

print(X_train.shape), print(y_train.shape)

print(X_test.shape), print(ytest.shape)

# reshape for LSTM

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

modelLSTM = LSTM_model(time_step,1)

modelLSTM.summary()


monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=30, 
        verbose=1, mode='auto', restore_best_weights=True)

history=modelLSTM.fit(X_train,y_train,validation_data=(X_test,ytest),
        callbacks=[monitor],verbose=1,epochs=50)


train_predict=modelLSTM.predict(X_train)
test_predict=modelLSTM.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


LSTM_mape, LSTM_rmse = metric_evals(test[:len(test)-21], test_predict)

eval_df.loc[2] = ['SARIMA'] + [SAR_mape] + [SAR_rmse]