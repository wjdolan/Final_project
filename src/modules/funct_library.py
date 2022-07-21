import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from regex import E
import requests
import psycopg2 as ps
import os
import json
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import requests
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from pmdarima.arima.utils import ndiffs
from prophet import Prophet
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM, TimeDistributed, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import ConvLSTM2D
from keras.callbacks import EarlyStopping




def plot_history(history):
    """
    Returns two graphs:
        Training and validation accuracy
        Traning and validation loss
        
    Parameters:
        Trained keras model.fit
    """
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def get_request(response):
    for value in response.json()['series'][0]['data']:
        df_date = value[0]
        df_value = value[1]

        df = df.append({'Date': df_date, 'Volume_kbbld': df_value}, ignore_index=True)

        return df

def request_to_df(root_URL, url_series, df, name):
    API_KEY = str(os.environ.get('EIA_API_KEY'))
    r_url = root_URL + API_KEY + url_series
    response = requests.get(r_url)
    
    for value in response.json()['series'][0]['data']:
        df_date = value[0]
        df_value = value[1]

        return df.append({'Date': df_date, 'Volume_kbbld': df_value}, ignore_index=True)
       
        

def resid_plot(model):
  """
    Plot residual errors
    Input: model.fit variable
  """
  plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
  
  residuals = pd.DataFrame(model.resid)
  fig, ax = plt.subplots(1,2)
  residuals.plot(title="Residuals", ax=ax[0])
  residuals.plot(kind='kde', title='Density', ax=ax[1])

  # fig.savefig('gas_residuals.png')

  plt.show()


def connect_to_db(host_name, dbname, port, username, password):
    try:
        conn = ps.connect(host=host_name, databse=dbname, user=username, password=password, port=port)

    except ps.OperationalError as e:
        raise e
    else:
        print('Connected!')
    return conn


def create_table(curr):
    create_table_command = (""" CREATE TABLE IF NOT EXISTS Fuel_demand (
                index VARCHAR (255) PRIMARY KEY,
                Date DATE NOT NULL,
                Volume_kbbld INTEGER NOT NULL)""")

    curr.execute(create_table_command)
    return

def check_if_row_exists(curr, Date):
    """
        Checks if row entry in database already exists based on date
    """
    query = (""" SELECT Date FROM Fuel_demand WHERE Date = %s""")
    curr.execute(query, (Date,))

    return curr.fetchone() is not None

def update_row(curr, Date, Volume_kbbld):
    """ 
        Updates row entry 
    """

    query = ("""UPDATE Fuel_demand
                    SET Date = %s,
                    Volume_kbbld = %s
                    WHERE Date = %s;""")

    updated_var = (Date, Volume_kbbld)
    curr.execute(query, updated_var)

    return

def update_db(curr, df):
    """
        Check if row exists and update data and return df of new data to add
    """

    temp_df = pd.DataFrame(columns=["Date", "Volume_kbbld"])

    for i, row in df.iterrows():
        if check_if_row_exists():
            pass
            # update_row(curr, row['Date'], row['Volume_kbbld']) # include if overwriting row
        else:
                temp_df = temp_df.append(row)
                
    return temp_df


def insert_data(curr, Date, Volume_kbbld):
    """
        Insert new data into db table
    """

    insert_data = ("""INSERT INTO Fuel_demand (Date, Volume_kbbld) 
                        VALUES(%s, %s);""")

    row_to_insert = (Date, Volume_kbbld)
    curr.execute(insert_data, row_to_insert)

    return

def append_from_df_to_db(curr, df):
    """
        Append data to db from temporary df
    """

    for i, row in df.iterrows():
        insert_data(curr, row['Date'], df['Volume_kbbld'])

    return

def data_preprocess(df):
    """
        Preprocess df for models
    """

    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')
    
    return df.sort_values(by=['Date'], ascending=True, inplace=True)



def tts(df,length,column):
    """
        Create training and test split of data
        Input: dataframe (df), length to split (int)
        Output: train and test dataframes
    """
    train = df.loc[:length,[column]]
    test = df.loc[length:, [column]]

    return train, test
    

def metric_evals(test_data, predictions,):
    """ 
        Evaluates MAPE and RMSE for model predictions.
        Data sets must be of the same length
    """

    mape = mean_absolute_percentage_error(test_data, predictions)
    rmse = mean_squared_error(test_data, predictions, square=False)

    return mape, rmse


def AR_plot(train_data, test_data, predictions, conf_interval):
    """
        Plot of actuals vs predictions
    """


    x_axis = np.arange(train_data.shape[0] + predictions.shape[0])
    x_years = x_axis

    plt.plot(x_years[x_axis[:train_data.shape[0]]], train_data, alpha=0.75)
    plt.plot(x_years[x_axis[train_data.shape[0]:]], predictions, alpha=0.75)  # Forecasts
    plt.scatter(x_years[x_axis[train_data.shape[0]:]], test_data, alpha=0.4, marker='x', color='g')  # Test data
    plt.fill_between(x_years[x_axis[-predictions.shape[0]:]], conf_interval[:, 0], conf_interval[:, 1],
                 alpha=0.1, color='b')
    plt.title("Model Test predictions")
    plt.xlabel("Year")


def create_dataset(dataset, time_step):
    """
        Create subset of data for LSTM model
    """

    dataX, dataY = [], []
    
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])

    return np.array(dataX), np.array(dataY)


def LSTM_model(n_steps, feature_length=1):
    """ 
        Build LSTM model
        Inputs: input1 and 2 are input shapes
    """

    model=Sequential()
    # Adding first LSTM layer
    model.add(LSTM(150,return_sequences=True, input_shape=(n_steps, feature_length)))
    model.add(Dropout(0.2)) 
    # second LSTM layer 
    model.add(LSTM(150,return_sequences=True))
    # Adding third LSTM layer 
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding fourth LSTM layer
    model.add(LSTM(150))
    model.add(Dropout(0.2))
    # Adding the Output Layer
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    return model

def CNN_model(n_seq, n_steps, feature_length=1):
    """ 
        Build CNN-LSTM model
        Inputs: input shapes required for model
    """

    model=Sequential()
    # Adding CNN layer
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(n_seq, n_steps, feature_length))))
    model.add(Dropout(0.2)) 
    # second LSTM layer 
    model.add(LSTM(150,return_sequences=True))
    # Adding third LSTM layer 
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding fourth LSTM layer
    model.add(LSTM(150))
    model.add(Dropout(0.2))
    # Adding the Output Layer
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    return model


def Conv_model(n_seq, n_steps, feature_length):
    """
        Build Conv-LSTM model
        Inputs: input shapes required for model
    """

    model=Sequential()
    # Adding Conv layer
    model.add(ConvLSTM2D(filters=128, kernel_size=(1,4), activation='relu', input_shape=(16,1,5,1)))
    # add flatten layer
    model.add(Flatten())
    # add layer 
    model.add(Dense(200, activation='relu'))
    # add layer
    model.add(Dense(100, activation='relu'))
    # add the Output Layer
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',optimizer='adam')

    return model

def prophet_process(data, column1, column2):
    """
        Prophet model preprocessing
    """

    return data.rename(columns={column1: 'ds', column2: 'y' })
    
    


def plot_loss(history, model_name):
    """
        Plot / compare training losses
    """

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name + ' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show();

    return

def plot_predictions(time_step, scaled_data, train_predict, test_predict, scaler):
    """
        Plot Train, test and actual data
    """

    fig, ax = plt.subplots(figsize=(10,7))
    trainPredictPlot = np.empty_like(scaled_data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[time_step:len(train_predict) + time_step, :] = train_predict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(scaled_data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(time_step*2)+1:len(scaled_data)-1, :] = test_predict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(scaled_data))

    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.legend(['Data','train','test'])
    plt.xlabel('Time Steps')
    plt.ylabel('Volume kbbl_d')
    plt.title('CNN-LSTM test Predictions')
    plt.show()

    return


def predict_inv_transform(model, X_train, X_test, scaler):
    """
        Create predictions from model and transform back to original data
        Inputs: model, X_train and test, scaler model
        Output: (train_pred, test_pred)
    """

    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    return train_predict, test_predict