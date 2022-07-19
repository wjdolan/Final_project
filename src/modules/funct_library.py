import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from regex import E
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM




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

def request_to_df(URL, url_series, df, name):
    API_KEY = str(os.environ.get('EIA_API_KEY'))
    r_url = URL + API_KEY + url_series
    #r_url2 = URL + API_KEY + url_series2
    response = requests.get(r_url)
    # response2 = requests.get(r_url2)

    for value in response.json()['series'][0]['data']:
        df_date = value[0]
        df_value = value[1]

        if name == "gasoline":
            df = df.append({'Date': df_date, 'Volume_kbbld_gas': df_value}, ignore_index=True)
       
        return df


def ADF_Stationarity_Test(timeseries, significance_level=0.05):
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')

    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        
    if result[1] > significance_level:
        result_str = 'NON-stationary timeseries'
    else:
        result_str = 'Stationary timeseries'
        
    return result_str


def differencing(series):
  """
    Difference time series and view autocorrelation

  """
  # Original series
  fig, axes = plt.subplots(3, 2, sharex=False)

  axes[0, 0].plot(series); 
  axes[0, 0].set_title('Original Series')
  plot_acf(series, ax=axes[0, 1])

  # 1st Differencing
  axes[1, 0].plot(series.diff()); 
  axes[1, 0].set_title('1st Order Differencing')
  plot_acf(series.diff().dropna(), ax=axes[1, 1])

  # 2nd Differencing
  axes[2, 0].plot(series.diff().diff()); 
  axes[2, 0].set_title('2nd Order Differencing')
  plot_acf(series.diff().diff().dropna(), ax=axes[2, 1])

  fig.tight_layout()

  # fig.savefig('gas_differencing.png')

  plt.show()


def n_diffs(series):
  """ 
    Returns multiple estimates of differencing order
  """
  ## Adf Test
  adf = ndiffs(series, test='adf')  

  # KPSS test
  kpss = ndiffs(series, test='kpss')  

  # PP test:
  pp = ndiffs(series, test='pp')  

  print(f'ADF test: {adf}\n KPSS test: {kpss}\n PP test: {pp}')


def graph_pacf(series):
  """
    Plot PACF of 1st differenced series
  """
  plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
  fig, axes = plt.subplots(1, 2, sharex=False)

  axes[0].plot(series.diff())
  axes[0].set_title('1st Differencing')
  axes[1].set(ylim=(0,5))

  plot_pacf(series.diff().dropna(), ax=axes[1])

#   fig.savefig('gas_pacf_p.png')

  plt.show();


  def graph_acf(series):
    """
    Plot ACF of 1st differenced series
    """
    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
    fig, axes = plt.subplots(1, 2, sharex=False)

    axes[0].plot(series.diff())
    axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0,5))

    plot_acf(series.diff().dropna(), ax=axes[1])

    # fig.savefig('gas_acf_q.png')

    plt.show();



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
                Volume_gas_kbbld INTEGER NOT NULL)""")

    curr.execute(create_table_command)


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


def LSTM_model(input1, input2):
    """ 
        Build LSTM model
        Inputs: input1 and 2 are input shapes
    """

    model=Sequential()
    # Adding first LSTM layer
    model.add(LSTM(150,return_sequences=True,input_shape=(20,1)))
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
