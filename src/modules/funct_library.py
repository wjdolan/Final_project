import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
import requests
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs




def clf_threshold(y_predict, y_prob, y_test):
    """
    Returns plot of precision / recall vs threshold
    
        Parameters:
            y_predict(float):  y_predict from fitted model
            y_prob(float):     y_probability from fitted model (predict_proba)
            y_test(array):     target values from test set
            
        Output:
            Plot      
    
    """

    precision, recall, thresholds = precision_recall_curve(y_test, probs_y[:, 1]) 

    pr_auc = metrics.auc(recall, precision)

    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0,1])

    return


def top_categories(X, perc):
    """
        Returns dominant groups in category by %
    
        Parameters:
            X (list):      column of values
            perc (float):  threshold for cutoff (percentage)
            
        Returns:
            List of top groups
    
    """
    
    val_list = []
    key_list = []
    denom = sum(X.values())
    
    for count, (keys, values) in enumerate(X.items()):
        if (sum(val_list)/denom) < perc:
            key_list.append(keys)
            val_list.append(values)
            
    return val_list, key_list


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
        if name == "jetfuel":
            df = df.append({'Date': df_date, 'Volume_kbbld_jet': df_value}, ignore_index=True)

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