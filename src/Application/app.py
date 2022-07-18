import streamlit as st
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

header = st.container()
dataset = st.container()
modelTrainer = st.container()

@st.cache()
def make_forecast(series):
    """
        Makes forecast from series input
        Input: selected from dropdown list (series)
    """
    if series == 'Ethane':
        series_name = 'Ethane'
        title = 'Ethane demand'
        x_label = 'Date'

    prophet_df = (df[series])


    plotly_fig = make_forecast(sel_series)
    st.plotly_chart(plotly_fig)

    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=36)
    forecast = model.predict(future)

    fig = plot_plotly(model, forecast)
    fig.update_layout(title=title, yaxis_title=label, xaxis_title='Date')

    return fig

with header:
    st.title('LighhouseLabs Final Project: Time Series Anlaysis')
    st.text('This is my project on time series analysis')


with dataset:
    st.header('EIA Energy Consumption')
    df = pd.read_csv('EIA volumes.csv', parse_dates=['Month'], index_col=['Month'])
    # st.write(df.head())
    st.line_chart(df, use_container_width=True)

with modelTrainer:
    st.header('Series forecasting by FBProphet')

    sel_series = st.selectbox('Choose a graph to forecast:', options=['Ethane', 'Propane', 'Gasoline', 'Jet Fuel'])

    