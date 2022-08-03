import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

header = st.container()
dataset = st.container()
modelTrainer = st.container()

@st.cache(allow_output_mutation=True)
def make_forecast(series, df):
    """
        Makes forecast from series input
        Input: selected from dropdown list (series, df)
    """
    
    prophet_df = (df.filter(items=['Date', series]).rename(columns={'Date': 'ds', series: 'y'}))

    title = series + ' demand (thousand bbls_d)'
    
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=60, include_history=True)
    forecast = model.predict(future)

    fig = plot_plotly(model, forecast)
    fig.update_layout(title=title, yaxis_title='kbbld_d', xaxis_title='Date')

    return fig

with header:
    st.title('Time Series Analysis and Forecast:')
    


with dataset:
    st.header('Data set: EIA Energy Demand')
    df = pd.read_csv('EIA_volumes.csv', parse_dates=['Date'])
    
    
    sel_series = st.selectbox('Choose a graph to plot:', options=['Ethane', 'Propane', 'Gasoline', 'Jet Fuel', 'Crude Oil'])
    
    fig_df = df.filter(items=['Date', sel_series])
    figp = px.line(fig_df, x='Date', y=sel_series, title=sel_series + ' demand')
    st.plotly_chart(figp, use_container_width=True)

with modelTrainer:
    st.header('Five Year forecast (FBProphet):')
 
    if st.button('Start Forecast'):
        
        plotly_fig = make_forecast(sel_series, df)
        st.plotly_chart(plotly_fig)

    else:
        st.write('Click button to forecast')