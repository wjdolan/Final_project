import pandas as pd
import numpy as np
import requests
import os
import json
import psycopg2 as ps
from Final_project.src.modules.funct_library import request_to_df


# import time series data

df_gas = pd.DataFrame(columns=["Date", "Volume_kbbld_gas"])

df_jet = pd.DataFrame(columns=["Date", "Volume_kbbld_jet"])


root_URL = "https://api.eia.gov/series/?api_key=" 

gasoline_series = "&series_id=PET.MGFUPUS2.M"

jetfuel_series = "&series_id=PET.MKJUPUS2.M"


request_to_df(root_URL, gasoline_series, "gasoline")

request_to_df(root_URL, gasoline_series, "jetfuel")



df_gasoline['Date'] = pd.to_datetime(df['Date'], format='%Y%m')


request_to_df(root_URL, jetfuel_series)









