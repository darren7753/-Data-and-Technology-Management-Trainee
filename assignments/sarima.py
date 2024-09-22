import time
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima

df = pd.read_csv("./datasets/AEP_hourly_updated.csv", index_col=0, parse_dates=True)
df = df.sort_index()
df.index.freq = "h"

df_sarima_train, df_sarima_test = train_test_split(df, test_size=0.2, shuffle=False)

start_time = time.time()

sarima = auto_arima(df_sarima_train["AEP_MW"], seasonal=True, m=24, trace=True)

end_time = time.time()
duration_sarima = end_time - start_time

print(sarima.summary())
print(duration_sarima)