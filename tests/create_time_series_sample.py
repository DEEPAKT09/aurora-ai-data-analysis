
import numpy as np
import pandas as pd

np.random.seed(42)

dates = pd.date_range(
    start="2021-01-01",
    periods=60,
    freq="MS"
)

trend = np.arange(60) * 10

seasonality = 100 * np.sin(
    2 * np.pi * np.arange(60) / 12
)

noise = np.random.normal(0, 20, 60)

sales = 1000 + trend + seasonality + noise

df = pd.DataFrame({
    "Date": dates,
    "Sales": sales
})

df.to_csv(
    "data/sample_datasets/time_series_sample.csv",
    index=False
)

print("Sample time-series dataset created!")
print(df.head())
