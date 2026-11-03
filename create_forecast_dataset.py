
import numpy as np
import pandas as pd

np.random.seed(42)

dates = pd.date_range(
    start="2021-01-01",
    periods=60,
    freq="MS"
)

trend = np.linspace(1000, 1600, 60)

seasonality = (
    100 * np.sin(
        2 * np.pi * np.arange(60) / 12
    )
)

noise = np.random.normal(
    loc=0,
    scale=25,
    size=60
)

sales = trend + seasonality + noise

df = pd.DataFrame({
    "Date": dates,
    "Sales": sales.round(2)
})

df.to_csv(
    "data/sample_datasets/monthly_sales_forecast.csv",
    index=False
)

print("Sample forecasting dataset created!")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nLast five records:")
print(df.tail())
