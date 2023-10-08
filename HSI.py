import pandas as pd
import seaborn as sns
import yfinance as yf
from datetime import datetime 
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime, timedelta


# Get Data
ticker = "^HSI"
# Download historical data
end_date = datetime.today().strftime("%Y-%m-%d")
data = yf.download(ticker, start="1997-01-01", end="2023-09-11")
data.tail()
# =============================================================================
# Return
# =============================================================================
data["PCT Cgange"] = 100*data["Adj Close"].pct_change()
data["PCT Cgange"] = data["PCT Cgange"].dropna()
sns.lineplot(data=data, x=data.index, y=data["PCT Cgange"])
plt.xlabel('Date')
plt.ylabel('Percent Change')
plt.title('Percent Change over Time')
plt.xticks(rotation=45)
plt.show()
# =============================================================================
# WAP
# =============================================================================
def calculate_vwap(data, num_days):
    vwap_values = []
    for i in range(len(data)):
        if i >= num_days - 1:
            price_slice = (data['Close'].iloc[i - num_days + 1:i + 1] + data['High'].iloc[i - num_days + 1:i + 1] + data['Low'].iloc[i - num_days + 1:i + 1])/3
            volume_slice = data['Volume'].iloc[i - num_days + 1:i + 1]
            vwap = (price_slice * volume_slice).sum() / volume_slice.sum()
            vwap_values.append(vwap)
        else:
            vwap_values.append(None)

    data['VWAP'] = vwap_values
    return data

r = data["PCT Cgange"][1:]

# Garch(3, 0)
model = arch_model(r, p=3, q=0)
model_fit = model.fit()
model_fit.summary()

rolling_predictions = []
test_size = 365*5

for i in range(test_size):
    train = r[:-(test_size-i)]
    model = arch_model(train, p=3, q=0)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

rolling_predictions = pd.Series(rolling_predictions, index=r.index[-365:])

plt.figure(figsize=(10,4))
true, = plt.plot(r[-365:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)

data["Pred_r"] = rolling_predictions
data.to_csv("Garch(3,0).csv", header = True)

# =============================================================================
# ERROR
# =============================================================================
actual_volatility = r[-365*5:]
# Calculate the mean absolute error (MAE)
forecast_error = rolling_predictions - actual_volatility
mae = np.mean(np.abs(forecast_error))

# Calculate the root mean square error (RMSE)
rmse = np.sqrt(np.mean(forecast_error**2))

# Calculate the mean absolute percentage error (MAPE)
mape = np.mean(np.abs(forecast_error / actual_volatility)) * 100

# Print the accuracy metrics
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

pred = model_fit.forecast(horizon=1)
next_day_prediction = pred.variance.values[-1:]
print("Next day prediction:", np.sqrt(next_day_prediction))

pred = model_fit.forecast(horizon=7)
future_dates = [r.index[-1] + timedelta(days=i) for i in range(1,8)]
pred = pd.Series(np.sqrt(pred.variance.values[-1,:]), index=future_dates)
plt.figure(figsize=(10,4))
plt.plot(pred)
plt.title('Volatility Prediction - Next 7 Days', fontsize=20)





















