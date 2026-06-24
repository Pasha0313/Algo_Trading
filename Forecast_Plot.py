import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Forecast_Models as forecastmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error , mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import os
Plot_folder = "Plots"
os.makedirs(Plot_folder, exist_ok=True)
Optimize_folder = "Optimize"
os.makedirs(Optimize_folder, exist_ok=True)
def decompose_time_series(time_series,symbol):
    decomposition = seasonal_decompose(time_series, model='additive', period=30)  #7 or 30
    
    plt.figure(figsize=(8, 6))
    
    plt.subplot(411)
    plt.plot(decomposition.observed, label='Observed')
    plt.title(f'{symbol}: Observed Time Series')
    plt.legend(loc='best')
    plt.xticks(rotation=30)

    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend', color='r')
    plt.title(f'{symbol}: Trend')
    plt.legend(loc='best')
    plt.xticks(rotation=30)

    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonality', color='g')
    plt.title(f'{symbol}: Seasonality')
    plt.legend(loc='best')
    plt.xticks(rotation=30)

    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals (Noise)', color='b')
    plt.title(f'{symbol}: Residuals (Noise)')
    plt.legend(loc='best')
    plt.xticks(rotation=30)

    plt.tight_layout()
    #plt.savefig(os.path.join("Plots", f'Decompose_the_Return_time_series{symbol}.png'))
    plt.savefig(os.path.join(Plot_folder, f'Decompose_the_Price_time_series{symbol}.png'))
    plt.show()

def test_stationarity(time_series):
    adf_result = adfuller(time_series)
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])

def plot_return(time_series,symbol):
    skip = int(len(time_series)/50)
    plt.figure(figsize=(8, 6))
    plt.plot(time_series.index, time_series , color='red', linestyle='-', linewidth=1, alpha=0.8)  
    plt.plot(time_series.index[::skip], time_series[::skip] , linestyle='' , marker='o', markersize=7, markerfacecolor='blue', alpha=0.8)      
    plt.title(f'Crypto: {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=30)  
    plt.legend()
    plt.grid()
    #plt.savefig(f'Return_{symbol}.png')
    plt.savefig(os.path.join(Plot_folder, f'Return_{symbol}.png'))
    plt.show()

def parse_interval(interval):
    # Mapping for frequency strings
    freq_map = {
        "m": "T",  # Minutes
        "h": "H",  # Hours
        "d": "D",  # Days
        "w": "W",  # Weeks
    }

    # Mapping for Timedelta arguments
    timedelta_map = {
        "m": "minutes",
        "h": "hours",
        "d": "days",
        "w": "weeks",}

    # Extract the unit and value
    unit = interval[-1]
    value = int(interval[:-1])

    if unit in freq_map and unit in timedelta_map:
        freq = f"{value}{freq_map[unit]}"
        timedelta = pd.Timedelta(**{timedelta_map[unit]: value})
        print(f'\n freq : {freq}, timedelta : {timedelta}')
        return freq, timedelta, unit
    else:
        raise ValueError(f"Unsupported interval: {interval}")


def forecast_and_evaluate(ForecastModelName, train_data, test_data, X_train, y_train, X_test, y_test,
                          model_function,bar_length, symbol, forecast_steps, future_forecast_steps =10, **model_params):
    
    print (f'\n Forecast & Evaluate with : {ForecastModelName}')
    freq, time_delta,unit = parse_interval(bar_length)
    start_date = test_data.index[0] + time_delta

    predictions_index = pd.date_range(start=start_date, periods=forecast_steps+future_forecast_steps, freq=freq)
    print (f'\n Start : {start_date} , End : {predictions_index[-1]}')
    print (f'\n forecast_steps : {forecast_steps} , future_forecast_steps : {future_forecast_steps} , Total : {forecast_steps+future_forecast_steps}')

    # Fit the model with given parameters
    if ForecastModelName in ['ARIMA', 'Prophet', 'Exponential_Smoothing']:
        fitted_model = model_function(train_data, **model_params)
    elif ForecastModelName in ['XGBoost', 'Gaussian']:
        fitted_model = model_function(X_train, y_train, **model_params)
    elif ForecastModelName in ['LSTM']:   
        fitted_model, LSTM_scaler, LSTM_n_steps = model_function(train_data, **model_params)
    elif ForecastModelName in ['GRU']:
        fitted_model, GRU_history, GRU_scaler, GRU_n_steps= model_function(train_data, **model_params)
    else:
        raise ValueError(f"Unsupported model: {ForecastModelName}")

    if ForecastModelName == 'ARIMA':
        predictions = fitted_model.forecast(steps=forecast_steps+future_forecast_steps)
    
    elif ForecastModelName == 'Prophet':
        future = fitted_model.make_future_dataframe(periods=forecast_steps+future_forecast_steps, freq=freq)
        forecast = fitted_model.predict(future)
        predictions = forecast['yhat'][-(forecast_steps+future_forecast_steps):].values

    elif ForecastModelName == 'Exponential_Smoothing':
        predictions = fitted_model.forecast(steps=forecast_steps+future_forecast_steps)

    elif ForecastModelName == 'XGBoost':
        predictions = fitted_model.predict(X_test)
        last_known_data = X_test[-1:]
        future_prediction = forecastmodels.forecast_with_xgboost(fitted_model, last_known_data, steps=future_forecast_steps,
                                                                  lag_features=X_train.shape[1])
    elif ForecastModelName == 'Gaussian':
        predictions = fitted_model.predict(X_test)
        last_known_data = X_test[-1:]
        future_prediction = forecastmodels.forecast_with_gaussian(model=fitted_model,last_known_data=last_known_data,
                                                        steps=future_forecast_steps,lag_features=X_train.shape[1],noise_std=0.1)
    elif ForecastModelName == 'LSTM':
        scaled_test_data = LSTM_scaler.transform(test_data.values.reshape(-1, 1))
        X_test = []
        for i in range(len(scaled_test_data) - LSTM_n_steps):
            X_test.append(scaled_test_data[i:i + LSTM_n_steps])
        X_test = np.array(X_test)
        lstm_predictions = fitted_model.predict(X_test).flatten()
        predictions = LSTM_scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()
        future_prediction = forecastmodels.forecast_with_lstm(fitted_model=fitted_model,LSTM_scaler=LSTM_scaler,
                                scaled_test_data=scaled_test_data,LSTM_n_steps=LSTM_n_steps,future_steps=future_forecast_steps)
    elif ForecastModelName == 'GRU':
        scaled_test_data = GRU_scaler.transform(test_data.values.reshape(-1, 1))
        X_test = []
        for i in range(len(scaled_test_data) - GRU_n_steps):
            X_test.append(scaled_test_data[i:i + GRU_n_steps])
        X_test = np.array(X_test)
        gru_predictions = fitted_model.predict(X_test).flatten()
        predictions = GRU_scaler.inverse_transform(gru_predictions.reshape(-1, 1)).flatten()
        future_prediction = forecastmodels.forecast_with_gru(fitted_model=fitted_model,
            GRU_scaler=GRU_scaler,scaled_test_data=scaled_test_data,GRU_n_steps=GRU_n_steps,future_steps=future_forecast_steps)        
    else:
        raise ValueError(f"Unsupported model for tuning: {ForecastModelName}")

    if ForecastModelName in ['ARIMA', 'Prophet', 'Exponential_Smoothing']:
        print(f"\npredictions shape: {predictions.shape}, type: {type(predictions)}")
        print(f"\predictions_index shape: {predictions_index.shape}, type: {type(predictions_index)}")
        print(predictions)

        predictions = pd.Series(predictions, index=predictions_index)   
    elif ForecastModelName in ['XGBoost', 'Gaussian','LSTM','GRU']:
        predictions = np.array(predictions)
        future_prediction = np.array(future_prediction)
        combined_predictions = np.concatenate((predictions, future_prediction))
        predictions_index = pd.date_range(start=start_date, periods=len(combined_predictions), freq=freq)
        predictions = pd.Series(combined_predictions, index=predictions_index)

    # Forecast
    print(f"Forecasting with {ForecastModelName}...")
    
    # Error Estimations
    predictions_series = pd.Series(predictions, index=test_data.index)
    common_index = test_data.index.intersection(predictions_series.index)
    aligned_actual = test_data.loc[common_index]
    aligned_predicted = predictions_series.loc[common_index]
    comparison_df = pd.DataFrame({'Actual': aligned_actual,'Predicted': aligned_predicted}).dropna()
    print("\nPredictions vs. Actuals:")
    print(comparison_df)
    csv_file_path = os.path.join(Optimize_folder, f'Actual_vs_Predicted_{ForecastModelName}_{symbol}.csv')
    comparison_df.to_csv(csv_file_path, index=True)
    print(f"Predictions saved to {csv_file_path}")
    mae = mean_absolute_error(comparison_df['Actual'], comparison_df['Predicted'])
    mse = mean_squared_error(comparison_df['Actual'], comparison_df['Predicted'])
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

    # Plot actual vs. predicted
    plt.figure(figsize=(8, 6)) 
    plt.plot(test_data.index, test_data, 
        label='Actual', color='blue', marker='o', markersize=7, markerfacecolor='blue', linestyle='-', linewidth=1, alpha=0.8)
    plt.plot(predictions.index, predictions, 
        label='Predicted', color='red', marker='s', markersize=7, markerfacecolor='red', linestyle='--', linewidth=1, alpha=0.8)
    plt.title(f'Actual vs Predicted with {ForecastModelName} ({symbol})', fontsize=16, fontweight='bold')
    plt.axvline(x=test_data.index[-1], color='green', linestyle='--', label='Test Data End')    
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12, loc='upper left', frameon=True)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()  
    plt.savefig(os.path.join(Plot_folder, f'Price_Prediction_{ForecastModelName}_{symbol}.png'))
    plt.show()

    return 


