import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Loading_Data_BN import fetch_historical_data
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import Forecast_Plot as forecastplot
import Forecast_Models as forecastmodels
import Forecast_Models_Tuner as forecastmodelstuner
import logging
import warnings
import os

plots_folder = "Plots"
os.makedirs(plots_folder, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class ForecastTesting:
    def __init__(self, client, symbol, bar_length, start, end=None, tc=0.0, leverage=5, strategy="PV", future_forecast_steps = 10):
        self.client = client
        self.symbol = str(symbol)
        self.start = str(start)
        self.end = str(end) if end else None
        self.tc = tc
        self.leverage = leverage
        self.strategy = strategy
        self.data = None
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h","6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.Error_MSE = True
        self.Error_MAE = False
        self.future_forecast_steps = future_forecast_steps
        try:
            self.data = self.get_data()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    def get_data(self):
        try:
            data = fetch_historical_data(client=self.client, symbol=self.symbol,
                bar_length=self.bar_length, start=self.start, end=self.end)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise

    def Forecast_price(self, ForecastModelName, parameters, param_ranges, Perform_Tuner=False):
        data = self.data.copy()
        print('\n',data.columns)
        print(data.head())

        #time_series = data['returns'].replace([np.inf, -np.inf], np.nan).dropna()
        time_series = data['Close'].replace([np.inf, -np.inf], np.nan).dropna()

        train_size = int(len(time_series) * 4 / 5)
        train_data = time_series.iloc[:train_size]
        test_data = time_series.iloc[train_size:]

        print(f"\ntrain_data shape: {train_data.shape}, type: {type(train_data)}")
        print(f"test_data shape: {test_data.shape}, type: {type(test_data)}")        
        print(f"Train Data Start: {train_data.index[0]}, Train Data End: {train_data.index[-1]}")
        print(f"Test Data Start: {test_data.index[0]}, Test Data End: {test_data.index[-1]}")

        # Prepare features and labels for ML models
        X_train, y_train = forecastmodels.prepare_features(train_data)
        X_test, y_test = forecastmodels.prepare_features(test_data)

        print(f"\nX_train shape: {X_train.shape}, type: {type(X_train)}")
        print(f"X_test shape: {X_test.shape}, type: {type(X_test)}")
        print(f"y_train shape: {y_train.shape}, type: {type(y_train)}")
        print(f"y_test shape: {y_test.shape}, type: {type(y_test)}\n")

        # Plotting and decomposition
        forecastplot.test_stationarity(time_series)
        forecastplot.plot_return(time_series, symbol=self.symbol)
        forecastplot.decompose_time_series(time_series, symbol=self.symbol)

        # Retrieve the model function
        model_function = forecastmodels.get_model_function(ForecastModelName)
        if not model_function:
            raise ValueError(f"Unsupported model: {ForecastModelName}")

        best_params = parameters
        if Perform_Tuner:
            print("\nTuner is active.\n")

            try:
                if ForecastModelName == 'ARIMA' : 
                    best_params, best_loss, results_df = forecastmodelstuner.arima_tuner_mango(
                        train_data=train_data,test_data=test_data,
                        param_ranges=param_ranges,error_metric="MSE",
                        output_file=f"{ForecastModelName}_tuning_results.csv")
                    
                elif ForecastModelName == 'Prophet':
                    best_params, best_loss, results_df = forecastmodelstuner.prophet_tuner_mango(
                        train_data=train_data,test_data=test_data,
                        param_ranges=param_ranges,error_metric="MSE",
                        output_file=f"{ForecastModelName}_tuning_results.csv")

                elif ForecastModelName == 'Exponential_Smoothing':
                    best_params, best_loss, results_df = forecastmodelstuner.exponential_smoothing_tuner_mango(
                        train_data=train_data,test_data=test_data,
                        param_ranges=param_ranges,error_metric="MSE",
                        output_file=f"{ForecastModelName}_tuning_results.csv")

                elif ForecastModelName == 'XGBoost':
                    best_params, best_loss, results_df = forecastmodelstuner.xgboost_tuner_mango(
                        X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test,
                        param_ranges=param_ranges,error_metric="MSE",
                        output_file=f"{ForecastModelName}_tuning_results.csv")
                    
                elif ForecastModelName == 'Gaussian':
                    best_params, best_loss, results_df = forecastmodelstuner.gaussian_tuner_mango(
                        X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test,
                        param_ranges=param_ranges,error_metric="MSE",
                        output_file=f"{ForecastModelName}_tuning_results.csv")
                    
                elif ForecastModelName == 'LSTM':
                    best_params, best_loss, results_df = forecastmodelstuner.lstm_tuner_mango(
                        train_data=train_data,test_data=test_data,
                        param_ranges=param_ranges,error_metric="MSE",
                        output_file=f"{ForecastModelName}_tuning_results.csv")  
                    
                elif ForecastModelName == 'GRU':
                    best_params, best_loss, results_df = forecastmodelstuner.gru_tuner_mango(
                        train_data=train_data,test_data=test_data,
                        param_ranges=param_ranges,error_metric="MSE",
                        output_file=f"{ForecastModelName}_tuning_results.csv")                    
          
                print(f"Best parameters for {ForecastModelName}: {best_params}")
                print(f"Best Loss {ForecastModelName}: {best_loss}")
                print(results_df.head())
            except Exception as e:
                print(f"Error during tuning: {e}")
                raise

            try:
                best_params_df = pd.read_csv(os.path.join(plots_folder,f"{ForecastModelName}_tuning_results.csv"))
                best_params = best_params_df.iloc[0].to_dict()  # Convert first row to dictionary
                print(f"Loaded best parameters: {best_params}")
            except Exception as e:
                print(f"Failed to read parameters from {ForecastModelName}_tuning_results.csv: {e}")
                raise

        # Ensure numeric parameters are properly converted
        for key in best_params:
            try:
                if isinstance(best_params[key], str) and best_params[key].isdigit():
                    best_params[key] = int(best_params[key])
            except AttributeError:
                pass

        if ForecastModelName in ['ARIMA', 'Prophet', 'Exponential_Smoothing', 'LSTM','GRU']:
            forecast_steps = len(test_data)
        elif ForecastModelName in ['XGBoost', 'Gaussian']:
            forecast_steps = len(X_test)

        # Evaluate the model
        forecastplot.forecast_and_evaluate(ForecastModelName=ForecastModelName,
                                           train_data=train_data,test_data=test_data,X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test,
                                           model_function=model_function,bar_length =self.bar_length,symbol=self.symbol,
                                           forecast_steps=forecast_steps, future_forecast_steps=self.future_forecast_steps,**best_params)