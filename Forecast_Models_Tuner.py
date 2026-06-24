from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from prophet import Prophet
from mango import Tuner
import pandas as pd
import numpy as np
import os
import Forecast_Models as forecastmodels

Optimize_folder = "Optimize"
os.makedirs(Optimize_folder, exist_ok=True)


def convert_param_ranges_to_param_space(param_ranges):
    param_space = {}
    for key, value in param_ranges.items():
        param_key = key.replace('_range', '')
        if isinstance(value[0], (int, float)) and len(value) == 3:
            if value[2] == 0:
                raise ValueError(f"Step size for parameter range '{key}' cannot be zero.")
            if isinstance(value[0], float):
                param_space[param_key] = list(np.arange(value[0], value[1] + value[2], value[2]))
            else:
                param_space[param_key] = list(range(*value))
        elif isinstance(value, list): 
            param_space[param_key] = value
        else:
            raise ValueError(f"Invalid parameter range format for key '{key}': {value}")
    return param_space


def evaluate_predictions(y_test, predictions, error_metric, params, params_evaluated, results):
    if error_metric == "MSE":
        error = mean_squared_error(y_test, predictions)
    elif error_metric == "MAE":
        error = mean_absolute_error(y_test, predictions)
    else:
        raise ValueError("Unsupported error metric.")
    params_evaluated.append(params)
    results.append(error)
    return params_evaluated, results

def arima_tuner_mango(train_data, test_data, param_ranges, error_metric="MSE", output_file=None):
    def tuner_function(args_list):
        params_evaluated = []
        results = []
        for params in args_list:            
            try:
                p, d, q = int(params["p"]), int(params["d"]), int(params["q"])
                trend = params["trend"]
                model = ARIMA(train_data, order=(p, d, q), trend=trend)
                fitted_model = model.fit()
                predictions = fitted_model.forecast(len(test_data))
                predictions.index = test_data.index
                params_evaluated, results = evaluate_predictions(y_test=test_data,predictions=predictions,
                            error_metric=error_metric,params=params,params_evaluated=params_evaluated,results=results)
            except Exception as e:
                print(f"Error during tuning with parameters {params}: {e}")
                params_evaluated.append(params)
                results.append(1e5)
        return params_evaluated, results
    mango_param_dict = convert_param_ranges_to_param_space(param_ranges)
    conf_Dict = dict()
    conf_Dict['num_iteration'] = 100
    tuner = Tuner(mango_param_dict, tuner_function, conf_Dict)
    results = tuner.minimize()
    print('best parameters:', results['best_params'])
    print('best loss:', results['best_objective'])
    best_params = results['best_params']
    best_loss = results['best_objective']
    results_df = pd.DataFrame({
    'p': [best_params['p']],
        'd': [best_params['d']],
        'q': [best_params['q']],
        'trend': [best_params['trend']],
        'best_loss': [best_loss]})
    if output_file: results_df.to_csv(os.path.join(Optimize_folder, output_file), index=False)
    return best_params, best_loss, results_df

def prophet_tuner_mango(train_data, test_data, param_ranges, error_metric="MSE", output_file=None):
    def tuner_function(args_list):
        params_evaluated = []
        results = []
        for params in args_list:
            try:
                seasonality_mode = params["seasonality_mode"]
                changepoint_prior_scale = params["changepoint_prior_scale"]
                seasonality_prior_scale = params["seasonality_prior_scale"]
                yearly_seasonality = params.get("yearly_seasonality", True)
                weekly_seasonality = params.get("weekly_seasonality", True)
                model = Prophet(seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality)
                model.fit(train_data)
                future = model.make_future_dataframe(periods=len(test_data))
                forecast = model.predict(future)
                predictions = forecast.loc[forecast['ds'].isin(test_data['ds']), 'yhat']
                params_evaluated, results = evaluate_predictions(y_test=test_data,predictions=predictions,
                            error_metric=error_metric,params=params,params_evaluated=params_evaluated,results=results)
            except Exception as e:
                print(f"Error during tuning with parameters {params}: {e}")
                params_evaluated.append(params)
                results.append(1e5)  
        return params_evaluated, results
    mango_param_dict = convert_param_ranges_to_param_space(param_ranges)
    conf_Dict = {'num_iteration': 100}
    tuner = Tuner(mango_param_dict, tuner_function, conf_Dict)
    results = tuner.minimize()
    print('Best parameters:', results['best_params'])
    print('Best loss:', results['best_objective'])
    best_params = results['best_params']
    best_loss = results['best_objective']
    results_df = pd.DataFrame({
        'seasonality_mode': [best_params['seasonality_mode']],
        'changepoint_prior_scale': [best_params['changepoint_prior_scale']],
        'seasonality_prior_scale': [best_params['seasonality_prior_scale']],
        'yearly_seasonality': [best_params.get('yearly_seasonality', True)],
        'weekly_seasonality': [best_params.get('weekly_seasonality', True)],
        'best_loss': [best_loss]})
    if output_file:
        results_df.to_csv(os.path.join(Optimize_folder, output_file), index=False)
    return best_params, best_loss, results_df

def exponential_smoothing_tuner_mango(train_data, test_data, param_ranges, error_metric="MSE", output_file=None):
    def tuner_function(args_list):
        params_evaluated = []
        results = []
        for params in args_list:
            try:
                trend = params["trend"]
                seasonal = params["seasonal"]
                seasonal_periods = params["seasonal_periods"]
                model = ExponentialSmoothing(train_data,
                    trend=trend if trend != "none" else None,
                    seasonal=seasonal if seasonal != "none" else None,
                    seasonal_periods=seasonal_periods).fit()
                predictions = model.forecast(len(test_data))
                params_evaluated, results = evaluate_predictions(y_test=test_data,predictions=predictions,
                            error_metric=error_metric,params=params,params_evaluated=params_evaluated,results=results)
            except Exception as e:
                print(f"Error during tuning with parameters {params}: {e}")
                params_evaluated.append(params)
                results.append(1e5)  
        return params_evaluated, results
    mango_param_dict = convert_param_ranges_to_param_space(param_ranges)
    conf_Dict = {'num_iteration': 100}
    tuner = Tuner(mango_param_dict, tuner_function, conf_Dict)
    results = tuner.minimize()
    print('Best parameters:', results['best_params'])
    print('Best loss:', results['best_objective'])
    best_params = results['best_params']
    best_loss = results['best_objective']
    results_df = pd.DataFrame({
        'trend': [best_params['trend']],
        'seasonal': [best_params['seasonal']],
        'seasonal_periods': [best_params['seasonal_periods']],
        'best_loss': [best_loss]})
    if output_file: results_df.to_csv(os.path.join(Optimize_folder, output_file), index=False)
    return best_params, best_loss, results_df

def xgboost_tuner_mango(X_train, y_train,X_test, y_test, param_ranges, error_metric="MSE", output_file=None):
    def tuner_function(args_list):
        params_evaluated = []
        results = []
        for params in args_list:
            try:
                n_estimators = params["n_estimators"]
                max_depth = params["max_depth"]
                learning_rate = params["learning_rate"]
                model = XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
                model.fit(X_train,y_train)
                predictions = model.predict(X_test)
                params_evaluated, results = evaluate_predictions(y_test=y_test,predictions=predictions,
                        error_metric=error_metric,params=params,params_evaluated=params_evaluated,results=results)
            except Exception as e:
                print(f"Error during tuning with parameters {params}: {e}")
                params_evaluated.append(params)
                results.append(1e5)  
        return params_evaluated, results
    mango_param_dict = convert_param_ranges_to_param_space(param_ranges)
    conf_Dict = {'num_iteration': 100}
    tuner = Tuner(mango_param_dict, tuner_function, conf_Dict)
    results = tuner.minimize()
    print('Best parameters:', results['best_params'])
    print('Best loss:', results['best_objective'])
    best_params = results['best_params']
    best_loss = results['best_objective']
    results_df = pd.DataFrame({
        'n_estimators': [best_params['n_estimators']],
        'max_depth': [best_params['max_depth']],
        'learning_rate': [best_params['learning_rate']],
        'best_loss': [best_loss]})
    if output_file: results_df.to_csv(os.path.join(Optimize_folder, output_file), index=False)
    return best_params, best_loss, results_df

def gaussian_tuner_mango(X_train, y_train, X_test, y_test, param_ranges, error_metric="MSE", output_file=None):
    def tuner_function(args_list):
        params_evaluated = []
        results = []
        for params in args_list:
            try:
                kernel = params["kernel"]
                alpha = params["alpha"]
                if kernel == "RBF":
                    kernel_obj = RBF()
                elif kernel == "Matern":
                    kernel_obj = Matern()
                elif kernel == "RationalQuadratic":
                    kernel_obj = RationalQuadratic()
                else:
                    raise ValueError(f"Unsupported kernel: {kernel}")
                model = GaussianProcessRegressor(kernel=kernel_obj,alpha=alpha)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                params_evaluated, results = evaluate_predictions(y_test=y_test,predictions=predictions,
                        error_metric=error_metric,params=params,params_evaluated=params_evaluated,results=results)
            except Exception as e:
                print(f"Error during tuning with parameters {params}: {e}")
                params_evaluated.append(params)
                results.append(1e5)  
        return params_evaluated, results
    mango_param_dict = convert_param_ranges_to_param_space(param_ranges)
    conf_Dict = {'num_iteration': 100}
    tuner = Tuner(mango_param_dict, tuner_function, conf_Dict)
    results = tuner.minimize()
    print('Best parameters:', results['best_params'])
    print('Best loss:', results['best_objective'])
    best_params = results['best_params']
    best_loss = results['best_objective']
    results_df = pd.DataFrame({
        'kernel': [best_params['kernel']],
        'alpha': [best_params['alpha']],
        'best_loss': [best_loss]})
    if output_file: results_df.to_csv(os.path.join(Optimize_folder, output_file), index=False)
    return best_params, best_loss, results_df

def lstm_tuner_mango(train_data, test_data, param_ranges, error_metric="MSE", output_file=None):
    def tuner_function(args_list):
        params_evaluated = []
        results = []
        for params in args_list:
            try:
                n_steps = int(params["n_steps"])
                n_units = int(params["n_units"])
                dropout_rate = float(params["dropout_rate"])
                epochs = int(params["epochs"])
                batch_size = int(params["batch_size"])
                optimizer = params["optimizer"]
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
                X_train, y_train = forecastmodels.prepare_lstm_features(scaled_train_data, n_steps)
                model = Sequential([
                    LSTM(n_units, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
                    Dropout(dropout_rate),
                    LSTM(n_units, activation='relu'),
                    Dropout(dropout_rate),
                    Dense(1)
                ])
                model.compile(optimizer=optimizer, loss=error_metric)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                scaled_test_data = scaler.transform(test_data.values.reshape(-1, 1))
                X_test, y_test = forecastmodels.prepare_lstm_features(scaled_test_data, n_steps)
                predictions = model.predict(X_test).flatten()
                predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
                error = evaluate_predictions(y_test=test_data.iloc[n_steps:], predictions=predictions.flatten(),
                                             error_metric=error_metric)
                params_evaluated.append(params)
                results.append(error)
            except Exception as e:
                print(f"Error during tuning with parameters {params}: {e}")
                params_evaluated.append(params)
                results.append(1e5)
        return params_evaluated, results
    mango_param_dict = convert_param_ranges_to_param_space(param_ranges)
    conf_Dict = dict()
    conf_Dict['num_iteration'] = 100
    tuner = Tuner(mango_param_dict, tuner_function, conf_Dict)
    results = tuner.minimize()
    print('best parameters:', results['best_params'])
    print('best loss:', results['best_objective'])
    best_params = results['best_params']
    best_loss = results['best_objective']
    results_df = pd.DataFrame({
        'n_steps': [best_params['n_steps']],
        'n_units': [best_params['n_units']],
        'dropout_rate': [best_params['dropout_rate']],
        'epochs': [best_params['epochs']],
        'batch_size': [best_params['batch_size']],
        'optimizer': [best_params['optimizer']],
        'best_loss': [best_loss]})
    if output_file:results_df.to_csv(os.path.join(Optimize_folder, output_file), index=False)
    return best_params, best_loss, results_df

def gru_tuner_mango(train_data, test_data, param_ranges, error_metric="MSE", output_file=None):
    def tuner_function(args_list):
        params_evaluated = []
        results = []
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
        test_scaled = scaler.transform(test_data.reshape(-1, 1))
        for params in args_list:
            try:
                n_steps = params["n_steps"]
                n_units = params["n_units"]
                dropout_rate = params["dropout_rate"]
                epochs = params.get("epochs", 10)
                batch_size = params.get("batch_size", 32)
                optimizer_name = params["optimizer"]
                
                if optimizer_name == "adam":
                    optimizer = Adam()
                elif optimizer_name == "rmsprop":
                    optimizer = RMSprop()
                elif optimizer_name == "sgd":
                    optimizer = SGD()
                else:
                    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

                X_train, y_train = forecastmodels.create_sequences_GRU(train_scaled, n_steps)
                X_test, y_test = forecastmodels.create_sequences_GRU(test_scaled, n_steps)

                model = Sequential([
                    GRU(n_units, return_sequences=True, dropout=dropout_rate, input_shape=(n_steps, 1)),
                    GRU(n_units, return_sequences=False, dropout=dropout_rate),
                    Dense(1)])
                
                model.compile(optimizer=optimizer, loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                y_test_actual = scaler.inverse_transform(y_test)
                
                if error_metric == "MSE":
                    loss = np.mean((predictions - y_test_actual) ** 2)
                elif error_metric == "MAE":
                    loss = np.mean(np.abs(predictions - y_test_actual))
                else:
                    raise ValueError(f"Unsupported error metric: {error_metric}")

                params_evaluated.append(params)
                results.append(loss)

            except Exception as e:
                print(f"Error during tuning with parameters {params}: {e}")
                params_evaluated.append(params)
                results.append(1e5)

        return params_evaluated, results

    mango_param_dict = {
        "n_steps": param_ranges["n_steps_range"],
        "n_units": param_ranges["n_units_range"],
        "dropout_rate": param_ranges["dropout_rate_range"],
        "epochs": param_ranges["epochs_range"],
        "batch_size": param_ranges["batch_size_range"],
        "optimizer": param_ranges["optimizer_range"]}

    conf_dict = {"num_iteration": 100}
    tuner = Tuner(mango_param_dict, tuner_function, conf_dict)
    results = tuner.minimize()
    best_params = results["best_params"]
    best_loss = results["best_objective"]
    results_df = pd.DataFrame([best_params])
    results_df["best_loss"] = best_loss
    if output_file: results_df.to_csv(os.path.join(Optimize_folder, output_file), index=False)
    return best_params, best_loss, results_df
