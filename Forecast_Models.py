from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,GRU
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def get_model_function(model_name):
    model_functions = {
        "ARIMA": arima_model,
        "Prophet": prophet_model,
        "Exponential_Smoothing": exponential_smoothing_model,
        "XGBoost": xgboost_model,
        "Gaussian": gaussian_process_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }
    return model_functions.get(model_name, None)

##################################################
# ARIMA Model
##################################################
def arima_model(train_data, **params):
    p, d, q = params.get("p",2), params.get("d",1), params.get("q",2)
    trend = params.get("trend", "n")  
    if d > 0 and trend in ['c', 'ct']:  
        trend = 'n'  
    model = ARIMA(train_data, order=(p, d, q), trend=trend)
    fitted_model = model.fit()
    return fitted_model

##################################################
# Prophet Model
##################################################
def prophet_model(train_data, **params):
    if train_data.index.name == 'Date':
        train_data = train_data.reset_index()
    else:
        raise ValueError("The index must be named 'Date' to use Prophet.")
    if 'returns' in train_data.columns:
        target_column = 'returns'
    elif 'Close' in train_data.columns:
        target_column = 'Close'
    else:
        raise ValueError("train_data must have either a 'returns' or 'Close' column for target values.")

    df = train_data.rename(columns={'Date': 'ds', target_column: 'y'})
    seasonality_mode = params.get("seasonality_mode", "additive")
    changepoint_prior_scale = params.get("changepoint_prior_scale", 0.05)
    seasonality_prior_scale = params.get("seasonality_prior_scale", 10.0)
    yearly_seasonality = params.get("yearly_seasonality", True)
    weekly_seasonality = params.get("weekly_seasonality", True)
    
    model = Prophet(seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality)
    
    model.fit(df)
    return model

##################################################
# Exponential Smoothing Model
##################################################
def exponential_smoothing_model(train_data, **params):
    trend = params.get("trend", "additive")
    seasonal = params.get("seasonal", "additive")
    seasonal_periods = params.get("seasonal_periods", 12)
    trend = None if trend.lower() == "none" else trend
    seasonal = None if seasonal.lower() == "none" else seasonal
    model = ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fitted_model = model.fit()
    return fitted_model

##################################################
# XGBoost Model
##################################################
def xgboost_model(X_train, y_train, **params):
    n_estimators = int(params.get("n_estimators", 100))
    max_depth = int(params.get("max_depth", 3))
    learning_rate = float(params.get("learning_rate", 0.1))  
    model = XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
    model.fit(X_train, y_train)
    return model

def forecast_with_xgboost(model, last_known_data, steps, lag_features):
    predictions = []
    current_input = last_known_data[-1].reshape(1, -1)
    for _ in range(steps):
        pred = model.predict(current_input)[0]
        predictions.append(pred)
        current_input = np.roll(current_input, -1)  
        current_input[0, -1] = pred  
    return predictions

##################################################
# Gaussian Process Regressor Model
##################################################
def gaussian_process_model(X_train, y_train, **params):
    kernel_type = params.get("kernel", "RBF")
    alpha = params.get("alpha", 1e-2)
    
    # Select the kernel
    if kernel_type == "RBF":
        selected_kernel = RBF()
    elif kernel_type == "Matern":
        selected_kernel = Matern()
    elif kernel_type == "RationalQuadratic":
        selected_kernel = RationalQuadratic()
    else:
        raise ValueError(f"Unsupported kernel: {kernel_type}")
    
    model = GaussianProcessRegressor(kernel=selected_kernel, alpha=alpha)
    model.fit(X_train, y_train)
    return model

def forecast_with_gaussian(model, last_known_data, steps, lag_features, noise_std=0.0):
    predictions = []
    current_input = last_known_data[-1].reshape(1, -1)  
    for _ in range(steps):
        pred_mean = model.predict(current_input)[0]
        pred = pred_mean + np.random.normal(0, noise_std)
        predictions.append(pred)
        current_input = np.roll(current_input, -1)  
        current_input[0, -1] = pred 
    return predictions

##################################################
# LSTM
##################################################
def lstm_model(train_data, **params):
    n_steps = params.get("n_steps", 30) 
    n_units = params.get("n_units", 50)  
    dropout_rate = params.get("dropout_rate", 0.2)  
    epochs = params.get("epochs", 20)  
    batch_size = params.get("batch_size", 32)  
    optimizer = params.get("optimizer", "adam") 
    loss_function = params.get("loss_function", "mse")  
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
    X_train, y_train = prepare_lstm_features(scaled_data, n_steps)
    model = Sequential([
        LSTM(n_units, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(dropout_rate),
        LSTM(n_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)])
    model.compile(optimizer=optimizer, loss=loss_function)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, scaler, n_steps

def forecast_with_lstm(fitted_model, LSTM_scaler, scaled_test_data, LSTM_n_steps, future_steps):
    future_prediction = []  
    current_input = scaled_test_data[-LSTM_n_steps:]  
    for _ in range(future_steps):
        current_input = current_input.reshape(1, LSTM_n_steps, 1)
        next_pred_scaled = fitted_model.predict(current_input).flatten()[0]  
        next_pred = LSTM_scaler.inverse_transform([[next_pred_scaled]])[0, 0]
        future_prediction.append(next_pred)
        current_input = np.append(current_input.flatten()[1:], [next_pred_scaled]).reshape(-1, 1)
    return future_prediction


def prepare_lstm_features(data, n_steps=30):
    """Prepare features and labels for LSTM."""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

##################################################
# GRU
##################################################
def gru_model(train_data, **params):
    n_steps = params.get("n_steps", 60)
    n_units = params.get("n_units", 100)
    dropout_rate = params.get("dropout_rate", 0.3)
    epochs = params.get("epochs", 50)
    batch_size = params.get("batch_size", 32)
    optimizer_name = params.get("optimizer", "adam")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))

    X_train, y_train = create_sequences_GRU(train_scaled, n_steps)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Add feature dimension

    if optimizer_name == "adam":
        optimizer = Adam()
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop()
    elif optimizer_name == "sgd":
        optimizer = SGD()
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model = Sequential([
        GRU(n_units, return_sequences=True, dropout=dropout_rate, input_shape=(n_steps, 1)),
        GRU(n_units, return_sequences=False, dropout=dropout_rate),
        Dense(1)
    ])

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    return model, history, scaler, n_steps

def forecast_with_gru(fitted_model, GRU_scaler, scaled_test_data, GRU_n_steps, future_steps):
    future_prediction = [] 
    current_input = scaled_test_data[-GRU_n_steps:]  
    for _ in range(future_steps):
        current_input = current_input.reshape(1, GRU_n_steps, 1)
        next_pred_scaled = fitted_model.predict(current_input, verbose=0).flatten()[0]
        next_pred = GRU_scaler.inverse_transform([[next_pred_scaled]])[0, 0]
        future_prediction.append(next_pred)
        current_input = np.append(current_input.flatten()[1:], [next_pred_scaled]).reshape(-1, 1)
    return future_prediction

def create_sequences_GRU(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

############################################################################################
############################################################################################
def prepare_features(data, lags=3):
    df = pd.DataFrame({'t': data})
    for lag in range(1, lags + 1):
        df[f't-{lag}'] = df['t'].shift(lag)
    df = df.dropna()
    X = df.iloc[:, 1:].values  # Lag features
    y = df.iloc[:, 0].values  # Current value
    return np.array(X), np.array(y)

def stacking_regressor_model(X_train, y_train, **params):
    estimators = params.get("estimators", [
        ('xgb', XGBRegressor()),
        ('rf', RandomForestRegressor())
    ])
    final_estimator = params.get("final_estimator", LinearRegression())
    n_jobs = params.get("n_jobs", -1)

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    return model

