from datetime import datetime, timedelta
from tabulate import tabulate
import math

def load_config_from_text(filename):
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip()
    return config

def get_control_settings(config):
    # Extract control settings
    Optimize_All = config.get("Optimize_All", "False").lower() == "true"
    Unsupervised_Learning = config.get("Unsupervised_Learning", "False").lower() == "true"
    Perform_BackTesting = config.get("Perform_BackTesting", "False").lower() == "true"
    Perform_MLBackTesting = config.get("Perform_MLBackTesting", "False").lower() == "true"
    Perform_MLFutureTesting = config.get("Perform_MLFutureTesting", "False").lower() == "true"
    Print_Data = config.get("Print_Data", "False").lower() == "true"
    Perform_Forecasting = config.get("Perform_Forecasting", "False").lower() == "true"
    Perform_Tuner = config.get("Perform_Tuner", "False").lower() == "true"
    Perform_Trading = config.get("Perform_Trading", "False").lower() == "true"

    return Optimize_All,Unsupervised_Learning, Perform_BackTesting, Perform_MLBackTesting, Perform_MLFutureTesting ,Print_Data,Perform_Forecasting, Perform_Tuner,Perform_Trading

def print_config_values(symbol, bar_length, leverage, strategy, tc, test_days):
    print("\nIndividual Values:")
    print(f"Symbol: {symbol}")
    print(f"Bar Length: {bar_length}")
    print(f"Leverage: {leverage}")
    print(f"Strategy: {strategy}")
    print(f"Trading Costs: {tc:.5f}")
    print(f"Days: {test_days}")

def prepare_data_from_config(config,Broker,mode):
    # Extract and convert values
    symbol = config.get("symbol", "BTCUSDT")
    bar_length = config.get("bar_length", "15m")
    leverage = int(config.get("leverage", 10))
    strategy = config.get("strategy", "RSI")
    tc = float(config.get("tc", -0.00085))
    test_days = int(config.get("test_days", 120))
    metric = config.get("metric", "Sharpe")
    ForecastModelName=config.get("ForecastModelName", "ARIMA") 
    future_forecast_steps=int(config.get("FutureForecastSteps", 50))
    
    # Define time Period
    Today = datetime.utcnow()
    historical_days = timedelta(days=test_days)
    start_date = Today - historical_days
    end_date = Today 

    # Prepare data
    input_data = [
        ["Broker", Broker],
        ["mode", mode],
        ["Symbol", symbol],
        ["Bar Length", bar_length],
        ["Leverage", leverage],
        ["Strategy", strategy],
        ["Trading Costs", f"{tc:.5f}"],
        ["Start Date", start_date.strftime('%Y-%m-%d %H:%M')],
        ["End Date", end_date.strftime('%Y-%m-%d %H:%M')],
        ["Metric", metric],
        ["Forecast Model Name", ForecastModelName]
    ]

    # Print table
    print_data_table(input_data)

    return start_date, end_date, symbol, bar_length, leverage, strategy, tc, test_days,metric,ForecastModelName,future_forecast_steps

def prepare_trade_from_config(config):
    # Extract and convert values
    history_days = int(config.get("history_days", 10))  # Default to 120 hours if not set
    trade_hours = float(config.get("trade_hours", 24))  # Default to 120 hours if not set
    minimum_future_trade_value = float(config.get("minimum_future_trade_value", 100))
    trade_value = float(config.get("trade_value", 250.0))
    TN_trades = int(config.get("TN_trades", 50))
    position = int(config.get("position", 0))
    stop_loss_pct = float(config.get("stop_loss_pct", 0.02))
    Total_stop_loss = float(config.get("Total_stop_loss", 200))
    Total_Take_Profit = float(config.get("Total_Take_Profit", 200))
    Position_Long = config.get("Position_Long", "True").lower() == "true"
    Position_Neutral = config.get("Position_Neutral", "True").lower() == "true"
    Position_Short = config.get("Position_Short", "True").lower() == "true"

    # Define time Period
    Today = datetime.utcnow()
    history_days = timedelta(days=history_days)
    loading_from_date = Today - history_days
    trading_period = timedelta(hours=trade_hours)
    stop_trade_date = Today + trading_period
    
    # Prepare trade data
    trade_data = [
        ["Minimum Future Trade Value", f"{minimum_future_trade_value:.2f}"],
        ["Trade Value", f"{trade_value:.2f}"],
        ["Number of Trades", TN_trades],
        ["Position", position],
        ["Stop Loss Percentage", f"{stop_loss_pct:.2f}"],
        ["Totla Stop Loss", f"{Total_stop_loss:.2f}"],
        ["Totla PNL", f"{Total_Take_Profit:.2f}"],
        ["Historical Period", f"{history_days}"],
        ["loading from Date", loading_from_date.strftime('%Y-%m-%d %H:%M')],
        ["Trade Start Date", Today.strftime('%Y-%m-%d %H:%M')],
        ["Trade Stop Date", stop_trade_date.strftime('%Y-%m-%d %H:%M')],
        ["Position_Long", Position_Long],
        ["Position_Neutral", Position_Neutral],
        ["Position_Short", Position_Short],
    ]

    # Print table
    print_data_table(trade_data)

    return  loading_from_date,Today,stop_trade_date, minimum_future_trade_value, trade_value, TN_trades,\
            position, stop_loss_pct,Total_stop_loss,Total_Take_Profit,Position_Long,Position_Neutral,Position_Short

def load_api_keys(filename):
    try:
        with open(filename, 'r') as file:
            api_key = file.readline().strip()
            secret_key = file.readline().strip()
            if not api_key or not secret_key:
                raise ValueError("API key or secret key is missing.")
            return api_key, secret_key
    except Exception as e:
        raise ValueError(f"Error loading API keys: {e}")

def print_data_table(data):
    print("\n")
    print(tabulate(data, headers=["Parameter", "Value"], tablefmt="grid"))
    
def round_up(value, decimals=3):
    factor = 10 ** decimals
    return math.ceil(value * factor) / factor


def get_optimization_settings(config):
    opt_metric = config.get("opt_metric", "Robust")
    opt_n_splits = int(config.get("opt_n_splits", 4))
    opt_min_trades = int(config.get("opt_min_trades", 10))
    opt_min_win_rate = float(config.get("opt_min_win_rate", 0.35))
    opt_max_drawdown = float(config.get("opt_max_drawdown", -0.70))
    opt_warmup_bars = int(config.get("opt_warmup_bars", 100))

    max_workers_val = config.get("opt_max_workers", "None")
    opt_max_workers = None if max_workers_val == "None" else int(max_workers_val)

    return (
        opt_metric,
        opt_n_splits,
        opt_min_trades,
        opt_min_win_rate,
        opt_max_drawdown,
        opt_warmup_bars,
        opt_max_workers,
    )