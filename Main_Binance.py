import Loading_Config_BN as Loading_Config_BN
#import Loading_Config_IB as Loading_Config_IB
from Back_Testing_ML_BN import ML_Strategies
import Strategy_Optimizer
import pandas as pd
import os

from Config_Check import make_request_with_retries,Debug_function
from tkinter import TRUE
from Trader_BN import FuturesTrader_BN
from Back_Testing_BN import BackTesting_BN
from Forecast_Testing import ForecastTesting  
from Loading_Strategy import StrategyLoader
from Loading_ForecastModel import LoadingForecastModel
from Unsupervised_learning_trading_strategy import Unsupervised_learning_trading_strategy

#from ib_insync import *
#from ib_insync import IB, Forex, Stock, Future, Contract
from binance import Client

import warnings
warnings.filterwarnings("ignore")

Path_Configs ="Configs"
os.makedirs(Path_Configs, exist_ok=True)

# Load configuration using os.path.join to concatenate the path and file name
def get_BN_price(symbol, client):
    return float(client.futures_symbol_ticker(symbol=symbol)['price'])

def run_binance(Broker):
    config = Loading_Config_BN.load_config_from_text(os.path.join(Path_Configs, "Config_BN.txt"))
    strategy_loader = StrategyLoader(os.path.join(Path_Configs,"strategies_config.json"))

    # Get control settings
    Optimize_All,Unsupervised_Learning, Perform_BackTesting,Perform_MLBackTesting, Perform_MLFutureTesting, Print_Data, Perform_Forecasting, Perform_Tuner, Perform_Trading\
    = Loading_Config_BN.get_control_settings(config)
    if Optimize_All : 
        mode = "Optimize"
    elif Perform_BackTesting :   
        mode = "BackTesting" 
    elif Perform_MLBackTesting :
        mode = "MLBackTesting"                 
    elif Perform_MLFutureTesting :
        mode = "MLFutureTesting"
    elif Perform_Trading :   
        mode = "LiveTrading"  

    # Prepare data and get individual values
    start_date, end_date, symbol, bar_length, leverage, strategy, tc, test_days,metric , ForecastModelName ,\
    future_forecast_steps = Loading_Config_BN.prepare_data_from_config(config,Broker=Broker,mode=mode)

    opt_metric, opt_n_splits, opt_min_trades, opt_min_win_rate, opt_max_drawdown, opt_warmup_bars, opt_max_workers = Loading_Config_BN.get_optimization_settings(config)

    # Load API keys from file
    try:
        api_key, secret_key = Loading_Config_BN.load_api_keys(os.path.join(Path_Configs, "KEY.txt"))
    except ValueError as e:
        print(e)
        exit(1)        

    print(f"\nAttempt : Trying to connect to Binance...")
    client = Client(api_key=api_key, api_secret=secret_key, tld="com", testnet=True)
    print("Successfully connected to Binance!\n")

################################################################################################################  
    if Optimize_All:
        print("\n🚀 Running optimizer: Evaluating multiple strategies and bar lengths to identify the top-performing configuration by Sharpe ratio.\n")
        Strategy_Optimizer.run_optimizer(client=client,start_date=start_date, end_date=end_date,symbol=symbol,tc=tc, \
                                         leverage=leverage,metric=metric, Print_Data = Print_Data)

################################################################################################################  
    if Unsupervised_Learning:
        print("\n Unsupervised Learning Trading Strategy")
        Unsupervised_learning_trading_strategy()

################################################################################################################   
    if Perform_BackTesting:
        print("\nFutures Back Testing is enabled\n")

        description, parameters_BT, param_ranges_BT = strategy_loader.process_strategy(strategy, Print_Data)

        print(f"\nStrategy: {strategy}")
        print(f"Description: {description}")
        print(f"Parameters Back Testing: {parameters_BT}")
        print(f"Parameter Ranges: {param_ranges_BT}\n")
        
        backtesting = BackTesting_BN(
            client=client,
            symbol=symbol,
            bar_length=bar_length,
            start=start_date,
            end=end_date,
            tc=tc,
            leverage=leverage,
            strategy=strategy,
        )

        # Initial backtest with default indicator parameters only
        backtesting.test_strategy(parameters_BT)
        backtesting.add_leverage(leverage=leverage)
        backtesting.plot_strategy_comparison(
            leverage=True,
            plot_name=f"{symbol}_{strategy}"
        )
        backtesting.plot_all_indicators(
            plot_name=f"{symbol}_{strategy}",
            Print_Data=Print_Data
        )
        print(backtesting.results.trades.value_counts())

        # Optimization: indicator params + SL/TP params
        parameters_BT = backtesting.optimize_strategy_parallel(
            param_ranges_BT,
            metric=opt_metric,
            output_file=f"{strategy}_optimize_results.csv",
            max_workers=opt_max_workers,
            n_splits=opt_n_splits,
            min_trades=opt_min_trades,
            min_win_rate=opt_min_win_rate,
            max_drawdown_limit=opt_max_drawdown,
            warmup_bars=opt_warmup_bars,
        )

        if parameters_BT is not None:
            # Do NOT call test_strategy(parameters_BT) here.
            # find_best_strategy() already ran the correct final backtest
            # with indicator params separated from SL/TP.

            backtesting.add_leverage(leverage=leverage)

            backtesting.plot_strategy_comparison(
                leverage=True,
                plot_name=f"WOpt_{symbol}_{strategy}"
            )

            backtesting.plot_all_indicators(
                plot_name=f"WOpt_{symbol}_{strategy}",
                Print_Data=Print_Data
            )

            print(backtesting.results.trades.value_counts())

        else:
            print("Parameters (BT) is : None")
################################################################################################################   
    if (Perform_MLBackTesting):
        ml_strategy = ML_Strategies(client=client, symbol=symbol, bar_length=bar_length,
                                    start=start_date, end=end_date, tc=tc)

        # Step 1: Feature engineering
        ml_strategy.ML_Strategy(CFModel=None, parameters=None)

        # Step 2: Train model
        ml_strategy.run_model(model_type='rf') #xgb

        # Step 3 (Optional): Run backtesting with `backtesting.py` Strategy
        ml_strategy.run_backtesting_strategy()

        # Step 4: Plot performance with leverage
        ml_strategy.plot_performance(leverage=leverage)

    if Perform_MLFutureTesting:
        ml = ML_Strategies(
            client=client,
            symbol=symbol,
            bar_length=bar_length,
            start=start_date,
            end=end_date,
            tc=tc
        )

        ml.ML_Strategy(CFModel=None, parameters=None)

        # 1) Conv1D
        forecast_df_conv1d = ml.run_future_prediction_conv1d_binance(
            future_steps=48,
            eval_h=24,
            n_trials=30,
            skip_tuning_if_best_exists=False,
            window_size=3000,
            step_size=750,
            wfv_test_slice=50,
            wfv_max_epochs=8,
            wfv_patience=2,
            enable_eager_debug=False,
            final_epochs=20,
            global_seed=42,
            output_dir="Forecasts",
        )

        # 2) Transformer
        forecast_df_tr = ml.run_future_prediction_transformer_binance_optuna(
            future_steps=48,
            eval_h=24,
            n_trials=30,
            skip_tuning_if_best_exists=False,
            window_size=3000,
            step_size=750,
            wfv_test_slice=50,
            wfv_max_epochs=8,
            wfv_patience=2,
            final_epochs=20,
            global_seed=42,
            enable_eager_debug=False,
            output_dir="Forecasts",
        )

        print("Conv1D forecast head:\n", forecast_df_conv1d.head())
        print("Transformer forecast head:\n", forecast_df_tr.head())
################################################################################################################   
    if (Perform_Forecasting) :
        print("\nFutures Forecast Testing is enabled\n")

        model_loader = LoadingForecastModel(os.path.join(Path_Configs,"forecast_models_config.json"))
        description, parameters_F, param_ranges_F = model_loader.process_model(ForecastModelName)

        model_loader.print_model_details(ForecastModelName)
        forecast_testing = ForecastTesting(client=client, symbol=symbol, bar_length=bar_length,
                                start=start_date, end=end_date, tc=tc,leverage=leverage, strategy=strategy
                                ,future_forecast_steps=future_forecast_steps)
        
        forecast_testing.Forecast_price(ForecastModelName,parameters_F,param_ranges_F,Perform_Tuner) 

################################################################################################################   
    if Perform_Trading:
        # Prepare trade parameters
        loading_from_date,Today,stop_trade_date, minimum_future_trade_value, trade_value, TN_trades, position,\
        stop_loss_pct ,Total_stop_loss, Total_Take_Profit ,Position_Long,Position_Neutral,Position_Short= \
        Loading_Config_BN.prepare_trade_from_config(config)
        current_price = get_BN_price(symbol, client)
       
        print("\nStart Trading")
        #print(f"\nTrade will continue from: {Today}, until: {stop_trade_date}, Max number of trades is: {TN_trades}")

        print(f"trade_value = {trade_value} , current_price = {current_price} ")
        units = round(trade_value / current_price,3)  
        print(f'units = {units}')

        description, parameters, param_ranges = strategy_loader.process_strategy(strategy)
        print(f"Strategy: {strategy}")
        print(f"Description: {description}")
        strategy_loader.print_strategy_details(strategy)

        if trade_value < minimum_future_trade_value:
            min_required_units = minimum_future_trade_value / current_price
            print(f"\nMinimum trade value {trade_value} is below {minimum_future_trade_value}. Adjust units or strategy.")
            print(f"Current Price: {current_price}")
            print(f"Minimum Required Units for Trade: {min_required_units}\n")
        else:
            trader = FuturesTrader_BN(client=client, symbol=symbol, bar_length=bar_length,parameters=parameters,
                                units=units,stop_trade_date=stop_trade_date, Total_stop_loss=Total_stop_loss,stop_loss_pct=stop_loss_pct,
                                Total_Take_Profit=Total_Take_Profit,Position_Long=Position_Long,Position_Neutral=Position_Neutral,
                                Position_Short=Position_Short, TN_trades=TN_trades, position=position, leverage=leverage,strategy=strategy)
            trader.start_trading(historical_days=loading_from_date)

            if hasattr(trader, 'Rep_Trade') and isinstance(trader.Rep_Trade, pd.DataFrame) and not trader.Rep_Trade.empty:
                print(trader.prepared_data.tail(trader.N_trades))
                print('\n' * 2)
                Report_Trades = trader.Rep_Trade.drop(['id', 'orderId', 'commissionAsset', 'positionSide', 'maker', 'buyer'], axis=1, errors='ignore')
                Report_Trades['time'] = pd.to_datetime(Report_Trades['time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M')
                columns_order = ['time'] + [col for col in Report_Trades.columns if col != 'time']
                Report_Trades = Report_Trades[columns_order]
                print(Report_Trades)

            separator = "-" * 70
            print("\n" * 2 + separator)
            print(f"| {'Final Trade Report'.center(66)} |")
            print(separator)
            print(f"| Trade Number          : {trader.N_trades:<42} |")
            print(f"| Cumulative Profit     : {trader.cum_profits:<42} |")
            print(separator + "\n")        

            # Uncomment to check account info
            # client.get_account()