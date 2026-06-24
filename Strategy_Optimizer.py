import os
import json
import pandas as pd
from Back_Testing_BN import BackTesting_BN
from Loading_Strategy import StrategyLoader
import warnings

Optimize_folder = "Optimize"
os.makedirs(Optimize_folder, exist_ok=True)

def run_optimizer(client, start_date, end_date, symbol, tc, leverage, metric, Print_Data):
    Path_Configs = "Configs"
    strategy_loader = StrategyLoader(os.path.join(Path_Configs, "strategies_config.json"))
    bar_lengths = ["5m", "15m", "30m", "1h"]

    # Format dates
    fmt_start = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M')
    fmt_end = pd.to_datetime(end_date).strftime('%Y-%m-%d %H:%M')
    print(f"\nSymbol = {symbol}, Start = {fmt_start}, End = {fmt_end}, Leverage = {leverage}\n")

    All_best_results = []

    for strategy_name in strategy_loader.strategies:
        for bar in bar_lengths:
            description, parameters_BT, param_ranges_BT = strategy_loader.process_strategy(strategy_name,Print_Data)

            print(f"\n>>>> strategy_name : {strategy_name} | bar : {bar} | Init: {parameters_BT} ")

            bt = BackTesting_BN(client=client, symbol=symbol, bar_length=bar,
                                start=start_date, end=end_date, tc=tc, leverage=leverage, strategy=strategy_name)

            bt.test_strategy(parameters_BT)
            bt.add_leverage(leverage=leverage)
            bt.plot_strategy_comparison(leverage=True, plot_name=f"{symbol}_{strategy_name}_{bar}",plot_show=False)
            bt.plot_all_indicators(plot_name=f"{symbol}_{strategy_name}_{bar}",plot_show=False, Print_Data = False)

            #print(bt.results.trades.value_counts())

            best_params = bt.optimize_strategy(param_ranges_BT, metric, output_file=f"{strategy_name}_{bar}_optimize_results.csv")
            if best_params is not None:
                if Print_Data : print(f"Collecting performance metrics for strategy: {strategy_name} | bar length: {bar}")
                bt.add_leverage(leverage=leverage, report=False)
                strategy_multiple, bh_multiple, outperf, cagr, ann_mean, ann_std, sharpe = bt.print_performance(leverage=True)
                if Print_Data : print(f"\nBest Params : {best_params} >> | sharpe: {sharpe} | outperf: {outperf}")
                All_best_results.append({
                    "strategy": strategy_name,
                    "bar_length": bar,
                    "params": dict(zip(param_ranges_BT.keys(), best_params[:-1])),  # exclude performance value
                    "Strategy Multiple": strategy_multiple,
                    "Buy-and-Hold Multiple": bh_multiple,
                    "Out-/Underperformance": outperf,
                    "CAGR": cagr,
                    "Annualized Mean Return": ann_mean,
                    "Annualized Standard Deviation": ann_std,
                    "sharpe": sharpe
                })

    # Save top strategies
    if All_best_results:
        # Save all results
        all_output_file = os.path.join(Optimize_folder, "all_strategies.json")

        with open(all_output_file, "w") as f:
            json.dump(All_best_results, f, indent=2)
        print(f"\nAll strategies saved to {all_output_file}")

        # Save top 20 sorted by sharpe
        best_results_sorted = sorted(All_best_results, key=lambda x: x["sharpe"], reverse=True)
        top_output_file = os.path.join(Optimize_folder, "best_strategies.json")
        with open(top_output_file, "w") as f:
            json.dump(best_results_sorted[:20], f, indent=2)
        print(f"Top 20 strategies saved to {top_output_file}")
    else:
        print("\nNo successful strategies found to save.")


