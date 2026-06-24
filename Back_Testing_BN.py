from concurrent.futures import ProcessPoolExecutor, as_completed
from Back_Testing_Base_BN import BackTestingBase_BN
import Strategy as strategy
from itertools import product
import numpy as np
import pandas as pd
import os

Optimize_folder = "Optimize"
os.makedirs(Optimize_folder, exist_ok=True)


def get_strategy_function(strategy_name):
    special_map = {
        "Bollinger_Bands_ADX": strategy.define_strategy_Bollinger_ADX,
        "Bollinger_Breakout_Momentum": strategy.define_strategy_Bollinger_Breakout_Momentum_Oscillator,
        "SuperTrend_RSI": strategy.define_strategy_ATR_RSI,
        "MA_Momentum": strategy.define_strategy_MA_Momentum_F,
        "Bollinger_Stochastic_RSI_Modified": strategy.define_strategy_Bollinger_Stochastic_RSI_modified,
        "HMA_Stochastic_RSI": strategy.define_strategy_HMA_StochRSI,
        "Supertrend_Stochastic_RSI": strategy.define_strategy_Supertrend_Stochastic_RSI,
        "Simple_Pivot_Points": strategy.define_strategy_Simple_Pivot_Points,
        "Volume_delta": strategy.define_strategy_volume_delta,
        "OBV": strategy.define_strategy_OBV,
    }

    if strategy_name in special_map:
        return special_map[strategy_name]

    func_name = f"define_strategy_{strategy_name}"

    if hasattr(strategy, func_name):
        return getattr(strategy, func_name)

    return None


def evaluate_param_set_robust(job):
    (
        data,
        strategy_name,
        params,
        param_names,
        tc,
        leverage,
        tp_year,
        n_splits,
        min_trades,
        min_win_rate,
        max_drawdown_limit,
        score_mode,
        warmup_bars,
    ) = job

    try:
        bt = BackTesting_BN.__new__(BackTesting_BN)
        bt.data = data.copy()
        bt.strategy = strategy_name
        bt.tc = tc
        bt.leverage = leverage
        bt.tp_year = tp_year
        bt.results = None
        bt.position = None

        n = len(data)

        if n < 200 or n_splits < 3:
            return params, None

        fold_size = n // n_splits
        #min_valid_folds = max(2, n_splits - 2)
        min_valid_folds = 2

        fold_sharpes = []
        fold_pnls = []
        fold_drawdowns = []
        fold_win_rates = []
        fold_trades = []

        for fold in range(1, n_splits):
            train_end = fold * fold_size
            test_end = min((fold + 1) * fold_size, n)

            if test_end <= train_end:
                continue

            start_idx = max(0, train_end - warmup_bars)
            fold_data = data.iloc[start_idx:test_end].copy()

            bt.data = fold_data
            bt.results = None

            param_dict = dict(zip(param_names, params))

            stop_loss_pct = param_dict.pop("stop_loss_pct", None)
            take_profit_pct = param_dict.pop("take_profit_pct", None)

            strategy_params = tuple(param_dict.values())

            use_sl_tp = stop_loss_pct is not None and take_profit_pct is not None

            bt.test_strategy(
                strategy_params,
                use_sl_tp=use_sl_tp,
                stop_loss_pct=stop_loss_pct if stop_loss_pct is not None else 0.03,
                take_profit_pct=take_profit_pct if take_profit_pct is not None else 0.06,
            )

            if bt.results is None or bt.results.empty:
                continue

            res = bt.results.copy()

            test_index = data.iloc[train_end:test_end].index
            res_test = res.loc[res.index.intersection(test_index)].copy()

            if len(res_test) < 10:
                continue

            strat = res_test["strategy"].replace([np.inf, -np.inf], np.nan).dropna()

            if strat.empty or strat.std() == 0:
                continue

            trades = res_test["trades"].fillna(0)
            trade_mask = trades != 0
            num_trades = int(trade_mask.sum())

            if num_trades == 0:
                continue

            ann_mean = strat.mean() * tp_year
            ann_std = strat.std() * np.sqrt(tp_year)

            if ann_std == 0 or np.isnan(ann_std):
                continue

            sharpe = float(ann_mean / ann_std)

            equity = np.exp(strat.cumsum())
            pnl_multiple = float(equity.iloc[-1])
            drawdown = float((equity / equity.cummax() - 1).min())

            trade_returns = strat.loc[
                strat.index.intersection(res_test.loc[trade_mask].index)
            ]

            win_rate = (
                float((trade_returns > 0).mean())
                if len(trade_returns) > 0
                else 0.0
            )

            if np.isnan(sharpe) or np.isnan(pnl_multiple) or np.isnan(drawdown):
                continue

            fold_sharpes.append(sharpe)
            fold_pnls.append(pnl_multiple)
            fold_drawdowns.append(drawdown)
            fold_win_rates.append(win_rate)
            fold_trades.append(float(num_trades))

        if len(fold_sharpes) < min_valid_folds:
            return params, None

        mean_sharpe = float(np.mean(fold_sharpes))
        min_sharpe = float(np.min(fold_sharpes))
        std_sharpe = float(np.std(fold_sharpes))
        median_pnl = float(np.median(fold_pnls))
        max_drawdown = float(np.min(fold_drawdowns))
        win_rate = float(np.mean(fold_win_rates))
        num_trades = float(np.sum(fold_trades))
        n_valid_folds = len(fold_sharpes)


        if mean_sharpe < 0:
            return params, None

        #if median_pnl < 1.0:
        #    return params, None

        if num_trades < min_trades:
            return params, None

        if win_rate < min_win_rate:
            return params, None

        if max_drawdown < max_drawdown_limit:
            return params, None

        if n_valid_folds < 1:
            return params, None

        if std_sharpe > 4.0:
            return params, None

        if score_mode == "Sharpe":
            score = (
                1.00 * mean_sharpe
                + 0.50 * min_sharpe
                - 0.50 * std_sharpe
                - 0.50 * abs(max_drawdown)
            )

        elif score_mode == "PnL":
            score = (
                0.50 * mean_sharpe
                + 0.50 * np.log(max(median_pnl, 1e-9))
                - 0.50 * abs(max_drawdown)
                - 0.25 * std_sharpe
            )

        else:  # Robust
            score = (
                1.00 * mean_sharpe
                + 0.50 * min_sharpe
                - 0.75 * std_sharpe
                + 0.30 * np.log(max(median_pnl, 1e-9))
                - 0.50 * abs(max_drawdown)
            )

        metrics = {
            "performance": float(score),
            "mean_sharpe": mean_sharpe,
            "min_sharpe": min_sharpe,
            "std_sharpe": std_sharpe,
            "median_pnl": median_pnl,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": num_trades,
            "n_valid_folds": n_valid_folds,
        }

        return params, metrics

    except Exception:
        return params, None

def evaluate_param_set(args):
    data, strategy_name, params, tc, leverage, metric = args

    try:
        data = data.copy()
        strategy_func = get_strategy_function(strategy_name)

        if strategy_func is None:
            return params, np.nan

        no_param_strategies = ["OBV", "Volume_delta", "Simple_Pivot_Points"]

        if strategy_name in no_param_strategies:
            results = strategy_func(data)
        else:
            results = strategy_func(data, params)

        if results is None or "position" not in results.columns:
            return params, np.nan

        if "returns" not in results.columns:
            results["returns"] = np.log(results["Close"] / results["Close"].shift(1))

        results = results.replace([np.inf, -np.inf], np.nan).dropna().copy()

        if results.empty:
            return params, np.nan

        results["strategy"] = results["position"].shift(1) * results["returns"]
        results["trades"] = results["position"].diff().fillna(0).abs()
        results["strategy"] += results["trades"] * tc

        if metric == "Sharpe":
            simple_ret = np.exp(results["strategy"]) - 1
            lev_ret = leverage * simple_ret
            lev_ret = np.where(lev_ret < -1, -1, lev_ret)

            log_ret = np.log(pd.Series(lev_ret, index=results.index).add(1))
            log_ret = log_ret.replace([np.inf, -np.inf], np.nan).dropna()

            if log_ret.empty:
                return params, np.nan

            years = (results.index[-1] - results.index[0]).days / 365.25

            if years <= 0:
                return params, np.nan

            tp_year = results["Close"].count() / years
            ann_std = log_ret.std() * np.sqrt(tp_year)

            if ann_std == 0 or np.isnan(ann_std):
                return params, np.nan

            cagr = np.exp(log_ret.sum()) ** (1 / years) - 1
            score = cagr / ann_std

        elif metric == "Multiple":
            score = np.exp(results["strategy"].sum())

        else:
            return params, np.nan

        if np.isnan(score) or np.isinf(score):
            return params, np.nan

        return params, score

    except Exception:
        return params, np.nan


class BackTesting_BN(BackTestingBase_BN):
    def __init__(
        self,
        client,
        symbol,
        bar_length,
        start,
        end=None,
        tc=0.0,
        leverage=5,
        strategy="PV",
    ):
        super().__init__(client, symbol, bar_length, start, end, tc, leverage, strategy)

    def __repr__(self):
        return "\nFutures Backtester (symbol = {}, start = {}, end = {})\n".format(
            self.symbol, self.start, self.end
        )

    def prepare_data(self, parameters):
        data = self.data.copy()
        strategy_func = get_strategy_function(self.strategy)

        if strategy_func is None:
            raise ValueError(f"No function found for strategy: {self.strategy}")

        no_param_strategies = ["OBV", "Volume_delta", "Simple_Pivot_Points"]

        if self.strategy in no_param_strategies:
            self.results = strategy_func(data)
        else:
            self.results = strategy_func(data, parameters)

    def optimize_strategy(
        self,
        param_ranges,
        metric="Multiple",
        output_file=None,
        Print_Data=False,
    ):
        if Print_Data:
            print("\nOptimize Strategy is running.")

        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        param_combinations = self._generate_param_combinations(param_ranges)

        performance = []
        valid_combinations = []

        for params in param_combinations:
            self.prepare_data(parameters=params)
            self.run_backtest()

            if metric == "Sharpe":
                self.add_leverage(self.leverage, report=False)
                result = self.calculate_sharpe(
                    np.log(self.results["strategy_leverage"].add(1))
                )
            else:
                result = performance_function(self.results.strategy)

            if not np.isnan(result):
                performance.append(result)
                valid_combinations.append(params)
            else:
                if Print_Data:
                    print("There is NaN in Performance Optimization")

        self.results_overview = pd.DataFrame(
            data=valid_combinations,
            columns=list(param_ranges.keys()),
        )

        self.results_overview["performance"] = performance

        if Print_Data:
            print(f"Performance values:\n{self.results_overview}")

        if output_file:
            self.results_overview.to_csv(
                os.path.join(Optimize_folder, "ALL_" + output_file),
                index=False,
            )

        best_params = self.find_best_strategy(output_file)
        return best_params

    def optimize_strategy_parallel(
        self,
        param_ranges,
        metric="Robust",
        output_file=None,
        max_workers=None,
        n_splits=5,
        min_trades=30,
        min_win_rate=0.45,
        max_drawdown_limit=-0.40,
        warmup_bars=150,
    ):
        """
        Parallel robust optimization for indicator-based strategies.

        metric options:
            "Robust" -> robust composite score
            "Sharpe" -> Sharpe-focused robust score
            "PnL"    -> PnL-focused robust score
        """

        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)

        os.makedirs(Optimize_folder, exist_ok=True)

        if self.data is None or self.data.empty:
            raise ValueError("No data available for optimization.")

        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex for robust optimization.")

        if not hasattr(self, "tp_year") or self.tp_year is None:
            years = (self.data.index[-1] - self.data.index[0]).days / 365.25

            if years <= 0:
                raise ValueError(
                    "Cannot calculate tp_year because data time span is too short."
                )

            self.tp_year = self.data["Close"].count() / years

        param_combinations = self._generate_param_combinations(param_ranges)

        param_names = list(param_ranges.keys())

        jobs = [
            (
                self.data.copy(),
                self.strategy,
                params,
                param_names,
                self.tc,
                self.leverage,
                self.tp_year,
                n_splits,
                min_trades,
                min_win_rate,
                max_drawdown_limit,
                metric,
                warmup_bars,
            )
            for params in param_combinations
        ]

        rows = []

        print(f"\nRunning robust parallel optimization with {max_workers} workers")
        print(f"Total combinations: {len(jobs)}")
        print(f"Metric mode: {metric}")
        print(f"Folds: {n_splits}")
        print(f"Warmup bars: {warmup_bars}")
        print(f"Min trades: {min_trades}")
        print(f"Min win rate: {min_win_rate}")
        print(f"Max drawdown limit: {max_drawdown_limit}\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(evaluate_param_set_robust, job)
                for job in jobs
            ]

            for i, future in enumerate(as_completed(futures), 1):
                params, metrics = future.result()

                if metrics is not None:
                    row = dict(zip(param_ranges.keys(), params))
                    row.update(metrics)
                    rows.append(row)

                if i % 100 == 0:
                    print(
                        f"Completed {i}/{len(jobs)} combinations... "
                        f"valid so far: {len(rows)}"
                    )

        self.results_overview = pd.DataFrame(rows)

        if self.results_overview.empty:
            print("No valid parameter combinations found.")
            return None

        self.results_overview.sort_values(
            "performance",
            ascending=False,
            inplace=True,
        )

        if output_file:
            self.results_overview.to_csv(
                os.path.join(Optimize_folder, "ALL_" + output_file),
                index=False,
            )

        best_params = self.find_best_strategy(output_file)
        return best_params

    def _generate_param_combinations(self, param_ranges):
        ranges = [param_ranges[param] for param in param_ranges]
        return list(product(*ranges))

    def find_best_strategy(self, output_file=None):
        try:
            if self.results_overview is None or self.results_overview.empty:
                print("Error: Results overview is empty.")
                return None

            if "performance" not in self.results_overview.columns:
                print("Error: Column 'performance' not found.")
                return None

            best = self.results_overview.nlargest(1, "performance").iloc[0]

            # -------------------------------
            # Separate metric vs param columns
            # -------------------------------
            metric_columns = {
                "performance",
                "mean_sharpe",
                "min_sharpe",
                "std_sharpe",
                "median_pnl",
                "max_drawdown",
                "win_rate",
                "num_trades",
                "n_valid_folds",
            }

            param_names = [
                col for col in self.results_overview.columns
                if col not in metric_columns
            ]

            # -------------------------------
            # Extract parameters
            # -------------------------------
            param_dict = {name: best[name] for name in param_names}

            # Separate risk params
            stop_loss_pct = param_dict.pop("stop_loss_pct", None)
            take_profit_pct = param_dict.pop("take_profit_pct", None)

            strategy_params_tuple = tuple(param_dict.values())

            use_sl_tp = stop_loss_pct is not None and take_profit_pct is not None

            # -------------------------------
            # Print best params
            # -------------------------------
            print("\n✅ Best Strategy Parameters:")
            for name in param_names:
                print(f"{name:<35}: {best[name]}")

            print("\n📊 Optimization Metrics:")
            for name in metric_columns:
                if name in best.index:
                    print(f"{name:<35}: {best[name]:>10.4f}")

            print("\\/" * 25)

            # -------------------------------
            # Run final backtest correctly
            # -------------------------------
            self.test_strategy(
                strategy_params_tuple,
                use_sl_tp=use_sl_tp,
                stop_loss_pct=stop_loss_pct if stop_loss_pct is not None else 0.03,
                take_profit_pct=take_profit_pct if take_profit_pct is not None else 0.06,
            )

            # -------------------------------
            # Save best result
            # -------------------------------
            if output_file:
                best_params_df = pd.DataFrame([best.to_dict()])
                best_params_df.to_csv(
                    os.path.join(Optimize_folder, output_file),
                    index=False,
                )

            # Return FULL tuple (including SL/TP for logging consistency)
            return tuple(best[name] for name in param_names)

        except IndexError:
            print("Error: No rows in results overview after applying nlargest.")
            return None

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None  