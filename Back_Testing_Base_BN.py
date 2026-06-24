import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import logging
from Loading_Data_BN import fetch_historical_data
import os
Plot_folder = "Plots"
os.makedirs(Plot_folder, exist_ok=True)
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

class BackTestingBase_BN:
    def __init__(self, client, symbol, bar_length, start, end=None, tc=0.0, leverage=5, strategy="PV"):
        self.client = client
        self.symbol = str(symbol)
        self.bar_length = str(bar_length)
        self.start = str(start)
        self.end = str(end) if end else None
        self.tc = tc
        self.leverage = leverage
        self.strategy = strategy
        self.data = None
        self.results = None

        # Validate input intervals
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        if self.bar_length not in self.available_intervals:
            raise ValueError(f"Invalid bar length: {self.bar_length}. Choose from {self.available_intervals}.")

        # Fetch data
        try:
            self.data = self.get_data()
            self.tp_year = self.calculate_trading_periodicity()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    def get_data(self):
        try:
            data = fetch_historical_data(client=self.client,symbol=self.symbol,bar_length=self.bar_length,
                                        start=self.start,end=self.end)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise

    def test_strategy(self, parameters, use_sl_tp=False, stop_loss_pct=0.03, take_profit_pct=0.06):
        self.prepare_data(parameters)

        self.run_backtest(
            use_sl_tp=use_sl_tp,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

    def run_backtest(self, use_sl_tp=False, stop_loss_pct=0.03, take_profit_pct=0.06):
        if self.results is None:
            raise ValueError("No strategy results available. Please generate results first.")

        if use_sl_tp:
            self.apply_stop_loss_take_profit(
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )

        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy += data.trades * self.tc
        self.results = data

    def add_leverage(self, leverage, report=True):
        self.leverage = leverage
        self.add_sessions()
        data = self.results.copy()
        data["simple_ret"] = np.exp(data.strategy) - 1
        data["eff_lev"] = leverage * (1 + data.session_compound) / (1 + data.session_compound * leverage)
        data.eff_lev.fillna(leverage, inplace=True)
        data.loc[data.trades != 0, "eff_lev"] = leverage
        leverage_returns = data.eff_lev.shift() * data.simple_ret
        leverage_returns = np.where(leverage_returns < -1, -1, leverage_returns)
        data["strategy_leverage"] = leverage_returns
        data["cstrategy_leverage"] = data.strategy_leverage.add(1).cumprod()
        self.results = data

        if report:
            self.print_performance(leverage=True)

    def add_sessions(self):
 
        if self.results is None:
            print("Run test_strategy() first.")
            
        data = self.results.copy()
        data["session"] = np.sign(data.trades).cumsum().shift().fillna(0)
        data["session_compound"] = data.groupby("session").strategy.cumsum().apply(np.exp) - 1
        self.results = data            

    def print_performance(self, leverage=False):
        if self.results is None:
            raise ValueError("No strategy results to analyze.")

        data = self.results.copy()
        to_analyze = np.log(data["strategy_leverage"].add(1)) if leverage else data.strategy

        strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple = round(self.calculate_multiple(data["returns"]), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(to_analyze), 6)
        ann_mean = round(self.calculate_annualized_mean(to_analyze), 6)
        ann_std = round(self.calculate_annualized_std(to_analyze), 6)
        sharpe = round(self.calculate_sharpe(to_analyze), 6)

        #print("=" * 50)
        #print(f"STRATEGY PERFORMANCE | INSTRUMENT = {self.symbol}")
        #print("-" * 50)
        #print(f"Strategy Multiple:           {round(strategy_multiple,2)}")
        #print(f"Buy-and-Hold Multiple:       {round(bh_multiple,2)}")
        #print(f"Out-/Underperformance:       {round(outperf,2)}")
        #print(f"CAGR:                        {round(cagr,2)}")
        #print(f"Annualized Mean Return:      {round(ann_mean,2)}")
        #print(f"Annualized Standard Deviation: {round(ann_std,2)}")
        #print(f"Sharpe Ratio:                {round(sharpe,2)}")
        #print("=" * 50)
        
        print("=" * 50)
        print(f"📊 STRATEGY PERFORMANCE | Symbol = {self.symbol}")
        print("-" * 50)

        metrics = {
            "Strategy Multiple": strategy_multiple,
            "Buy-and-Hold Multiple": bh_multiple,
            "Out-/Underperformance": outperf,
            "CAGR": cagr,
            "Annualized Mean Return": ann_mean,
            "Annualized Std. Dev.": ann_std,
            "Sharpe Ratio": sharpe
        }

        for name, value in metrics.items():
            print(f"{name:<35}: {value:>10.2f}")

        print("=" * 50)


        return strategy_multiple,bh_multiple,outperf,cagr,ann_mean,ann_std,sharpe
    def calculate_multiple(self, series):
        return np.exp(series.sum())

    def calculate_cagr(self, series):
        return np.exp(series.sum()) ** (1 / ((series.index[-1] - series.index[0]).days / 365.25)) - 1

    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year

    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)

    def calculate_sharpe(self, series):
        return self.calculate_cagr(series) / self.calculate_annualized_std(series)  
    
    def calculate_trading_periodicity(self):
        if self.data is None:
            raise ValueError("No data available to calculate periodicity.")
        return self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25)    
    
    def apply_stop_loss_take_profit(self, stop_loss_pct=0.03, take_profit_pct=0.06):
        data = self.results.copy()

        position = data["position"].copy()
        close = data["Close"]

        active_pos = 0
        entry_price = None

        for i in range(1, len(data)):
            signal_pos = position.iloc[i]

            if active_pos == 0:
                if signal_pos != 0:
                    active_pos = signal_pos
                    entry_price = close.iloc[i]
                position.iloc[i] = active_pos
                continue

            if active_pos == 1:
                if close.iloc[i] <= entry_price * (1 - stop_loss_pct) or close.iloc[i] >= entry_price * (1 + take_profit_pct):
                    active_pos = 0
                    entry_price = None
                position.iloc[i] = active_pos

            elif active_pos == -1:
                if close.iloc[i] >= entry_price * (1 + stop_loss_pct) or close.iloc[i] <= entry_price * (1 - take_profit_pct):
                    active_pos = 0
                    entry_price = None
                position.iloc[i] = active_pos

            if active_pos == 0 and signal_pos != 0:
                active_pos = signal_pos
                entry_price = close.iloc[i]

        data["position_raw"] = data["position"]
        data["position"] = position
        self.results = data

#######################################################################################################
#                                       plot_results
#######################################################################################################

    def plot_strategy_comparison(self, leverage=False, plot_name=None, plot_show=True):
        if self.results is None:
            logger.warning("Run test_strategy() first.")
            return

        data = self.results.copy()

        title = f"{self.strategy} | {self.symbol} | TC = {self.tc}"
        if leverage:
            title += f" | Leverage = {self.leverage}"

        equity_col = "cstrategy_leverage" if leverage and "cstrategy_leverage" in data.columns else "cstrategy"

        data["drawdown"] = data[equity_col] / data[equity_col].cummax() - 1

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )

        axes[0].plot(data.index, data["creturns"], label="Buy & Hold", linewidth=1.5)
        axes[0].plot(data.index, data["cstrategy"], label="Strategy", linewidth=1.5)

        if leverage and "cstrategy_leverage" in data.columns:
            axes[0].plot(
                data.index,
                data["cstrategy_leverage"],
                label=f"Strategy x{self.leverage}",
                linewidth=1.8
            )

        axes[0].set_title(title)
        axes[0].set_ylabel("Cumulative Multiple")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].fill_between(data.index, data["drawdown"], 0, alpha=0.4)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = os.path.join(
            Plot_folder,
            f"Comparison_{plot_name}.png" if plot_name else "Comparison_plot.png"
        )

        plt.savefig(plot_path, dpi=200, bbox_inches="tight")

        if plot_show:
            plt.show()
        else:
            plt.close()

    def plot_results_II(self,plot_show=True):
        ''' Plots a scatter plot of volume change against returns.
        '''
        if self.results is None:
            print("No data to plot. Please provide data.")
        else:
            plt.scatter(x=self.results['vol_ch'], y=self.results['returns'])
            plt.xlabel("Volume Change")
            plt.ylabel("Returns")
            if plot_show:
                plt.show()
            else:
                plt.close()

    def plot_heatmap(self,plot_show=True):
        ''' Bins returns and volume change, creates a crosstab matrix, and plots a heatmap.
        '''
        if self.results is None:
            print("No data to process. Please provide data.")
        else:
            # Binning returns and volume change
            self.results["ret_cat"] = pd.qcut(self.results['returns'], q=10, labels=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            self.results["vol_cat"] = pd.qcut(self.results['vol_ch'], q=10, labels=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            
            # Creating crosstab matrix
            matrix_I = pd.crosstab(self.results['vol_cat'], self.results['ret_cat'])
            
            # Plotting the first heatmap
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1)
            sns.heatmap(matrix_I, cmap="RdYlBu_r", annot=True, robust=True, fmt=".0f")
            plt.title(f"Heatmap of Volume Change vs Returns | {self.symbol} | TC = {self.tc}")
            plt.xlabel("Return cat")
            plt.ylabel("Volume cat")
            if plot_show:
                plt.show()
            else:
                plt.close()

            #matrix_II = pd.crosstab(self.results['vol_cat'].shift(), self.results['ret_cat'].shift(),values = self.results, aggfunc =np.mean)

            # Shifting categories and calculating the mean of the desired values
            shifted_results = self.results.shift()

            # Creating crosstab matrix for shifted data
            # need to be checked
            matrix_II = pd.crosstab(shifted_results['vol_cat'], shifted_results['ret_cat'], values=shifted_results['returns'], aggfunc=np.mean)
        

            # Plotting the second heatmap
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=0.75)
            sns.heatmap(matrix_II, cmap="RdYlBu", annot=True, robust=True, fmt=".3f")
            plt.title(f"Heatmap of Volume Change vs Returns | {self.symbol} | TC = {self.tc}")
            plt.xlabel("Return cat")
            plt.ylabel("Volume cat")
            if plot_show:
                plt.show()
            else:
                plt.close()   

    def plot_all_indicators(self, plot_name=None, plot_show=True, Print_Data=False):
        if self.results is None:
            logger.warning("Run test_strategy() first.")
            return

        data = self.results.copy()

        if data.empty:
            logger.warning("Results are empty.")
            return

        available_columns = data.columns.tolist()

        if Print_Data:
            print(f"Available columns in results:\n{available_columns}")

        # ============================
        # Normalized / derived columns
        # ============================
        if "Stoch_RSI" in data.columns:
            data["Stoch_RSI_n"] = data["Stoch_RSI"] * 100

        if "std_dev" in data.columns and "SMA" in data.columns:
            data["std_dev_n"] = data["std_dev"] / data["SMA"]

        if "returns" in data.columns:
            data["returns_pct"] = data["returns"] * 100

        # ============================
        # Column groups
        # ============================
        price_cols = [
            "Close", "Open", "High", "Low",
            "SMA", "SMA_S", "SMA_M", "SMA_L",
            "EMA", "EMA_S", "EMA_L", "EMA_Fast", "EMA_Slow",
            "TEMA", "Hull_MA", "VWAP", "VWMA",
            "Upper_Band", "Lower_Band", "Middle_Band",
            "Upper_Envelope", "Lower_Envelope",
            "High_Max", "Low_Min",
            "Donchian_High", "Donchian_Low",
            "Keltner_Upper", "Keltner_Lower",
            "Chandelier_Exit",
            "PSAR",
            "Supertrend",
            "Linear_Regression",
            "Pivot_Point", "R1", "S1",
            "Fibonacci_23_6", "Fibonacci_38_2", "Fibonacci_50",
            "Fibonacci_61_8", "Fibonacci_78_6",
            "Gann_1x1", "Gann_2x1", "Gann_3x1",
        ]

        oscillator_cols = [
            "RSI", "Stoch_RSI_n", "Williams_R", "CCI",
            "CMO", "ROC", "UO", "Momentum",
            "Momentum_Oscillator", "Price_Oscillator",
            "z_score", "Trix",
        ]

        trend_cols = [
            "ADX", "DI_plus", "DI_minus",
            "MACD", "MACD_Signal", "MACD_Histogram",
            "Aroon_Up", "Aroon_Down",
            "Klinger", "Chaikin_Oscillator",
            "BullPower", "BearPower",
            "Force_Index",
        ]

        volume_cols = [
            "Volume", "vol_ch", "CMF", "MFI", "ADL",
            "OBV", "Volume_Delta", "dollar_volume",
            "Taker Buy Base Asset Volume",
            "Taker Buy Quote Asset Volume",
        ]

        volatility_cols = [
            "ATR", "BB_Width", "std_dev_n",
            "Garman_Klass", "variance",
            "hl_range", "atr14",
        ]

        performance_cols = [
            "creturns", "cstrategy", "cstrategy_leverage",
            "strategy", "strategy_leverage",
            "returns_pct",
        ]

        signal_cols = [
            "position", "trades", "prediction",
        ]

        # ============================
        # Keep only existing columns
        # ============================
        def existing(cols):
            return [c for c in cols if c in data.columns]

        groups = {
            "Price / Bands / Moving Averages": existing(price_cols),
            "Oscillators / Momentum": existing(oscillator_cols),
            "Trend / Directional Indicators": existing(trend_cols),
            "Volume / Money Flow": existing(volume_cols),
            "Volatility / Risk Indicators": existing(volatility_cols),
            "Performance": existing(performance_cols),
            "Trading Signals": existing(signal_cols),
        }

        # Remove empty groups
        groups = {k: v for k, v in groups.items() if len(v) > 0}

        # ============================
        # Unknown indicator columns
        # ============================
        known_cols = set()
        for cols in groups.values():
            known_cols.update(cols)

        exclude_cols = {
            "Open Time", "Close Time", "Complete",
            "Quote Asset Volume", "Number of Trades",
            "Ignore",
        }

        unknown_cols = [
            c for c in data.columns
            if c not in known_cols
            and c not in exclude_cols
            and pd.api.types.is_numeric_dtype(data[c])
        ]

        # Avoid plotting too many unknown columns
        unknown_cols = unknown_cols[:8]

        if unknown_cols:
            groups["Other Numeric Indicators"] = unknown_cols

        if not groups:
            logger.warning("No numeric indicator columns found to plot.")
            return

        # ============================
        # Plot
        # ============================
        num_plots = len(groups)
        fig_height = max(6, 1.8 * num_plots)

        fig, axes = plt.subplots(
            num_plots,
            1,
            figsize=(14, fig_height),
            sharex=True
        )

        if num_plots == 1:
            axes = [axes]

        title = f"{self.strategy} | {self.symbol} | TC = {self.tc}"
        fig.suptitle(title, fontsize=14, fontweight="bold")

        for ax, (group_name, cols) in zip(axes, groups.items()):

            for col in cols:
                if col not in data.columns:
                    continue

                series = data[col].replace([np.inf, -np.inf], np.nan)

                if series.dropna().empty:
                    continue

                if col == "position":
                    ax.step(data.index, series, where="post", label=col, linewidth=1.5)
                    ax.axhline(1, linestyle="--", alpha=0.4)
                    ax.axhline(0, linestyle="--", alpha=0.3)
                    ax.axhline(-1, linestyle="--", alpha=0.4)

                elif col == "trades":
                    ax.bar(data.index, series, label=col, alpha=0.4)

                else:
                    ax.plot(data.index, series, label=col, linewidth=1.2)

            ax.set_title(group_name, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        plot_path = os.path.join(
            Plot_folder,
            f"Indicators_{plot_name}.png" if plot_name else "Indicators_plot.png"
        )

        plt.savefig(plot_path, dpi=200, bbox_inches="tight")

        if plot_show:
            plt.show()
        else:
            plt.close()