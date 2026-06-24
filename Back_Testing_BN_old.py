from concurrent.futures import ProcessPoolExecutor, as_completed
from Back_Testing_Base_BN import BackTestingBase_BN
import Strategy as strategy  
from itertools import product
import numpy as np
import pandas as pd
import os


Optimize_folder = "Optimize"
os.makedirs(Optimize_folder, exist_ok=True)

def evaluate_param_set(args):
    data, strategy_name, params, tc, leverage, metric = args

    data = data.copy()

    if strategy_name == "EMA_MACD_ADX":
        results = strategy.define_strategy_EMA_MACD_ADX(data, params)
    elif strategy_name == "Supertrend_Stochastic_RSI":
        results = strategy.define_strategy_Supertrend_Stochastic_RSI(data, params)
    elif strategy_name == "Bollinger_Stochastic_RSI_Modified":
        results = strategy.define_strategy_Bollinger_Stochastic_RSI_modified(data, params)
    else:
        return params, np.nan

    results["strategy"] = results["position"].shift(1) * results["returns"]
    results["trades"] = results.position.diff().fillna(0).abs()
    results["strategy"] += results["trades"] * tc

    if metric == "Sharpe":
        simple_ret = np.exp(results["strategy"]) - 1
        lev_ret = leverage * simple_ret
        lev_ret = np.where(lev_ret < -1, -1, lev_ret)
        log_ret = np.log(pd.Series(lev_ret, index=results.index).add(1))

        years = (results.index[-1] - results.index[0]).days / 365.25
        tp_year = results.Close.count() / years

        ann_std = log_ret.std() * np.sqrt(tp_year)
        if ann_std == 0 or np.isnan(ann_std):
            return params, np.nan

        cagr = np.exp(log_ret.sum()) ** (1 / years) - 1
        score = cagr / ann_std

    else:
        score = np.exp(results["strategy"].sum())

    return params, score


class BackTesting_BN(BackTestingBase_BN):
    def __init__(self, client, symbol, bar_length, start, end=None, tc=0.0, leverage=5, strategy="PV"):
        super().__init__(client, symbol, bar_length, start, end, tc, leverage, strategy)

    def __repr__(self):
        return "\nFutures Backtester (symbol = {}, start = {}, end = {})\n".format(self.symbol, self.start, self.end)
    
    def prepare_data(self, parameters): #(return_prec_min,return_prec_high,volume_prec_min,volume_prec_high)
        data = self.data.copy()
        if self.strategy == "PV":  # 0. **Simple Price and Volume**
            self.results = strategy.define_strategy_PV(data, parameters)
        elif self.strategy == "SMA":  # 1. **Simple Moving Average**
            self.results = strategy.define_strategy_SMA(data, parameters)
        elif self.strategy == "MACD":  # 2. **Moving Average Convergence Divergence (MACD) Histogram**
            self.results = strategy.define_strategy_MACD(data, parameters)
        elif self.strategy == "RSI_MA":  # 3. **RSI with Moving Average**
            self.results = strategy.define_strategy_RSI_MA(data, parameters)
        elif self.strategy == "RSI":  # 4. **Relative Strength Index with Divergence (RSI Divergence)**
            self.results = strategy.define_strategy_RSI(data, parameters)
        elif self.strategy == "Stochastic_RSI":  # 5. **Stochastic RSI**
            self.results = strategy.define_strategy_Stochastic_RSI(data, parameters)
        elif self.strategy == "Bollinger_Bands_ADX":  # 6. **Bollinger Bands with ADX Trend Filter**
            self.results = strategy.define_strategy_Bollinger_ADX(data, parameters)
        elif self.strategy == "TEMA_momentum":  # 7. **Triple Exponential Moving Average (TEMA) with Momentum Filter**
            self.results = strategy.define_strategy_TEMA_momentum(data, parameters)
        elif self.strategy == "VWAP":  # 8. **Volume Weighted Average Price (VWAP)**
            self.results = strategy.define_strategy_VWAP(data, parameters)
        elif self.strategy == "VWAP_momentum":  # 9. **Volume Weighted Average Price (VWAP) with Momentum**
            self.results = strategy.define_strategy_VWAP_momentum(data, parameters)
        elif self.strategy == "Bollinger_breakout":  # 10. **Bollinger Bands Breakout**
            self.results = strategy.define_strategy_Bollinger_breakout(data, parameters)
        elif self.strategy == "Bollinger_squeeze":  # 11. **Bollinger Bands Squeeze**
            self.results = strategy.define_strategy_Bollinger_squeeze(data, parameters)
        elif self.strategy == "EMA_cross":  # 12. **Exponential Moving Average Cross Strategy**
            self.results = strategy.define_strategy_EMA_cross(data, parameters)
        elif self.strategy == "EMA_envelope":  # 13. **Exponential Moving Average Envelope**
            self.results = strategy.define_strategy_EMA_envelope(data, parameters)
        elif self.strategy == "TEMA":  # 14. **Triple Exponential Moving Average (TEMA)**
            self.results = strategy.define_strategy_TEMA(data, parameters)
        elif self.strategy == "Donchian":  # 15. **Donchian Channel**
            self.results = strategy.define_strategy_Donchian(data, parameters)
        elif self.strategy == "Aroon":  # 16. **Aroon Indicator**
            self.results = strategy.define_strategy_Aroon(data, parameters)
        elif self.strategy == "WilliamsR":  # 17. **Williams %R**
            self.results = strategy.define_strategy_WilliamsR(data, parameters)
        elif self.strategy == "Elder_Ray":  # 18. **Elder Ray Index**
            self.results = strategy.define_strategy_Elder_Ray(data, parameters)
        elif self.strategy == "Klinger":  # 19. **Klinger Oscillator**
            self.results = strategy.define_strategy_Klinger(data, parameters)
        elif self.strategy == "CMO":  # 20. **Chande Momentum Oscillator (CMO)**
            self.results = strategy.define_strategy_CMO(data, parameters)
        elif self.strategy == "Price_Oscillator":  # 21. **Price Oscillator**
            self.results = strategy.define_strategy_Price_Oscillator(data, parameters)
        elif self.strategy == "Ultimate_Oscillator":  # 22. **Ultimate Oscillator**
            self.results = strategy.define_strategy_Ultimate_Oscillator(data, parameters)
        elif self.strategy == "Chaikin":  # 23. **Chaikin Oscillator**
            self.results = strategy.define_strategy_Chaikin(data, parameters)
        elif self.strategy == "CMF":  # 24. **Chaikin Money Flow (CMF)**
            self.results = strategy.define_strategy_CMF(data, parameters)
        elif self.strategy == "Fractal_Chaos":  # 25. **Fractal Chaos Bands**
            self.results = strategy.define_strategy_Fractal_Chaos(data, parameters)
        elif self.strategy == "SuperTrend":  # 26. **SuperTrend**
            self.results = strategy.define_strategy_SuperTrend(data, parameters)
        elif self.strategy == "ZigZag":  # 27. **ZigZag Indicator**
            self.results = strategy.define_strategy_ZigZag(data, parameters)
        elif self.strategy == "Hull_MA":  # 28. **Hull Moving Average**
            self.results = strategy.define_strategy_Hull_MA(data, parameters)
        elif self.strategy == "Gann_Fan":  # 29. **Gann Fan**
            self.results = strategy.define_strategy_Gann_Fan(data, parameters)
        elif self.strategy == "ROC":  # 30. **Price Rate of Change (ROC)**
            self.results = strategy.define_strategy_ROC(data, parameters)
        elif self.strategy == "MFI_divergence":  # 31. **MFI (Money Flow Index) Divergence**
            self.results = strategy.define_strategy_MFI_divergence(data, parameters)
        elif self.strategy == "PSAR_simple":  # 32. **Parabolic SAR (PSAR)**
            self.results = strategy.define_strategy_PSAR_simple(data, parameters)
        elif self.strategy == "CMF_ADX":  # 33. **Chaikin Money Flow (CMF) with ADX**
            self.results = strategy.define_strategy_CMF_ADX(data, parameters)
        elif self.strategy == "PSAR_momentum":  # 34. **Parabolic SAR (PSAR) with Momentum**
            self.results = strategy.define_strategy_PSAR_momentum(data, parameters)
        elif self.strategy == "Trix":  # 35. **Trix Indicator**
            self.results = strategy.define_strategy_Trix(data, parameters)
        elif self.strategy == "Keltner_channel":  # 36. **Keltner Channel Breakout**
            self.results = strategy.define_strategy_Keltner_channel(data, parameters)
        elif self.strategy == "Momentum":  # 37. **Momentum Strategy**
            self.results = strategy.define_strategy_Momentum(data, parameters)
        elif self.strategy == "Ichimoku":  # 38. **Ichimoku Cloud**
            self.results = strategy.define_strategy_Ichimoku(data, parameters)
        elif self.strategy == "Zscore":  # 39. **Z-Score Mean Reversion**
            self.results = strategy.define_strategy_Zscore(data, parameters)
        elif self.strategy == "MA_envelope":  # 40. **Moving Average Envelope**
            self.results = strategy.define_strategy_MA_envelope(data, parameters)
        elif self.strategy == "ATR":  # 41. **Average True Range (ATR) Breakout**
            self.results = strategy.define_strategy_ATR(data, parameters)
        elif self.strategy == "ADX":  # 42. **Average Directional Index (ADX)**
            self.results = strategy.define_strategy_ADX(data, parameters)
        elif self.strategy == "CCI":  # 43. **Commodity Channel Index (CCI)**
            self.results = strategy.define_strategy_CCI(data, parameters)
        elif self.strategy == "Linear_Regression":  # 44. **Linear Regression Channel**
            self.results = strategy.define_strategy_Linear_Regression(data, parameters)
        elif self.strategy == "VWMA_Price_Oscillator":  # 45. **Volume Weighted Moving Average (VWMA) with Price Oscillator**
            self.results = strategy.define_strategy_VWMA_Price_Oscillator(data, parameters)
        elif self.strategy == "Dynamic_Pivot_Points":  # 46. **Dynamic Pivot Points Classic Strategy**
            self.results = strategy.define_strategy_Dynamic_Pivot_Points_Classic(data, parameters)
        elif self.strategy == "Force_Index":  # 47. **Force Index**
            self.results = strategy.define_strategy_Force_Index(data, parameters)
        elif self.strategy == "Chandelier_Exit":  # 48. **Chandelier Exit**
            self.results = strategy.define_strategy_Chandelier_Exit(data, parameters)
        elif self.strategy == "Fibonacci":  # 49. **Fibonacci Retracement Levels**
            self.results = strategy.define_strategy_Fibonacci(data, parameters)
        elif self.strategy == "ADL":  # 50. **Accumulation/Distribution Line (A/D Line)**
            self.results = strategy.define_strategy_ADL(data, parameters)
        elif self.strategy == "RSI_Bollinger":  # 51. **RSI with Bollinger Bands**
            self.results = strategy.define_strategy_RSI_Bollinger(data, parameters)
        elif self.strategy == "Turtle_Trading":  # 52. **Turtle Trading**
            self.results = strategy.define_strategy_Turtle_Trading(data, parameters)
        elif self.strategy == "Mean_Reversion":  # 53. **Mean Reversion Strategy**
            self.results = strategy.define_strategy_Mean_Reversion(data, parameters)
        elif self.strategy == "Breakout":  # 54. **Breakout Strategy**
            self.results = strategy.define_strategy_Breakout(data, parameters)
        elif self.strategy == "RSI_Divergence":  # 55. **RSI Divergence Strategy**
            self.results = strategy.define_strategy_RSI_Divergence(data, parameters)
        elif self.strategy == "MA_Cross_RSI":  # 56. **Moving Average Cross with RSI Filter**
            self.results = strategy.define_strategy_MA_Cross_RSI(data, parameters)
        elif self.strategy == "ADX_MA":  # 57. **ADX with Moving Averages**
            self.results = strategy.define_strategy_ADX_MA(data, parameters)
        elif self.strategy == "Bollinger_Breakout_Momentum":  # 58. **Bollinger Bands Breakout with Momentum Oscillator**
            self.results = strategy.define_strategy_Bollinger_Breakout_Momentum_Oscillator(data, parameters)
        elif self.strategy == "Fibonacci_MA":  # 59. **Fibonacci Retracement with Moving Average Filter**
            self.results = strategy.define_strategy_Fibonacci_MA(data, parameters)
        elif self.strategy == "Mean_Variance_Optimization":  # 60. **Mean-Variance Optimization Strategy**
            self.results = strategy.define_strategy_Mean_Variance_Optimization(data, parameters)
        elif self.strategy == "MA_ribbon":  # 61. **Moving Average Ribbon**
            self.results = strategy.define_strategy_MA_ribbon(data, parameters)
        elif self.strategy == "ADX_DI":  # 62. **ADX + DI (Directional Indicators)**
            self.results = strategy.define_strategy_ADX_DI(data, parameters)
        elif self.strategy == "MACD_RSI":  # 63. **MACD Histogram with RSI**
            self.results = strategy.define_strategy_MACD_RSI(data, parameters)
        elif self.strategy == "Fibonacci_retracement":  # 64. **Fibonacci Retracement Strategy**
            self.results = strategy.define_strategy_Fibonacci_retracement(data, parameters)
        elif self.strategy == "RSI_trend_reversal":  # 65. **Relative Strength Index (RSI) Trend Reversal Strategy**
            self.results = strategy.define_strategy_RSI_trend_reversal(data, parameters)
        elif self.strategy == "CMO_EMA":  # 66. **Chande Momentum Oscillator with EMA**
            self.results = strategy.define_strategy_CMO_EMA(data, parameters)
        elif self.strategy == "MA_momentum":  # 67. **Moving Average Cross with Momentum**
            self.results = strategy.define_strategy_MA_momentum(data, parameters)
        elif self.strategy == "RSI_Stochastic":  # 68. **RSI and Stochastic Oscillator**
            self.results = strategy.define_strategy_RSI_Stochastic(data, parameters)
        elif self.strategy == "Garman_Klass":  # 69. **Garman-Klass Volatility Strategy**
            self.results = strategy.define_strategy_Garman_Klass_Volatility(data, parameters)
        elif self.strategy == "Momentum_MACD":  # 70. **Momentum Oscillator with MACD**
            self.results = strategy.define_strategy_Momentum_MACD(data, parameters)
        elif self.strategy == "Bollinger_Stochastic":  # 71. **Bollinger Bands with Stochastic Oscillator**
            self.results = strategy.define_strategy_Bollinger_Stochastic(data, parameters)
        elif self.strategy == "Momentum_Breakout":  # 72. **Momentum Breakout Strategy**
            self.results = strategy.define_strategy_Momentum_Breakout(data, parameters)
        elif self.strategy == "EMA_MACD":  # 73. **Exponential Moving Average Convergence Divergence (EMA MACD)**
            self.results = strategy.define_strategy_EMA_MACD(data, parameters)
        elif self.strategy == "Bollinger_EMA":  # 74. **Bollinger Bands + EMA Cross**
            self.results = strategy.define_strategy_Bollinger_EMA(data, parameters)
        elif self.strategy == "MA_Momentum":  # 75. **Moving Average Cross with Momentum Filter**
            self.results = strategy.define_strategy_MA_Momentum_F(data, parameters)
        elif self.strategy == "Pivot_Stochastic":  # 76. **Pivot Points with Stochastic Oscillator**
            self.results = strategy.define_strategy_Pivot_Stochastic(data, parameters)
        elif self.strategy == "VWMA":  # 77. **Volume Weighted Moving Average (VWMA)**
            self.results = strategy.define_strategy_VWMA(data, parameters)
        elif self.strategy == "EMA_Momentum":  # 78. **Exponential Moving Average with Momentum**
            self.results = strategy.define_strategy_EMA_Momentum(data, parameters)
        elif self.strategy == "RSI_A_MA":  # 79. **RSI and Moving Average**
            self.results = strategy.define_strategy_RSI_A_MA(data, parameters)
        elif self.strategy == "EMA_Ribbon":  # 80. **Exponential Moving Average Ribbon**
            self.results = strategy.define_strategy_EMA_Ribbon(data, parameters)
        elif self.strategy == "RSI_MA_Envelope":  # 81. **RSI and Moving Average Envelope**
            self.results = strategy.define_strategy_RSI_MA_Envelope(data, parameters)
        elif self.strategy == "OBV_RSI":  # 82. **On-Balance Volume (OBV) with RSI**
            self.results = strategy.define_strategy_OBV_RSI(data, parameters)
        elif self.strategy == "SuperTrend_RSI":  # 83. **SuperTrend with RSI**
            self.results = strategy.define_strategy_ATR_RSI(data, parameters)
        elif self.strategy == "EMA_Bollinger":  # 84. **Exponential Moving Average with Bollinger Bands**
            self.results = strategy.define_strategy_EMA_Bollinger(data, parameters)
        elif self.strategy == "RSI_MA_Ribbon":  # 85. **RSI and Moving Average Ribbon**
            self.results = strategy.define_strategy_RSI_MA_Ribbon(data, parameters)
        elif self.strategy == "EMA_ADX":  # 86. **EMA Crossover with ADX Filter**
            self.results = strategy.define_strategy_EMA_ADX(data, parameters)
        elif self.strategy == "RSI_Bollinger_Momentum":  # 87. **RSI with Bollinger Bands and Momentum Filter**
            self.results = strategy.define_strategy_RSI_Bollinger_Momentum(data, parameters)
        elif self.strategy == "Renko_Box":  # 88. **Renko Box Trading Strategy**
            self.results = strategy.define_strategy_Renko_Box_Trading(data, parameters)
        elif self.strategy == "ADX_Stochastic":  # 89. **ADX with Stochastic Oscillator**
            self.results = strategy.define_strategy_ADX_Stochastic(data, parameters)
        elif self.strategy == "MA_Ribbon_ADX":  # 90. **Moving Average Ribbon with ADX Filter**
            self.results = strategy.define_strategy_MA_Ribbon_ADX(data, parameters)
        elif self.strategy == "EMA_Stochastic":  # 91. **EMA with Stochastic Oscillator**
            self.results = strategy.define_strategy_EMA_Stochastic(data, parameters)
        elif self.strategy == "RSI_ADX":  # 92. **RSI with ADX Filter**
            self.results = strategy.define_strategy_RSI_ADX(data, parameters)
        elif self.strategy == "MACD_Stochastic":  # 93. **MACD with Stochastic Oscillator**
            self.results = strategy.define_strategy_MACD_Stochastic(data, parameters)
        elif self.strategy == "MACD_Bollinger":  # 94. **MACD with Bollinger Bands Filter**
            self.results = strategy.define_strategy_MACD_Bollinger(data, parameters)
        elif self.strategy == "EMA_Stochastic_Filter":  # 95. **EMA Cross with Stochastic Filter**
            self.results = strategy.define_strategy_EMA_Stochastic_Filter(data, parameters)
        elif self.strategy == "MACD_MA_Ribbon":  # 96. **MACD with Moving Average Ribbon**
            self.results = strategy.define_strategy_MACD_MA_Ribbon(data, parameters)
        elif self.strategy == "RSI_MACD_Combo":  # 97. **RSI and MACD Combo Strategy**
            self.results = strategy.define_strategy_RSI_MACD_Combo(data, parameters)
        elif self.strategy == "Heikin_Ashi_Trend":  # 98. **Heikin Ashi Trend Continuation Strategy**
            self.results = strategy.define_strategy_Heikin_Ashi_Trend_Continuation(data, parameters)
        elif self.strategy == "Bollinger_Stochastic_RSI":  # 99. **Bollinger Bands with Stochastic RSI**
            self.results = strategy.define_strategy_Bollinger_Stochastic_RSI(data, parameters)
        elif self.strategy == "Trend_Reversal_RSI":  # 100. **Trend Reversal with RSI**
            self.results = strategy.define_strategy_Trend_Reversal_RSI(data, parameters)
        elif self.strategy == "Volume_Profile":  # 101. **Volume Profile Strategy**
            self.results = strategy.define_strategy_Volume_Profile(data, parameters)  
        elif self.strategy == "Grid_Trading":  # 102. **Grid Trading Strategy**
            self.results = strategy.define_strategy_Grid_Trading(data, parameters)                                          
        elif self.strategy == "EMA_MACD_ADX":  # 103. **EMA + MACD + ADX Hybrid Strategy**
            self.results = strategy.define_strategy_EMA_MACD_ADX(data, parameters)                                          
        elif self.strategy == "Trend_Momentum_Volatility":  # 104. **EMA + MACD + ADX + ATR Stochastic + RSI Hybrid Strategy**
            self.results = strategy.define_strategy_Trend_Momentum_Volatility(data, parameters)   
        elif self.strategy == "Stochastic_RSI_Bollinger_VWAP":  # 105. ** Stochastic RSI Bollinger VWAP Hybrid Strategy**
            self.results = strategy.define_strategy_Stochastic_RSI_Bollinger_VWAP(data, parameters)       
        elif self.strategy == "Stochastic_RSI_FULL":  #106. ** Stochastic RSI Strategy with %K and %D smoothing **
            self.results = strategy.define_strategy_Stochastic_RSI_FULL(data, parameters)  
        elif self.strategy == "Gaussian_Channel_FULL":  #107. **  Gaussian Channel strategy **
            self.results = strategy.define_strategy_Gaussian_Channel_FULL(data, parameters) 
        elif self.strategy == "Combined_Gaussian_Stochastic_RSI_FULL":  #108. ** combined the Gaussian Channel and Stochastic RSI strategies **
            self.results = strategy.define_strategy_Combined_Gaussian_Stochastic_RSI_FULL(data, parameters) 
        elif self.strategy == "OBV": # 109. **On-Balance Volume (OBV) **
            self.results = strategy.define_strategy_OBV(data)
        elif self.strategy == "Volume_delta": # 110. ** Volum Delta **
            self.results = strategy.define_strategy_volume_delta(data)
        elif self.strategy == "Ease_of_Movement": # 111. ** Ease of Movement **
            self.results = strategy.define_strategy_ease_of_movement(data, parameters)
        elif self.strategy == "WMA": # 112. ** Weight Moving Average **
            self.results = strategy.define_strategy_wma(data,parameters)
        elif self.strategy == "EMA": # 113. ** Exponential Moving Average **
            self.results = strategy.define_strategy_ema(data,parameters)
        elif self.strategy == "DEMA": # 114. ** Double Exponential Moving Average **
            self.results = strategy.define_strategy_dema(data, parameters)
        elif self.strategy == "AMA": # 115. ** Adaptive Moving Average **
            self.results = strategy.define_strategy_ama(data, parameters)
        elif self.strategy == "VIDYA": # 116. ** Variable Index Dynamic Average (VIDYA) **
            self.results = strategy.define_strategy_vidya(data, parameters)
        elif self.strategy == "SMA_cross":  # 117. ** Simple Moving Average Cross **
            self.results = strategy.define_strategy_SMA_cross(data, parameters)
        elif self.strategy == "Stochastic": # 118. **Stochastic Oscillator Strategy**
           self.results = strategy.define_strategy_Stochastic(data, parameters)
        elif self.strategy == "AO": # 119. **Awesome Oscillator (AO) Strategy**
            self.results = strategy.define_strategy_AO(data, parameters)
        elif self.strategy == "KST": # 120. **Know Sure Thing (KST) Strategy**
            self.results = strategy.define_strategy_KST(data, parameters)
        elif self.strategy == "Bollinger_SMA": # 121. **Bollinger Bands with SMA Strategy**
            self.results = strategy.define_strategy_Bollinger(data, parameters)
        elif self.strategy == "Bollinger_Keltner_Squeeze": # 122. **Bollinger Bands & Keltner Channel Squeeze Strategy**
           self.results = strategy.define_strategy_Squeeze(data, parameters)
        elif self.strategy == "StdDev_Channel": # 123. **Standard Deviation Channel Strategy**
            self.results = strategy.define_strategy_StdDev_Channel(data, parameters)
        elif self.strategy == "HV": # 124. **Historical Volatility (HV) Strategy**
            self.results = strategy.define_strategy_HV(data, parameters)
        elif self.strategy == "VR": # 125. ** Volatility Ratio (VR) Strategy **
            self.results = strategy.define_strategy_VR(data, parameters)
        elif self.strategy == "Simple_Pivot_Points": # 126. ** Simple_Pivot_Points Strategy **
            self.results = strategy.define_strategy_Simple_Pivot_Points(data)
        elif self.strategy == "DI": # 127. ** Directional Indicator (DI-Only) Strategy **
            self.results = strategy.define_strategy_DI(data, parameters)
        elif self.strategy == "Stochastic_RSI_StdDev_Channel": # 128. ** Stochastic RSI with Standard Deviation Channel Strategy **
            self.results = strategy.define_strategy_Stochastic_RSI_StdDev_Channel(data, parameters)    
        elif self.strategy == "Bollinger_Stochastic_RSI_Modified": #129. **Bollinger Bands with Stochastic RSI modified**
            self.results = strategy.define_strategy_Bollinger_Stochastic_RSI_modified(data, parameters)  
        elif self.strategy == "Keltner_Stochastic_RSI": #130. **Keltner Channel Calculation Bands with Stochastic RSI**
            self.results = strategy.define_strategy_Keltner_Stochastic_RSI(data, parameters)      
        elif self.strategy == "HMA_Stochastic_RSI": # 131. **Hull Moving Average Channel with Stochastic RSI**
            self.results = strategy.define_strategy_HMA_StochRSI(data, parameters)    
        elif self.strategy == "ADX_ATR_Bollinger_Stochastic_RSI":  # 132. **ADX ATR Bollinger Bands with Stochastic RSI**
            self.results = strategy.define_strategy_ADX_ATR_Bollinger_Stochastic_RSI(data, parameters)    
        elif self.strategy == "Supertrend_Stochastic_RSI":  # 133. **SuperTrend with Stochastic RSI**
            self.results = strategy.define_strategy_Supertrend_Stochastic_RSI(data, parameters)                                                                                     

    def optimize_strategy(self, param_ranges, metric="Multiple", output_file=None, Print_Data = False):
        if Print_Data : print("\nOptimize Strategy is running.")
        # Select the performance function based on the metric
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Generate parameter combinations using the provided ranges
        param_combinations = self._generate_param_combinations(param_ranges)

        performance = []
        valid_combinations = []        
        for params in param_combinations:
            self.prepare_data(parameters=params)
            self.run_backtest()

            if metric == "Sharpe":
                self.add_leverage(self.leverage, report=False)
                result = self.calculate_sharpe(np.log(self.results["strategy_leverage"].add(1)))
            else:
                result = performance_function(self.results.strategy)
            
            if not np.isnan(result):
                performance.append(result)
                valid_combinations.append(params)
            else:                
                if Print_Data : print(f"There is Nan in Performace Optimization")

        self.results_overview = pd.DataFrame(data=valid_combinations, columns=list(param_ranges.keys()))
        self.results_overview["performance"] = performance
        if Print_Data : print(f"Performance values:\n{self.results_overview}")
        self.results_overview.to_csv(os.path.join(Optimize_folder,'ALL_'+output_file), index=False)
        best_params = self.find_best_strategy(output_file)
        return best_params
    
    def _generate_param_combinations(self, param_ranges):
        ranges = [param_ranges[param] for param in param_ranges]
        return list(product(*ranges))

    def find_best_strategy(self, output_file):
        try:
            if self.results_overview.empty:
                print("Error: Results overview is empty. Ensure your strategy generates valid results.")
                return None  

            if "performance" not in self.results_overview.columns:
                print("Error: Column 'performance' not found in results overview.")
                return None

            best = self.results_overview.nlargest(1, "performance").iloc[0]
            best_params = best.to_dict()

            print("✅ Best Strategy Parameters:")
            for name, value in best_params.items():
                print(f"{name:<35}: {value:>10.2f}")
            print("\/" * 25)

            param_names = [c for c in self.results_overview.columns if c != "performance"]
            best_params_tuple = tuple(best[name] for name in param_names)

            self.test_strategy(best_params_tuple)

            if output_file:
                best_params_df = pd.DataFrame([best_params])
                best_params_df.to_csv(os.path.join(Optimize_folder, output_file), index=False)

            return best_params_tuple

        except IndexError:
            print("Error: No rows in results overview after applying nlargest.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def optimize_strategy_parallel(self, param_ranges, metric="Sharpe", output_file=None, max_workers=None):
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)

        param_combinations = list(product(*[param_ranges[p] for p in param_ranges]))

        jobs = [
            (self.data, self.strategy, params, self.tc, self.leverage, metric)
            for params in param_combinations
        ]

        performance = []
        valid_combinations = []

        print(f"Running parallel optimization with {max_workers} workers")
        print(f"Total combinations: {len(jobs)}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_param_set, job) for job in jobs]

            for future in as_completed(futures):
                params, score = future.result()

                if not np.isnan(score):
                    valid_combinations.append(params)
                    performance.append(score)

        self.results_overview = pd.DataFrame(
            data=valid_combinations,
            columns=list(param_ranges.keys())
        )
        self.results_overview["performance"] = performance

        if output_file:
            self.results_overview.to_csv(
                os.path.join(Optimize_folder, "ALL_" + output_file),
                index=False
            )

        best_params = self.find_best_strategy(output_file)
        return best_params