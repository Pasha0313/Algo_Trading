import numpy as np
import pandas as pd
import ta
from scipy.ndimage import gaussian_filter1d

# 0. **Simple Price and Volume**
def define_strategy_PV(data, parameters):
    data["returns"] = np.log(data.Close / data.Close.shift(1))
    data["vol_ch"] = np.log(data.Volume / data.Volume.shift(1))    
    return_thresh = np.percentile(data.returns.dropna(), [parameters[0], parameters[1]])
    volume_thresh = np.percentile(data.vol_ch.dropna(), [parameters[2], parameters[3]])
    data.dropna(inplace=True)

    cond1 = data.returns <= return_thresh[0]
    cond2 = data.vol_ch.between(volume_thresh[0], volume_thresh[1])
    cond3 = data.returns >= return_thresh[1]

    data["position"] = 0
    data.loc[cond1 & cond2, "position"] = 1
    data.loc[cond3 & cond2, "position"] = -1
    return data

# 1. **Simple Moving Average**
def define_strategy_SMA(data, parameters):
    data["SMA_S"] = data.Close.rolling(window=int(parameters[0])).mean()
    data["SMA_M"] = data.Close.rolling(window=int(parameters[1])).mean()
    data["SMA_L"] = data.Close.rolling(window=int(parameters[2])).mean()
    data.dropna(inplace=True)

    cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
    cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 2. **Moving Average Convergence Divergence (MACD) Histogram**
def define_strategy_MACD(data, parameters):
    data['MACD'] = ta.trend.macd(data['Close'], window_slow=int(parameters[0]), window_fast=int(parameters[1]))
    data['MACD_Signal'] = ta.trend.macd_signal(data['Close'], window_slow=int(parameters[0]), \
                                               window_fast=int(parameters[1]), window_sign=int(parameters[2]))
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    data.dropna(inplace=True)

    cond1 = (data['MACD_Histogram'] > 0)
    cond2 = (data['MACD_Histogram'] < 0)

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 3. **RSI with Moving Average**
def define_strategy_RSI_MA(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['SMA_RSI'] = ta.trend.SMAIndicator(data['RSI'], window=int(parameters[1])).sma_indicator()
    data.dropna(inplace=True)

    cond1 = (data['RSI'] > parameters[2]) & (data['RSI'] > data['SMA_RSI'])
    cond2 = (data['RSI'] < parameters[3]) & (data['RSI'] < data['SMA_RSI'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 4. **Relative Strength Index with Divergence (RSI Divergence)**
def define_strategy_RSI(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['RSI'] < parameters[1])
    cond2 = (data['RSI'] > parameters[2])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 5. **Stochastic RSI**
def define_strategy_Stochastic_RSI(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window=int(parameters[1]))
    data.dropna(inplace=True)

    #cond1 = (data['Stoch_RSI'] < parameters[2])
    #cond2 = (data['Stoch_RSI'] > parameters[3])

    cond1 = (data['Stoch_RSI'] < parameters[2]) & (data['Stoch_RSI'] > 0.01)
    cond2 = (data['Stoch_RSI'] > parameters[3]) & (data['Stoch_RSI'] < 0.99)


    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 6. **Bollinger Bands with ADX Trend Filter**
def define_strategy_Bollinger_ADX(data, parameters):
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Band'] = data['SMA'] + parameters[1] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[1] * data['std_dev']
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Band']) & (data['ADX'] > parameters[3])  
    cond2 = (data['Close'] < data['Lower_Band']) & (data['ADX'] > parameters[3])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1  
    return data

# 7. **Triple Exponential Moving Average (TEMA) with Momentum Filter**
def define_strategy_TEMA_momentum(data, parameters):
    data['TEMA'] = calculate_tema(data['Close'], window=int(parameters[0]))
    data['momentum'] = data['Close'].pct_change(periods=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['TEMA']) & (data['momentum'] > parameters[2])  
    cond2 = (data['Close'] < data['TEMA']) & (data['momentum'] < -parameters[2])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1  
    return data
def calculate_tema(series, window):
    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

# 8. **Volume Weighted Average Price (VWAP)**
def define_strategy_VWAP(data, parameters):
    data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], 
                                  data['Close'], data['Volume'], int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['VWAP'] + parameters[1])
    cond2 = (data['Close'] < data['VWAP'] - parameters[1])

    data['position'] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 9. **Volume Weighted Average Price (VWAP) with Momentum**
def define_strategy_VWAP_momentum(data, parameters):
    data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'], window=int(parameters[0]))
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))  
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['VWAP']) & (data['returns'] > parameters[1])
    cond2 = (data['Close'] < data['VWAP']) & (data['returns'] < parameters[2])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 10. **Bollinger Bands Breakout**
def define_strategy_Bollinger_breakout(data, parameters):
    data["SMA"] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data["std_dev"] = data["Close"].rolling(window=int(parameters[0])).std()
    data["Upper_Band"] = data["SMA"] + (parameters[1] * data["std_dev"])
    data["Lower_Band"] = data["SMA"] - (parameters[1] * data["std_dev"])
    data.dropna(inplace=True)  

    cond1 = (data["Close"] > data["Upper_Band"])
    cond2 = (data["Close"] < data["Lower_Band"])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

