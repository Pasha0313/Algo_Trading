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

# 11. **Bollinger Bands Squeeze**
def define_strategy_Bollinger_squeeze(data, parameters):  
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Band'] = data['SMA'] + parameters[1] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[1] * data['std_dev']
    data.dropna(inplace=True)

    cond1 = (data['Upper_Band'] - data['Lower_Band'] < parameters[2])  
    cond2 = (data['Close'] > data['Upper_Band'])  
    cond3 = (data['Close'] < data['Lower_Band'])  

    data["position"] = 0
    data.loc[cond1 & cond2, "position"] = 1
    data.loc[cond1 & cond3, "position"] = -1
    return data

# 12. **Exponential Moving Average Cross Strategy**
def define_strategy_EMA_cross(data, parameters):
    data['EMA_S'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['EMA_L'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['EMA_S'] > data['EMA_L'])  
    cond2 = (data['EMA_S'] < data['EMA_L'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 13. **Exponential Moving Average Envelope**
def define_strategy_EMA_envelope(data, parameters):
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['Upper_Envelope'] = data['EMA'] * (1 + parameters[1] / 100)
    data['Lower_Envelope'] = data['EMA'] * (1 - parameters[1] / 100)
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Envelope'])  
    cond2 = (data['Close'] < data['Lower_Envelope'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 14. **Triple Exponential Moving Average (TEMA)**
def define_strategy_TEMA(data, parameters):
    data['TEMA'] = calculate_tema(data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['TEMA'])   
    cond2 = (data['Close'] < data['TEMA'])   

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data
def calculate_tema(series, window):
    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

# 15. **Donchian Channel**
def define_strategy_Donchian(data, parameters):
    data['Upper_Band'] = data['High'].rolling(window=int(parameters[0])).max()
    data['Lower_Band'] = data['Low'].rolling(window=int(parameters[0])).min()
    data['Middle_Band'] = (data['Upper_Band'] + data['Lower_Band']) / 2
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Band'])   
    cond2 = (data['Close'] < data['Lower_Band'])   

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 16. **Aroon Indicator**
def define_strategy_Aroon(data, parameters):
    data['Aroon_Up'] = ta.trend.aroon_up(high=data['High'], low=data['Low'], window=int(parameters[0]))
    data['Aroon_Down'] = ta.trend.aroon_down(high=data['High'], low=data['Low'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['Aroon_Up'] > parameters[1])  
    cond2 = (data['Aroon_Down'] > parameters[2])   

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 17. **Williams %R**
def define_strategy_WilliamsR(data, parameters):
    data['Williams_R'] = calculate_williams_r(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['Williams_R'] < parameters[1])  
    cond2 = (data['Williams_R'] > parameters[2])   

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data
def calculate_williams_r(high, low, close, window):
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    return ((highest_high - close) / (highest_high - lowest_low)) * -100

# 18. **Elder Ray Index**
def define_strategy_Elder_Ray(data, parameters):
    data['BullPower'] = data['High'] - ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['BearPower'] = data['Low'] - ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['BullPower'] > 0)  
    cond2 = (data['BearPower'] < 0)   

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 19. **Klinger Oscillator**
def define_strategy_Klinger(data, parameters):
    data['Klinger'] = calculate_klinger_oscillator(data['High'], data['Low'], data['Close'], data['Volume'], 
                                window_fast=int(parameters[0]), window_slow=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['Klinger'] > 0)   
    cond2 = (data['Klinger'] < 0)  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data
def calculate_klinger_oscillator(high, low, close, volume, window_fast, window_slow):
    tp = (high + low + close) / 3
    vf = tp.diff() * volume
    ema_fast = vf.ewm(span=window_fast, adjust=False).mean()
    ema_slow = vf.ewm(span=window_slow, adjust=False).mean()
    return ema_fast - ema_slow

# 20. **Chande Momentum Oscillator (CMO)**
def define_strategy_CMO(data, parameters):
    data['CMO'] = calculate_cmo(data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['CMO'] < parameters[1])   
    cond2 = (data['CMO'] > parameters[2])   

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data
def calculate_cmo(series, window):
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    sum_gains = gains.rolling(window=window).sum()
    sum_losses = losses.rolling(window=window).sum()

    cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    return cmo

# 21. **Price Oscillator**
def define_strategy_Price_Oscillator(data, parameters):
    data['EMA_Fast'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['EMA_Slow'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data['Price_Oscillator'] = ((data['EMA_Fast'] - data['EMA_Slow']) / data['EMA_Slow']) * 100
    data.dropna(inplace=True)

    cond1 = (data['Price_Oscillator'] > float(parameters[2])) 
    cond2 = (data['Price_Oscillator'] < -float(parameters[2])) 

    #cond1 = (data['Price_Oscillator'] > float(parameters[2])) & (data['Price_Oscillator'].shift(1) <= float(parameters[2]))
    #cond2 = (data['Price_Oscillator'] < -float(parameters[2])) & (data['Price_Oscillator'].shift(1) >= -float(parameters[2]))

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data

# 22. **Ultimate Oscillator**
def define_strategy_Ultimate_Oscillator(data, parameters):
    data['UO'] = ta.momentum.ultimate_oscillator(data['High'], data['Low'], data['Close'],\
                     window1=int(parameters[0]), window2=int(parameters[1]), window3=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['UO'] < parameters[3])
    cond2 = (data['UO'] > parameters[4])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 23. **Chaikin Oscillator**
def define_strategy_Chaikin(data, parameters):
    data['Chaikin_Oscillator'] = calculate_chaikin_oscillator(data['High']
                                , data['Low'], data['Close'], data['Volume'],
                                 window_slow=int(parameters[0]), window_fast=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['Chaikin_Oscillator'] > 0)
    cond2 = (data['Chaikin_Oscillator'] < 0)

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data
def calculate_chaikin_oscillator(high, low, close, volume, window_slow, window_fast):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1)  
    mfv = mfm * volume
    adl = mfv.cumsum()
    ema_fast = adl.ewm(span=window_fast, adjust=False).mean()
    ema_slow = adl.ewm(span=window_slow, adjust=False).mean()
    return ema_fast - ema_slow    

# 24. **Chaikin Money Flow (CMF)**
def define_strategy_CMF(data, parameters):
    data['CMF'] = ta.volume.chaikin_money_flow(data['High'], data['Low'],
                                                data['Close'], data['Volume'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['CMF'] > parameters[1])
    cond2 = (data['CMF'] < -parameters[1])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 25. **Fractal Chaos Bands**
def define_strategy_Fractal_Chaos(data, parameters):
    data['Fractal_High'] = detect_fractal_high(data).astype(int)
    data['Fractal_Low'] = detect_fractal_low(data).astype(int)
    data['Upper_Band'] = data['High'].rolling(window=int(parameters[0])).max() * data['Fractal_High']
    data['Lower_Band'] = data['Low'].rolling(window=int(parameters[0])).min() * data['Fractal_Low']
    data['Upper_Band'].replace(0, np.nan, inplace=True)
    data['Lower_Band'].replace(0, np.nan, inplace=True)
    data['Upper_Band'].fillna(method="ffill", inplace=True)
    data['Lower_Band'].fillna(method="ffill", inplace=True)
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Band']) 
    cond2 = (data['Close'] < data['Lower_Band'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1   
    data.loc[cond2, "position"] = -1  
    return data

def detect_fractal_high(data):
    return (data['High'] > data['High'].shift(1)) & \
        (data['High'] > data['High'].shift(2)) & \
        (data['High'] > data['High'].shift(-1)) & \
        (data['High'] > data['High'].shift(-2))

def detect_fractal_low(data):
    return (data['Low'] < data['Low'].shift(1)) & \
        (data['Low'] < data['Low'].shift(2)) & \
        (data['Low'] < data['Low'].shift(-1)) & \
        (data['Low'] < data['Low'].shift(-2))


# 26. **SuperTrend**
def define_strategy_SuperTrend(data, parameters):
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], 
                                     data['Close'], window=int(parameters[0]))
    data = calculate_supertrend(data, multiplier=float(parameters[1]))
    data.dropna(inplace=True)

    cond1 = data['Close'] < data['Lower_Band']
    cond2 = data['Close'] > data['Upper_Band']

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data
def calculate_supertrend(data, multiplier):
    high = data['High']
    low = data['Low']
    close = data['Close']
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * data['ATR'] )
    lower_band = hl2 - (multiplier * data['ATR'] )
    supertrend = np.zeros(len(data))
    supertrend[0] = close[0]

    for i in range(1, len(data)):
        if close[i] > upper_band[i - 1]:
            supertrend[i] = lower_band[i]  
        elif close[i] < lower_band[i - 1]:
            supertrend[i] = upper_band[i]  
        else:
            supertrend[i] = supertrend[i - 1]  

    data['Supertrend'] = supertrend
    data['Upper_Band'] = upper_band
    data['Lower_Band'] = lower_band
    return data


# 27. **ZigZag**
def define_strategy_ZigZag(data, parameters):
    data['ZigZag'] = calculate_zigzag(data['Close'], float(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['ZigZag'].notna()) & (data['ZigZag'].shift(1).isna()) & (data['Close'] > data['ZigZag'])  
    cond2 = (data['ZigZag'].notna()) & (data['ZigZag'].shift(1).isna()) & (data['Close'] < data['ZigZag'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1   
    data.loc[cond2, "position"] = -1  
    return data
def calculate_zigzag(data, threshold):
    zigzag = [None] * len(data)
    last_extreme = data.iloc[0]
    is_peak = True  
    for i in range(1, len(data)):
        price = data.iloc[i]
        change = (price - last_extreme) / last_extreme * 100
        if is_peak and change >= threshold:
            zigzag[i] = price  
            last_extreme = price
            is_peak = False  
        elif not is_peak and change <= -threshold:
            zigzag[i] = price 
            last_extreme = price
            is_peak = True  
        else:
            zigzag[i] = zigzag[i - 1]  
    return pd.Series(zigzag, index=data.index)



# 28. **Hull Moving Average**
def define_strategy_Hull_MA(data, parameters):
    data['Hull_MA'] = hull_moving_average(data['Close'], int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Hull_MA'])
    cond2 = (data['Close'] < data['Hull_MA'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data
def hull_moving_average(close, window):
    """ Calculate Hull Moving Average (HMA) """
    wma_half = ta.trend.wma_indicator(close, window=window//2)
    wma_full = ta.trend.wma_indicator(close, window=window)
    hma = ta.trend.wma_indicator(2 * wma_half - wma_full, window=int(np.sqrt(window)))
    return hma

# 29. **Gann Fan**
def define_strategy_Gann_Fan(data, parameters):
    gann_data = calculate_gann_fan(data['Close'], int(parameters[0]) )
    data = data.join(gann_data)
    data.dropna(subset=['Gann_1x1'], inplace=True)

    cond1 = (data['Close'] > data['Gann_1x1']) & (data['Close'].shift(1) <= data['Gann_1x1'])
    cond2 = (data['Close'] < data['Gann_1x1']) & (data['Close'].shift(1) >= data['Gann_1x1'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1  
    return data
def calculate_gann_fan(data, window):
    gann = pd.DataFrame(index=data.index)
    gann['Pivot_High'] = data.rolling(window=window, min_periods=1).max()
    gann['Pivot_Low'] = data.rolling(window=window, min_periods=1).min()

    step = np.linspace(0, 1, num=len(data))

    gann['Gann_1x1'] = gann['Pivot_Low'] + step * (gann['Pivot_High'] - gann['Pivot_Low'])
    gann['Gann_2x1'] = gann['Pivot_Low'] + step * (gann['Pivot_High'] - gann['Pivot_Low']) / 2
    gann['Gann_3x1'] = gann['Pivot_Low'] + step * (gann['Pivot_High'] - gann['Pivot_Low']) / 3
    return gann



    
# 30. **Price Rate of Change (ROC)**
def define_strategy_ROC(data, parameters):
    data['ROC'] = ta.momentum.roc(data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['ROC'] > parameters[1])
    cond2 = (data['ROC'] < parameters[2])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 31. **MFI (Money Flow Index) Divergence**
def define_strategy_MFI_divergence(data, parameters):
    data['MFI'] = ta.volume.money_flow_index(data['High'], data['Low'],
                      data['Close'], data['Volume'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['MFI'] < parameters[1])
    cond2 = (data['MFI'] > parameters[2])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 32. **Parabolic SAR (PSAR)**
def define_strategy_PSAR_simple(data, parameters):
    data['PSAR'], data['Trend'] = calculate_psar(data['High'], data['Low'], float(parameters[0]), float(parameters[1]))

    cond1 = (data['Close'] > data['PSAR']) & (data['Trend'] == 1) & (data['Trend'].shift(1) == -1)  
    cond2 = (data['Close'] < data['PSAR']) & (data['Trend'] == -1) & (data['Trend'].shift(1) == 1)  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 

    return data
def calculate_psar(high, low, acceleration=0.02, maximum=0.2):
    psar = np.zeros(len(high))
    trend = np.ones(len(high))  
    af = acceleration  
    ep = high.iloc[0]  
    psar[0] = low.iloc[0] 
    for i in range(1, len(high)):
        psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
        if trend[i - 1] == 1:  
            if low.iloc[i] < psar[i]:  
                trend[i] = -1
                psar[i] = ep  
                af = acceleration
                ep = low.iloc[i]
            else:
                trend[i] = 1
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + acceleration, maximum)
        else:  
            if high.iloc[i] > psar[i]:  
                trend[i] = 1
                psar[i] = ep
                af = acceleration
                ep = high.iloc[i]
            else:
                trend[i] = -1
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + acceleration, maximum)
    return pd.Series(psar, index=high.index), pd.Series(trend, index=high.index)

# 33. **Chaikin Money Flow (CMF) Calculation**
def define_strategy_CMF_ADX(data, parameters):
    data['CMF'] = ta.volume.chaikin_money_flow(data['High'], data['Low'], data['Close'], data['Volume'], window=int(parameters[0]))
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['CMF'] > 0) & (data['ADX'] > parameters[2])  
    cond2 = (data['CMF'] < 0) & (data['ADX'] > parameters[2])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data

# 34. **Parabolic SAR (PSAR) with Momentum**
def define_strategy_PSAR_momentum(data, parameters):
    data['PSAR'] = calculate_psar(data['High'], data['Low'], data['Close'], parameters[0], parameters[1])    
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['PSAR']) & (data['returns'] > parameters[2])
    cond2 = (data['Close'] < data['PSAR']) & (data['returns'] < parameters[3])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 35. **Trix Indicator**
def define_strategy_Trix(data, parameters):
    data['Trix'] = ta.trend.trix(data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['Trix'] > 0)
    cond2 = (data['Trix'] < 0)

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 36. **Keltner Channel Breakout**
def define_strategy_Keltner_channel(data, parameters):
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], \
                                         data['Close'], window=int(parameters[0]))
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data['Upper_Band'] = data['EMA'] + parameters[2] * data['ATR']
    data['Lower_Band'] = data['EMA'] - parameters[2] * data['ATR']
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Band'])
    cond2 = (data['Close'] < data['Lower_Band'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 37. **Momentum Strategy**
def define_strategy_Momentum(data, parameters):
    data['Momentum'] = data['Close'] - data['Close'].shift(parameters[0])
    data.dropna(inplace=True)

    cond1 = (data['Momentum'] > parameters[1])
    cond2 = (data['Momentum'] < -parameters[1])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 38. **Ichimoku Cloud**
def define_strategy_Ichimoku(data, parameters):
    data['conversion_line'] = ichimoku_conversion_line(data['High'], data['Low'], int(parameters[0]))
    data['base_line'] = ichimoku_base_line(data['High'], data['Low'], int(parameters[1]))
    data['leading_span_a'] = ichimoku_leading_span_a(data['conversion_line'], data['base_line'])
    data['leading_span_b'] = ichimoku_leading_span_b(data['High'], data['Low'], int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['conversion_line'] > data['base_line'])
    cond2 = (data['conversion_line'] < data['base_line'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data
def ichimoku_conversion_line(high, low, period):
    return (high.rolling(window=period).max() + low.rolling(window=period).min()) / 2
def ichimoku_base_line(high, low, period):
    return (high.rolling(window=period).max() + low.rolling(window=period).min()) / 2
def ichimoku_leading_span_a(conversion_line, base_line):
    return (conversion_line + base_line) / 2
def ichimoku_leading_span_b(high, low, period):
    return (high.rolling(window=period).max() + low.rolling(window=period).min()) / 2

# 39. **Z-Score Mean Reversion**
def define_strategy_Zscore(data, parameters):
    data['mean'] = data['Close'].rolling(window=int(parameters[0])).mean()
    data['std'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['z_score'] = (data['Close'] - data['mean']) / data['std']
    data.dropna(inplace=True)

    cond1 = (data['z_score'] > parameters[1])
    cond2 = (data['z_score'] < -parameters[1])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 40. **Moving Average Envelope**
def define_strategy_MA_envelope(data, parameters):
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['Upper_Envelope'] = data['SMA'] * (1 + parameters[1] / 100)
    data['Lower_Envelope'] = data['SMA'] * (1 - parameters[1] / 100)
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Envelope'])
    cond2 = (data['Close'] < data['Lower_Envelope'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 41. **Average True Range (ATR) Breakout**
def define_strategy_ATR(data, parameters):
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data['Upper_Band'] = data['Close'] + parameters[1] * data['ATR']
    data['Lower_Band'] = data['Close'] - parameters[1] * data['ATR']
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Band'])
    cond2 = (data['Close'] < data['Lower_Band'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 42. **Average Directional Index (ADX)**
def define_strategy_ADX(data, parameters):
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['ADX'] > parameters[1])
    cond2 = (data['ADX'] < parameters[2])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 43. **Commodity Channel Index (CCI)**
def define_strategy_CCI(data, parameters):
    data['CCI'] = ta.trend.cci(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['CCI'] > parameters[1])
    cond2 = (data['CCI'] < -parameters[1])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 44. **Linear Regression Channel**
def define_strategy_Linear_Regression(data, parameters):
    data['Linear_Regression'], data['Std_Dev'] = calculate_linear_regression(data['Close'],  int(parameters[0]) )
    data['Upper_Band'] = data['Linear_Regression'] + (parameters[1] * data['Std_Dev'])
    data['Lower_Band'] = data['Linear_Regression'] - (parameters[1] * data['Std_Dev'])
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Band'])  
    cond2 = (data['Close'] < data['Lower_Band'])  

    data["position"] = 0
    data.loc[cond1, "position"] = -1  
    data.loc[cond2, "position"] = 1   
    return data
def calculate_linear_regression(series, window):
    from scipy.stats import linregress
    x = np.arange(window)
    lr_values = []
    std_dev = []

    for i in range(len(series)):
        if i < window - 1:
            lr_values.append(np.nan)
            std_dev.append(np.nan)
        else:
            y = series[i - window + 1:i + 1].values
            slope, intercept, _, _, _ = linregress(x, y)
            predicted = slope * x + intercept
            lr_values.append(predicted[-1])
            std_dev.append(np.std(y - predicted))  
    return pd.Series(lr_values, index=series.index), pd.Series(std_dev, index=series.index)


# 45. **Volume Weighted Moving Average (VWMA) Calculation** 
def define_strategy_VWMA_Price_Oscillator(data, parameters):
    data['VWMA'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'], window=int(parameters[0]))
    data['EMA_Fast'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data['EMA_Slow'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[2]))
    data['Price_Oscillator'] = data['EMA_Fast'] - data['EMA_Slow']
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['VWMA']) & (data['Price_Oscillator'] > 0)  
    cond2 = (data['Close'] < data['VWMA']) & (data['Price_Oscillator'] < 0)  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1  
    return data

# 46. **Dynamic Pivot Points Classic Strategy**
def define_strategy_Dynamic_Pivot_Points_Classic(data, parameters):
    data['Pivot_Point'] = (data['High'].rolling(window=int(parameters[1])).max() +
        data['Low'].rolling(window=int(parameters[1])).min() +
        data['Close'].rolling(window=int(parameters[1])).mean()) / 3
    data['R1'] = data['Pivot_Point'] + parameters[0] * (data['High'] - data['Low'])
    data['S1'] = data['Pivot_Point'] - parameters[0] * (data['High'] - data['Low'])
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['R1'])
    cond2 = (data['Close'] < data['S1'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 47. **Force Index**
def define_strategy_Force_Index(data, parameters):
    data['Force_Index'] = calculate_force_index(data['Close'], data['Volume'],
                                                 int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['Force_Index'] > 0)
    cond2 = (data['Force_Index'] < 0)

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data
def calculate_force_index(close, volume, window):
    fi = (close.diff() * volume).rolling(window=window).sum()
    return fi

# 48. **Chandelier Exit**
def define_strategy_Chandelier_Exit(data, parameters):
    data['ATR'] = ta.volatility.average_true_range(data['High'],
                         data['Low'], data['Close'], window=int(parameters[0]))
    data['Chandelier_Exit'] = data['Close'] - parameters[1] * data['ATR']
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Chandelier_Exit'])
    cond2 = (data['Close'] < data['Chandelier_Exit'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 49. **Fibonacci**
def define_strategy_Fibonacci(data, parameters):
    data['High_Max'] = data['High'].rolling(window=int(parameters[0])).max()
    data['Low_Min'] = data['Low'].rolling(window=int(parameters[0])).min()
    data['Fibonacci_23_6'] = data['Low_Min'] + 0.236 * (data['High_Max'] - data['Low_Min'])
    data['Fibonacci_38_2'] = data['Low_Min'] + 0.382 * (data['High_Max'] - data['Low_Min'])
    data['Fibonacci_50'] = data['Low_Min'] + 0.5 * (data['High_Max'] - data['Low_Min'])
    data['Fibonacci_61_8'] = data['Low_Min'] + 0.618 * (data['High_Max'] - data['Low_Min'])
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Fibonacci_50']) & (data['Close'] < data['Fibonacci_61_8'])
    cond2 = (data['Close'] < data['Fibonacci_38_2']) & (data['Close'] > data['Fibonacci_23_6'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 50. **Accumulation/Distribution Line (A/D Line)**
def define_strategy_ADL(data, parameters):
    data['ADL'] = ta.volume.acc_dist_index(data['High'], data['Low'],
                                            data['Close'], data['Volume'])
    data.dropna(inplace=True)

    cond1 = (data['ADL'] > parameters[0])
    cond2 = (data['ADL'] < parameters[1])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 51. **RSI with Bollinger Bands**
def define_strategy_RSI_Bollinger(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[1])).std()
    data['Upper_Band'] = data['SMA'] + parameters[2] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[2] * data['std_dev']
    data.dropna(inplace=True)

    cond1 = (data['RSI'] < parameters[3]) & (data['Close'] < data['Lower_Band'])  
    cond2 = (data['RSI'] > parameters[4]) & (data['Close'] > data['Upper_Band'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 52. **Turtle Trading**
def define_strategy_Turtle_Trading(data, parameters):
    data['High_Max'] = data['High'].rolling(window=int(parameters[0])).max()
    data['Low_Min'] = data['Low'].rolling(window=int(parameters[1])).min()
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['High_Max'].shift(1))  
    cond2 = (data['Close'] < data['Low_Min'].shift(1))   

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 53. **Mean Reversion Strategy**
def define_strategy_Mean_Reversion(data, parameters):
    data['mean'] = data['Close'].rolling(window=int(parameters[0])).mean()
    data['std'] = data['Close'].rolling(window=int(parameters[0])).std()
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['mean'] + parameters[1] * data['std'])  
    cond2 = (data['Close'] < data['mean'] - parameters[1] * data['std'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 54. **Breakout Strategy**
def define_strategy_Breakout(data, parameters):
    data['High_Max'] = data['High'].rolling(window=int(parameters[0])).max()
    data['Low_Min'] = data['Low'].rolling(window=int(parameters[1])).min()
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['High_Max'].shift(1))  
    cond2 = (data['Close'] < data['Low_Min'].shift(1))   

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 55. **RSI Divergence Strategy**
def define_strategy_RSI_Divergence(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['RSI_Signal'] = ta.trend.SMAIndicator(data['RSI'], window=int(parameters[1])).sma_indicator()
    data.dropna(inplace=True)

    cond1 = (data['RSI'] < parameters[2])  
    cond2 = (data['RSI'] > parameters[3])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 56. **Moving Average Cross with RSI Filter**
def define_strategy_MA_Cross_RSI(data, parameters):
    data['SMA_S'] = data['Close'].rolling(window=int(parameters[0])).mean()
    data['SMA_L'] = data['Close'].rolling(window=int(parameters[1])).mean()
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['SMA_S'] > data['SMA_L']) & (data['RSI'] < parameters[3])  
    cond2 = (data['SMA_S'] < data['SMA_L']) & (data['RSI'] > parameters[4])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 57. **ADX with Moving Averages**
def define_strategy_ADX_MA(data, parameters):
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data['SMA_S'] = data['Close'].rolling(window=int(parameters[1])).mean()
    data['SMA_L'] = data['Close'].rolling(window=int(parameters[2])).mean()
    data.dropna(inplace=True)

    cond1 = (data['ADX'] > parameters[3]) & (data['SMA_S'] > data['SMA_L'])  
    cond2 = (data['ADX'] < parameters[4]) & (data['SMA_S'] < data['SMA_L'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 58. **Bollinger Bands Breakout with Momentum Oscillator**
def define_strategy_Bollinger_Breakout_Momentum_Oscillator(data, parameters):
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Band'] = data['SMA'] + parameters[1] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[1] * data['std_dev']
    data['Momentum_Oscillator'] = ta.momentum.roc(data['Close'], window=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Band']) & (data['Momentum_Oscillator'] > 0)  
    cond2 = (data['Close'] < data['Lower_Band']) & (data['Momentum_Oscillator'] < 0)  

    data["position"] = 0
    data.loc[cond1, "position"] = 1 
    data.loc[cond2, "position"] = -1
    return data

# 59. **Fibonacci Retracement with Moving Average Filter**
def define_strategy_Fibonacci_MA(data, parameters):
    data['High_Max'] = data['High'].rolling(window=int(parameters[0])).max()
    data['Low_Min'] = data['Low'].rolling(window=int(parameters[0])).min()
    data['Fibonacci_23_6'] = data['Low_Min'] + 0.236 * (data['High_Max'] - data['Low_Min'])
    data['Fibonacci_38_2'] = data['Low_Min'] + 0.382 * (data['High_Max'] - data['Low_Min'])
    data['Fibonacci_50'] = data['Low_Min'] + 0.5 * (data['High_Max'] - data['Low_Min'])
    data['Fibonacci_61_8'] = data['Low_Min'] + 0.618 * (data['High_Max'] - data['Low_Min'])
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data.dropna(inplace=True)

    cond1 = ((data['Close'] > data['Fibonacci_38_2']) & (data['Close'] < data['Fibonacci_50'])) & (data['Close'] > data['SMA'])
    cond2 = ((data['Close'] < data['Fibonacci_38_2']) & (data['Close'] > data['Fibonacci_23_6'])) & (data['Close'] < data['SMA'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1  
    return data


# 60. **Mean-Variance Optimization Strategy**
def define_strategy_Mean_Variance_Optimization(data, parameters):
    data['mean'] = data['Close'].rolling(window=int(parameters[0])).mean()
    data['variance'] = data['Close'].rolling(window=int(parameters[1])).var()
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['mean'] + parameters[2] * data['variance'])  
    cond2 = (data['Close'] < data['mean'] - parameters[2] * data['variance'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 61. **Moving Average Ribbon**
def define_strategy_MA_ribbon(data, parameters):
    data["SMA_short"] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data["SMA_long"] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data.dropna(inplace=True)

    cond1 = (data['SMA_short'] > data['SMA_long'])  
    cond2 = (data['SMA_short'] < data['SMA_long'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 62. **ADX + DI (Directional Indicators)**
def define_strategy_ADX_DI(data, parameters):
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data['DI_plus'] = ta.trend.adx_pos(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data['DI_minus'] = ta.trend.adx_neg(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['DI_plus'] > data['DI_minus']) & (data['ADX'] > parameters[1])  
    cond2 = (data['DI_plus'] < data['DI_minus']) & (data['ADX'] > parameters[1])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 63. **MACD Histogram with RSI**
def define_strategy_MACD_RSI(data, parameters):
    data['MACD'] = ta.trend.macd(data['Close'], window_slow=int(parameters[0]), window_fast=int(parameters[1]))
    data['MACD_Signal'] = ta.trend.macd_signal(data['Close'], window_slow=int(parameters[0]), window_fast=int(parameters[1]), window_sign=int(parameters[2]))
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[3]))
    data.dropna(inplace=True)

    cond1 = (data['MACD_Histogram'] > 0) & (data['RSI'] < parameters[4])  
    cond2 = (data['MACD_Histogram'] < 0) & (data['RSI'] > parameters[5])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 64. **Fibonacci Retracement Strategy** # Simpler Version
def define_strategy_Fibonacci_retracement(data, parameters):
    data['High_Swing'] = data['High'].rolling(window=int(parameters[0])).max()
    data['Low_Swing'] = data['Low'].rolling(window=int(parameters[0])).min()

    data['Fibonacci_23_6'] = data['Low_Swing'] + 0.236 * (data['High_Swing'] - data['Low_Swing'])
    data['Fibonacci_38_2'] = data['Low_Swing'] + 0.382 * (data['High_Swing'] - data['Low_Swing'])
    data['Fibonacci_50']   = data['Low_Swing'] + 0.5   * (data['High_Swing'] - data['Low_Swing'])
    data['Fibonacci_61_8'] = data['Low_Swing'] + 0.618 * (data['High_Swing'] - data['Low_Swing'])
    data['Fibonacci_78_6'] = data['Low_Swing'] + 0.786 * (data['High_Swing'] - data['Low_Swing'])

    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Fibonacci_50']) & (data['Close'].shift(1) <= data['Fibonacci_50'])  
    cond2 = (data['Close'] < data['Fibonacci_38_2']) & (data['Close'].shift(1) >= data['Fibonacci_38_2']) 
    cond3 = (data['Close'] > data['Fibonacci_61_8']) & (data['Close'].shift(1) <= data['Fibonacci_61_8'])  
    cond4 = (data['Close'] < data['Fibonacci_23_6']) & (data['Close'].shift(1) >= data['Fibonacci_23_6'])  

    # Initialize Position Column
    data["position"] = 0
    data.loc[cond1, "position"] = 1   # Buy at Fibonacci 50% Breakout
    data.loc[cond2, "position"] = -1  # Sell at Fibonacci 38.2% Breakdown
    data.loc[cond3, "position"] = 2   # Strong Buy at Fibonacci 61.8% (Golden Ratio)
    data.loc[cond4, "position"] = -2  # Strong Sell at Fibonacci 23.6% Breakdown
    return data

# 65. **Relative Strength Index (RSI) Trend Reversal Strategy**
def define_strategy_RSI_trend_reversal(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)
    
    cond1 = (data['RSI'] < parameters[2]) & (data['Close'] > data['EMA'])  
    cond2 = (data['RSI'] > parameters[3]) & (data['Close'] < data['EMA'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data

# 66. **Chande Momentum Oscillator with EMA**
def define_strategy_CMO_EMA(data, parameters):
    data['CMO'] = calculate_cmo(data['Close'], window=int(parameters[0]))
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['CMO'] > parameters[2]) & (data['Close'] > data['EMA'])  
    cond2 = (data['CMO'] < parameters[3]) & (data['Close'] < data['EMA'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 67. **Moving Average Cross with Momentum**
def define_strategy_MA_momentum(data, parameters):
    data['SMA_Short'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['SMA_Long'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data['momentum'] = data['Close'] - data['Close'].shift(parameters[2])
    data.dropna(inplace=True)

    cond1 = (data['SMA_Short'] > data['SMA_Long']) & (data['momentum'] > parameters[3])  
    cond2 = (data['SMA_Short'] < data['SMA_Long']) & (data['momentum'] < parameters[4])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 68. **RSI and Stochastic Oscillator**
def define_strategy_RSI_Stochastic(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['Stochastic'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['RSI'] < parameters[2]) & (data['Stochastic'] < parameters[3])  
    cond2 = (data['RSI'] > parameters[4]) & (data['Stochastic'] > parameters[5])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 69. **Garman-Klass Volatility Strategy** # 
def define_strategy_Garman_Klass_Volatility(data, parameters):
    data['GK_Volatility'] = 0.5 * (((data['High'] - data['Low']) ** 2) \
            - (0.25 * ((2 * data['Close'] - data['High'] - data['Low']) ** 2))) / data['Close']
    data.dropna(inplace=True)
    threshold = data['GK_Volatility'].mean() + parameters[0] * data['GK_Volatility'].std()

    cond1 = data['GK_Volatility'] > threshold  
    cond2 = data['GK_Volatility'] < threshold  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data

# 70. **Momentum Oscillator with MACD**
def define_strategy_Momentum_MACD(data, parameters):
    data['Momentum'] = data['Close'] - data['Close'].shift(parameters[0])
    data['MACD'] = ta.trend.macd(data['Close'], window_slow=int(parameters[1]), window_fast=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['Momentum'] > parameters[3]) & (data['MACD'] > 0)  
    cond2 = (data['Momentum'] < -parameters[4]) & (data['MACD'] < 0)  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 71. **Bollinger Bands with Stochastic Oscillator**
def define_strategy_Bollinger_Stochastic(data, parameters):
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Band'] = data['SMA'] + parameters[1] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[1] * data['std_dev']
    data['Stochastic'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['Stochastic'] < parameters[3]) & (data['Close'] < data['Lower_Band'])
    cond2 = (data['Stochastic'] > parameters[4]) & (data['Close'] > data['Upper_Band'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 72. **Momentum Breakout Strategy**
def define_strategy_Momentum_Breakout(data, parameters):
    data['returns'] = data['Close'].pct_change()
    data['momentum'] = data['returns'].rolling(window=int(parameters[0])).sum()
    data.dropna(inplace=True)

    cond1 = data['momentum'] > parameters[1]
    cond2 = data['momentum'] < -parameters[1]

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 73. **Exponential Moving Average Convergence Divergence (EMA MACD)**
def define_strategy_EMA_MACD(data, parameters):
    data['EMA_fast'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['EMA_slow'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data['MACD'] = data['EMA_fast'] - data['EMA_slow']
    data.dropna(inplace=True)

    cond1 = data['MACD'] > parameters[2]
    cond2 = data['MACD'] < -parameters[3]

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 74. **Bollinger Bands + EMA Cross**
def define_strategy_Bollinger_EMA(data, parameters):
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Band'] = data['SMA'] + parameters[1] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[1] * data['std_dev']
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Band']) & (data['Close'] > data['EMA'])
    cond2 = (data['Close'] < data['Lower_Band']) & (data['Close'] < data['EMA'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 75. **Moving Average Cross with Momentum Filter**
def define_strategy_MA_Momentum_F(data, parameters):
    data['SMA_short'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['SMA_long'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data['momentum'] = data['Close'].pct_change(periods=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['SMA_short'] > data['SMA_long']) & (data['momentum'] > parameters[3])
    cond2 = (data['SMA_short'] < data['SMA_long']) & (data['momentum'] < -parameters[3])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 76. **Pivot Points with Stochastic Oscillator**
def define_strategy_Pivot_Stochastic(data, parameters):
    data['Pivot_Point'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['R1'] = data['Pivot_Point'] + (data['High'] - data['Low'])
    data['S1'] = data['Pivot_Point'] - (data['High'] - data['Low'])
    data['Stochastic'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = (data['Close'] < data['S1']) & (data['Stochastic'] < parameters[1])
    cond2 = (data['Close'] > data['R1']) & (data['Stochastic'] > parameters[2])

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1  
    return data

# 77. **Volume Weighted Moving Average (VWMA)**
def define_strategy_VWMA(data, parameters):
    data['VWMA'] = (data['Close'] * data['Volume']).rolling(window=int(parameters[0])).sum() \
                                      / data['Volume'].rolling(window=int(parameters[0])).sum()
    data.dropna(inplace=True)

    cond1 = data['Close'] > data['VWMA']
    cond2 = data['Close'] < data['VWMA']

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 78. **Exponential Moving Average with Momentum**
def define_strategy_EMA_Momentum(data, parameters):
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['momentum'] = data['Close'].pct_change(periods=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = data['momentum'] > parameters[2]
    cond2 = data['momentum'] < -parameters[2]

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 79. **RSI and Moving Average**
def define_strategy_RSI_A_MA(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data.dropna(inplace=True)

    cond1 = data['RSI'] < parameters[2]
    cond2 = data['RSI'] > parameters[3]

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 80. **Exponential Moving Average Ribbon**
def define_strategy_EMA_Ribbon(data, parameters):
    data['EMA_short'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['EMA_medium'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data['EMA_long'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['EMA_short'] > data['EMA_medium']) & (data['EMA_medium'] > data['EMA_long'])
    cond2 = (data['EMA_short'] < data['EMA_medium']) & (data['EMA_medium'] < data['EMA_long'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 81. **RSI and Moving Average Envelope**
def define_strategy_RSI_MA_Envelope(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data['upper_envelope'] = data['SMA'] * (1 + parameters[2] / 100)
    data['lower_envelope'] = data['SMA'] * (1 - parameters[2] / 100)
    data.dropna(inplace=True)

    cond1 = (data['RSI'] < parameters[3]) & (data['Close'] < data['lower_envelope'])
    cond2 = (data['RSI'] > parameters[4]) & (data['Close'] > data['upper_envelope'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 82. **On-Balance Volume (OBV) with RSI**
def define_strategy_OBV_RSI(data, parameters):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=int(parameters[0])).rsi()
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    data.dropna(inplace=True)

    rsi_buy_signal = data['RSI'] < parameters[1]
    rsi_sell_signal = data['RSI'] > parameters[2]
    obv_buy_signal = data['OBV'] > data['OBV'].shift(1)
    obv_sell_signal = data['OBV'] < data['OBV'].shift(1)

    data["position"] = 0
    data.loc[rsi_buy_signal & obv_buy_signal, "position"] = 1
    data.loc[rsi_sell_signal & obv_sell_signal, "position"] = -1
    return data

# 83. **ATR with RSI**
def define_strategy_ATR_RSI(data, parameters):
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=int(parameters[0])) * parameters[1]
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['ATR'] > data['Close']) & (data['RSI'] < parameters[3])
    cond2 = (data['ATR'] < data['Close']) & (data['RSI'] > parameters[4])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 84. **Exponential Moving Average with Bollinger Bands**
def define_strategy_EMA_Bollinger(data, parameters):
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[1])).std()
    data['Upper_Band'] = data['SMA'] + parameters[2] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[2] * data['std_dev']
    data.dropna(inplace=True)

    cond1 = (data['EMA'] > data['Upper_Band'])
    cond2 = (data['EMA'] < data['Lower_Band'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 85. **RSI and Moving Average Ribbon**
def define_strategy_RSI_MA_Ribbon(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['SMA_short'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data['SMA_medium'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[2])).sma_indicator()
    data['SMA_long'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[3])).sma_indicator()
    data.dropna(inplace=True)

    cond1 = (data['RSI'] < parameters[4]) & (data['SMA_short'] > data['SMA_medium']) & (data['SMA_medium'] > data['SMA_long'])
    cond2 = (data['RSI'] > parameters[5]) & (data['SMA_short'] < data['SMA_medium']) & (data['SMA_medium'] < data['SMA_long'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 86. **EMA Crossover with ADX Filter**
def define_strategy_EMA_ADX(data, parameters):
    data['EMA_short'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['EMA_long'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['EMA_short'] > data['EMA_long']) & (data['ADX'] > parameters[3])
    cond2 = (data['EMA_short'] < data['EMA_long']) & (data['ADX'] < parameters[3])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 87. **RSI with Bollinger Bands and Momentum Filter**
def define_strategy_RSI_Bollinger_Momentum(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[1])).std()
    data['Upper_Band'] = data['SMA'] + parameters[2] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[2] * data['std_dev']
    data['momentum'] = data['Close'].pct_change(periods=int(parameters[3]))
    data.dropna(inplace=True)

    cond1 = (data['RSI'] < parameters[4]) & (data['Close'] < data['Lower_Band']) & (data['momentum'] > parameters[5])
    cond2 = (data['RSI'] > parameters[6]) & (data['Close'] > data['Upper_Band']) & (data['momentum'] < -parameters[5])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 88. **Renko Box Trading Strategy**
def define_strategy_Renko_Box_Trading(data, parameters):
    renko = pd.DataFrame()
    renko['Close'] = data['Close']
    renko['Renko_Box'] = np.floor(renko['Close'] / int(parameters[0]) ) * int(parameters[0])  

    cond1 = (renko['Renko_Box'] > renko['Renko_Box'].shift(1)) 
    cond2 = (renko['Renko_Box'] < renko['Renko_Box'].shift(1)) 

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data



# 89. **ADX with Stochastic Oscillator**
def define_strategy_ADX_Stochastic(data, parameters):
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data['Stochastic'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['ADX'] > parameters[2]) & (data['Stochastic'] < parameters[3])
    cond2 = (data['ADX'] < parameters[4]) & (data['Stochastic'] > parameters[5])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 90. **Moving Average Ribbon with ADX Filter**
def define_strategy_MA_Ribbon_ADX(data, parameters):
    data['SMA_short'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['SMA_medium'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data['SMA_long'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[2])).sma_indicator()
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=parameters[3])
    data.dropna(inplace=True)

    cond1 = (data['SMA_short'] > data['SMA_medium']) & (data['SMA_medium'] > data['SMA_long']) & (data['ADX'] > parameters[4])
    cond2 = (data['SMA_short'] < data['SMA_medium']) & (data['SMA_medium'] < data['SMA_long']) & (data['ADX'] < parameters[5])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 91. **EMA with Stochastic Oscillator**
def define_strategy_EMA_Stochastic(data, parameters):
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['Stochastic'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['EMA'] > data['Close']) & (data['Stochastic'] < parameters[2])
    cond2 = (data['EMA'] < data['Close']) & (data['Stochastic'] > parameters[3])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 92. **RSI with ADX Filter**
def define_strategy_RSI_ADX(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['RSI'] < parameters[2]) & (data['ADX'] > parameters[3])
    cond2 = (data['RSI'] > parameters[4]) & (data['ADX'] < parameters[5])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 93. **MACD with Stochastic Oscillator**
def define_strategy_MACD_Stochastic(data, parameters):
    data['MACD'] = ta.trend.macd(data['Close'], window_slow=int(parameters[0]), window_fast=int(parameters[1]))
    data['Stochastic'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['MACD'] > parameters[3]) & (data['Stochastic'] < parameters[4])
    cond2 = (data['MACD'] < -parameters[5]) & (data['Stochastic'] > parameters[6])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 94. **MACD with Bollinger Bands Filter**
def define_strategy_MACD_Bollinger(data, parameters):
    data['MACD'] = ta.trend.macd(data['Close'], window_slow=int(parameters[0]), window_fast=int(parameters[1]))
    data['MACD_signal'] = ta.trend.macd_signal(data['Close'], window_slow=int(parameters[0]), window_fast=int(parameters[1]), window_sign=int(parameters[2]))
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[3])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[3])).std()
    data['Upper_Band'] = data['SMA'] + parameters[4] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[4] * data['std_dev']
    data.dropna(inplace=True)

    cond1 = (data['MACD'] > data['MACD_signal']) & (data['Close'] < data['Lower_Band'])  
    cond2 = (data['MACD'] < data['MACD_signal']) & (data['Close'] > data['Upper_Band'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 95. **EMA Cross with Stochastic Filter**
def define_strategy_EMA_Stochastic_Filter(data, parameters):
    data['EMA_short'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['EMA_long'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data['Stochastic'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['EMA_short'] > data['EMA_long']) & (data['Stochastic'] < parameters[3])
    cond2 = (data['EMA_short'] < data['EMA_long']) & (data['Stochastic'] > parameters[4])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 96. **MACD with Moving Average Ribbon**
def define_strategy_MACD_MA_Ribbon(data, parameters):
    data['MACD'] = ta.trend.macd(data['Close'], window_slow=int(parameters[0]), window_fast=int(parameters[1]))
    data['SMA_short'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[2])).sma_indicator()
    data['SMA_medium'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[3])).sma_indicator()
    data['SMA_long'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[4])).sma_indicator()
    data.dropna(inplace=True)

    cond1 = (data['MACD'] > 0) & (data['SMA_short'] > data['SMA_medium']) & (data['SMA_medium'] > data['SMA_long'])
    cond2 = (data['MACD'] < 0) & (data['SMA_short'] < data['SMA_medium']) & (data['SMA_medium'] < data['SMA_long'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 97. **RSI and MACD Combo Strategy**
def define_strategy_RSI_MACD_Combo(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['MACD'] = ta.trend.macd(data['Close'], window_slow=int(parameters[1]), window_fast=int(parameters[2]))
    data.dropna(inplace=True)

    cond1 = (data['RSI'] < parameters[3]) & (data['MACD'] > 0)
    cond2 = (data['RSI'] > parameters[4]) & (data['MACD'] < 0)

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 98. **Heikin Ashi Trend Continuation Strategy**
def define_strategy_Heikin_Ashi_Trend_Continuation(data, parameters):
    data['HA_Close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    data['HA_Open'] = (data['Open'].shift(1) + data['Close'].shift(1)) / 2
    data['HA_High'] = data[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    data['HA_Low'] = data[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
    data['SMA'] = ta.trend.sma_indicator(data['Close'], window=int(parameters[0]))
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['HA_Close'] > data['HA_Open']) & (data['HA_Open'] > data['HA_Close'].shift(1)) & (data['ADX'] > parameters[2]) 
    cond2 = (data['HA_Close'] < data['HA_Open']) & (data['HA_Open'] < data['HA_Close'].shift(1)) 

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data

# 99. **Bollinger Bands with Stochastic RSI**
def define_strategy_Bollinger_Stochastic_RSI(data, parameters):
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Band'] = data['SMA'] + parameters[1] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[1] * data['std_dev']
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[2]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window=int(parameters[3]))
    data.dropna(inplace=True)

    cond1 = (data['Stoch_RSI'] < parameters[4]) & (data['Close'] < data['Lower_Band'])
    cond2 = (data['Stoch_RSI'] > parameters[5]) & (data['Close'] > data['Upper_Band'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 100. **Trend Reversal with RSI**
def define_strategy_Trend_Reversal_RSI(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['RSI'] < parameters[2]) & (data['EMA'] > data['Close'])
    cond2 = (data['RSI'] > parameters[3]) & (data['EMA'] < data['Close'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 101. **Volume Profile Strategy**
def define_strategy_Volume_Profile(data, parameters):
    from scipy.stats import gaussian_kde
    price_range = np.linspace(data["Close"].min(), data["Close"].max(), int(parameters[0]))
    volume_kde = gaussian_kde(data["Volume"] * data["Close"])
    volume_profile = volume_kde(price_range)
    data["Volume_Profile"] = np.interp(data["Close"], price_range, volume_profile)
    data.dropna(inplace=True)

    cond1 = data["Volume_Profile"] > np.percentile(data["Volume_Profile"], parameters[1])
    cond2 = data["Volume_Profile"] < np.percentile(data["Volume_Profile"], parameters[2])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 102. **Grid Trading Strategy**
def define_strategy_Grid_Trading(data, parameters):
    grid_size = float(parameters[0])  
    num_levels = int(parameters[1])   
    initial_price = data["Close"].iloc[0]
    levels = [initial_price + i * grid_size for i in range(-num_levels, num_levels + 1)]
    data["Grid_Level"] = 0
    for level in levels:
        data.loc[(data["Close"] >= level - grid_size / 2) & \
                 (data["Close"] < level + grid_size / 2), "Grid_Level"] = level

    cond1 = data["Close"] > data["Grid_Level"]  
    cond2 = data["Close"] < data["Grid_Level"]  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 103. **EMA + MACD + ADX Hybrid Strategy**
def define_strategy_EMA_MACD_ADX(data, parameters):
    data['EMA_Short'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['EMA_Long'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    macd = ta.trend.MACD(data['Close'], 
                         window_slow=int(parameters[3]), 
                         window_fast=int(parameters[2]), 
                         window_sign=int(parameters[4]))
    data['MACD'] = macd.macd()
    data['Signal_Line'] = macd.macd_signal()
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[5]) )
    data.dropna(inplace=True)

    cond1 = (data['EMA_Short'] > data['EMA_Long']) & \
            (data['MACD'] > data['Signal_Line']) & \
            (data['ADX'] > parameters[6])  
    cond2 = (data['EMA_Short'] < data['EMA_Long']) & \
            (data['MACD'] < data['Signal_Line']) & \
            (data['ADX'] > parameters[6])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data

# 104. ** Trend_Momentum_Volatility : EMA + MACD + ADX + ATR Stochastic + RSI Hybrid Strategy**
def define_strategy_Trend_Momentum_Volatility(data, parameters):
    data['EMA_Short'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['EMA_Long'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    macd = ta.trend.MACD(data['Close'],  int(parameters[2]), int(parameters[3]),  int(parameters[4]))
    data['MACD'] = macd.macd()
    data['Signal_Line'] = macd.macd_signal()
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[5]))
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=int(parameters[6]))
    stochastic = ta.momentum.stochrsi(data['Close'], window=int(parameters[7]))
    data['Stoch_RSI'] = stochastic

    data.dropna(inplace=True)

    cond_long = (data['EMA_Short'] > data['EMA_Long']) & \
                (data['MACD'] > data['Signal_Line']) & \
                (data['Stoch_RSI'] < 30) & \
                (data['ADX'] > 20) & \
                (data['ATR'] > float(parameters[8]))

    cond_short = (data['EMA_Short'] < data['EMA_Long']) & \
                 (data['MACD'] < data['Signal_Line']) & \
                 (data['Stoch_RSI'] > 70) & \
                 (data['ADX'] > 20) & \
                 (data['ATR'] > float(parameters[8]))

    data["position"] = 0
    data.loc[cond_long, "position"] = 1
    data.loc[cond_short, "position"] = -1
    return data

# 105. ** Stochastic RSI Bollinger VWAP Hybrid Strategy**
def define_strategy_Stochastic_RSI_Bollinger_VWAP(data, parameters):
    # 1. Stochastic RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window=int(parameters[1]))

    # 2. Bollinger Bands
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[2])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[2])).std()
    data['Upper_Band'] = data['SMA'] + parameters[3] * data['std_dev']
    data['Lower_Band'] = data['SMA'] - parameters[3] * data['std_dev']

    # 3. VWAP
    data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'], int(parameters[4]))
    data.dropna(inplace=True)


    cond1 = (data['Stoch_RSI'] < parameters[5]) & (data['Close'] < data['Lower_Band']) & (data['Close'] > data['VWAP'] + parameters[7])
    cond2 = (data['Stoch_RSI'] > parameters[6]) & (data['Close'] > data['Upper_Band']) & (data['Close'] < data['VWAP'] - parameters[7])
    
    data["position"] = 0
    data.loc[cond1, "position"] = 1  # Buy signal
    data.loc[cond2, "position"] = -1  # Sell signal
    return data

#106 Stochastic RSI Strategy with %K and %D smoothing.
def define_strategy_Stochastic_RSI_FULL(data, parameters):
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[0]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window=int(parameters[1]))
    data['%K'] = data['Stoch_RSI'].rolling(window=int(parameters[2])).mean()
    data['%D'] = data['%K'].rolling(window=int(parameters[3])).mean()
    data.dropna(inplace=True)

    cond1 = (data['%K'] < parameters[4]) & (data['%D'] < parameters[4])  
    cond2 = (data['%K'] > parameters[5]) & (data['%D'] > parameters[5]) 

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1  
    return data

# 107  Gaussian Channel strategy
def define_strategy_Gaussian_Channel_FULL(data, parameters):
    data['Gaussian_Smoothed'] = gaussian_filter1d(data['Close'], sigma=int(parameters[1]))
    data['Rolling_STD'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Band'] = data['Gaussian_Smoothed'] + float(parameters[2]) * data['Rolling_STD']
    data['Lower_Band'] = data['Gaussian_Smoothed'] - float(parameters[3]) * data['Rolling_STD']
    data.dropna(inplace=True)

    cond1 = (data['Close'] < data['Lower_Band'])  
    cond2 = (data['Close'] > data['Upper_Band'])  

    data['position'] = 0
    data.loc[cond1, 'position'] = 1   
    data.loc[cond2, 'position'] = -1  
    return data

#108 combined the Gaussian Channel and Stochastic RSI strategies
def define_strategy_Combined_Gaussian_Stochastic_RSI_FULL(data, parameters):
    # Gaussian Channel Calculation
    data['Gaussian_Smoothed'] = gaussian_filter1d(data['Close'], sigma= int(parameters[1]))
    data['Rolling_STD'] = data['Close'].rolling(window= int(parameters[0])).std()
    data['Upper_Band'] = data['Gaussian_Smoothed'] + float(parameters[2]) * data['Rolling_STD']
    data['Lower_Band'] = data['Gaussian_Smoothed'] - float(parameters[3]) * data['Rolling_STD']

    # Stochastic RSI Calculation
    data['RSI'] = ta.momentum.rsi(data['Close'], window= int(parameters[4]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window= int(parameters[5]))
    data['%K'] = data['Stoch_RSI'].rolling(window= int(parameters[6])).mean()
    data['%D'] = data['%K'].rolling(window=int(parameters[7])).mean()

    data.dropna(inplace=True)

    # Combined Conditions
    cond_buy = (data['Close'] < data['Lower_Band']) & (data['%K'] < float(parameters[8])) & (data['%D'] < float(parameters[8]))
    cond_sell = (data['Close'] > data['Upper_Band']) & (data['%K'] > float(parameters[9])) & (data['%D'] > float(parameters[9]))

    data['position'] = 0
    data.loc[cond_buy, 'position'] = 1   # Long Entry
    data.loc[cond_sell, 'position'] = -1  # Short Entry
    return data

# 109. **On-Balance Volume (OBV) **
def define_strategy_OBV(data):
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    data.dropna(inplace=True)

    cond1 = data['OBV'] > data['OBV'].shift(1)
    cond2 = data['OBV'] < data['OBV'].shift(1)

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 110. ** Volum Delta **
def define_strategy_volume_delta(data):
    data['Up_Volume'] = np.where(data['Close'] > data['Close'].shift(1), data['Volume'], 0)
    data['Down_Volume'] = np.where(data['Close'] < data['Close'].shift(1), data['Volume'], 0)
    data['Volume_Delta'] = data['Up_Volume'] - data['Down_Volume']
    data.dropna(inplace=True)

    cond1 = data['Volume_Delta'] > 0
    cond2 = data['Volume_Delta'] < 0

    data['position'] = 0  
    data.loc[ cond1 , 'position'] = 1  
    data.loc[ cond2 , 'position'] = -1 
    return data

# 111. ** Ease of Movement **
def define_strategy_ease_of_movement(data, parameters):
    data['Midpoint_Move'] = (data['High'] + data['Low']) / 2 - \
                            (data['High'].shift(1) + data['Low'].shift(1)) / 2
    data['Box_Ratio'] = data['Volume'] / (data['High'] - data['Low'])
    data['EoM'] = data['Midpoint_Move'] / data['Box_Ratio']
    data['EoM_MA'] = data['EoM'].rolling(window= int(parameters[0])).mean()
    data.dropna(inplace=True)

    cond1 = data['EoM_MA'] > 0  
    cond2 = data['EoM_MA'] < 0  

    data['position'] = 0  
    data.loc[cond1, 'position'] = 1  
    data.loc[cond2, 'position'] = -1  
    return data    

# 112. ** Weight Moving Average **
def define_strategy_wma(data,parameters):
    weights = np.arange(1, int(parameters[0]) + 1)
    data['WMA'] = data['Close'].rolling(window=int(parameters[0]))\
        .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    data.dropna(inplace=True)

    cond1 = data['Close'] > data['WMA']  
    cond2 = data['Close'] < data['WMA']  

    data['position'] = 0  
    data.loc[cond1, 'position'] = 1  
    data.loc[cond2, 'position'] = -1  
    return data

# 113. ** Exponential Moving Average **
def define_strategy_ema(data,parameters):
    data['EMA'] = data['Close'].ewm(span=int(parameters[0]), adjust=False).mean()
    data.dropna(inplace=True)

    cond1 = data['Close'] > data['EMA']  
    cond2 = data['Close'] < data['EMA']  

    data['position'] = 0  
    data.loc[cond1, 'position'] = 1  
    data.loc[cond2, 'position'] = -1  
    return data

# 114. ** Double Exponential Moving Average **
def define_strategy_dema(data, parameters):
    ema1 = data['Close'].ewm(span=int(parameters[0]), adjust=False).mean()
    ema2 = ema1.ewm(span=int(parameters[0]), adjust=False).mean()
    data['DEMA'] = 2 * ema1 - ema2
    data.dropna(inplace=True)

    cond1 = data['Close'] > data['DEMA']  
    cond2 = data['Close'] < data['DEMA']  

    data['position'] = 0  
    data.loc[cond1, 'position'] = 1  
    data.loc[cond2, 'position'] = -1 
    return data

# 115. ** Adaptive Moving Average **
def define_strategy_ama(data, parameters):
    price_change = abs(data['Close'] - data['Close'].shift(int(parameters[0])))
    volatility = data['Close'].diff().abs().rolling(window=int(parameters[0])).sum()
    data['ER'] = price_change / volatility
    fast_sc = 2 / (int(parameters[1]) + 1)
    slow_sc = 2 / (int(parameters[2]) + 1)
    data['SC'] = (data['ER'] * (fast_sc - slow_sc) + slow_sc) ** 2
    data['AMA'] = np.nan  
    if int(parameters[0]) < len(data):  
        data.iloc[int(parameters[0]), data.columns.get_loc('AMA')] = data.iloc[int(parameters[0]), data.columns.get_loc('Close')]
    for i in range(int(parameters[0]) + 1, len(data)):
        data.iloc[i, data.columns.get_loc('AMA')] = (
            data.iloc[i - 1, data.columns.get_loc('AMA')] +
            data.iloc[i, data.columns.get_loc('SC')] * 
            (data.iloc[i, data.columns.get_loc('Close')] - data.iloc[i - 1, data.columns.get_loc('AMA')]))
    data.dropna(inplace=True)

    cond1 = data['Close'] > data['AMA']  
    cond2 = data['Close'] < data['AMA']  

    data['position'] = 0  
    data.loc[cond1, 'position'] = 1  
    data.loc[cond2, 'position'] = -1  
    return data

# 116. ** Variable Index Dynamic Average (VIDYA) **
def define_strategy_vidya(data, parameters):
    period = int(parameters[0]) 
    volatility_period = int(parameters[1])
    data['Volatility'] = data['Close'].rolling(window=volatility_period).std()
    data['Volatility_Factor'] = data['Volatility'] / data['Volatility'].sum()
    data = data.reset_index(drop=True)  
    data['VIDYA'] = np.nan
    if period >= len(data):
        raise ValueError(f"Period {period} is out of range for data with {len(data)} rows")
    data.loc[period, 'VIDYA'] = data.loc[period, 'Close']
    for i in range(period + 1, len(data)):
        smoothing_factor = (2 / (period + 1)) * data.loc[i, 'Volatility_Factor']
        data.loc[i, 'VIDYA'] = data.loc[i - 1, 'VIDYA'] + smoothing_factor * (data.loc[i, 'Close'] - data.loc[i - 1, 'VIDYA'])
    data.dropna(inplace=True)

    cond1 = data['Close'] > data['VIDYA']  
    cond2 = data['Close'] < data['VIDYA']  

    data['position'] = 0  
    data.loc[cond1, 'position'] = 1  
    data.loc[cond2, 'position'] = -1  
    return data

 # 117. ** Simple Moving Average Cross **
def define_strategy_SMA_cross(data, parameters):
    data['SMA_S'] = ta.trend.sma_indicator(data['Close'], window=int(parameters[0]))
    data['SMA_L'] = ta.trend.sma_indicator(data['Close'], window=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['SMA_S'] > data['SMA_L'])  
    cond2 = (data['SMA_S'] < data['SMA_L'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 118. **Stochastic Oscillator Strategy**
def define_strategy_Stochastic(data, parameters):
    data['%K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], 
                                   window=int(parameters[0]), smooth_window=int(parameters[1]))
    data['%D'] = data['%K'].rolling(window=int(parameters[2])).mean()
    data.dropna(inplace=True)

    cond1 = (data['%K'] > data['%D']) & (data['%K'] < int(parameters[3]))  
    cond2 = (data['%K'] < data['%D']) & (data['%K'] > int(parameters[4])) 

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data
    
# 119. **Awesome Oscillator (AO) Strategy**
def define_strategy_AO(data, parameters):
    data['AO'] = ta.momentum.awesome_oscillator(data['High'], data['Low'], 
                       window1=int(parameters[0]), window2=int(parameters[1]))
    data.dropna(inplace=True)

    cond1 = (data['AO'] > float(parameters[2])) & (data['AO'].shift(1) <= float(parameters[2]))  
    cond2 = (data['AO'] < -float(parameters[2])) & (data['AO'].shift(1) >= -float(parameters[2]))  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  # Buy signal
    data.loc[cond2, "position"] = -1  # Sell signal
    return data

# 120. **Know Sure Thing (KST) Strategy**
def define_strategy_KST(data, parameters):
    roc1 = ta.momentum.roc(data['Close'], window=int(parameters[0]))
    roc2 = ta.momentum.roc(data['Close'], window=int(parameters[1]))
    roc3 = ta.momentum.roc(data['Close'], window=int(parameters[2]))
    roc4 = ta.momentum.roc(data['Close'], window=int(parameters[3]))

    data['KST'] = roc1 + (2 * roc2) + (3 * roc3) + (4 * roc4)
    data['Signal'] = data['KST'].rolling(window=int(parameters[4])).mean()
    data.dropna(inplace=True)

    cond1 = (data['KST'] > data['Signal']) & (data['KST'].shift(1) <= data['Signal']) & (data['KST'] > float(parameters[5]))
    cond2 = (data['KST'] < data['Signal']) & (data['KST'].shift(1) >= data['Signal']) & (data['KST'] < -float(parameters[5]))

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1  
    return data

# 121. **Bollinger Bands with SMA Strategy**
def define_strategy_Bollinger(data, parameters):
    data['SMA'] = ta.trend.sma_indicator(data['Close'], window=int(parameters[0]))
    data['std_dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Band'] = data['SMA'] + float(parameters[1]) * data['std_dev']
    data['Lower_Band'] = data['SMA'] - float(parameters[1]) * data['std_dev']
    data.dropna(inplace=True)

    cond1 = (data['Close'] > data['Upper_Band'])  
    cond2 = (data['Close'] < data['Lower_Band'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data    


# 122. **Bollinger Bands Squeeze Strategy**
def define_strategy_Squeeze(data, parameters):
    data['SMA'] = ta.trend.sma_indicator(data['Close'], window=int(parameters[0]))
    data['std_dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_BB'] = data['SMA'] + float(parameters[1]) * data['std_dev']
    data['Lower_BB'] = data['SMA'] - float(parameters[1]) * data['std_dev']

    data['EMA_KC'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[2]))
    atr = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=int(parameters[2]))
    data['Upper_KC'] = data['EMA_KC'] + float(parameters[3]) * atr
    data['Lower_KC'] = data['EMA_KC'] - float(parameters[3]) * atr

    data.dropna(inplace=True)

    squeeze_on = (data['Lower_BB'] > data['Lower_KC']) & (data['Upper_BB'] < data['Upper_KC'])
    squeeze_off = (data['Lower_BB'] < data['Lower_KC']) & (data['Upper_BB'] > data['Upper_KC'])

    cond1 = squeeze_on & (data['Close'] > data['Upper_BB'])  
    cond2 = squeeze_off & (data['Close'] < data['Lower_BB'])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data    

# 123. **Standard Deviation Channel Strategy**
def define_strategy_StdDev_Channel(data, parameters):
    x = np.arange(int(parameters[0]))
    def linreg(y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope * x + intercept
    data['Central_Line'] = data['Close'].rolling(window=int(parameters[0]))\
        .apply(lambda y: linreg(y)[-1], raw=True)
    data['Std_Dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Channel'] = data['Central_Line'] + float(parameters[1]) * data['Std_Dev']
    data['Lower_Channel'] = data['Central_Line'] - float(parameters[1]) * data['Std_Dev']

    data.dropna(inplace=True)

    cond1 = data['Close'] < data['Lower_Channel']  
    cond2 = data['Close'] > data['Upper_Channel']  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data   

# 124. **Historical Volatility (HV) Strategy**
def define_strategy_HV(data, parameters):
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['HV'] = data['Log_Returns'].rolling(window=int(parameters[0])).std() * np.sqrt(252)  
    data.dropna(inplace=True)

    cond1 = data['HV'] < float(parameters[2])  
    cond2 = data['HV'] > float(parameters[1])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data   

# 125. ** Volatility Ratio (VR) Strategy **
def define_strategy_VR(data, parameters):
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Short_Vol'] = data['Log_Returns'].rolling(window=int(parameters[0])).std()
    data['Long_Vol'] = data['Log_Returns'].rolling(window=int(parameters[1])).std()
    data['VR'] = data['Short_Vol'] / data['Long_Vol']
    data.dropna(inplace=True)

    cond1 = data['VR'] > float(parameters[2])
    cond2 = data['VR'] < float(parameters[3])

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data   

# 126. ** Simple_Pivot_Points Strategy **
def define_strategy_Simple_Pivot_Points(data):
    data['Pivot_Point'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['R1'] = 2 * data['Pivot_Point'] - data['Low']
    data['S1'] = 2 * data['Pivot_Point'] - data['High']
    data.dropna(inplace=True)

    cond1 = data['Close'] > data['R1']  
    cond2 = data['Close'] < data['S1']  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data

# 127. ** Directional Indicator (DI-Only) Strategy **
def define_strategy_DI(data, parameters):
    data['+DI'] = ta.trend.adx_pos(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data['-DI'] = ta.trend.adx_neg(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data.dropna(inplace=True)

    cond1 = data['+DI'] > data['-DI']  
    cond2 = data['+DI'] < data['-DI']  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data

# 128. **Stochastic_RSI with Standard Deviation Channel Strategy**
def define_strategy_Stochastic_RSI_StdDev_Channel(data, parameters):
    x = np.arange(int(parameters[0]))
    def linreg(y):
        slope, intercept = np.polyfit(x, y, 1)
        return slope * x + intercept
    data['Central_Line'] = data['Close'].rolling(window=int(parameters[0]))\
        .apply(lambda y: linreg(y)[-1], raw=True)
    data['Std_Dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Channel'] = data['Central_Line'] + float(parameters[1]) * data['Std_Dev']
    data['Lower_Channel'] = data['Central_Line'] - float(parameters[1]) * data['Std_Dev']

    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[2]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window=int(parameters[3]))
    data.dropna(inplace=True)

    cond1 = (data['Stoch_RSI'] < parameters[4]) & (data['Stoch_RSI'] > 0.01) & data['Close'] < data['Lower_Channel']  
    cond2 = (data['Stoch_RSI'] > parameters[5]) & (data['Stoch_RSI'] < 0.99) & data['Close'] > data['Upper_Channel']  


    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data   

# 129. **Bollinger Bands with Stochastic RSI modified**
def define_strategy_Bollinger_Stochastic_RSI_modified(data, parameters):
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[0])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[0])).std()
    data['Upper_Band'] = data['SMA'] + float(parameters[1]) * data['std_dev']
    data['Lower_Band'] = data['SMA'] - float(parameters[1]) * data['std_dev']
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[2]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window=int(parameters[3]))
    data['Stoch_RSI'] = data['Stoch_RSI'].rolling(window= int(parameters[4])).mean()
    data.dropna(inplace=True)

    cond1 = (data['Stoch_RSI'] < parameters[5]) & (data['Close'] < data['Lower_Band'])
    cond2 = (data['Stoch_RSI'] > parameters[6]) & (data['Close'] > data['Upper_Band'])

    #cond1 = (data['Stoch_RSI'] < parameters[2]) & (data['Stoch_RSI'] > 0.01) & (data['Close'] < data['Lower_Band'])
    #cond2 = (data['Stoch_RSI'] > parameters[3]) & (data['Stoch_RSI'] < 0.99) & (data['Close'] > data['Upper_Band'])

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data

# 130. **Keltner Channel Calculation Bands with Stochastic RSI**
def define_strategy_Keltner_Stochastic_RSI(data, parameters):
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))
    data['Upper_Band'] = data['EMA'] + float(parameters[2]) * data['ATR']
    data['Lower_Band'] = data['EMA'] - float(parameters[2]) * data['ATR']
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[3]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window=int(parameters[4]))
    data['Stoch_RSI'] = data['Stoch_RSI'].rolling(window=int(parameters[5])).mean()
    data.dropna(inplace=True)

    cond1 = (data['Close'] < data['Lower_Band']) & (data['Stoch_RSI'] < parameters[6])  
    cond2 = (data['Close'] > data['Upper_Band']) & (data['Stoch_RSI'] > parameters[7])  

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1  
    return data

# 131. **Hull Moving Average Channel with Stochastic RSI**
def define_strategy_HMA_StochRSI(data, parameters):
    data['HMA'] = hull_moving_average(data['Close'], window=int(parameters[0]))
    data['HMA_STD'] = data['HMA'].rolling(window=int(parameters[1])).std()
    data['Upper_Band'] = data['HMA'] + float(parameters[2]) * data['HMA_STD']
    data['Lower_Band'] = data['HMA'] - float(parameters[2]) * data['HMA_STD']
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[3]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window=int(parameters[4]))
    data['Stoch_RSI'] = data['Stoch_RSI'].rolling(window=int(parameters[5])).mean()
    data.dropna(inplace=True)

    cond1 = (data['Close'] < data['Lower_Band']) & (data['Stoch_RSI'] < parameters[6])
    cond2 = (data['Close'] > data['Upper_Band']) & (data['Stoch_RSI'] > parameters[7])

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1  
    return data

# 132. **ADX ATR Bollinger Bands with Stochastic RSI**
def define_strategy_ADX_ATR_Bollinger_Stochastic_RSI(data, parameters):
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[0]))
    
    data['SMA'] = ta.trend.SMAIndicator(data['Close'], window=int(parameters[1])).sma_indicator()
    data['std_dev'] = data['Close'].rolling(window=int(parameters[1])).std()
    data['Upper_Band'] = data['SMA'] + float(parameters[2]) * data['std_dev']
    data['Lower_Band'] = data['SMA'] - float(parameters[2]) * data['std_dev']
    data['BB_Width'] = (data['Upper_Band'] - data['Lower_Band']) / data['SMA']

    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[3]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window=int(parameters[4]))
    data['Stoch_RSI'] = data['Stoch_RSI'].rolling(window=int(parameters[5])).mean()

    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=int(parameters[8]))/ data["Close"]
    data.dropna(inplace=True)

    if (data['ADX'].iloc[-1] > parameters[9]) and (data['ATR'].iloc[-1] > parameters[10]):  
        cond1 = (data['Stoch_RSI'] < parameters[6]) & (data['Close'] < data['Lower_Band'])
        cond2 = (data['Stoch_RSI'] > parameters[7]) & (data['Close'] > data['Upper_Band'])
    else:
        cond1 = (data['Close'] > data['SMA'] + parameters[1] * data['std_dev']) & \
                (data['BB_Width'] > parameters[11]) & (data['ATR'] > parameters[10])

        cond2 = (data['Close'] < data['SMA'] - parameters[1] * data['std_dev']) & \
                (data['BB_Width'] > parameters[11]) & (data['ATR'] > parameters[10])

    data["position"] = 0
    data.loc[cond1, "position"] = 1  
    data.loc[cond2, "position"] = -1 
    return data

# 133. **SuperTrend with Stochastic RSI**
def define_strategy_Supertrend_Stochastic_RSI(data, parameters):
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], 
                                     data['Close'], window=int(parameters[0]))
    data = calculate_supertrend(data, multiplier=float(parameters[1]))
    # Calculate RSI and Stochastic RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[2]))
    data['Stoch_RSI'] = ta.momentum.stochrsi(data['RSI'], window=int(parameters[3]))
    data['Stoch_RSI'] = data['Stoch_RSI'].rolling(window=int(parameters[4])).mean()
    data.dropna(inplace=True)

    # Define conditions for buy and sell signals
    cond1 = (data['Stoch_RSI'] < parameters[5]) & (data['Close'] < data['Lower_Band'])
    cond2 = (data['Stoch_RSI'] > parameters[6]) & (data['Close'] > data['Upper_Band'])

    # Assign positions
    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1

    return data

def define_strategy_Trend_Momentum_RSI_Volatility(data, parameters):
    data['EMA_Fast'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[0]))
    data['EMA_Slow'] = ta.trend.ema_indicator(data['Close'], window=int(parameters[1]))

    macd = ta.trend.MACD(data['Close'], window_slow=int(parameters[2]), window_fast=int(parameters[2]), window_sign=int(parameters[4]))
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()

    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], window=int(parameters[5]))
    data['RSI'] = ta.momentum.rsi(data['Close'], window=int(parameters[7]))
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=int(parameters[10]))

    data.dropna(inplace=True)

    cond1 = ((data['EMA_Fast'] > data['EMA_Slow']) &
        (data['MACD_Hist'] > 0) &
        (data['ADX'] > float(parameters[6])) &
        (data['RSI'] > int(parameters[8])) & (data['RSI'] < int(parameters[9])))

    cond2 = ((data['EMA_Fast'] < data['EMA_Slow']) &
        (data['MACD_Hist'] < 0) &
        (data['ADX'] > float(parameters[6])) &
        (data['RSI'] > int(parameters[8])) & (data['RSI'] < int(parameters[9])))

    data["position"] = 0
    data.loc[cond1, "position"] = 1
    data.loc[cond2, "position"] = -1
    return data