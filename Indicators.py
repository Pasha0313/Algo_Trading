"""
indicators.py

Reusable indicator/calculation library for Strategy.py.

Purpose
-------
Instead of recalculating RSI, Bollinger Bands, MACD, ADX, ATR, etc.
inside every strategy, import these functions and build strategies from
shared indicator blocks.

Convention
----------
Most `add_*` functions:
    1. Modify the provided DataFrame in place.
    2. Create uniquely named columns based on parameters.
    3. Return `(data, column_name...)` so the strategy can reference columns safely.

Example
-------
from indicators import add_bollinger, add_stoch_rsi, make_position

def define_strategy_Bollinger_StochRSI(data, parameters):
    data = data.copy()
    data, upper, lower, width = add_bollinger(data, parameters[0], parameters[1])
    data, stoch = add_stoch_rsi(data, parameters[2], parameters[3], smoothing=parameters[4])

    long_cond = (data[stoch] < parameters[5]) & (data["Close"] < data[lower])
    short_cond = (data[stoch] > parameters[6]) & (data["Close"] > data[upper])

    data = make_position(data, long_cond, short_cond)
    return clean_indicator_frame(data)
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import ta
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde, linregress


# ============================================================
# 0. Core helpers
# ============================================================

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _fmt(value) -> str:
    """Safe string used in generated column names."""
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_")


def require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_indicator_frame(data: pd.DataFrame, subset: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Drop inf and NaN rows after indicator calculation."""
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=subset, inplace=True)
    return data


def make_position(data: pd.DataFrame, long_cond: pd.Series, short_cond: pd.Series, column: str = "position") -> pd.DataFrame:
    """Standard long/short/flat position assignment."""
    data[column] = 0
    data.loc[long_cond.fillna(False), column] = 1
    data.loc[short_cond.fillna(False), column] = -1
    return data


def crossed_above(series: pd.Series, level_or_series) -> pd.Series:
    other = level_or_series
    return (series > other) & (series.shift(1) <= other if not isinstance(other, pd.Series) else series.shift(1) <= other.shift(1))


def crossed_below(series: pd.Series, level_or_series) -> pd.Series:
    other = level_or_series
    return (series < other) & (series.shift(1) >= other if not isinstance(other, pd.Series) else series.shift(1) >= other.shift(1))


# ============================================================
# 1. Price, return, volume, volatility basics
# ============================================================


def add_log_returns(data: pd.DataFrame, column: str = "Close", name: str = "Log_Returns") -> Tuple[pd.DataFrame, str]:
    data[name] = np.log(data[column] / data[column].shift(1))
    return data, name


def add_pct_returns(data: pd.DataFrame, column: str = "Close", periods: int = 1) -> Tuple[pd.DataFrame, str]:
    col = f"Pct_Returns_{_fmt(periods)}"
    data[col] = data[column].pct_change(periods=int(periods))
    return data, col


def add_volume_change(data: pd.DataFrame, column: str = "Volume", name: str = "vol_ch") -> Tuple[pd.DataFrame, str]:
    data[name] = np.log(data[column] / data[column].shift(1))
    return data, name


def add_momentum(data: pd.DataFrame, window: int, column: str = "Close", pct: bool = False) -> Tuple[pd.DataFrame, str]:
    col = f"Momentum_{_fmt(window)}" if not pct else f"Momentum_PCT_{_fmt(window)}"
    if pct:
        data[col] = data[column].pct_change(periods=int(window))
    else:
        data[col] = data[column] - data[column].shift(int(window))
    return data, col


def add_rolling_sum_returns(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str, str]:
    data, ret_col = add_pct_returns(data)
    col = f"Momentum_ReturnSum_{_fmt(window)}"
    data[col] = data[ret_col].rolling(int(window)).sum()
    return data, ret_col, col


def add_historical_volatility(data: pd.DataFrame, window: int, annualization: float = 252.0) -> Tuple[pd.DataFrame, str, str]:
    data, ret_col = add_log_returns(data)
    col = f"HV_{_fmt(window)}"
    data[col] = data[ret_col].rolling(int(window)).std() * np.sqrt(float(annualization))
    return data, ret_col, col


def add_volatility_ratio(data: pd.DataFrame, short_window: int, long_window: int) -> Tuple[pd.DataFrame, str, str, str, str]:
    data, ret_col = add_log_returns(data)
    short_col = f"Short_Vol_{_fmt(short_window)}"
    long_col = f"Long_Vol_{_fmt(long_window)}"
    vr_col = f"VR_{_fmt(short_window)}_{_fmt(long_window)}"
    data[short_col] = data[ret_col].rolling(int(short_window)).std()
    data[long_col] = data[ret_col].rolling(int(long_window)).std()
    data[vr_col] = data[short_col] / data[long_col]
    return data, ret_col, short_col, long_col, vr_col


def add_garman_klass_volatility(data: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    require_columns(data, ["High", "Low", "Close"])
    col = "GK_Volatility"
    data[col] = 0.5 * (((data["High"] - data["Low"]) ** 2) - 0.25 * ((2 * data["Close"] - data["High"] - data["Low"]) ** 2)) / data["Close"]
    return data, col


# ============================================================
# 2. Moving averages and trend indicators
# ============================================================


def add_sma(data: pd.DataFrame, window: int, column: str = "Close", prefix: str = "SMA") -> Tuple[pd.DataFrame, str]:
    col = f"{prefix}_{_fmt(window)}"
    if col not in data.columns:
        data[col] = data[column].rolling(int(window)).mean()
    return data, col


def add_ema(data: pd.DataFrame, window: int, column: str = "Close", prefix: str = "EMA") -> Tuple[pd.DataFrame, str]:
    col = f"{prefix}_{_fmt(window)}"
    if col not in data.columns:
        data[col] = ta.trend.ema_indicator(data[column], window=int(window))
    return data, col


def add_wma(data: pd.DataFrame, window: int, column: str = "Close") -> Tuple[pd.DataFrame, str]:
    col = f"WMA_{_fmt(window)}"
    if col not in data.columns:
        weights = np.arange(1, int(window) + 1)
        data[col] = data[column].rolling(window=int(window)).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return data, col


def add_dema(data: pd.DataFrame, window: int, column: str = "Close") -> Tuple[pd.DataFrame, str]:
    col = f"DEMA_{_fmt(window)}"
    if col not in data.columns:
        ema1 = data[column].ewm(span=int(window), adjust=False).mean()
        ema2 = ema1.ewm(span=int(window), adjust=False).mean()
        data[col] = 2 * ema1 - ema2
    return data, col


def add_tema(data: pd.DataFrame, window: int, column: str = "Close") -> Tuple[pd.DataFrame, str]:
    col = f"TEMA_{_fmt(window)}"
    if col not in data.columns:
        ema1 = data[column].ewm(span=int(window), adjust=False).mean()
        ema2 = ema1.ewm(span=int(window), adjust=False).mean()
        ema3 = ema2.ewm(span=int(window), adjust=False).mean()
        data[col] = 3 * (ema1 - ema2) + ema3
    return data, col


def add_hma(data: pd.DataFrame, window: int, column: str = "Close") -> Tuple[pd.DataFrame, str]:
    col = f"HMA_{_fmt(window)}"
    if col not in data.columns:
        w = int(window)
        wma_half = ta.trend.wma_indicator(data[column], window=max(1, w // 2))
        wma_full = ta.trend.wma_indicator(data[column], window=w)
        data[col] = ta.trend.wma_indicator(2 * wma_half - wma_full, window=max(1, int(np.sqrt(w))))
    return data, col


def add_ama(data: pd.DataFrame, period: int, fast_ema: int, slow_ema: int, column: str = "Close") -> Tuple[pd.DataFrame, str, str, str]:
    period = int(period)
    er_col = f"AMA_ER_{_fmt(period)}"
    sc_col = f"AMA_SC_{_fmt(period)}_{_fmt(fast_ema)}_{_fmt(slow_ema)}"
    ama_col = f"AMA_{_fmt(period)}_{_fmt(fast_ema)}_{_fmt(slow_ema)}"

    price_change = (data[column] - data[column].shift(period)).abs()
    volatility = data[column].diff().abs().rolling(window=period).sum()
    data[er_col] = price_change / volatility

    fast_sc = 2 / (int(fast_ema) + 1)
    slow_sc = 2 / (int(slow_ema) + 1)
    data[sc_col] = (data[er_col] * (fast_sc - slow_sc) + slow_sc) ** 2

    ama = np.full(len(data), np.nan, dtype=float)
    if period < len(data):
        close_values = data[column].to_numpy(dtype=float)
        sc_values = data[sc_col].to_numpy(dtype=float)
        ama[period] = close_values[period]
        for i in range(period + 1, len(data)):
            ama[i] = ama[i - 1] + sc_values[i] * (close_values[i] - ama[i - 1])
    data[ama_col] = ama
    return data, er_col, sc_col, ama_col


def add_vidya(data: pd.DataFrame, period: int, volatility_period: int, column: str = "Close") -> Tuple[pd.DataFrame, str, str, str]:
    period = int(period)
    volatility_period = int(volatility_period)
    vol_col = f"VIDYA_Volatility_{_fmt(volatility_period)}"
    factor_col = f"VIDYA_Factor_{_fmt(volatility_period)}"
    vidya_col = f"VIDYA_{_fmt(period)}_{_fmt(volatility_period)}"

    data[vol_col] = data[column].rolling(window=volatility_period).std()
    data[factor_col] = data[vol_col] / data[vol_col].sum()

    vidya = np.full(len(data), np.nan, dtype=float)
    if period >= len(data):
        raise ValueError(f"Period {period} is out of range for data with {len(data)} rows")
    close_values = data[column].to_numpy(dtype=float)
    factor_values = data[factor_col].to_numpy(dtype=float)
    vidya[period] = close_values[period]
    for i in range(period + 1, len(data)):
        smoothing_factor = (2 / (period + 1)) * factor_values[i]
        vidya[i] = vidya[i - 1] + smoothing_factor * (close_values[i] - vidya[i - 1])

    data[vidya_col] = vidya
    return data, vol_col, factor_col, vidya_col


def add_ma_ribbon(data: pd.DataFrame, windows: Sequence[int], ma_type: str = "SMA", column: str = "Close") -> Tuple[pd.DataFrame, list[str]]:
    cols = []
    for w in windows:
        if ma_type.upper() == "EMA":
            data, col = add_ema(data, int(w), column=column, prefix="EMA_RIBBON")
        else:
            data, col = add_sma(data, int(w), column=column, prefix="SMA_RIBBON")
        cols.append(col)
    return data, cols


# ============================================================
# 3. Bands, channels, envelopes, pivots
# ============================================================


def add_bollinger(data: pd.DataFrame, window: int, multiplier: float, column: str = "Close", prefix: str = "BB") -> Tuple[pd.DataFrame, str, str, str, str]:
    window = int(window)
    multiplier = float(multiplier)
    sma_col = f"{prefix}_SMA_{_fmt(window)}"
    std_col = f"{prefix}_STD_{_fmt(window)}"
    upper_col = f"{prefix}_Upper_{_fmt(window)}_{_fmt(multiplier)}"
    lower_col = f"{prefix}_Lower_{_fmt(window)}_{_fmt(multiplier)}"
    width_col = f"{prefix}_Width_{_fmt(window)}_{_fmt(multiplier)}"

    if sma_col not in data.columns:
        data[sma_col] = data[column].rolling(window).mean()
    if std_col not in data.columns:
        data[std_col] = data[column].rolling(window).std()
    data[upper_col] = data[sma_col] + multiplier * data[std_col]
    data[lower_col] = data[sma_col] - multiplier * data[std_col]
    data[width_col] = (data[upper_col] - data[lower_col]) / data[sma_col]
    return data, upper_col, lower_col, sma_col, width_col


def add_keltner_channel(data: pd.DataFrame, atr_window: int, ema_window: int, multiplier: float, normalize_atr: bool = False) -> Tuple[pd.DataFrame, str, str, str, str]:
    data, atr_col = add_atr(data, atr_window, normalize=normalize_atr)
    data, ema_col = add_ema(data, ema_window)
    upper_col = f"KC_Upper_{_fmt(atr_window)}_{_fmt(ema_window)}_{_fmt(multiplier)}"
    lower_col = f"KC_Lower_{_fmt(atr_window)}_{_fmt(ema_window)}_{_fmt(multiplier)}"
    data[upper_col] = data[ema_col] + float(multiplier) * data[atr_col]
    data[lower_col] = data[ema_col] - float(multiplier) * data[atr_col]
    return data, upper_col, lower_col, ema_col, atr_col


def add_ma_envelope(data: pd.DataFrame, window: int, percentage: float, ma_type: str = "SMA", column: str = "Close") -> Tuple[pd.DataFrame, str, str, str]:
    if ma_type.upper() == "EMA":
        data, ma_col = add_ema(data, window, column=column, prefix="Envelope_EMA")
    else:
        data, ma_col = add_sma(data, window, column=column, prefix="Envelope_SMA")
    upper_col = f"Envelope_Upper_{ma_col}_{_fmt(percentage)}"
    lower_col = f"Envelope_Lower_{ma_col}_{_fmt(percentage)}"
    data[upper_col] = data[ma_col] * (1 + float(percentage) / 100)
    data[lower_col] = data[ma_col] * (1 - float(percentage) / 100)
    return data, upper_col, lower_col, ma_col


def add_donchian(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str, str, str]:
    upper_col = f"Donchian_Upper_{_fmt(window)}"
    lower_col = f"Donchian_Lower_{_fmt(window)}"
    mid_col = f"Donchian_Middle_{_fmt(window)}"
    data[upper_col] = data["High"].rolling(int(window)).max()
    data[lower_col] = data["Low"].rolling(int(window)).min()
    data[mid_col] = (data[upper_col] + data[lower_col]) / 2
    return data, upper_col, lower_col, mid_col


def add_stddev_channel(data: pd.DataFrame, window: int, multiplier: float, column: str = "Close") -> Tuple[pd.DataFrame, str, str, str, str]:
    window = int(window)
    x = np.arange(window)

    def linreg_last(y):
        slope, intercept = np.polyfit(x, y, 1)
        return (slope * x + intercept)[-1]

    central_col = f"StdDev_Central_{_fmt(window)}"
    std_col = f"StdDev_STD_{_fmt(window)}"
    upper_col = f"StdDev_Upper_{_fmt(window)}_{_fmt(multiplier)}"
    lower_col = f"StdDev_Lower_{_fmt(window)}_{_fmt(multiplier)}"

    data[central_col] = data[column].rolling(window).apply(linreg_last, raw=True)
    data[std_col] = data[column].rolling(window).std()
    data[upper_col] = data[central_col] + float(multiplier) * data[std_col]
    data[lower_col] = data[central_col] - float(multiplier) * data[std_col]
    return data, upper_col, lower_col, central_col, std_col


def add_linear_regression_channel(data: pd.DataFrame, window: int, multiplier: float, column: str = "Close") -> Tuple[pd.DataFrame, str, str, str, str]:
    lr_col = f"LinReg_{_fmt(window)}"
    std_col = f"LinReg_STD_{_fmt(window)}"
    upper_col = f"LinReg_Upper_{_fmt(window)}_{_fmt(multiplier)}"
    lower_col = f"LinReg_Lower_{_fmt(window)}_{_fmt(multiplier)}"

    x = np.arange(int(window))
    lr_values = []
    std_dev = []
    for i in range(len(data[column])):
        if i < int(window) - 1:
            lr_values.append(np.nan)
            std_dev.append(np.nan)
        else:
            y = data[column].iloc[i - int(window) + 1:i + 1].values
            slope, intercept, _, _, _ = linregress(x, y)
            predicted = slope * x + intercept
            lr_values.append(predicted[-1])
            std_dev.append(np.std(y - predicted))

    data[lr_col] = pd.Series(lr_values, index=data.index)
    data[std_col] = pd.Series(std_dev, index=data.index)
    data[upper_col] = data[lr_col] + float(multiplier) * data[std_col]
    data[lower_col] = data[lr_col] - float(multiplier) * data[std_col]
    return data, upper_col, lower_col, lr_col, std_col


def add_zscore(data: pd.DataFrame, window: int, column: str = "Close") -> Tuple[pd.DataFrame, str, str, str]:
    mean_col = f"Z_Mean_{_fmt(window)}"
    std_col = f"Z_STD_{_fmt(window)}"
    z_col = f"ZScore_{_fmt(window)}"
    data[mean_col] = data[column].rolling(int(window)).mean()
    data[std_col] = data[column].rolling(int(window)).std()
    data[z_col] = (data[column] - data[mean_col]) / data[std_col]
    return data, z_col, mean_col, std_col


def add_gaussian_channel(data: pd.DataFrame, window: int, sigma: float, upper_multiplier: float, lower_multiplier: float, column: str = "Close") -> Tuple[pd.DataFrame, str, str, str, str]:
    smooth_col = f"Gaussian_Smoothed_{_fmt(sigma)}"
    std_col = f"Gaussian_STD_{_fmt(window)}"
    upper_col = f"Gaussian_Upper_{_fmt(window)}_{_fmt(sigma)}_{_fmt(upper_multiplier)}"
    lower_col = f"Gaussian_Lower_{_fmt(window)}_{_fmt(sigma)}_{_fmt(lower_multiplier)}"
    data[smooth_col] = gaussian_filter1d(data[column].to_numpy(dtype=float), sigma=float(sigma))
    data[std_col] = data[column].rolling(int(window)).std()
    data[upper_col] = data[smooth_col] + float(upper_multiplier) * data[std_col]
    data[lower_col] = data[smooth_col] - float(lower_multiplier) * data[std_col]
    return data, upper_col, lower_col, smooth_col, std_col


def add_pivot_points(data: pd.DataFrame, window: Optional[int] = None, multiplier: float = 1.0, prefix: str = "Pivot") -> Tuple[pd.DataFrame, str, str, str]:
    if window is None:
        pivot = (data["High"] + data["Low"] + data["Close"]) / 3
    else:
        pivot = (data["High"].rolling(int(window)).max() + data["Low"].rolling(int(window)).min() + data["Close"].rolling(int(window)).mean()) / 3
    pivot_col = f"{prefix}_Point" if window is None else f"{prefix}_Point_{_fmt(window)}"
    r1_col = f"{prefix}_R1_{_fmt(multiplier)}" if window is None else f"{prefix}_R1_{_fmt(window)}_{_fmt(multiplier)}"
    s1_col = f"{prefix}_S1_{_fmt(multiplier)}" if window is None else f"{prefix}_S1_{_fmt(window)}_{_fmt(multiplier)}"
    data[pivot_col] = pivot
    data[r1_col] = data[pivot_col] + float(multiplier) * (data["High"] - data["Low"])
    data[s1_col] = data[pivot_col] - float(multiplier) * (data["High"] - data["Low"])
    return data, pivot_col, r1_col, s1_col


def add_classic_pivot_points(data: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str]:
    pivot_col = "Classic_Pivot_Point"
    r1_col = "Classic_R1"
    s1_col = "Classic_S1"
    data[pivot_col] = (data["High"] + data["Low"] + data["Close"]) / 3
    data[r1_col] = 2 * data[pivot_col] - data["Low"]
    data[s1_col] = 2 * data[pivot_col] - data["High"]
    return data, pivot_col, r1_col, s1_col


def add_chandelier_exit(data: pd.DataFrame, atr_window: int, multiplier: float) -> Tuple[pd.DataFrame, str, str]:
    data, atr_col = add_atr(data, atr_window, normalize=False)
    col = f"Chandelier_Exit_{_fmt(atr_window)}_{_fmt(multiplier)}"
    data[col] = data["Close"] - float(multiplier) * data[atr_col]
    return data, col, atr_col


# ============================================================
# 4. Oscillators and momentum indicators
# ============================================================


def add_rsi(data: pd.DataFrame, window: int, column: str = "Close") -> Tuple[pd.DataFrame, str]:
    col = f"RSI_{_fmt(window)}"
    if col not in data.columns:
        data[col] = ta.momentum.rsi(data[column], window=int(window))
    return data, col


def add_sma_of_series(data: pd.DataFrame, source_col: str, window: int, prefix: str = "SMA") -> Tuple[pd.DataFrame, str]:
    col = f"{prefix}_{source_col}_{_fmt(window)}"
    data[col] = data[source_col].rolling(int(window)).mean()
    return data, col


def add_stoch_rsi(data: pd.DataFrame, rsi_window: int, stoch_window: int, smoothing: int = 1, column: str = "Close") -> Tuple[pd.DataFrame, str, str]:
    data, rsi_col = add_rsi(data, rsi_window, column=column)
    stoch_col = f"Stoch_RSI_{_fmt(rsi_window)}_{_fmt(stoch_window)}_{_fmt(smoothing)}"
    raw_col = f"Stoch_RSI_RAW_{_fmt(rsi_window)}_{_fmt(stoch_window)}"
    if raw_col not in data.columns:
        # Kept consistent with your Strategy.py usage: ta.momentum.stochrsi(data['RSI'], ...)
        data[raw_col] = ta.momentum.stochrsi(data[rsi_col], window=int(stoch_window))
    if int(smoothing) > 1:
        data[stoch_col] = data[raw_col].rolling(int(smoothing)).mean()
    else:
        data[stoch_col] = data[raw_col]
    return data, stoch_col, rsi_col


def add_stoch_rsi_kd(data: pd.DataFrame, rsi_window: int, stoch_window: int, k_smoothing: int, d_smoothing: int, column: str = "Close") -> Tuple[pd.DataFrame, str, str, str, str]:
    data, stoch_col, rsi_col = add_stoch_rsi(data, rsi_window, stoch_window, smoothing=1, column=column)
    k_col = f"Stoch_RSI_K_{_fmt(rsi_window)}_{_fmt(stoch_window)}_{_fmt(k_smoothing)}"
    d_col = f"Stoch_RSI_D_{_fmt(rsi_window)}_{_fmt(stoch_window)}_{_fmt(k_smoothing)}_{_fmt(d_smoothing)}"
    data[k_col] = data[stoch_col].rolling(int(k_smoothing)).mean()
    data[d_col] = data[k_col].rolling(int(d_smoothing)).mean()
    return data, k_col, d_col, stoch_col, rsi_col


def add_stochastic(data: pd.DataFrame, window: int, smooth_window: int = 3, d_window: Optional[int] = None) -> Tuple[pd.DataFrame, str, Optional[str]]:
    k_col = f"Stochastic_K_{_fmt(window)}_{_fmt(smooth_window)}"
    data[k_col] = ta.momentum.stoch(data["High"], data["Low"], data["Close"], window=int(window), smooth_window=int(smooth_window))
    d_col = None
    if d_window is not None:
        d_col = f"Stochastic_D_{_fmt(window)}_{_fmt(smooth_window)}_{_fmt(d_window)}"
        data[d_col] = data[k_col].rolling(int(d_window)).mean()
    return data, k_col, d_col


def add_macd(data: pd.DataFrame, fast: int, slow: int, signal: int, column: str = "Close") -> Tuple[pd.DataFrame, str, str, str]:
    macd_col = f"MACD_{_fmt(fast)}_{_fmt(slow)}_{_fmt(signal)}"
    signal_col = f"MACD_Signal_{_fmt(fast)}_{_fmt(slow)}_{_fmt(signal)}"
    hist_col = f"MACD_Hist_{_fmt(fast)}_{_fmt(slow)}_{_fmt(signal)}"
    if hist_col not in data.columns:
        macd = ta.trend.MACD(data[column], window_fast=int(fast), window_slow=int(slow), window_sign=int(signal))
        data[macd_col] = macd.macd()
        data[signal_col] = macd.macd_signal()
        data[hist_col] = macd.macd_diff()
    return data, macd_col, signal_col, hist_col


def add_price_oscillator(data: pd.DataFrame, short_ema: int, long_ema: int, percent: bool = True) -> Tuple[pd.DataFrame, str, str, str]:
    data, fast_col = add_ema(data, short_ema, prefix="PO_EMA_Fast")
    data, slow_col = add_ema(data, long_ema, prefix="PO_EMA_Slow")
    col = f"Price_Oscillator_{_fmt(short_ema)}_{_fmt(long_ema)}"
    if percent:
        data[col] = ((data[fast_col] - data[slow_col]) / data[slow_col]) * 100
    else:
        data[col] = data[fast_col] - data[slow_col]
    return data, col, fast_col, slow_col


def add_roc(data: pd.DataFrame, window: int, column: str = "Close") -> Tuple[pd.DataFrame, str]:
    col = f"ROC_{_fmt(window)}"
    data[col] = ta.momentum.roc(data[column], window=int(window))
    return data, col


def add_ultimate_oscillator(data: pd.DataFrame, window1: int, window2: int, window3: int) -> Tuple[pd.DataFrame, str]:
    col = f"UO_{_fmt(window1)}_{_fmt(window2)}_{_fmt(window3)}"
    data[col] = ta.momentum.ultimate_oscillator(data["High"], data["Low"], data["Close"], window1=int(window1), window2=int(window2), window3=int(window3))
    return data, col


def add_williams_r(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str]:
    col = f"Williams_R_{_fmt(window)}"
    highest_high = data["High"].rolling(window=int(window)).max()
    lowest_low = data["Low"].rolling(window=int(window)).min()
    data[col] = ((highest_high - data["Close"]) / (highest_high - lowest_low)) * -100
    return data, col


def add_cmo(data: pd.DataFrame, window: int, column: str = "Close") -> Tuple[pd.DataFrame, str]:
    col = f"CMO_{_fmt(window)}"
    delta = data[column].diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    sum_gains = gains.rolling(window=int(window)).sum()
    sum_losses = losses.rolling(window=int(window)).sum()
    data[col] = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    return data, col


def add_cci(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str]:
    col = f"CCI_{_fmt(window)}"
    data[col] = ta.trend.cci(data["High"], data["Low"], data["Close"], window=int(window))
    return data, col


def add_trix(data: pd.DataFrame, window: int, column: str = "Close") -> Tuple[pd.DataFrame, str]:
    col = f"Trix_{_fmt(window)}"
    data[col] = ta.trend.trix(data[column], window=int(window))
    return data, col


def add_awesome_oscillator(data: pd.DataFrame, short_window: int, long_window: int) -> Tuple[pd.DataFrame, str]:
    col = f"AO_{_fmt(short_window)}_{_fmt(long_window)}"
    data[col] = ta.momentum.awesome_oscillator(data["High"], data["Low"], window1=int(short_window), window2=int(long_window))
    return data, col


def add_kst(data: pd.DataFrame, roc1: int, roc2: int, roc3: int, roc4: int, signal_window: int) -> Tuple[pd.DataFrame, str, str]:
    kst_col = f"KST_{_fmt(roc1)}_{_fmt(roc2)}_{_fmt(roc3)}_{_fmt(roc4)}"
    signal_col = f"KST_Signal_{_fmt(signal_window)}"
    r1 = ta.momentum.roc(data["Close"], window=int(roc1))
    r2 = ta.momentum.roc(data["Close"], window=int(roc2))
    r3 = ta.momentum.roc(data["Close"], window=int(roc3))
    r4 = ta.momentum.roc(data["Close"], window=int(roc4))
    data[kst_col] = r1 + 2 * r2 + 3 * r3 + 4 * r4
    data[signal_col] = data[kst_col].rolling(int(signal_window)).mean()
    return data, kst_col, signal_col


# ============================================================
# 5. Directional, trend strength, volatility indicators
# ============================================================


def add_adx(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str]:
    col = f"ADX_{_fmt(window)}"
    if col not in data.columns:
        data[col] = ta.trend.adx(data["High"], data["Low"], data["Close"], window=int(window))
    return data, col


def add_di(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str, str]:
    plus_col = f"DI_plus_{_fmt(window)}"
    minus_col = f"DI_minus_{_fmt(window)}"
    data[plus_col] = ta.trend.adx_pos(data["High"], data["Low"], data["Close"], window=int(window))
    data[minus_col] = ta.trend.adx_neg(data["High"], data["Low"], data["Close"], window=int(window))
    return data, plus_col, minus_col


def add_adx_di(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str, str, str]:
    data, adx_col = add_adx(data, window)
    data, plus_col, minus_col = add_di(data, window)
    return data, adx_col, plus_col, minus_col


def add_atr(data: pd.DataFrame, window: int, normalize: bool = False) -> Tuple[pd.DataFrame, str]:
    suffix = "N" if normalize else ""
    col = f"ATR{suffix}_{_fmt(window)}"
    if col not in data.columns:
        atr = ta.volatility.average_true_range(data["High"], data["Low"], data["Close"], window=int(window))
        data[col] = atr / data["Close"] if normalize else atr
    return data, col


def add_aroon(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str, str]:
    up_col = f"Aroon_Up_{_fmt(window)}"
    down_col = f"Aroon_Down_{_fmt(window)}"
    data[up_col] = ta.trend.aroon_up(high=data["High"], low=data["Low"], window=int(window))
    data[down_col] = ta.trend.aroon_down(high=data["High"], low=data["Low"], window=int(window))
    return data, up_col, down_col


def add_psar(data: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> Tuple[pd.DataFrame, str, str]:
    psar_col = f"PSAR_{_fmt(acceleration)}_{_fmt(maximum)}"
    trend_col = f"PSAR_Trend_{_fmt(acceleration)}_{_fmt(maximum)}"
    high = data["High"]
    low = data["Low"]
    psar = np.zeros(len(high))
    trend = np.ones(len(high))
    af = float(acceleration)
    ep = high.iloc[0]
    psar[0] = low.iloc[0]

    for i in range(1, len(high)):
        psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
        if trend[i - 1] == 1:
            if low.iloc[i] < psar[i]:
                trend[i] = -1
                psar[i] = ep
                af = float(acceleration)
                ep = low.iloc[i]
            else:
                trend[i] = 1
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + float(acceleration), float(maximum))
        else:
            if high.iloc[i] > psar[i]:
                trend[i] = 1
                psar[i] = ep
                af = float(acceleration)
                ep = high.iloc[i]
            else:
                trend[i] = -1
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + float(acceleration), float(maximum))

    data[psar_col] = pd.Series(psar, index=data.index)
    data[trend_col] = pd.Series(trend, index=data.index)
    return data, psar_col, trend_col


def add_supertrend(data: pd.DataFrame, atr_window: int, multiplier: float) -> Tuple[pd.DataFrame, str, str, str, str]:
    data, atr_col = add_atr(data, atr_window, normalize=False)
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    hl2 = (high + low) / 2
    upper_band = hl2 + float(multiplier) * data[atr_col]
    lower_band = hl2 - float(multiplier) * data[atr_col]

    supertrend = np.zeros(len(data))
    if len(data) > 0:
        supertrend[0] = close.iloc[0]
    for i in range(1, len(data)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            supertrend[i] = lower_band.iloc[i]
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            supertrend[i] = upper_band.iloc[i]
        else:
            supertrend[i] = supertrend[i - 1]

    upper_col = f"Supertrend_Upper_{_fmt(atr_window)}_{_fmt(multiplier)}"
    lower_col = f"Supertrend_Lower_{_fmt(atr_window)}_{_fmt(multiplier)}"
    st_col = f"Supertrend_{_fmt(atr_window)}_{_fmt(multiplier)}"
    data[upper_col] = upper_band
    data[lower_col] = lower_band
    data[st_col] = pd.Series(supertrend, index=data.index)
    return data, st_col, upper_col, lower_col, atr_col


# ============================================================
# 6. Volume and money-flow indicators
# ============================================================


def add_vwap(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str]:
    col = f"VWAP_{_fmt(window)}"
    data[col] = ta.volume.volume_weighted_average_price(data["High"], data["Low"], data["Close"], data["Volume"], window=int(window))
    return data, col


def add_vwma(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str]:
    col = f"VWMA_{_fmt(window)}"
    data[col] = (data["Close"] * data["Volume"]).rolling(window=int(window)).sum() / data["Volume"].rolling(window=int(window)).sum()
    return data, col


def add_obv(data: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    col = "OBV"
    data[col] = ta.volume.OnBalanceVolumeIndicator(data["Close"], data["Volume"]).on_balance_volume()
    return data, col


def add_volume_delta(data: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str]:
    up_col = "Up_Volume"
    down_col = "Down_Volume"
    delta_col = "Volume_Delta"
    data[up_col] = np.where(data["Close"] > data["Close"].shift(1), data["Volume"], 0)
    data[down_col] = np.where(data["Close"] < data["Close"].shift(1), data["Volume"], 0)
    data[delta_col] = data[up_col] - data[down_col]
    return data, up_col, down_col, delta_col


def add_mfi(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str]:
    col = f"MFI_{_fmt(window)}"
    data[col] = ta.volume.money_flow_index(data["High"], data["Low"], data["Close"], data["Volume"], window=int(window))
    return data, col


def add_cmf(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str]:
    col = f"CMF_{_fmt(window)}"
    data[col] = ta.volume.chaikin_money_flow(data["High"], data["Low"], data["Close"], data["Volume"], window=int(window))
    return data, col


def add_adl(data: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    col = "ADL"
    data[col] = ta.volume.acc_dist_index(data["High"], data["Low"], data["Close"], data["Volume"])
    return data, col


def add_chaikin_oscillator(data: pd.DataFrame, fast_window: int, slow_window: int) -> Tuple[pd.DataFrame, str, str]:
    adl_col = f"Chaikin_ADL_{_fmt(fast_window)}_{_fmt(slow_window)}"
    osc_col = f"Chaikin_Osc_{_fmt(fast_window)}_{_fmt(slow_window)}"
    mfm = ((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / (data["High"] - data["Low"]).replace(0, 1)
    mfv = mfm * data["Volume"]
    data[adl_col] = mfv.cumsum()
    ema_fast = data[adl_col].ewm(span=int(fast_window), adjust=False).mean()
    ema_slow = data[adl_col].ewm(span=int(slow_window), adjust=False).mean()
    data[osc_col] = ema_fast - ema_slow
    return data, osc_col, adl_col


def add_klinger_oscillator(data: pd.DataFrame, fast_window: int, slow_window: int) -> Tuple[pd.DataFrame, str]:
    col = f"Klinger_{_fmt(fast_window)}_{_fmt(slow_window)}"
    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    vf = tp.diff() * data["Volume"]
    ema_fast = vf.ewm(span=int(fast_window), adjust=False).mean()
    ema_slow = vf.ewm(span=int(slow_window), adjust=False).mean()
    data[col] = ema_fast - ema_slow
    return data, col


def add_force_index(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str]:
    col = f"Force_Index_{_fmt(window)}"
    data[col] = (data["Close"].diff() * data["Volume"]).rolling(window=int(window)).sum()
    return data, col


def add_ease_of_movement(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str, str, str, str]:
    midpoint_col = "EoM_Midpoint_Move"
    box_col = "EoM_Box_Ratio"
    eom_col = "EoM"
    ma_col = f"EoM_MA_{_fmt(window)}"
    data[midpoint_col] = (data["High"] + data["Low"]) / 2 - (data["High"].shift(1) + data["Low"].shift(1)) / 2
    data[box_col] = data["Volume"] / (data["High"] - data["Low"]).replace(0, np.nan)
    data[eom_col] = data[midpoint_col] / data[box_col]
    data[ma_col] = data[eom_col].rolling(window=int(window)).mean()
    return data, midpoint_col, box_col, eom_col, ma_col


def add_volume_profile(data: pd.DataFrame, price_bins: int) -> Tuple[pd.DataFrame, str]:
    col = f"Volume_Profile_{_fmt(price_bins)}"
    price_range = np.linspace(data["Close"].min(), data["Close"].max(), int(price_bins))
    volume_kde = gaussian_kde(data["Volume"] * data["Close"])
    volume_profile = volume_kde(price_range)
    data[col] = np.interp(data["Close"], price_range, volume_profile)
    return data, col


# ============================================================
# 7. Pattern/price transforms and special tools
# ============================================================


def add_heikin_ashi(data: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str, str]:
    ha_close = "HA_Close"
    ha_open = "HA_Open"
    ha_high = "HA_High"
    ha_low = "HA_Low"
    data[ha_close] = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4
    data[ha_open] = (data["Open"].shift(1) + data["Close"].shift(1)) / 2
    data[ha_high] = data[["High", ha_open, ha_close]].max(axis=1)
    data[ha_low] = data[["Low", ha_open, ha_close]].min(axis=1)
    return data, ha_open, ha_high, ha_low, ha_close


def add_renko_box(data: pd.DataFrame, box_size: float) -> Tuple[pd.DataFrame, str]:
    col = f"Renko_Box_{_fmt(box_size)}"
    data[col] = np.floor(data["Close"] / float(box_size)) * float(box_size)
    return data, col


def add_grid_levels(data: pd.DataFrame, grid_size: float, num_levels: int) -> Tuple[pd.DataFrame, str]:
    col = f"Grid_Level_{_fmt(grid_size)}_{_fmt(num_levels)}"
    initial_price = data["Close"].iloc[0]
    levels = [initial_price + i * float(grid_size) for i in range(-int(num_levels), int(num_levels) + 1)]
    data[col] = 0.0
    for level in levels:
        data.loc[(data["Close"] >= level - float(grid_size) / 2) & (data["Close"] < level + float(grid_size) / 2), col] = level
    return data, col


def detect_fractal_high(data: pd.DataFrame) -> pd.Series:
    return (
        (data["High"] > data["High"].shift(1))
        & (data["High"] > data["High"].shift(2))
        & (data["High"] > data["High"].shift(-1))
        & (data["High"] > data["High"].shift(-2))
    )


def detect_fractal_low(data: pd.DataFrame) -> pd.Series:
    return (
        (data["Low"] < data["Low"].shift(1))
        & (data["Low"] < data["Low"].shift(2))
        & (data["Low"] < data["Low"].shift(-1))
        & (data["Low"] < data["Low"].shift(-2))
    )


def add_fractal_chaos_bands(data: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str, str, str, str]:
    high_flag = f"Fractal_High_{_fmt(window)}"
    low_flag = f"Fractal_Low_{_fmt(window)}"
    upper_col = f"Fractal_Upper_{_fmt(window)}"
    lower_col = f"Fractal_Lower_{_fmt(window)}"
    data[high_flag] = detect_fractal_high(data).astype(int)
    data[low_flag] = detect_fractal_low(data).astype(int)
    data[upper_col] = data["High"].rolling(window=int(window)).max() * data[high_flag]
    data[lower_col] = data["Low"].rolling(window=int(window)).min() * data[low_flag]
    data[upper_col] = data[upper_col].replace(0, np.nan).ffill()
    data[lower_col] = data[lower_col].replace(0, np.nan).ffill()
    return data, upper_col, lower_col, high_flag, low_flag


def add_zigzag(data: pd.DataFrame, threshold: float, column: str = "Close") -> Tuple[pd.DataFrame, str]:
    col = f"ZigZag_{_fmt(threshold)}"
    zigzag = [None] * len(data)
    if len(data) == 0:
        data[col] = pd.Series(dtype=float, index=data.index)
        return data, col
    last_extreme = data[column].iloc[0]
    is_peak = True
    for i in range(1, len(data)):
        price = data[column].iloc[i]
        change = (price - last_extreme) / last_extreme * 100
        if is_peak and change >= float(threshold):
            zigzag[i] = price
            last_extreme = price
            is_peak = False
        elif (not is_peak) and change <= -float(threshold):
            zigzag[i] = price
            last_extreme = price
            is_peak = True
        else:
            zigzag[i] = zigzag[i - 1]
    data[col] = pd.Series(zigzag, index=data.index)
    return data, col


def add_gann_fan(data: pd.DataFrame, window: int, column: str = "Close") -> Tuple[pd.DataFrame, str, str, str, str, str]:
    ph_col = f"Gann_Pivot_High_{_fmt(window)}"
    pl_col = f"Gann_Pivot_Low_{_fmt(window)}"
    g1_col = f"Gann_1x1_{_fmt(window)}"
    g2_col = f"Gann_2x1_{_fmt(window)}"
    g3_col = f"Gann_3x1_{_fmt(window)}"
    data[ph_col] = data[column].rolling(window=int(window), min_periods=1).max()
    data[pl_col] = data[column].rolling(window=int(window), min_periods=1).min()
    step = np.linspace(0, 1, num=len(data))
    data[g1_col] = data[pl_col] + step * (data[ph_col] - data[pl_col])
    data[g2_col] = data[pl_col] + step * (data[ph_col] - data[pl_col]) / 2
    data[g3_col] = data[pl_col] + step * (data[ph_col] - data[pl_col]) / 3
    return data, g1_col, g2_col, g3_col, ph_col, pl_col


def add_fibonacci_levels(data: pd.DataFrame, window: int, prefix: str = "Fib") -> Tuple[pd.DataFrame, dict]:
    high_col = f"{prefix}_High_Max_{_fmt(window)}"
    low_col = f"{prefix}_Low_Min_{_fmt(window)}"
    data[high_col] = data["High"].rolling(window=int(window)).max()
    data[low_col] = data["Low"].rolling(window=int(window)).min()
    levels = {
        "23_6": f"{prefix}_23_6_{_fmt(window)}",
        "38_2": f"{prefix}_38_2_{_fmt(window)}",
        "50": f"{prefix}_50_{_fmt(window)}",
        "61_8": f"{prefix}_61_8_{_fmt(window)}",
        "78_6": f"{prefix}_78_6_{_fmt(window)}",
    }
    span = data[high_col] - data[low_col]
    data[levels["23_6"]] = data[low_col] + 0.236 * span
    data[levels["38_2"]] = data[low_col] + 0.382 * span
    data[levels["50"]] = data[low_col] + 0.5 * span
    data[levels["61_8"]] = data[low_col] + 0.618 * span
    data[levels["78_6"]] = data[low_col] + 0.786 * span
    levels["high"] = high_col
    levels["low"] = low_col
    return data, levels


def add_ichimoku(data: pd.DataFrame, conversion_window: int, base_window: int, span_b_window: int) -> Tuple[pd.DataFrame, str, str, str, str]:
    conv_col = f"Ichimoku_Conversion_{_fmt(conversion_window)}"
    base_col = f"Ichimoku_Base_{_fmt(base_window)}"
    span_a_col = f"Ichimoku_Span_A_{_fmt(conversion_window)}_{_fmt(base_window)}"
    span_b_col = f"Ichimoku_Span_B_{_fmt(span_b_window)}"
    data[conv_col] = (data["High"].rolling(window=int(conversion_window)).max() + data["Low"].rolling(window=int(conversion_window)).min()) / 2
    data[base_col] = (data["High"].rolling(window=int(base_window)).max() + data["Low"].rolling(window=int(base_window)).min()) / 2
    data[span_a_col] = (data[conv_col] + data[base_col]) / 2
    data[span_b_col] = (data["High"].rolling(window=int(span_b_window)).max() + data["Low"].rolling(window=int(span_b_window)).min()) / 2
    return data, conv_col, base_col, span_a_col, span_b_col


def add_elder_ray(data: pd.DataFrame, ema_window: int) -> Tuple[pd.DataFrame, str, str, str]:
    data, ema_col = add_ema(data, ema_window, prefix="Elder_EMA")
    bull_col = f"BullPower_{_fmt(ema_window)}"
    bear_col = f"BearPower_{_fmt(ema_window)}"
    data[bull_col] = data["High"] - data[ema_col]
    data[bear_col] = data["Low"] - data[ema_col]
    return data, bull_col, bear_col, ema_col


# ============================================================
# 8. Convenience: add all common indicators for a given strategy
# ============================================================


def add_common_indicators_for_strategy(data: pd.DataFrame, strategy_name: str, parameters: Sequence[float]) -> pd.DataFrame:
    """
    Optional dispatcher to precompute indicators for selected strategies.

    This is intentionally small. For clean code, prefer calling the exact
    add_* functions inside each strategy.
    """
    name = strategy_name.lower()

    if "bollinger" in name:
        window = int(parameters[0]) if len(parameters) > 0 else 20
        mult = float(parameters[1]) if len(parameters) > 1 else 2.0
        add_bollinger(data, window, mult)

    if "stochastic_rsi" in name or "stochrsi" in name:
        # Try common parameter locations, but exact strategies should call add_stoch_rsi explicitly.
        rsi_window = int(parameters[2]) if len(parameters) > 2 else 14
        stoch_window = int(parameters[3]) if len(parameters) > 3 else 14
        smoothing = int(parameters[4]) if len(parameters) > 4 else 1
        add_stoch_rsi(data, rsi_window, stoch_window, smoothing)

    if "rsi" in name and "stochastic_rsi" not in name and "stochrsi" not in name:
        add_rsi(data, int(parameters[0]) if len(parameters) > 0 else 14)

    if "macd" in name:
        # Common MACD defaults; exact strategies should call add_macd explicitly.
        add_macd(data, fast=12, slow=26, signal=9)

    if "adx" in name:
        add_adx(data, int(parameters[0]) if len(parameters) > 0 else 14)

    if "atr" in name:
        add_atr(data, int(parameters[0]) if len(parameters) > 0 else 14)

    return data


__all__ = [name for name in globals() if name.startswith("add_") or name in {
    "make_position",
    "clean_indicator_frame",
    "require_columns",
    "crossed_above",
    "crossed_below",
    "detect_fractal_high",
    "detect_fractal_low",
}]
