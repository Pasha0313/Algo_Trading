"""
Strategy_modular.py

Modular strategy layer built on top of indicators.py.

This file keeps the same style as your original Strategy.py: every strategy
returns the input DataFrame with a `position` column where:
    1  = long
   -1  = short
    0  = flat

The difference is that indicators are now calculated through reusable functions
from indicators.py, so RSI / MACD / Bollinger / ADX / ATR / etc. are not
reimplemented inside every strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import (
    clean_indicator_frame,
    make_position,
    crossed_above,
    crossed_below,
    add_log_returns,
    add_volume_change,
    add_pct_returns,
    add_momentum,
    add_rolling_sum_returns,
    add_historical_volatility,
    add_volatility_ratio,
    add_garman_klass_volatility,
    add_sma,
    add_ema,
    add_wma,
    add_dema,
    add_tema,
    add_hma,
    add_ama,
    add_vidya,
    add_ma_ribbon,
    add_bollinger,
    add_keltner_channel,
    add_ma_envelope,
    add_donchian,
    add_stddev_channel,
    add_linear_regression_channel,
    add_zscore,
    add_gaussian_channel,
    add_pivot_points,
    add_classic_pivot_points,
    add_chandelier_exit,
    add_rsi,
    add_sma_of_series,
    add_stoch_rsi,
    add_stoch_rsi_kd,
    add_stochastic,
    add_macd,
    add_price_oscillator,
    add_roc,
    add_ultimate_oscillator,
    add_williams_r,
    add_cmo,
    add_cci,
    add_trix,
    add_awesome_oscillator,
    add_kst,
    add_adx,
    add_di,
    add_adx_di,
    add_atr,
    add_aroon,
    add_psar,
    add_supertrend,
    add_vwap,
    add_vwma,
    add_obv,
    add_volume_delta,
    add_mfi,
    add_cmf,
    add_adl,
    add_chaikin_oscillator,
    add_klinger_oscillator,
    add_force_index,
    add_ease_of_movement,
    add_volume_profile,
    add_heikin_ashi,
    add_renko_box,
    add_grid_levels,
    add_fractal_chaos_bands,
    add_zigzag,
    add_gann_fan,
    add_fibonacci_levels,
    add_ichimoku,
    add_elder_ray,
)


def _copy(data: pd.DataFrame) -> pd.DataFrame:
    return data.copy()


def _finalize(data: pd.DataFrame, long_cond: pd.Series, short_cond: pd.Series, subset=None) -> pd.DataFrame:
    data = make_position(data, long_cond, short_cond)
    return clean_indicator_frame(data, subset=subset)


# ============================================================
# 0-20. Basic / trend / oscillator strategies
# ============================================================


def define_strategy_PV(data, parameters):
    data = _copy(data)
    data, ret_col = add_log_returns(data, name="returns")
    data, vol_col = add_volume_change(data, name="vol_ch")
    return_thresh = np.percentile(data[ret_col].dropna(), [parameters[0], parameters[1]])
    volume_thresh = np.percentile(data[vol_col].dropna(), [parameters[2], parameters[3]])
    cond_long = (data[ret_col] <= return_thresh[0]) & data[vol_col].between(volume_thresh[0], volume_thresh[1])
    cond_short = (data[ret_col] >= return_thresh[1]) & data[vol_col].between(volume_thresh[0], volume_thresh[1])
    return _finalize(data, cond_long, cond_short, subset=[ret_col, vol_col])


def define_strategy_SMA(data, parameters):
    data = _copy(data)
    data, sma_s = add_sma(data, parameters[0], prefix="SMA_S")
    data, sma_m = add_sma(data, parameters[1], prefix="SMA_M")
    data, sma_l = add_sma(data, parameters[2], prefix="SMA_L")
    return _finalize(data, (data[sma_s] > data[sma_m]) & (data[sma_m] > data[sma_l]),
                     (data[sma_s] < data[sma_m]) & (data[sma_m] < data[sma_l]), [sma_s, sma_m, sma_l])


def define_strategy_MACD(data, parameters):
    data = _copy(data)
    data, macd_col, signal_col, hist_col = add_macd(data, parameters[1], parameters[0], parameters[2])
    return _finalize(data, data[hist_col] > 0, data[hist_col] < 0, [hist_col])


def define_strategy_RSI_MA(data, parameters):
    data = _copy(data)
    data, rsi_col = add_rsi(data, parameters[0])
    data, sma_rsi = add_sma_of_series(data, rsi_col, parameters[1], prefix="SMA_RSI")
    return _finalize(data, (data[rsi_col] > parameters[2]) & (data[rsi_col] > data[sma_rsi]),
                     (data[rsi_col] < parameters[3]) & (data[rsi_col] < data[sma_rsi]), [rsi_col, sma_rsi])


def define_strategy_RSI(data, parameters):
    data = _copy(data)
    data, rsi_col = add_rsi(data, parameters[0])
    return _finalize(data, data[rsi_col] < parameters[1], data[rsi_col] > parameters[2], [rsi_col])


def define_strategy_Stochastic_RSI(data, parameters):
    data = _copy(data)
    data, stoch_col, rsi_col = add_stoch_rsi(data, parameters[0], parameters[1])
    return _finalize(data, (data[stoch_col] < parameters[2]) & (data[stoch_col] > 0.01),
                     (data[stoch_col] > parameters[3]) & (data[stoch_col] < 0.99), [stoch_col, rsi_col])


def define_strategy_Bollinger_ADX(data, parameters):
    data = _copy(data)
    data, upper, lower, sma, width = add_bollinger(data, parameters[0], parameters[1])
    data, adx = add_adx(data, parameters[2])
    return _finalize(data, (data["Close"] > data[upper]) & (data[adx] > parameters[3]),
                     (data["Close"] < data[lower]) & (data[adx] > parameters[3]), [upper, lower, adx])


def define_strategy_TEMA_momentum(data, parameters):
    data = _copy(data)
    data, tema = add_tema(data, parameters[0])
    data, mom = add_momentum(data, parameters[1], pct=True)
    return _finalize(data, (data["Close"] > data[tema]) & (data[mom] > parameters[2]),
                     (data["Close"] < data[tema]) & (data[mom] < -parameters[2]), [tema, mom])


def define_strategy_VWAP(data, parameters):
    data = _copy(data)
    data, vwap = add_vwap(data, parameters[0])
    return _finalize(data, data["Close"] > data[vwap] + parameters[1],
                     data["Close"] < data[vwap] - parameters[1], [vwap])


def define_strategy_VWAP_momentum(data, parameters):
    data = _copy(data)
    data, vwap = add_vwap(data, parameters[0])
    data, ret = add_log_returns(data, name="returns")
    return _finalize(data, (data["Close"] > data[vwap]) & (data[ret] > parameters[1]),
                     (data["Close"] < data[vwap]) & (data[ret] < parameters[2]), [vwap, ret])


def define_strategy_Bollinger_breakout(data, parameters):
    data = _copy(data)
    data, upper, lower, sma, width = add_bollinger(data, parameters[0], parameters[1])
    return _finalize(data, data["Close"] > data[upper], data["Close"] < data[lower], [upper, lower])


def define_strategy_Bollinger_squeeze(data, parameters):
    data = _copy(data)
    data, upper, lower, sma, width = add_bollinger(data, parameters[0], parameters[1])
    squeeze = (data[upper] - data[lower]) < parameters[2]
    return _finalize(data, squeeze & (data["Close"] > data[upper]), squeeze & (data["Close"] < data[lower]), [upper, lower])


