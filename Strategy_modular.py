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


def define_strategy_EMA_cross(data, parameters):
    data = _copy(data)
    data, ema_s = add_ema(data, parameters[0], prefix="EMA_S")
    data, ema_l = add_ema(data, parameters[1], prefix="EMA_L")
    return _finalize(data, data[ema_s] > data[ema_l], data[ema_s] < data[ema_l], [ema_s, ema_l])


def define_strategy_EMA_envelope(data, parameters):
    data = _copy(data)
    data, upper, lower, ema = add_ma_envelope(data, parameters[0], parameters[1], ma_type="EMA")
    return _finalize(data, data["Close"] > data[upper], data["Close"] < data[lower], [upper, lower])


def define_strategy_TEMA(data, parameters):
    data = _copy(data)
    data, tema = add_tema(data, parameters[0])
    return _finalize(data, data["Close"] > data[tema], data["Close"] < data[tema], [tema])


def define_strategy_Donchian(data, parameters):
    data = _copy(data)
    data, upper, lower, mid = add_donchian(data, parameters[0])
    return _finalize(data, data["Close"] > data[upper], data["Close"] < data[lower], [upper, lower])


def define_strategy_Aroon(data, parameters):
    data = _copy(data)
    data, up, down = add_aroon(data, parameters[0])
    return _finalize(data, data[up] > parameters[1], data[down] > parameters[2], [up, down])


def define_strategy_WilliamsR(data, parameters):
    data = _copy(data)
    data, wr = add_williams_r(data, parameters[0])
    return _finalize(data, data[wr] < parameters[1], data[wr] > parameters[2], [wr])


def define_strategy_Elder_Ray(data, parameters):
    data = _copy(data)
    data, bull, bear, ema = add_elder_ray(data, parameters[0])
    return _finalize(data, data[bull] > 0, data[bear] < 0, [bull, bear])


def define_strategy_Klinger(data, parameters):
    data = _copy(data)
    data, klinger = add_klinger_oscillator(data, parameters[0], parameters[1])
    return _finalize(data, data[klinger] > 0, data[klinger] < 0, [klinger])


def define_strategy_CMO(data, parameters):
    data = _copy(data)
    data, cmo = add_cmo(data, parameters[0])
    return _finalize(data, data[cmo] < parameters[1], data[cmo] > parameters[2], [cmo])


# ============================================================
# 21-50. Channels / volatility / volume
# ============================================================


def define_strategy_Price_Oscillator(data, parameters):
    data = _copy(data)
    data, fast, slow, po = add_price_oscillator(data, parameters[0], parameters[1], percent=True)
    return _finalize(data, data[po] > float(parameters[2]), data[po] < -float(parameters[2]), [po])


def define_strategy_Ultimate_Oscillator(data, parameters):
    data = _copy(data)
    data, uo = add_ultimate_oscillator(data, parameters[0], parameters[1], parameters[2])
    return _finalize(data, data[uo] < parameters[3], data[uo] > parameters[4], [uo])


def define_strategy_Chaikin(data, parameters):
    data = _copy(data)
    data, osc, adl = add_chaikin_oscillator(data, fast_window=parameters[1], slow_window=parameters[0])
    return _finalize(data, data[osc] > 0, data[osc] < 0, [osc])


def define_strategy_CMF(data, parameters):
    data = _copy(data)
    data, cmf = add_cmf(data, parameters[0])
    return _finalize(data, data[cmf] > parameters[1], data[cmf] < -parameters[1], [cmf])


def define_strategy_Fractal_Chaos(data, parameters):
    data = _copy(data)
    data, upper, lower, high_flag, low_flag = add_fractal_chaos_bands(data, parameters[0])
    return _finalize(data, data["Close"] > data[upper], data["Close"] < data[lower], [upper, lower])


def define_strategy_SuperTrend(data, parameters):
    data = _copy(data)
    data, st, upper, lower, atr = add_supertrend(data, parameters[0], parameters[1])
    return _finalize(data, data["Close"] < data[lower], data["Close"] > data[upper], [upper, lower, atr])


def define_strategy_ZigZag(data, parameters):
    data = _copy(data)
    data, zz = add_zigzag(data, parameters[0])
    return _finalize(data, data[zz].notna() & data[zz].shift(1).isna() & (data["Close"] > data[zz]),
                     data[zz].notna() & data[zz].shift(1).isna() & (data["Close"] < data[zz]), [zz])


def define_strategy_Hull_MA(data, parameters):
    data = _copy(data)
    data, hma = add_hma(data, parameters[0])
    return _finalize(data, data["Close"] > data[hma], data["Close"] < data[hma], [hma])


def define_strategy_Gann_Fan(data, parameters):
    data = _copy(data)
    data, g1, g2, g3, ph, pl = add_gann_fan(data, parameters[0])
    return _finalize(data, crossed_above(data["Close"], data[g1]), crossed_below(data["Close"], data[g1]), [g1])


def define_strategy_ROC(data, parameters):
    data = _copy(data)
    data, roc = add_roc(data, parameters[0])
    return _finalize(data, data[roc] > parameters[1], data[roc] < parameters[2], [roc])


def define_strategy_MFI_divergence(data, parameters):
    data = _copy(data)
    data, mfi = add_mfi(data, parameters[0])
    return _finalize(data, data[mfi] < parameters[1], data[mfi] > parameters[2], [mfi])


def define_strategy_PSAR_simple(data, parameters):
    data = _copy(data)
    data, psar, trend = add_psar(data, parameters[0], parameters[1])
    return _finalize(data, (data["Close"] > data[psar]) & (data[trend] == 1) & (data[trend].shift(1) == -1),
                     (data["Close"] < data[psar]) & (data[trend] == -1) & (data[trend].shift(1) == 1), [psar, trend])


def define_strategy_CMF_ADX(data, parameters):
    data = _copy(data)
    data, cmf = add_cmf(data, parameters[0])
    data, adx = add_adx(data, parameters[1])
    return _finalize(data, (data[cmf] > 0) & (data[adx] > parameters[2]),
                     (data[cmf] < 0) & (data[adx] > parameters[2]), [cmf, adx])


def define_strategy_PSAR_momentum(data, parameters):
    data = _copy(data)
    data, psar, trend = add_psar(data, parameters[0], parameters[1])
    data, ret = add_log_returns(data, name="returns")
    return _finalize(data, (data["Close"] > data[psar]) & (data[ret] > parameters[2]),
                     (data["Close"] < data[psar]) & (data[ret] < parameters[3]), [psar, ret])


def define_strategy_Trix(data, parameters):
    data = _copy(data)
    data, trix = add_trix(data, parameters[0])
    return _finalize(data, data[trix] > 0, data[trix] < 0, [trix])


def define_strategy_Keltner_channel(data, parameters):
    data = _copy(data)
    data, upper, lower, ema, atr = add_keltner_channel(data, parameters[0], parameters[1], parameters[2])
    return _finalize(data, data["Close"] > data[upper], data["Close"] < data[lower], [upper, lower])


def define_strategy_Momentum(data, parameters):
    data = _copy(data)
    data, mom = add_momentum(data, parameters[0])
    return _finalize(data, data[mom] > parameters[1], data[mom] < -parameters[1], [mom])


def define_strategy_Ichimoku(data, parameters):
    data = _copy(data)
    data, conv, base, span_a, span_b = add_ichimoku(data, parameters[0], parameters[1], parameters[2])
    return _finalize(data, data[conv] > data[base], data[conv] < data[base], [conv, base])


def define_strategy_Zscore(data, parameters):
    data = _copy(data)
    data, z, mean, std = add_zscore(data, parameters[0])
    return _finalize(data, data[z] > parameters[1], data[z] < -parameters[1], [z])


def define_strategy_MA_envelope(data, parameters):
    data = _copy(data)
    data, upper, lower, sma = add_ma_envelope(data, parameters[0], parameters[1], ma_type="SMA")
    return _finalize(data, data["Close"] > data[upper], data["Close"] < data[lower], [upper, lower])


def define_strategy_ATR(data, parameters):
    data = _copy(data)
    data, atr = add_atr(data, parameters[0], normalize=False)
    upper = data["Close"] + parameters[1] * data[atr]
    lower = data["Close"] - parameters[1] * data[atr]
    return _finalize(data, data["Close"] > upper, data["Close"] < lower, [atr])


def define_strategy_ADX(data, parameters):
    data = _copy(data)
    data, adx = add_adx(data, parameters[0])
    return _finalize(data, data[adx] > parameters[1], data[adx] < parameters[2], [adx])


def define_strategy_CCI(data, parameters):
    data = _copy(data)
    data, cci = add_cci(data, parameters[0])
    return _finalize(data, data[cci] > parameters[1], data[cci] < -parameters[1], [cci])


def define_strategy_Linear_Regression(data, parameters):
    data = _copy(data)
    data, upper, lower, lr, std = add_linear_regression_channel(data, parameters[0], parameters[1])
    return _finalize(data, data["Close"] > data[upper], data["Close"] < data[lower], [upper, lower])


def define_strategy_VWMA_Price_Oscillator(data, parameters):
    data = _copy(data)
    data, vwma = add_vwap(data, parameters[0])
    data, fast, slow, po = add_price_oscillator(data, parameters[1], parameters[2], percent=False)
    return _finalize(data, (data["Close"] > data[vwma]) & (data[po] > 0),
                     (data["Close"] < data[vwma]) & (data[po] < 0), [vwma, po])


def define_strategy_Dynamic_Pivot_Points_Classic(data, parameters):
    data = _copy(data)
    data, pivot, r1, s1 = add_pivot_points(data, window=parameters[1], multiplier=parameters[0], prefix="Dynamic_Pivot")
    return _finalize(data, data["Close"] > data[r1], data["Close"] < data[s1], [r1, s1])


def define_strategy_Force_Index(data, parameters):
    data = _copy(data)
    data, fi = add_force_index(data, parameters[0])
    return _finalize(data, data[fi] > 0, data[fi] < 0, [fi])


def define_strategy_Chandelier_Exit(data, parameters):
    data = _copy(data)
    data, ce, atr = add_chandelier_exit(data, parameters[0], parameters[1])
    return _finalize(data, data["Close"] > data[ce], data["Close"] < data[ce], [ce])


def define_strategy_Fibonacci(data, parameters):
    data = _copy(data)
    data, levels = add_fibonacci_levels(data, parameters[0], prefix="Fib")
    return _finalize(data, (data["Close"] > data[levels["50"]]) & (data["Close"] < data[levels["61_8"]]),
                     (data["Close"] < data[levels["38_2"]]) & (data["Close"] > data[levels["23_6"]]),
                     [levels["23_6"], levels["38_2"], levels["50"], levels["61_8"]])


def define_strategy_ADL(data, parameters):
    data = _copy(data)
    data, adl = add_adl(data)
    return _finalize(data, data[adl] > parameters[0], data[adl] < parameters[1], [adl])


# ============================================================
# 51-88. Combinations
# ============================================================


def define_strategy_RSI_Bollinger(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, upper, lower, sma, width = add_bollinger(data, parameters[1], parameters[2])
    return _finalize(data, (data[rsi] < parameters[3]) & (data["Close"] < data[lower]),
                     (data[rsi] > parameters[4]) & (data["Close"] > data[upper]), [rsi, upper, lower])


def define_strategy_Turtle_Trading(data, parameters):
    data = _copy(data)
    data, upper, lower, mid = add_donchian(data, parameters[0])
    return _finalize(data, data["Close"] > data[upper].shift(1), data["Close"] < data[lower].shift(1), [upper, lower])


def define_strategy_Mean_Reversion(data, parameters):
    data = _copy(data)
    data, z, mean, std = add_zscore(data, parameters[0])
    upper = data[mean] + parameters[1] * data[std]
    lower = data[mean] - parameters[1] * data[std]
    return _finalize(data, data["Close"] > upper, data["Close"] < lower, [mean, std])


def define_strategy_Breakout(data, parameters):
    data = _copy(data)
    data, upper, lower, mid = add_donchian(data, parameters[0])
    return _finalize(data, data["Close"] > data[upper].shift(1), data["Close"] < data[lower].shift(1), [upper, lower])


def define_strategy_RSI_Divergence(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, rsi_signal = add_sma_of_series(data, rsi, parameters[1], prefix="RSI_Signal")
    return _finalize(data, data[rsi] < parameters[2], data[rsi] > parameters[3], [rsi, rsi_signal])


def define_strategy_MA_Cross_RSI(data, parameters):
    data = _copy(data)
    data, sma_s = add_sma(data, parameters[0], prefix="SMA_S")
    data, sma_l = add_sma(data, parameters[1], prefix="SMA_L")
    data, rsi = add_rsi(data, parameters[2])
    return _finalize(data, (data[sma_s] > data[sma_l]) & (data[rsi] < parameters[3]),
                     (data[sma_s] < data[sma_l]) & (data[rsi] > parameters[4]), [sma_s, sma_l, rsi])


def define_strategy_ADX_MA(data, parameters):
    data = _copy(data)
    data, adx = add_adx(data, parameters[0])
    data, sma_s = add_sma(data, parameters[1], prefix="SMA_S")
    data, sma_l = add_sma(data, parameters[2], prefix="SMA_L")
    return _finalize(data, (data[adx] > parameters[3]) & (data[sma_s] > data[sma_l]),
                     (data[adx] < parameters[4]) & (data[sma_s] < data[sma_l]), [adx, sma_s, sma_l])


def define_strategy_Bollinger_Breakout_Momentum_Oscillator(data, parameters):
    data = _copy(data)
    data, upper, lower, sma, width = add_bollinger(data, parameters[0], parameters[1])
    data, roc = add_roc(data, parameters[2])
    return _finalize(data, (data["Close"] > data[upper]) & (data[roc] > 0),
                     (data["Close"] < data[lower]) & (data[roc] < 0), [upper, lower, roc])


def define_strategy_Fibonacci_MA(data, parameters):
    data = _copy(data)
    data, levels = add_fibonacci_levels(data, parameters[0], prefix="FibMA")
    data, sma = add_sma(data, parameters[1])
    long_cond = (data["Close"].between(data[levels["38_2"]], data[levels["50"]])) & (data["Close"] > data[sma])
    short_cond = (data["Close"].between(data[levels["23_6"]], data[levels["38_2"]])) & (data["Close"] < data[sma])
    return _finalize(data, long_cond, short_cond, [levels["23_6"], levels["38_2"], levels["50"], sma])


def define_strategy_Mean_Variance_Optimization(data, parameters):
    data = _copy(data)
    mean_col = f"Mean_{int(parameters[0])}"
    var_col = f"Variance_{int(parameters[1])}"
    data[mean_col] = data["Close"].rolling(int(parameters[0])).mean()
    data[var_col] = data["Close"].rolling(int(parameters[1])).var()
    return _finalize(data, data["Close"] > data[mean_col] + parameters[2] * data[var_col],
                     data["Close"] < data[mean_col] - parameters[2] * data[var_col], [mean_col, var_col])


def define_strategy_MA_ribbon(data, parameters):
    data = _copy(data)
    data, cols = add_ma_ribbon(data, [parameters[0], parameters[1]], ma_type="SMA")
    return _finalize(data, data[cols[0]] > data[cols[1]], data[cols[0]] < data[cols[1]], cols)


def define_strategy_ADX_DI(data, parameters):
    data = _copy(data)
    data, adx, plus, minus = add_adx_di(data, parameters[0])
    return _finalize(data, (data[plus] > data[minus]) & (data[adx] > parameters[1]),
                     (data[plus] < data[minus]) & (data[adx] > parameters[1]), [adx, plus, minus])


def define_strategy_MACD_RSI(data, parameters):
    data = _copy(data)
    data, macd, sig, hist = add_macd(data, parameters[1], parameters[0], parameters[2])
    data, rsi = add_rsi(data, parameters[3])
    return _finalize(data, (data[hist] > 0) & (data[rsi] < parameters[4]),
                     (data[hist] < 0) & (data[rsi] > parameters[5]), [hist, rsi])


def define_strategy_Fibonacci_retracement(data, parameters):
    data = _copy(data)
    data, levels = add_fibonacci_levels(data, parameters[0], prefix="FibRet")
    data["position"] = 0
    data.loc[crossed_above(data["Close"], data[levels["50"]]), "position"] = 1
    data.loc[crossed_below(data["Close"], data[levels["38_2"]]), "position"] = -1
    data.loc[crossed_above(data["Close"], data[levels["61_8"]]), "position"] = 2
    data.loc[crossed_below(data["Close"], data[levels["23_6"]]), "position"] = -2
    return clean_indicator_frame(data)


def define_strategy_RSI_trend_reversal(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, ema = add_ema(data, parameters[1])
    return _finalize(data, (data[rsi] < parameters[2]) & (data["Close"] > data[ema]),
                     (data[rsi] > parameters[3]) & (data["Close"] < data[ema]), [rsi, ema])


def define_strategy_CMO_EMA(data, parameters):
    data = _copy(data)
    data, cmo = add_cmo(data, parameters[0])
    data, ema = add_ema(data, parameters[1])
    return _finalize(data, (data[cmo] > parameters[2]) & (data["Close"] > data[ema]),
                     (data[cmo] < parameters[3]) & (data["Close"] < data[ema]), [cmo, ema])


def define_strategy_MA_momentum(data, parameters):
    data = _copy(data)
    data, sma_s = add_sma(data, parameters[0], prefix="SMA_Short")
    data, sma_l = add_sma(data, parameters[1], prefix="SMA_Long")
    data, mom = add_momentum(data, parameters[2])
    return _finalize(data, (data[sma_s] > data[sma_l]) & (data[mom] > parameters[3]),
                     (data[sma_s] < data[sma_l]) & (data[mom] < parameters[4]), [sma_s, sma_l, mom])


def define_strategy_RSI_Stochastic(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, stoch, _ = add_stochastic(data, parameters[1])
    return _finalize(data, (data[rsi] < parameters[2]) & (data[stoch] < parameters[3]),
                     (data[rsi] > parameters[4]) & (data[stoch] > parameters[5]), [rsi, stoch])


def define_strategy_Garman_Klass_Volatility(data, parameters):
    data = _copy(data)
    data, gk = add_garman_klass_volatility(data)
    threshold = data[gk].mean() + parameters[0] * data[gk].std()
    return _finalize(data, data[gk] > threshold, data[gk] < threshold, [gk])


def define_strategy_Momentum_MACD(data, parameters):
    data = _copy(data)
    data, mom = add_momentum(data, parameters[0])
    data, macd, sig, hist = add_macd(data, parameters[2], parameters[1], 9)
    return _finalize(data, (data[mom] > parameters[3]) & (data[macd] > 0),
                     (data[mom] < -parameters[4]) & (data[macd] < 0), [mom, macd])


def define_strategy_Bollinger_Stochastic(data, parameters):
    data = _copy(data)
    data, upper, lower, sma, width = add_bollinger(data, parameters[0], parameters[1])
    data, stoch, _ = add_stochastic(data, parameters[2])
    return _finalize(data, (data[stoch] < parameters[3]) & (data["Close"] < data[lower]),
                     (data[stoch] > parameters[4]) & (data["Close"] > data[upper]), [stoch, upper, lower])


def define_strategy_Momentum_Breakout(data, parameters):
    data = _copy(data)
    data, ret, mom = add_rolling_sum_returns(data, parameters[0])
    return _finalize(data, data[mom] > parameters[1], data[mom] < -parameters[1], [mom])


def define_strategy_EMA_MACD(data, parameters):
    data = _copy(data)
    data, ema_f = add_ema(data, parameters[0], prefix="EMA_fast")
    data, ema_s = add_ema(data, parameters[1], prefix="EMA_slow")
    macd_col = "EMA_MACD_Diff"
    data[macd_col] = data[ema_f] - data[ema_s]
    return _finalize(data, data[macd_col] > parameters[2], data[macd_col] < -parameters[3], [macd_col])


def define_strategy_Bollinger_EMA(data, parameters):
    data = _copy(data)
    data, upper, lower, sma, width = add_bollinger(data, parameters[0], parameters[1])
    data, ema = add_ema(data, parameters[2])
    return _finalize(data, (data["Close"] > data[upper]) & (data["Close"] > data[ema]),
                     (data["Close"] < data[lower]) & (data["Close"] < data[ema]), [upper, lower, ema])


def define_strategy_MA_Momentum_F(data, parameters):
    data = _copy(data)
    data, sma_s = add_sma(data, parameters[0], prefix="SMA_short")
    data, sma_l = add_sma(data, parameters[1], prefix="SMA_long")
    data, mom = add_momentum(data, parameters[2], pct=True)
    return _finalize(data, (data[sma_s] > data[sma_l]) & (data[mom] > parameters[3]),
                     (data[sma_s] < data[sma_l]) & (data[mom] < -parameters[3]), [sma_s, sma_l, mom])


def define_strategy_Pivot_Stochastic(data, parameters):
    data = _copy(data)
    data, pivot, r1, s1 = add_pivot_points(data, window=None, multiplier=1, prefix="Pivot")
    data, stoch, _ = add_stochastic(data, parameters[0])
    return _finalize(data, (data["Close"] < data[s1]) & (data[stoch] < parameters[1]),
                     (data["Close"] > data[r1]) & (data[stoch] > parameters[2]), [s1, r1, stoch])


def define_strategy_VWMA(data, parameters):
    data = _copy(data)
    data, vwma = add_vwma(data, parameters[0])
    return _finalize(data, data["Close"] > data[vwma], data["Close"] < data[vwma], [vwma])


def define_strategy_EMA_Momentum(data, parameters):
    data = _copy(data)
    data, ema = add_ema(data, parameters[0])
    data, mom = add_momentum(data, parameters[1], pct=True)
    return _finalize(data, data[mom] > parameters[2], data[mom] < -parameters[2], [ema, mom])


def define_strategy_RSI_A_MA(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, sma = add_sma(data, parameters[1])
    return _finalize(data, data[rsi] < parameters[2], data[rsi] > parameters[3], [rsi, sma])


def define_strategy_EMA_Ribbon(data, parameters):
    data = _copy(data)
    data, cols = add_ma_ribbon(data, [parameters[0], parameters[1], parameters[2]], ma_type="EMA")
    return _finalize(data, (data[cols[0]] > data[cols[1]]) & (data[cols[1]] > data[cols[2]]),
                     (data[cols[0]] < data[cols[1]]) & (data[cols[1]] < data[cols[2]]), cols)


def define_strategy_RSI_MA_Envelope(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, upper, lower, sma = add_ma_envelope(data, parameters[1], parameters[2], ma_type="SMA")
    return _finalize(data, (data[rsi] < parameters[3]) & (data["Close"] < data[lower]),
                     (data[rsi] > parameters[4]) & (data["Close"] > data[upper]), [rsi, upper, lower])


def define_strategy_OBV_RSI(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, obv = add_obv(data)
    return _finalize(data, (data[rsi] < parameters[1]) & (data[obv] > data[obv].shift(1)),
                     (data[rsi] > parameters[2]) & (data[obv] < data[obv].shift(1)), [rsi, obv])


def define_strategy_ATR_RSI(data, parameters):
    data = _copy(data)
    data, atr = add_atr(data, parameters[0])
    data, rsi = add_rsi(data, parameters[2])
    atr_scaled = data[atr] * parameters[1]
    return _finalize(data, (atr_scaled > data["Close"]) & (data[rsi] < parameters[3]),
                     (atr_scaled < data["Close"]) & (data[rsi] > parameters[4]), [atr, rsi])


def define_strategy_EMA_Bollinger(data, parameters):
    data = _copy(data)
    data, ema = add_ema(data, parameters[0])
    data, upper, lower, sma, width = add_bollinger(data, parameters[1], parameters[2])
    return _finalize(data, data[ema] > data[upper], data[ema] < data[lower], [ema, upper, lower])


def define_strategy_RSI_MA_Ribbon(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, cols = add_ma_ribbon(data, [parameters[1], parameters[2], parameters[3]], ma_type="SMA")
    return _finalize(data, (data[rsi] < parameters[4]) & (data[cols[0]] > data[cols[1]]) & (data[cols[1]] > data[cols[2]]),
                     (data[rsi] > parameters[5]) & (data[cols[0]] < data[cols[1]]) & (data[cols[1]] < data[cols[2]]), [rsi] + cols)


def define_strategy_EMA_ADX(data, parameters):
    data = _copy(data)
    data, ema_s = add_ema(data, parameters[0], prefix="EMA_short")
    data, ema_l = add_ema(data, parameters[1], prefix="EMA_long")
    data, adx = add_adx(data, parameters[2])
    return _finalize(data, (data[ema_s] > data[ema_l]) & (data[adx] > parameters[3]),
                     (data[ema_s] < data[ema_l]) & (data[adx] < parameters[3]), [ema_s, ema_l, adx])


def define_strategy_RSI_Bollinger_Momentum(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, upper, lower, sma, width = add_bollinger(data, parameters[1], parameters[2])
    data, mom = add_momentum(data, parameters[3], pct=True)
    return _finalize(data, (data[rsi] < parameters[4]) & (data["Close"] < data[lower]) & (data[mom] > parameters[5]),
                     (data[rsi] > parameters[6]) & (data["Close"] > data[upper]) & (data[mom] < -parameters[5]), [rsi, upper, lower, mom])


def define_strategy_Renko_Box_Trading(data, parameters):
    data = _copy(data)
    data, box = add_renko_box(data, parameters[0])
    return _finalize(data, data[box] > data[box].shift(1), data[box] < data[box].shift(1), [box])


# ============================================================
# 89-133. Later hybrid strategies
# ============================================================


def define_strategy_ADX_Stochastic(data, parameters):
    data = _copy(data)
    data, adx = add_adx(data, parameters[0])
    data, stoch, _ = add_stochastic(data, parameters[1])
    return _finalize(data, (data[adx] > parameters[2]) & (data[stoch] < parameters[3]),
                     (data[adx] < parameters[4]) & (data[stoch] > parameters[5]), [adx, stoch])


def define_strategy_MA_Ribbon_ADX(data, parameters):
    data = _copy(data)
    data, cols = add_ma_ribbon(data, [parameters[0], parameters[1], parameters[2]], ma_type="SMA")
    data, adx = add_adx(data, parameters[3])
    return _finalize(data, (data[cols[0]] > data[cols[1]]) & (data[cols[1]] > data[cols[2]]) & (data[adx] > parameters[4]),
                     (data[cols[0]] < data[cols[1]]) & (data[cols[1]] < data[cols[2]]) & (data[adx] < parameters[5]), cols + [adx])


def define_strategy_EMA_Stochastic(data, parameters):
    data = _copy(data)
    data, ema = add_ema(data, parameters[0])
    data, stoch, _ = add_stochastic(data, parameters[1])
    return _finalize(data, (data[ema] > data["Close"]) & (data[stoch] < parameters[2]),
                     (data[ema] < data["Close"]) & (data[stoch] > parameters[3]), [ema, stoch])


def define_strategy_RSI_ADX(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, adx = add_adx(data, parameters[1])
    return _finalize(data, (data[rsi] < parameters[2]) & (data[adx] > parameters[3]),
                     (data[rsi] > parameters[4]) & (data[adx] < parameters[5]), [rsi, adx])


def define_strategy_MACD_Stochastic(data, parameters):
    data = _copy(data)
    data, macd, sig, hist = add_macd(data, parameters[1], parameters[0], 9)
    data, stoch, _ = add_stochastic(data, parameters[2])
    return _finalize(data, (data[macd] > parameters[3]) & (data[stoch] < parameters[4]),
                     (data[macd] < -parameters[5]) & (data[stoch] > parameters[6]), [macd, stoch])


def define_strategy_MACD_Bollinger(data, parameters):
    data = _copy(data)
    data, macd, sig, hist = add_macd(data, parameters[1], parameters[0], parameters[2])
    data, upper, lower, sma, width = add_bollinger(data, parameters[3], parameters[4])
    return _finalize(data, (data[macd] > data[sig]) & (data["Close"] < data[lower]),
                     (data[macd] < data[sig]) & (data["Close"] > data[upper]), [macd, sig, upper, lower])


def define_strategy_EMA_Stochastic_Filter(data, parameters):
    data = _copy(data)
    data, ema_s = add_ema(data, parameters[0], prefix="EMA_short")
    data, ema_l = add_ema(data, parameters[1], prefix="EMA_long")
    data, stoch, _ = add_stochastic(data, parameters[2])
    return _finalize(data, (data[ema_s] > data[ema_l]) & (data[stoch] < parameters[3]),
                     (data[ema_s] < data[ema_l]) & (data[stoch] > parameters[4]), [ema_s, ema_l, stoch])


def define_strategy_MACD_MA_Ribbon(data, parameters):
    data = _copy(data)
    data, macd, sig, hist = add_macd(data, parameters[1], parameters[0], 9)
    data, cols = add_ma_ribbon(data, [parameters[2], parameters[3], parameters[4]], ma_type="SMA")
    return _finalize(data, (data[macd] > 0) & (data[cols[0]] > data[cols[1]]) & (data[cols[1]] > data[cols[2]]),
                     (data[macd] < 0) & (data[cols[0]] < data[cols[1]]) & (data[cols[1]] < data[cols[2]]), [macd] + cols)


def define_strategy_RSI_MACD_Combo(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, macd, sig, hist = add_macd(data, parameters[2], parameters[1], 9)
    return _finalize(data, (data[rsi] < parameters[3]) & (data[macd] > 0),
                     (data[rsi] > parameters[4]) & (data[macd] < 0), [rsi, macd])


def define_strategy_Heikin_Ashi_Trend_Continuation(data, parameters):
    data = _copy(data)
    data, ha_open, ha_high, ha_low, ha_close = add_heikin_ashi(data)
    data, sma = add_sma(data, parameters[0])
    data, adx = add_adx(data, parameters[1])
    return _finalize(data, (data[ha_close] > data[ha_open]) & (data[ha_open] > data[ha_close].shift(1)) & (data[adx] > parameters[2]),
                     (data[ha_close] < data[ha_open]) & (data[ha_open] < data[ha_close].shift(1)), [ha_open, ha_close, adx])


def define_strategy_Bollinger_Stochastic_RSI(data, parameters):
    data = _copy(data)
    data, upper, lower, sma, width = add_bollinger(data, parameters[0], parameters[1])
    data, stoch, rsi = add_stoch_rsi(data, parameters[2], parameters[3])
    return _finalize(data, (data[stoch] < parameters[4]) & (data["Close"] < data[lower]),
                     (data[stoch] > parameters[5]) & (data["Close"] > data[upper]), [stoch, upper, lower])


def define_strategy_Trend_Reversal_RSI(data, parameters):
    data = _copy(data)
    data, rsi = add_rsi(data, parameters[0])
    data, ema = add_ema(data, parameters[1])
    return _finalize(data, (data[rsi] < parameters[2]) & (data[ema] > data["Close"]),
                     (data[rsi] > parameters[3]) & (data[ema] < data["Close"]), [rsi, ema])


def define_strategy_Volume_Profile(data, parameters):
    data = _copy(data)
    data, vp = add_volume_profile(data, parameters[0])
    return _finalize(data, data[vp] > np.percentile(data[vp].dropna(), parameters[1]),
                     data[vp] < np.percentile(data[vp].dropna(), parameters[2]), [vp])


def define_strategy_Grid_Trading(data, parameters):
    data = _copy(data)
    data, grid = add_grid_levels(data, parameters[0], parameters[1])
    return _finalize(data, data["Close"] > data[grid], data["Close"] < data[grid], [grid])


def define_strategy_EMA_MACD_ADX(data, parameters):
    data = _copy(data)
    data, ema_s = add_ema(data, parameters[0], prefix="EMA_Short")
    data, ema_l = add_ema(data, parameters[1], prefix="EMA_Long")
    data, macd, sig, hist = add_macd(data, parameters[2], parameters[3], parameters[4])
    data, adx = add_adx(data, parameters[5])
    return _finalize(data, (data[ema_s] > data[ema_l]) & (data[macd] > data[sig]) & (data[adx] > parameters[6]),
                     (data[ema_s] < data[ema_l]) & (data[macd] < data[sig]) & (data[adx] > parameters[6]), [ema_s, ema_l, macd, sig, adx])


def define_strategy_Trend_Momentum_Volatility(data, parameters):
    data = _copy(data)
    data, ema_s = add_ema(data, parameters[0], prefix="EMA_Short")
    data, ema_l = add_ema(data, parameters[1], prefix="EMA_Long")
    data, macd, sig, hist = add_macd(data, parameters[2], parameters[3], parameters[4])
    data, adx = add_adx(data, parameters[5])
    data, atr = add_atr(data, parameters[6])
    data, stoch, rsi = add_stoch_rsi(data, parameters[7], parameters[7])
    return _finalize(data, (data[ema_s] > data[ema_l]) & (data[macd] > data[sig]) & (data[stoch] < 0.30) & (data[adx] > 20) & (data[atr] > float(parameters[8])),
                     (data[ema_s] < data[ema_l]) & (data[macd] < data[sig]) & (data[stoch] > 0.70) & (data[adx] > 20) & (data[atr] > float(parameters[8])),
                     [ema_s, ema_l, macd, sig, stoch, adx, atr])


def define_strategy_Stochastic_RSI_Bollinger_VWAP(data, parameters):
    data = _copy(data)
    data, stoch, rsi = add_stoch_rsi(data, parameters[0], parameters[1])
    data, upper, lower, sma, width = add_bollinger(data, parameters[2], parameters[3])
    data, vwap = add_vwap(data, parameters[4])
    return _finalize(data, (data[stoch] < parameters[5]) & (data["Close"] < data[lower]) & (data["Close"] > data[vwap] + parameters[7]),
                     (data[stoch] > parameters[6]) & (data["Close"] > data[upper]) & (data["Close"] < data[vwap] - parameters[7]),
                     [stoch, upper, lower, vwap])


def define_strategy_Stochastic_RSI_FULL(data, parameters):
    data = _copy(data)
    data, stoch, rsi, k, d = add_stoch_rsi_kd(data, parameters[0], parameters[1], parameters[2], parameters[3])
    return _finalize(data, (data[k] < parameters[4]) & (data[d] < parameters[4]),
                     (data[k] > parameters[5]) & (data[d] > parameters[5]), [k, d])


def define_strategy_Gaussian_Channel_FULL(data, parameters):
    data = _copy(data)
    data, upper, lower, smooth, std = add_gaussian_channel(data, parameters[0], parameters[1], parameters[2], parameters[3])
    return _finalize(data, data["Close"] < data[lower], data["Close"] > data[upper], [upper, lower])


def define_strategy_Combined_Gaussian_Stochastic_RSI_FULL(data, parameters):
    data = _copy(data)
    data, upper, lower, smooth, std = add_gaussian_channel(data, parameters[0], parameters[1], parameters[2], parameters[3])
    data, stoch, rsi, k, d = add_stoch_rsi_kd(data, parameters[4], parameters[5], parameters[6], parameters[7])
    return _finalize(data, (data["Close"] < data[lower]) & (data[k] < parameters[8]) & (data[d] < parameters[8]),
                     (data["Close"] > data[upper]) & (data[k] > parameters[9]) & (data[d] > parameters[9]), [upper, lower, k, d])


def define_strategy_OBV(data, parameters=None):
    data = _copy(data)
    data, obv = add_obv(data)
    return _finalize(data, data[obv] > data[obv].shift(1), data[obv] < data[obv].shift(1), [obv])


def define_strategy_volume_delta(data, parameters=None):
    data = _copy(data)
    data, up, down, delta = add_volume_delta(data)
    return _finalize(data, data[delta] > 0, data[delta] < 0, [delta])


def define_strategy_ease_of_movement(data, parameters):
    data = _copy(data)
    data, midpoint, box, eom, ma = add_ease_of_movement(data, parameters[0])
    return _finalize(data, data[ma] > 0, data[ma] < 0, [ma])


def define_strategy_wma(data, parameters):
    data = _copy(data)
    data, wma = add_wma(data, parameters[0])
    return _finalize(data, data["Close"] > data[wma], data["Close"] < data[wma], [wma])


def define_strategy_ema(data, parameters):
    data = _copy(data)
    data, ema = add_ema(data, parameters[0])
    return _finalize(data, data["Close"] > data[ema], data["Close"] < data[ema], [ema])


def define_strategy_dema(data, parameters):
    data = _copy(data)
    data, dema = add_dema(data, parameters[0])
    return _finalize(data, data["Close"] > data[dema], data["Close"] < data[dema], [dema])


def define_strategy_ama(data, parameters):
    data = _copy(data)
    data, er, sc, ama = add_ama(data, parameters[0], parameters[1], parameters[2])
    return _finalize(data, data["Close"] > data[ama], data["Close"] < data[ama], [ama])


def define_strategy_vidya(data, parameters):
    data = _copy(data)
    data, vol, factor, vidya = add_vidya(data, parameters[0], parameters[1])
    return _finalize(data, data["Close"] > data[vidya], data["Close"] < data[vidya], [vidya])


def define_strategy_SMA_cross(data, parameters):
    data = _copy(data)
    data, sma_s = add_sma(data, parameters[0], prefix="SMA_S")
    data, sma_l = add_sma(data, parameters[1], prefix="SMA_L")
    return _finalize(data, data[sma_s] > data[sma_l], data[sma_s] < data[sma_l], [sma_s, sma_l])


def define_strategy_Stochastic(data, parameters):
    data = _copy(data)
    data, k, d = add_stochastic(data, parameters[0], parameters[1], parameters[2])
    return _finalize(data, (data[k] > data[d]) & (data[k] < parameters[3]),
                     (data[k] < data[d]) & (data[k] > parameters[4]), [k, d])


def define_strategy_AO(data, parameters):
    data = _copy(data)
    data, ao = add_awesome_oscillator(data, parameters[0], parameters[1])
    return _finalize(data, crossed_above(data[ao], float(parameters[2])), crossed_below(data[ao], -float(parameters[2])), [ao])


def define_strategy_KST(data, parameters):
    data = _copy(data)
    data, kst, sig = add_kst(data, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4])
    return _finalize(data, crossed_above(data[kst], data[sig]) & (data[kst] > float(parameters[5])),
                     crossed_below(data[kst], data[sig]) & (data[kst] < -float(parameters[5])), [kst, sig])


def define_strategy_Bollinger(data, parameters):
    data = _copy(data)
    data, upper, lower, sma, width = add_bollinger(data, parameters[0], parameters[1])
    return _finalize(data, data["Close"] > data[upper], data["Close"] < data[lower], [upper, lower])


def define_strategy_Squeeze(data, parameters):
    data = _copy(data)
    data, bb_upper, bb_lower, bb_sma, bb_width = add_bollinger(data, parameters[0], parameters[1], prefix="Squeeze_BB")
    data, kc_upper, kc_lower, kc_ema, kc_atr = add_keltner_channel(data, parameters[2], parameters[2], parameters[3])
    squeeze_on = (data[bb_lower] > data[kc_lower]) & (data[bb_upper] < data[kc_upper])
    squeeze_off = (data[bb_lower] < data[kc_lower]) & (data[bb_upper] > data[kc_upper])
    return _finalize(data, squeeze_on & (data["Close"] > data[bb_upper]), squeeze_off & (data["Close"] < data[bb_lower]),
                     [bb_upper, bb_lower, kc_upper, kc_lower])


def define_strategy_StdDev_Channel(data, parameters):
    data = _copy(data)
    data, upper, lower, central, std = add_stddev_channel(data, parameters[0], parameters[1])
    return _finalize(data, data["Close"] < data[lower], data["Close"] > data[upper], [upper, lower])


def define_strategy_HV(data, parameters):
    data = _copy(data)
    data, ret, hv = add_historical_volatility(data, parameters[0])
    return _finalize(data, data[hv] < float(parameters[2]), data[hv] > float(parameters[1]), [hv])


def define_strategy_VR(data, parameters):
    data = _copy(data)
    data, ret, short_vol, long_vol, vr = add_volatility_ratio(data, parameters[0], parameters[1])
    return _finalize(data, data[vr] > float(parameters[2]), data[vr] < float(parameters[3]), [vr])


def define_strategy_Simple_Pivot_Points(data, parameters=None):
    data = _copy(data)
    data, pivot, r1, s1 = add_classic_pivot_points(data)
    return _finalize(data, data["Close"] > data[r1], data["Close"] < data[s1], [r1, s1])


def define_strategy_DI(data, parameters):
    data = _copy(data)
    data, plus, minus = add_di(data, parameters[0])
    return _finalize(data, data[plus] > data[minus], data[plus] < data[minus], [plus, minus])


def define_strategy_Stochastic_RSI_StdDev_Channel(data, parameters):
    data = _copy(data)
    data, upper, lower, central, std = add_stddev_channel(data, parameters[0], parameters[1])
    data, stoch, rsi = add_stoch_rsi(data, parameters[2], parameters[3])
    return _finalize(data, (data[stoch] < parameters[4]) & (data[stoch] > 0.01) & (data["Close"] < data[lower]),
                     (data[stoch] > parameters[5]) & (data[stoch] < 0.99) & (data["Close"] > data[upper]), [stoch, upper, lower])


def define_strategy_Bollinger_Stochastic_RSI_modified(data, parameters):
    data = _copy(data)
    data, upper, lower, sma, width = add_bollinger(data, parameters[0], parameters[1])
    data, stoch, rsi = add_stoch_rsi(data, parameters[2], parameters[3], smoothing=parameters[4])
    return _finalize(data, (data[stoch] < parameters[5]) & (data["Close"] < data[lower]),
                     (data[stoch] > parameters[6]) & (data["Close"] > data[upper]), [stoch, upper, lower])


def define_strategy_Keltner_Stochastic_RSI(data, parameters):
    data = _copy(data)
    data, upper, lower, ema, atr = add_keltner_channel(data, parameters[0], parameters[1], parameters[2])
    data, stoch, rsi = add_stoch_rsi(data, parameters[3], parameters[4], smoothing=parameters[5])
    return _finalize(data, (data["Close"] < data[lower]) & (data[stoch] < parameters[6]),
                     (data["Close"] > data[upper]) & (data[stoch] > parameters[7]), [upper, lower, stoch])


def define_strategy_HMA_StochRSI(data, parameters):
    data = _copy(data)
    data, hma = add_hma(data, parameters[0])
    std_col = f"HMA_STD_{int(parameters[1])}"
    data[std_col] = data[hma].rolling(int(parameters[1])).std()
    upper = f"HMA_Upper_{int(parameters[0])}_{float(parameters[2])}"
    lower = f"HMA_Lower_{int(parameters[0])}_{float(parameters[2])}"
    data[upper] = data[hma] + float(parameters[2]) * data[std_col]
    data[lower] = data[hma] - float(parameters[2]) * data[std_col]
    data, stoch, rsi = add_stoch_rsi(data, parameters[3], parameters[4], smoothing=parameters[5])
    return _finalize(data, (data["Close"] < data[lower]) & (data[stoch] < parameters[6]),
                     (data["Close"] > data[upper]) & (data[stoch] > parameters[7]), [upper, lower, stoch])


def define_strategy_ADX_ATR_Bollinger_Stochastic_RSI(data, parameters):
    data = _copy(data)
    data, adx = add_adx(data, parameters[0])
    data, upper, lower, sma, width = add_bollinger(data, parameters[1], parameters[2])
    data, stoch, rsi = add_stoch_rsi(data, parameters[3], parameters[4], smoothing=parameters[5])
    data, atr = add_atr(data, parameters[8], normalize=True)
    trend_filter = (data[adx] > parameters[9]) & (data[atr] > parameters[10])
    long_mean_rev = (data[stoch] < parameters[6]) & (data["Close"] < data[lower])
    short_mean_rev = (data[stoch] > parameters[7]) & (data["Close"] > data[upper])
    long_breakout = (data["Close"] > data[upper]) & (data[width] > parameters[11]) & (data[atr] > parameters[10])
    short_breakout = (data["Close"] < data[lower]) & (data[width] > parameters[11]) & (data[atr] > parameters[10])
    return _finalize(data, (trend_filter & long_mean_rev) | (~trend_filter & long_breakout),
                     (trend_filter & short_mean_rev) | (~trend_filter & short_breakout), [adx, atr, upper, lower, stoch])


def define_strategy_Supertrend_Stochastic_RSI(data, parameters):
    data = _copy(data)
    data, st, upper, lower, atr = add_supertrend(data, parameters[0], parameters[1])
    data, stoch, rsi = add_stoch_rsi(data, parameters[2], parameters[3], smoothing=parameters[4])
    return _finalize(data, (data[stoch] < parameters[5]) & (data["Close"] < data[lower]),
                     (data[stoch] > parameters[6]) & (data["Close"] > data[upper]), [upper, lower, stoch])


def define_strategy_Trend_Momentum_RSI_Volatility(data, parameters):
    data = _copy(data)
    data, ema_f = add_ema(data, parameters[0], prefix="EMA_Fast")
    data, ema_s = add_ema(data, parameters[1], prefix="EMA_Slow")
    data, macd, sig, hist = add_macd(data, parameters[2], parameters[3], parameters[4])
    data, adx = add_adx(data, parameters[5])
    data, rsi = add_rsi(data, parameters[7])
    data, atr = add_atr(data, parameters[10])
    rsi_filter = (data[rsi] > int(parameters[8])) & (data[rsi] < int(parameters[9]))
    return _finalize(data, (data[ema_f] > data[ema_s]) & (data[hist] > 0) & (data[adx] > float(parameters[6])) & rsi_filter,
                     (data[ema_f] < data[ema_s]) & (data[hist] < 0) & (data[adx] > float(parameters[6])) & rsi_filter,
                     [ema_f, ema_s, hist, adx, rsi, atr])


# Backward-compatible aliases for names used in your JSON/config.
def define_strategy_Bollinger_Bands_ADX(data, parameters):
    return define_strategy_Bollinger_ADX(data, parameters)


def define_strategy_Bollinger_Breakout_Momentum(data, parameters):
    return define_strategy_Bollinger_Breakout_Momentum_Oscillator(data, parameters)


def define_strategy_Renko_Box(data, parameters):
    return define_strategy_Renko_Box_Trading(data, parameters)


def define_strategy_Heikin_Ashi_Trend(data, parameters):
    return define_strategy_Heikin_Ashi_Trend_Continuation(data, parameters)


def define_strategy_Volume_delta(data, parameters=None):
    return define_strategy_volume_delta(data, parameters)


def define_strategy_Ease_of_Movement(data, parameters):
    return define_strategy_ease_of_movement(data, parameters)


def define_strategy_WMA(data, parameters):
    return define_strategy_wma(data, parameters)


def define_strategy_EMA(data, parameters):
    return define_strategy_ema(data, parameters)


def define_strategy_DEMA(data, parameters):
    return define_strategy_dema(data, parameters)


def define_strategy_AMA(data, parameters):
    return define_strategy_ama(data, parameters)


def define_strategy_VIDYA(data, parameters):
    return define_strategy_vidya(data, parameters)



# Registry: use this instead of a long if/elif chain.
STRATEGY_REGISTRY = {
    name.replace("define_strategy_", ""): func
    for name, func in list(globals().items())
    if name.startswith("define_strategy_") and callable(func)
}


def run_strategy(data: pd.DataFrame, strategy_name: str, parameters=None) -> pd.DataFrame:
    """Run any strategy by name using the registry."""
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {available}")
    if parameters is None:
        return STRATEGY_REGISTRY[strategy_name](data)
    return STRATEGY_REGISTRY[strategy_name](data, parameters)
