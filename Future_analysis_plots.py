# analysis_plots.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# BASIC PLOTS
# ============================================================
def plot_price_with_ema(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame | None,
    pair_name: str,
    timeframe: str,
    output_dir: str,
    suffix: str = "",
    last_days: int | None = None,
):
    # ============================
    # Time filtering (LAST N DAYS)
    # ============================
    if last_days is not None:
        end_ts = df_actual.index.max()
        start_ts = end_ts - pd.Timedelta(days=last_days)

        df_actual = df_actual.loc[start_ts:end_ts]
        if df_pred is not None:
            df_pred = df_pred.loc[start_ts:end_ts]

    plt.figure(figsize=(12, 6))

    # ============================
    # ACTUAL (BLUE)
    # ============================
    plt.plot(
        df_actual.index,
        df_actual["close"],
        color="blue",
        linestyle="-",
        linewidth=1.4,
        label="Close (actual)",
    )
    plt.plot(
        df_actual.index,
        df_actual["ema20"],
        color="blue",
        linestyle="--",
        linewidth=1.1,
        label="EMA20 (actual)",
    )
    plt.plot(
        df_actual.index,
        df_actual["ema50"],
        color="blue",
        linestyle="-.",
        linewidth=1.1,
        label="EMA50 (actual)",
    )

    # ============================
    # PREDICTED (RED)
    # ============================
    if df_pred is not None and not df_pred.empty:
        plt.plot(
            df_pred.index,
            df_pred["close"],
            color="red",
            linestyle="-",
            linewidth=1.3,
            alpha=0.85,
            label="Close (pred)",
        )
        plt.plot(
            df_pred.index,
            df_pred["ema20"],
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
            label="EMA20 (pred)",
        )
        plt.plot(
            df_pred.index,
            df_pred["ema50"],
            color="red",
            linestyle="-.",
            linewidth=1.0,
            alpha=0.85,
            label="EMA50 (pred)",
        )

    plt.title(f"{pair_name} Trend View [{timeframe}]"
              + (f" – last {last_days} days" if last_days else ""))
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fname = f"{pair_name}_trend_ema{suffix}.png"
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"[{pair_name}] Saved trend EMA plot:", out)


def plot_forward_return_comparison(idx, y_true_1, y_pred_1, pair_name: str, timeframe: str, output_dir: str, skip: int = 50):
    """
    Plots:
      - True 1-step log-return as a RED solid line
      - Pred 1-step log-return as a BLUE dashed line
      - Pred markers every `skip` points as BLUE circles (for readability)
    """
    idx = pd.Index(idx)
    y_true_1 = np.asarray(y_true_1)
    y_pred_1 = np.asarray(y_pred_1)

    plt.figure(figsize=(12, 6))
    plt.plot(idx, y_true_1, label="True 1-step fwd log-ret", color="red", linestyle="-", linewidth=1)
    plt.plot(idx, y_pred_1, label="Pred 1-step fwd log-ret (line)", color="blue", linestyle="--", linewidth=1)
    #plt.plot(idx[::skip], y_pred_1[::skip], label="Pred 1-step fwd log-ret (markers)",
    #         color="blue", linestyle="None", marker="o", markersize=3)

    plt.title(f"{pair_name} 1-step Forward Return (True vs Pred) [{timeframe}]")
    plt.xlabel("Time")
    plt.ylabel("log return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(output_dir, f"{pair_name}_fwdret_1step_true_vs_pred.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[{pair_name}] Saved 1-step return comparison plot:", out)


def plot_trend_comparison(idx, y_true_path, y_pred_path, pair_name: str, timeframe: str, output_dir: str, horizon_label: str):
    """
    Trend comparison over the full horizon:
      y_true_path / y_pred_path: shape (N, FUTURE_STEPS)
      We plot the *last* column -> cumulative log return to horizon
    """
    true_trend = y_true_path[:, -1]
    pred_trend = y_pred_path[:, -1]

    plt.figure(figsize=(12, 6))
    plt.plot(idx, true_trend, label="True cumulative forward return", linewidth=1)
    plt.plot(idx, pred_trend, label="Pred cumulative forward return", linewidth=1)
    plt.axhline(0, color="black", linewidth=0.8)

    plt.title(
        f"{pair_name} Trend Comparison (Real vs Predicted)\n"
        f"Horizon = {horizon_label} [{timeframe}]"
    )
    plt.xlabel("Time")
    plt.ylabel("Cumulative log return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(output_dir, f"{pair_name}_trend_comparison_{timeframe}_h{horizon_label}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[{pair_name}] Saved trend comparison plot:", out)


def plot_direction_accuracy_series(idx, y_true_1, y_pred_1, pair_name: str, timeframe: str, output_dir: str, roll_window: int = 100):
    """
    Rolling direction accuracy:
      correct[t] = 1 if sign(true_ret[t]) == sign(pred_ret[t]) else 0
      then rolling mean over `roll_window`
    """
    idx = pd.Index(idx)
    y_true_1 = np.asarray(y_true_1)
    y_pred_1 = np.asarray(y_pred_1)

    true_dir = np.sign(y_true_1)
    pred_dir = np.sign(y_pred_1)
    correct = (true_dir == pred_dir).astype(int)

    roll = pd.Series(correct, index=idx).rolling(roll_window).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(roll.index, roll.values, label=f"Rolling direction accuracy (window={roll_window})", linewidth=1)
    plt.ylim(0, 1)
    plt.title(f"{pair_name} Rolling Direction Accuracy (sign of 1-step fwd ret) [{timeframe}]")
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(output_dir, f"{pair_name}_direction_accuracy_roll.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[{pair_name}] Saved direction accuracy plot:", out)


# ============================================================
# CONFIDENCE FILTERING + ACCURACY vs |PRED|
# ============================================================

def compute_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def summarize_confidence_filter(
    y_true=None,
    y_pred=None,
    pair_name: str = "",
    timeframe: str | None = None,
    output_dir: str = ".",
    percentiles=(50, 60, 70, 80, 90),
    thresholds=None,
    idx=None,
    save_csv: bool = True,
    # Backward compatible positional names (if you still call it the old way)
    y_true_1=None,
    y_pred_1=None,
):
    # Backward compatibility: if caller passed y_true_1/y_pred_1 instead of y_true/y_pred
    if y_true is None and y_true_1 is not None:
        y_true = y_true_1
    if y_pred is None and y_pred_1 is not None:
        y_pred = y_pred_1

    if y_true is None or y_pred is None:
        raise ValueError("summarize_confidence_filter: y_true and y_pred must be provided.")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"[{pair_name}] Empty arrays. Skipping confidence summary.")
        return pd.DataFrame()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"[{pair_name}] Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )

    abs_pred = np.abs(y_pred)

    if thresholds is None:
        thresholds = [float(np.percentile(abs_pred, p)) for p in percentiles]

    rows = []
    for thr in thresholds:
        mask = abs_pred >= thr
        coverage = float(np.mean(mask))

        if mask.sum() == 0:
            dir_acc = np.nan
            n = 0
        else:
            dir_acc = float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))
            n = int(mask.sum())

        rows.append({
            "pair": pair_name,
            "timeframe": timeframe,
            "threshold": float(thr),
            "coverage": coverage,
            "n": n,
            "dir_acc": dir_acc,
        })

    df_sum = pd.DataFrame(rows)

    print(f"\n[{pair_name}] Confidence filtering summary (1-step):")
    print(df_sum.to_string(index=False))

    if save_csv:
        out = os.path.join(output_dir, f"{pair_name}_confidence_filter_summary.csv")
        df_sum.to_csv(out, index=False)
        print(f"[{pair_name}] Saved confidence filter summary CSV:", out)

    return df_sum


def plot_accuracy_vs_abs_pred(
    y_true_1,
    y_pred_1,
    pair_name: str,
    timeframe: str,
    output_dir: str,
    n_bins: int = 12,
    min_count_per_bin: int = 50,
):

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    y_true_1 = np.asarray(y_true_1).reshape(-1)
    y_pred_1 = np.asarray(y_pred_1).reshape(-1)

    # Safety: drop NaNs / infs
    mask = np.isfinite(y_true_1) & np.isfinite(y_pred_1)
    y_true_1 = y_true_1[mask]
    y_pred_1 = y_pred_1[mask]

    if len(y_true_1) == 0:
        print(f"[{pair_name}] plot_accuracy_vs_abs_pred: no valid samples.")
        return

    abs_pred = np.abs(y_pred_1)
    correct = (np.sign(y_true_1) == np.sign(y_pred_1)).astype(int)

    # Quantile bins => balanced sample sizes
    # If too many duplicates, qcut may fail; fallback to linspace bins.
    try:
        bins = pd.qcut(abs_pred, q=n_bins, duplicates="drop")
    except Exception:
        edges = np.linspace(abs_pred.min(), abs_pred.max(), n_bins + 1)
        bins = pd.cut(abs_pred, bins=edges, include_lowest=True)

    df = pd.DataFrame({
        "abs_pred": abs_pred,
        "correct": correct,
        "bin": bins.astype(str),
    })

    grp = df.groupby("bin", observed=False).agg(
        count=("correct", "size"),
        accuracy=("correct", "mean"),
        abs_pred_mean=("abs_pred", "mean"),
        abs_pred_min=("abs_pred", "min"),
        abs_pred_max=("abs_pred", "max"),
    ).reset_index()

    # Filter bins with too few samples
    grp["keep"] = grp["count"] >= int(min_count_per_bin)
    grp_keep = grp[grp["keep"]].copy()

    # Save bin table CSV (always)
    csv_path = os.path.join(output_dir, f"{pair_name}_acc_vs_abs_pred_bins.csv")
    grp.to_csv(csv_path, index=False)
    print(f"[{pair_name}] Saved accuracy-vs-|pred| bins CSV:", csv_path)

    if len(grp_keep) == 0:
        print(f"[{pair_name}] Not enough samples per bin to plot (min_count_per_bin={min_count_per_bin}).")
        return

    # Plot accuracy vs mean abs_pred per bin
    plt.figure(figsize=(12, 6))
    plt.plot(grp_keep["abs_pred_mean"].values, grp_keep["accuracy"].values, marker="o", linewidth=1)
    plt.axhline(0.5, linewidth=1)

    plt.title(f"{pair_name} Direction Accuracy vs |Prediction| [{timeframe}]")
    plt.xlabel("Mean |y_pred_1| (per bin)")
    plt.ylabel("Direction Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.tight_layout()

    out = os.path.join(output_dir, f"{pair_name}_acc_vs_abs_pred.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[{pair_name}] Saved accuracy-vs-|pred| plot:", out)

def plot_price_prediction_nextclose(
    timestamps,
    actual_prices,
    train_plot,
    test_plot,
    pair_name: str,
    timeframe: str,
    output_dir: str,
    marker_skip: int = 100,     # <-- ADD THIS
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18, 6))

    # Actual (black solid)
    plt.plot(
        timestamps, actual_prices,
        label="Actual Close",
        color="black",
        linestyle="-",
        linewidth=1
    )

    # Train (red dashed)
    plt.plot(
        timestamps, train_plot,
        label="Pred Next-Close (Train)",
        color="red",
        linestyle="--",
        linewidth=1
    )

    # Test (blue dash-dot)
    plt.plot(
        timestamps, test_plot,
        label="Pred Next-Close (Test)",
        color="blue",
        linestyle="-.",
        linewidth=1
    )

    # Markers (skip)
    #if marker_skip is not None and marker_skip > 0:
    #    ts_m = timestamps[::marker_skip]

    #    train_m = train_plot[::marker_skip]
    #    mask_tr = ~np.isnan(train_m)
    #    plt.plot(
    #        ts_m[mask_tr], train_m[mask_tr],
    #        label=f"Train markers (skip={marker_skip})",
    #        color="red",
    #        linestyle="None",
    #        marker="s",
    #        markersize=6
    #    )

    #    test_m = test_plot[::marker_skip]
    #    mask_te = ~np.isnan(test_m)
    #    plt.plot(
    #        ts_m[mask_te], test_m[mask_te],
    #        label=f"Test markers (skip={marker_skip})",
    #        color="blue",
    #        linestyle="None",
    #        marker="o",
    #        markersize=6
    #    )

    plt.title(f"{pair_name} Price & Predicted Next-Close [{timeframe}]")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(output_dir, f"{pair_name}_price_prediction_nextclose.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[{pair_name}] Saved price prediction plot:", out)

