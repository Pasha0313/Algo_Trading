import pandas as pd
import numpy as np
import time

def fetch_historical_data(client, symbol, bar_length, start, end=None):
    start_str = start
    end_str = end
    all_bars = []
    current_start_str = start_str
    previous_candles_count = 0

    while True:
        print(f"Requesting data from {pd.to_datetime(current_start_str).strftime('%Y-%m-%d %H:%M')}...")
        bars = client.futures_historical_klines(
            symbol=symbol,
            interval=bar_length,
            start_str=current_start_str,
            end_str=end_str,
            limit=1000)

        if not bars:
            print("No more data available or the API limit has been reached.")
            break

        all_bars.extend(bars)
        last_timestamp = pd.to_datetime(bars[-1][0], unit="ms")
        current_start_str = (last_timestamp + pd.Timedelta(milliseconds=1)).strftime('%Y-%m-%d %H:%M')
        print(f"Collected {len(all_bars)} candles so far...")

        if len(all_bars) == previous_candles_count + 1:
            print("Only one new candle collected, exiting loop.")
            break
        previous_candles_count = len(all_bars)
        time.sleep(1)

    print(f"Total of {len(all_bars)} candles collected.\n")

    data = pd.DataFrame(all_bars)
    data["Date"] = pd.to_datetime(data.iloc[:, 0], unit="ms")
    data.columns = [
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"
    ]
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    data.set_index("Date", inplace=True)
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Complete"] = [True for _ in range(len(data) - 1)] + [False]

    return data
