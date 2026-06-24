from socket import close
import ta


def Unsupervised_learning_trading_strategy():
    """
    # Unsupervised Learning Trading Strategy

    * Download/Load Crypto prices data.
    * Calculate different features and indicators on each Crypto.
    * Aggregate on monthly level and filter top 150 most liquid Crypto.
    * Calculate Monthly Returns for different time-horizons.
    * Download Fama-French Factors and Calculate Rolling Factor Betas.
    * For each month fit a K-Means Clustering Algorithm to group similar assets based on their features.
    * For each month select assets based on the cluster and form a portfolio based on Efficient Frontier max sharpe ratio optimization.
    * Visualize Portfolio returns and compare to SP500 returns.

    # All Packages Needed:
    * pandas, numpy, matplotlib, statsmodels, pandas_datareader, datetime, yfinance, sklearn, PyPortfolioOpt

    ## 1. Download/Load Crypto prices data.
    """

    from statsmodels.regression.rolling import RollingOLS
    import pandas_datareader.data as web
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import pandas as pd
    import numpy as np
    import datetime as dt
    import yfinance as yf
    import datetime as dt
    import ta
    import warnings
    warnings.filterwarnings('ignore')

    def fetch_crypto_symbols(start, count=100):
        #url = f'https://finance.yahoo.com/markets/crypto/all/?start={start}&count={count}&guccounter=1'
        url = f'https://finance.yahoo.com/markets/crypto/all/?start=0&count=100'

        crypto_symbols = pd.read_html(url)[0]
        crypto_symbols['Symbol'] = crypto_symbols['Symbol'].str.replace('.', '-')
        return crypto_symbols['Symbol'].unique().tolist()

    symbols_list = []
    for start in range(0, 500, 100):
        symbols_list.extend(fetch_crypto_symbols(start))

    symbols_list = list(set(symbols_list))
    end_date = '2024-11-30'
    start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * 4)
    df = yf.download(tickers=symbols_list, start=start_date, end=end_date).stack()
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    print(df)
    df.to_csv('crypto_data.csv')
    df = df.drop(columns=['adj close'], errors='ignore')
    print(df.head())

    """ 2. Calculate features and technical indicators for each Crypto.
    * Garman-Klass Volatility
    * RSI
    * Bollinger Bands
    * ATR
    * MACD
    * Dollar Volume
    """

    df = df.dropna(subset=['close', 'high', 'low', 'open', 'volume'])
    # Garman-Klass volatility
    df['garman_klass_vol'] = ((np.log(df['high']) - np.log(df['low']))**2) / 2 - (2 * np.log(2) - 1) * ((np.log(df['close']) - np.log(df['open']))**2)
    # RSI
    df['rsi'] = df.groupby(level=1)['close'].transform(lambda x: ta.momentum.rsi(x, window=20))

    # Bollinger Bands

    def compute_bbands(x, col_idx):
        try:
            s = np.log1p(x).astype(float)
            bb = ta.volatility.BollingerBands(close=s, window=20, window_dev=2)

            cols = [bb.bollinger_lband(), bb.bollinger_mavg(), bb.bollinger_hband()]
            return cols[col_idx]
        except Exception as e:
            print(f"Error in bbands calculation: {e}")
            return pd.Series(np.nan, index=x.index)

    df['bb_low'] = df.groupby(level=1)['close'].transform(lambda x: compute_bbands(x, 0))
    df['bb_mid'] = df.groupby(level=1)['close'].transform(lambda x: compute_bbands(x, 1))
    df['bb_high'] = df.groupby(level=1)['close'].transform(lambda x: compute_bbands(x, 2))

    # ATR
    def compute_atr(crypto_data):
        atr = ta.volatility.average_true_range(
            high=crypto_data["high"].astype(float),
            low=crypto_data["low"].astype(float),
            close=crypto_data["close"].astype(float),
            window=14)
        return atr.sub(atr.mean()).div(atr.std())

    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    # MACD
    def compute_macd(close):
        macd = ta.trend.macd(close.astype(float), window_slow=26, window_fast=12)
        return macd.sub(macd.mean()).div(macd.std())

    df['macd'] = df.groupby(level=1, group_keys=False)['close'].apply(compute_macd)

    # Dollar Volume
    df['dollar_volume'] = (df['close'] * df['volume']) / 1e6

    df = df.dropna()
    print(df)

    """## 3. Aggregate to monthly level and filter top 50 most liquid Crypto for each month.
    * To reduce training time and experiment with features and strategies, we convert the business-daily data to month-end frequency.
    """

    last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open','high', 'low']]

    data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                    df.unstack()[last_cols].resample('M').last().stack('ticker')],
                    axis=1)).dropna()

    """* Calculate 5-year rolling average of dollar volume for each Crypto before filtering."""

    Number_Of_Year = 5
    filter_top = 150

    data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(Number_Of_Year*12, min_periods=12).mean().stack())
    data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
    data = data[data['dollar_vol_rank']<filter_top].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

    """## 4. Calculate Monthly Returns for different time horizons as features.
    * To capture time series dynamics that reflect, for example, momentum patterns, we compute historical returns using the method .pct_change(lag), that is, returns over various monthly periods as identified by lags.
    """

    def calculate_returns(df):
        outlier_cutoff = 0.0001
        lags = [1, 2, 3, 6, 9, 12]
        for lag in lags:
            df[f'return_{lag}m'] = (df['close'].pct_change(lag).pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                        upper=x.quantile(1-outlier_cutoff))).add(1).pow(1/lag).sub(1))
        return df

    data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

    """## 5. Download Fama-French Factors and Calculate Rolling Factor Betas.
    * We will introduce the Fama—French data to estimate the exposure of assets to common risk factors using linear regression.
    * The five Fama—French factors, namely market risk, size, value, operating profitability, and investment have been shown empirically to explain asset returns and are commonly used to assess the risk/return profile of portfolios. Hence, it is natural to include past factor exposures as financial features in models.
    * We can access the historical factor returns using the pandas-datareader and estimate historical exposures using the RollingOLS rolling linear regression.
    """

    factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                                'famafrench',
                                start='2018')[0].drop('RF', axis=1)

    factor_data.index = factor_data.index.to_timestamp()
    factor_data = factor_data.resample('M').last().div(100)
    factor_data.index.name = 'date'
    factor_data = factor_data.join(data['return_1m']).sort_index()

    """* Filter out cryptos with less than 10 months of data."""

    observations = factor_data.groupby(level=1).size()
    valid_cryptos = observations[observations >= 5]
    factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_cryptos.index)]

    """* Calculate Rolling Factor Betas."""

    # Ensure the constraints for window and min_nobs
    def rolling_ols(group):
        num_regressors = len(group.columns)
        min_nobs = num_regressors + 1
        window = max(min(10, group.shape[0]), min_nobs)

        if group.shape[0] < min_nobs:
            return pd.DataFrame(np.nan, index=group.index, columns=group.columns.drop("return_1m"))

        model = RollingOLS(endog=group["return_1m"],
            exog=sm.add_constant(group.drop("return_1m", axis=1)),
            window=window,
            min_nobs=min_nobs,).fit(params_only=True)

        return model.params.drop("const", axis=1)

    betas = factor_data.groupby(level=1, group_keys=False).apply(rolling_ols)

    betas = betas.dropna()
    print(betas)

    """* Join the rolling factors data to the main features dataframe."""

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    data = (data.join(betas.groupby('ticker').shift()))
    data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
    data = data.drop('close', axis=1)
    data = data.dropna()
    data.info()

    """### At this point we have to decide on what ML model and approach to use for predictions etc.
    ## 6. For each month fit a K-Means Clustering Algorithm to group similar assets based on their features.
    ### K-Means Clustering
    * You may want to initialize predefined centroids for each cluster based on your research.
    * For visualization purpose of this tutorial we will initially rely on the ‘k-means++’ initialization.
    * Then we will pre-define our centroids for each cluster.
    """

    #data = data.drop('cluster', axis=1)
    print(data.columns)
    RSI_Column  = 1
    ATR_Column  = 5

    def plot_clusters(data):

        cluster_0 = data[data['cluster']==0]
        cluster_1 = data[data['cluster']==1]
        cluster_2 = data[data['cluster']==2]
        cluster_3 = data[data['cluster']==3]

        plt.scatter(cluster_0.iloc[:,ATR_Column] , cluster_0.iloc[:,RSI_Column] , color = 'red', label='cluster 0')
        plt.scatter(cluster_1.iloc[:,ATR_Column] , cluster_1.iloc[:,RSI_Column] , color = 'green', label='cluster 1')
        plt.scatter(cluster_2.iloc[:,ATR_Column] , cluster_2.iloc[:,RSI_Column] , color = 'blue', label='cluster 2')
        plt.scatter(cluster_3.iloc[:,ATR_Column] , cluster_3.iloc[:,RSI_Column] , color = 'black', label='cluster 3')

        plt.legend()
        plt.show()
        return

    """### Apply pre-defined centroids."""

    print(data.columns)
    num_features = data.shape[1]
    print(data.shape[1])
    RSI_Column = 1
    target_rsi_values = [30, 45, 55, 70]
    initial_centroids = np.zeros((len(target_rsi_values), num_features))
    initial_centroids[:, RSI_Column] = target_rsi_values
    print(initial_centroids.shape[1])
    print(initial_centroids)

    from sklearn.cluster import KMeans
    #data = data.drop('cluster', axis=1)
    def get_clusters(df):
        df['cluster'] = KMeans(n_clusters=4,random_state=0,init=initial_centroids).fit(df).labels_
        return df

    data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

    plot_numbner = 10
    plt.style.use('ggplot')
    unique_dates = data.index.get_level_values('date').unique().tolist()
    print("Total plot number is :", len(unique_dates))
    limited_dates = unique_dates[:plot_numbner]
    for i in limited_dates:
        g = data.xs(i, level=0)
        plt.title(f'Date {i}')
        plot_clusters(g)

    """## 7. For each month select assets based on the cluster and form a portfolio based on Efficient Frontier max sharpe ratio optimization
    * First we will filter only cryptos corresponding to the cluster we choose based on our hypothesis.
    * Momentum is persistent and my idea would be that cryptos clustered around RSI 70 centroid should continue to outperform in the following month - thus I would select cryptos corresponding to cluster 3.
    """

    filtered_df = data[data['cluster']==3].copy()
    filtered_df = filtered_df.reset_index(level=1)
    filtered_df.index = filtered_df.index+pd.DateOffset(1)
    filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
    dates = filtered_df.index.get_level_values('date').unique().tolist()
    fixed_dates = {}

    for d in dates:
        fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
    print(fixed_dates)

    print(f"Length: {len(fixed_dates)}")
    print(f"Keys: {fixed_dates.keys()}\n")
    for key, value in fixed_dates.items():
        print(f"Key: {key}, Value: {value}")

    """### Define portfolio optimization function
    * We will define a function which optimizes portfolio weights using PyPortfolioOpt package and EfficientFrontier optimizer to maximize the sharpe ratio.
    * To optimize the weights of a given portfolio we would need to supply last 1 year prices to the function.
    * Apply signle crypto weight bounds constraint for diversification (minimum half of equaly weight and maximum 10% of portfolio).
    """

    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns

    def optimize_weights(prices, lower_bound=0):
        returns = expected_returns.mean_historical_return(prices=prices,frequency=252)
        cov = risk_models.sample_cov(prices=prices,frequency=252)
        ef = EfficientFrontier(expected_returns=returns,cov_matrix=cov,weight_bounds=(lower_bound, .1),solver='SCS')
        weights = ef.max_sharpe()
        return ef.clean_weights()

    """* Download Fresh Daily Prices Data only for short listed cryptos."""

    cryptos = data.index.get_level_values('ticker').unique().tolist()
    new_df = yf.download(tickers=cryptos,start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                        end=data.index.get_level_values('date').unique()[-1])

    print(new_df)

    """* Calculate daily returns for each crypto which could land up in our portfolio.
    * Then loop over each month start, select the cryptos for the month and calculate their weights for the next month.
    * If the maximum sharpe ratio optimization fails for a given month, apply equally-weighted weights.
    * Calculated each day portfolio return.
    """

    from itertools import islice
    subset_fixed_dates = dict(islice(fixed_dates.items(), 1))
    print(list(subset_fixed_dates.keys()))
    print(np.log(new_df['Close']).diff())
    returns_dataframe = np.log(new_df['Close']).diff()
    portfolio_df = pd.DataFrame()

    for start_date in fixed_dates.keys():
        try:
            end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
            cols = fixed_dates[start_date]
            optimization_start_date = (pd.to_datetime(start_date)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')
            optimization_end_date = (pd.to_datetime(start_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')
            optimization_df = new_df[optimization_start_date:optimization_end_date]['Close'][cols]
            success = False
            try:
                weights = optimize_weights(prices=optimization_df,
                                    lower_bound=round(1/(len(optimization_df.columns)*2),3))
                weights = pd.DataFrame(weights, index=pd.Series(0))
                success = True
            except:
                print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights')

            if success==False:
                # if Optimization fails
                weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                                        index=optimization_df.columns.tolist(),columns=pd.Series(0)).T

            temp_df = returns_dataframe[start_date:end_date]
            temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                    .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True),
                            left_index=True,right_index=True)\
                    .reset_index().set_index(['Date', 'Ticker'])#.unstack().stack()

            temp_df.index.names = ['Date', 'ticker']
            temp_df['weighted_return'] = temp_df['return']*temp_df['weight']
            temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')
            portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

        except Exception as e:
            print(e)

    portfolio_df = portfolio_df.drop_duplicates()
    portfolio_df
    print(portfolio_df)
    portfolio_df.plot()

    """## 8. Visualize Portfolio returns and compare to SP500 returns."""

    tickers='SPY'
    spy = yf.download(tickers=tickers,start='2020-01-01',end=dt.date.today())
    print(spy)
    spy_ret = np.log(spy[['Close']]).diff().dropna()
    print(spy_ret)
    df_close = spy_ret.loc[:, ('Close', tickers)]
    df_close = df_close.rename('Buy & Hold '+tickers)
    df_close.index.name = 'Date'
    df_close = df_close.to_frame()
    print(df_close.head())
    portfolio_df = portfolio_df.merge(df_close,left_index=True,right_index=True)
    print(portfolio_df)

    tickers='BTC-USD'

    spy1 = yf.download(tickers=tickers,start='2020-01-01',end=dt.date.today())
    print(spy1)
    spy_ret2 = np.log(spy1[['Close']]).diff().dropna()
    print(spy_ret2)
    df_close11 = spy_ret2.loc[:, ('Close', tickers)]
    df_close11 = df_close11.rename('Buy & Hold '+ tickers)
    df_close11.index.name = 'Date'
    df_close11 = df_close11.to_frame()
    spy_ret2 = df_close11
    print(df_close11.head())

    try:
        portfolio_df.index = pd.to_datetime(portfolio_df.index)
    except Exception as e:
        print("Error converting index to datetime:", e)
        portfolio_df.index = pd.to_datetime(portfolio_df.index, errors='coerce')
        portfolio_df = portfolio_df[portfolio_df.index.notna()]  # Drop invalid rows

    spy_ret2.index = pd.to_datetime(spy_ret2.index)
    print(spy_ret2)
    portfolio_df = portfolio_df.merge(spy_ret2, left_index=True, right_index=True)
    print(portfolio_df)
    
    import matplotlib.ticker as mtick
    end_date = '2024-11-29'
    plt.style.use('ggplot')
    portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1
    portfolio_cumulative_return[:end_date].plot(figsize=(16,6))
    plt.title('Unsupervised Learning Trading Strategy Returns Over Time')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel('Return')
    plt.show()