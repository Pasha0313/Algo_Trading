import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Loading_Data_BN import fetch_historical_data
import logging
import warnings
import os

plots_folder = "Plots"
os.makedirs(plots_folder, exist_ok=True)


from Future_analysis_plots import (
    plot_price_with_ema,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class ML_Strategies:
    def __init__(self, client, symbol, bar_length, start, end=None, tc=0.0):
        self.client = client
        self.symbol = str(symbol)
        self.start = str(start)
        self.end = str(end) if end else None
        self.tc = tc
        self.data = None
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h","6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.MLName = None
        self.ML = None
        self.already_bought = False
        try:
            self.data = self.get_data()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise
    # ------------------------------------------------------------
    # HELPER FOR PLOTTING (Binance OHLCV → analysis_plots schema)
    # ------------------------------------------------------------
    def _add_plot_features_from_binance_ohlcv(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Binance OHLCV schema -> plot schema expected by analysis_plots:
        requires: Close, High, Low
        outputs: close, ema20, ema50, ema20_slope, atr14,
                close_minus_ema20, log_close, ret1, hl_range
        """
        df = df_in.copy()

        for c in ["Close", "High", "Low"]:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' for plotting features.")

        df["close"] = df["Close"].astype(float)

        df["log_close"] = np.log(df["close"])
        df["ret1"] = df["log_close"].diff()

        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema20_slope"] = df["ema20"].diff()

        df["hl_range"] = (df["High"] - df["Low"]).astype(float)
        df["atr14"] = df["hl_range"].rolling(14).mean()

        df["close_minus_ema20"] = df["close"] - df["ema20"]
        return df
    
    def get_data(self):
        try:
            data = fetch_historical_data(client=self.client, symbol=self.symbol,
                bar_length=self.bar_length, start=self.start, end=self.end)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise

    def ML_Strategy(self, CFModel, parameters, Perform_Tuner=False):
        data = self.data.copy()
        data['change_tomorrow'] = data.Close.pct_change(-1) * 100 * -1
        data = data.dropna().copy()
        data['change_tomorrow_direction'] = np.where(data.change_tomorrow > 0, 1, 0)

        self.data = data                
        self.create_advanced_features() 

        self.data.Close.plot()

    def create_advanced_features(self):
        from Loading_Strategy import StrategyLoader
        import Strategy as STRATEGY  

        Path_Configs = "Configs"
        strategy_loader = StrategyLoader(os.path.join(Path_Configs, "strategies_config.json"))

        data = self.data.copy()

        strategy = "Stochastic_RSI"
        description, parameters, _ = strategy_loader.process_strategy(strategy,Print_Data=False)
        #print(f"\nStrategy: {strategy}, Description: {description}")
        data = STRATEGY.define_strategy_Stochastic_RSI(data, parameters)

        if 'position' in data.columns:
            data = data.drop(columns='position')
        
        #print("\n📋 Feature columns:", data.columns.tolist())
        
        strategy = "Bollinger_EMA"
        description, parameters, _ = strategy_loader.process_strategy(strategy,Print_Data=False)
        #print(f"\nStrategy: {strategy}, Description: {description}")

        data = STRATEGY.define_strategy_Bollinger_EMA(data, parameters)

        if 'position' in data.columns:
            data = data.drop(columns='position')            

        print("\n📋 Final feature columns:", data.columns.tolist())

        self.data = data  

    def run_model(self, model_type='rf'):
        y = self.data["change_tomorrow_direction"].copy()
        X = self.data.drop(columns=['change_tomorrow_direction'])
        self.feature_names = X.columns.tolist()

        # 🧠 Time series split (no shuffling)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        if model_type == 'rf':
            self.train_random_forest(X, X_train, X_val, y_train, y_val, Tuning=True)
        elif model_type == 'xgb':
            self.train_xgboost(X, X_train, X_val, y_train, y_val, Tuning=True)
        else:
            raise ValueError("Invalid model_type. Choose 'rf', 'xgb', or 'both'.")

    def train_random_forest(
        self,
        X, X_train, X_val,
        y_train, y_val,
        Tuning=False,
        best_params=None,          # <- pass your best params here
        refit_on_full=True         # <- after evaluation, refit on (train+val)
    ):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        import numpy as np

        # -----------------------------
        # 1) Get params
        # -----------------------------
        if Tuning:
            print("\n🔍 Starting Random Forest hyperparameter tuning...")
            rf = self.fine_tune_random_forest_mango(X_train, y_train)  # or fine_tune_random_forest(...)
            # If your tuner returns an estimator, just use it.
            # If it returns params, then create rf from params (see Option B below).
            tuned = True
        else:
            tuned = False
            if best_params is None:
                # fallback defaults (keep your current ones)
                best_params = dict(
                    bootstrap=True,
                    criterion="entropy",
                    max_depth=10,
                    max_features="sqrt",
                    min_samples_leaf=5,
                    min_samples_split=10,
                    n_estimators=100
                )

            rf = RandomForestClassifier(
                **best_params,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

        # -----------------------------
        # 2) Evaluate on validation
        # -----------------------------
        y_pred = rf.predict(X_val)
        print(classification_report(y_val, y_pred, digits=4))

        # -----------------------------
        # 3) Refit on full data (train+val) BEFORE producing X predictions
        #    This is typically what you want for backtesting forward.
        # -----------------------------
        if refit_on_full:
            X_full = np.vstack([X_train, X_val])
            y_full = np.hstack([y_train, y_val])
            rf.fit(X_full, y_full)

        # -----------------------------
        # 4) Store predictions + model
        # -----------------------------
        self.data["prediction"] = rf.predict(X)
        self.ML = rf

        # Optional: store tuned params if model exposes them
        if hasattr(rf, "get_params"):
            self.rf_params_ = rf.get_params()
        self.rf_tuned_ = tuned

        return rf


    def fine_tune_random_forest_mango(
        self,
        X_train,
        y_train,
        cv_splits=5,
        num_iteration=40,
        scoring="f1",          # "f1", "roc_auc", "accuracy", ...
        time_series_cv=True,   # IMPORTANT for trading data
        random_state=42
    ):
        import numpy as np
        from mango import Tuner
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, cross_val_score

        # -----------------------------
        # 1) Param space (Mango expects lists)
        # -----------------------------
        param_space = {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [None, 5, 10, 20, 30],
            "max_features": ["sqrt", "log2", None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "bootstrap": [True, False],
            "class_weight": [None, "balanced", "balanced_subsample"],
            # Optional knobs that often help stability
            "max_samples": [None, 0.5, 0.7, 0.9]  # only used when bootstrap=True
        }

        # -----------------------------
        # 2) Choose CV (time-series safe)
        # -----------------------------
        if time_series_cv:
            cv = TimeSeriesSplit(n_splits=cv_splits)
        else:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        # -----------------------------
        # 3) Objective function Mango calls with a LIST of param dicts
        #    Must return: (params_evaluated_list, objective_list)
        # -----------------------------
        def tuner_function(args_list):
            params_evaluated = []
            objectives = []

            for params in args_list:
                try:
                    # --- enforce valid combinations ---
                    # max_samples only valid when bootstrap=True
                    if not params["bootstrap"]:
                        params["max_samples"] = None

                    # Build model
                    model = RandomForestClassifier(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=params["max_depth"],
                        max_features=params["max_features"],
                        min_samples_split=int(params["min_samples_split"]),
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        bootstrap=bool(params["bootstrap"]),
                        class_weight=params["class_weight"],
                        max_samples=params["max_samples"],
                        n_jobs=-1,
                        random_state=random_state
                    )

                    # Cross-validated score (Mango minimizes, so we use negative score)
                    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                    mean_score = float(np.mean(scores))

                    params_evaluated.append(params)
                    objectives.append(-mean_score)  # minimize negative score

                except Exception as e:
                    # Bad configs get punished
                    params_evaluated.append(params)
                    objectives.append(1e6)

            return params_evaluated, objectives

        # -----------------------------
        # 4) Run Mango
        # -----------------------------
        conf = {"num_iteration": int(num_iteration)}
        tuner = Tuner(param_space, tuner_function, conf)
        results = tuner.minimize()

        best_params = results["best_params"]
        best_objective = results["best_objective"]  # negative score

        print("🔧 Best RF Params (Mango):", best_params)
        print(f"🏁 Best CV {scoring}: {-best_objective:.6f}")

        # -----------------------------
        # 5) Fit final model on full train
        # -----------------------------
        if not best_params["bootstrap"]:
            best_params["max_samples"] = None

        best_model = RandomForestClassifier(
            n_estimators=int(best_params["n_estimators"]),
            max_depth=best_params["max_depth"],
            max_features=best_params["max_features"],
            min_samples_split=int(best_params["min_samples_split"]),
            min_samples_leaf=int(best_params["min_samples_leaf"]),
            bootstrap=bool(best_params["bootstrap"]),
            class_weight=best_params["class_weight"],
            max_samples=best_params["max_samples"],
            n_jobs=-1,
            random_state=random_state
        )
        best_model.fit(X_train, y_train)
        return best_model

    def train_xgboost(self,X, X_train, X_val, y_train, y_val, Tuning = False):
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        
        if Tuning :
            xgb_model = self.fine_tune_xgboost(X_train, y_train)
        else :
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=2024
            )
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = xgb_model.predict(X_val)
        print(classification_report(y_val, y_pred, digits=4))        
        self.data["prediction"] = xgb_model.predict(X)
        self.ML = xgb_model

    def fine_tune_xgboost(self, X_train, y_train, cv=3, n_iter=20):
        import xgboost as xgb
        from sklearn.model_selection import RandomizedSearchCV

        param_distributions = {
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5, 10],
            'gamma': [0, 0.2, 0.5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [1, 3, 5, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200]
        }

        search = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        search.fit(X_train, y_train)
        print("🔧 Best XGB Params:", search.best_params_)
        return search.best_estimator_

    def run_backtesting_strategy(self):
        from backtesting import Backtest
        from Back_Testing_MLClass_BN import SimpleClassificationUD
        if self.data is None or self.ML is None or not hasattr(self, "feature_names"):
            raise ValueError("Ensure data, model, and feature list are initialized.")

        X = self.data.copy()
        model = self.ML
        features = self.feature_names

        # ✅ Dynamically inject model and features into a subclass
        class StrategyWrapper(SimpleClassificationUD):
            def init(inner_self):
                super().init()
                inner_self.model = model
                inner_self.features = features

        bt = Backtest(
            X,
            StrategyWrapper,
            cash=1000,
            commission=self.tc,
            exclusive_orders=True
        )

        results = bt.run()
        print("\n=== Backtest Summary ===")
        print(results)
        return results

    def plot_performance(self, leverage=1.0, save=False, filename=None):
        if "prediction" not in self.data.columns:
            raise ValueError("Prediction column not found. Run DecisionTreeML() first.")
        
        if "change_tomorrow" not in self.data.columns:
            raise ValueError("change_tomorrow column missing. Run ML_Strategy() first.")

        df = self.data.copy()

        # Shift prediction to avoid look-ahead bias
        df["position"] = df["prediction"].shift()

        # Calculate daily returns
        df["strategy_return"] = df["position"] * df["change_tomorrow"] / 100
        df["buy_and_hold"] = df["change_tomorrow"] / 100

        # Apply leverage
        df["strategy_return_leveraged"] = df["strategy_return"] * leverage

        # Cumulative performance
        df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()
        df["cumulative_leverage"] = (1 + df["strategy_return_leveraged"]).cumprod()
        df["cumulative_bh"] = (1 + df["buy_and_hold"]).cumprod()

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["cumulative_bh"], label="Buy & Hold", linestyle="--")
        plt.plot(df.index, df["cumulative_strategy"], label="ML Strategy", linewidth=2)
        if leverage != 1.0:
            plt.plot(df.index, df["cumulative_leverage"], label=f"ML Strategy ×{leverage:.1f}", linewidth=2, alpha=0.8)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"📈 Strategy Performance | {self.symbol} | Leverage = {leverage:.1f}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save:
            fname = filename or f"ML_Strategy_Performance_{self.symbol}.png"
            plt.savefig(os.path.join("Plots", fname))
        plt.show()

    # ============================================================
    # ============================================================
    # ============================================================
    # ============================================================
    # ============================================================
    # ============================================================


    # ============================================================
    # TF GPU helper (shared)
    # ============================================================
    def _tf_setup_gpu(self, tf_memory_growth: bool = True, tf_mixed_precision: bool = False):
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                if tf_memory_growth:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                if tf_mixed_precision:
                    from tensorflow.keras import mixed_precision as mp
                    mp.set_global_policy("mixed_float16")
                print(f"[TF] GPUs detected: {gpus}")
            except Exception as e:
                print(f"[TF] GPU setup warning: {e}")
        else:
            print("[TF] No GPU detected. Running on CPU.")


    # ============================================================
    # Conv1D | Optuna + WFV | Direction objective
    # ============================================================
    def run_future_prediction_conv1d_binance(
        self,
        future_steps: int = 48,
        feature_cols: list = None,
        output_dir: str = "Forecasts",
        use_date_split: bool = False,
        train_end: str = None,
        test_end: str = None,

        # Optuna / WFV
        n_trials: int = 30,
        skip_tuning_if_best_exists: bool = False,
        window_size: int = 3000,
        step_size: int = 750,
        wfv_test_slice: int = 50,
        wfv_val_split: float = 0.2,
        wfv_min_samples: int = 300,
        wfv_max_epochs: int = 8,
        wfv_patience: int = 2,

        # Final training
        final_epochs: int = 20,
        global_seed: int = 42,

        # Objective alignment
        eval_h: int = 24,
        conf_percentile: int = 75,

        # TF / GPU
        tf_memory_growth: bool = True,
        tf_mixed_precision: bool = False,
        enable_eager_debug: bool = False,
    ):
        """
        Conv1D future prediction for Binance data.
        Optuna objective: MAXIMIZE median direction accuracy at horizon eval_h using walk-forward validation.
        Exports: forecast CSV + history+forecast plot + EMA actual-vs-pred plot.
        """

        import os, json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        import optuna
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.callbacks import EarlyStopping

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        from Future_analysis_plots import plot_price_with_ema

        self._tf_setup_gpu(tf_memory_growth=tf_memory_growth, tf_mixed_precision=tf_mixed_precision)

        if enable_eager_debug:
            tf.config.run_functions_eagerly(True)
            tf.data.experimental.enable_debug_mode()

        os.makedirs(output_dir, exist_ok=True)

        # ----------------------------
        # Binance bar_length -> pandas freq
        # ----------------------------
        def _bar_length_to_pandas_freq(bar_length: str) -> str:
            bl = str(bar_length).strip()
            if bl.endswith("m") and bl[:-1].isdigit():
                return f"{int(bl[:-1])}min"
            if bl.endswith("h") and bl[:-1].isdigit():
                return f"{int(bl[:-1])}H"
            if bl.endswith("d") and bl[:-1].isdigit():
                return f"{int(bl[:-1])}D"
            if bl.endswith("w") and bl[:-1].isdigit():
                return f"{int(bl[:-1])}W"
            if bl == "1M":
                return "MS"
            raise ValueError(f"Unsupported bar_length='{bar_length}' for pandas date_range.")

        pandas_freq = _bar_length_to_pandas_freq(self.bar_length)

        # ----------------------------
        # Validate / clean data
        # ----------------------------
        if self.data is None or len(self.data) < 500:
            raise ValueError("self.data is empty/too small. Ensure Binance API data was loaded first.")

        df = self.data.copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("self.data must have a DatetimeIndex.")
        if "Close" not in df.columns:
            raise ValueError("Expected 'Close' column in self.data (Binance OHLCV).")

        df = df.sort_index()
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="last")]

        # ----------------------------
        # Feature columns
        # ----------------------------
        if feature_cols is None:
            drop_cols = {"Open", "High", "Low", "Close", "Volume", "ml_target", "prediction"}
            drop_cols |= {"change_tomorrow", "change_tomorrow_direction"}
            feature_cols = [c for c in df.columns if c not in drop_cols]

        if not feature_cols:
            raise ValueError("feature_cols is empty. Run ML_Strategy() first or pass feature_cols explicitly.")

        # ----------------------------
        # Split
        # ----------------------------
        if use_date_split:
            if train_end is None or test_end is None:
                raise ValueError("use_date_split=True requires train_end and test_end.")
            train_end_ts = pd.to_datetime(train_end)
            test_end_ts = pd.to_datetime(test_end)
            train_df = df.loc[:train_end_ts].copy()
            test_df = df.loc[train_end_ts:test_end_ts].copy()
        else:
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

        if len(train_df) < 1000:
            raise ValueError("Not enough training bars after cleaning/splitting.")

        # ----------------------------
        # Scale features (train-only)
        # ----------------------------
        feat_scaler = MinMaxScaler()
        train_feat_scaled = feat_scaler.fit_transform(train_df[feature_cols].values)
        test_feat_scaled = feat_scaler.transform(test_df[feature_cols].values)

        train_df_scaled = pd.DataFrame(train_feat_scaled, index=train_df.index, columns=feature_cols)
        test_df_scaled = pd.DataFrame(test_feat_scaled, index=test_df.index, columns=feature_cols)
        train_df_scaled["close"] = train_df["Close"].astype(float).values
        test_df_scaled["close"] = test_df["Close"].astype(float).values

        # ----------------------------
        # Sequences
        # ----------------------------
        def create_supervised_sequences(local_df: pd.DataFrame, cols: list, time_step: int, f_steps: int):
            feat_vals = local_df[cols].to_numpy(dtype=np.float32, copy=False)
            close_vals = local_df["close"].to_numpy(dtype=np.float64, copy=False)
            ts_index = local_df.index

            n = len(local_df)
            last_i = n - f_steps - 1
            if last_i <= time_step:
                return (
                    np.empty((0, time_step, len(cols)), dtype=np.float32),
                    np.empty((0, f_steps), dtype=np.float32),
                    pd.Index([]),
                )

            X, Y, idx = [], [], []
            for i in range(time_step, last_i + 1):
                X.append(feat_vals[i - time_step:i])
                c0 = close_vals[i]
                future = close_vals[i + 1:i + 1 + f_steps]
                y_path = np.log(future / c0).astype(np.float32)
                Y.append(y_path)
                idx.append(ts_index[i])

            return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32), pd.Index(idx)

        # ----------------------------
        # Conv1D builder
        # ----------------------------
        def build_conv1d(time_step: int, n_features: int, f_steps: int, filters: int, kernel: int,
                        dense_units: int, dropout: float, lr_: float):
            inputs = Input(shape=(time_step, n_features))

            x = Conv1D(filters, kernel, padding="causal", activation="relu")(inputs)
            x = Conv1D(filters, kernel, padding="causal", activation="relu")(x)
            x = MaxPooling1D(2)(x)

            x = Conv1D(filters * 2, kernel, padding="causal", activation="relu")(x)
            x = MaxPooling1D(2)(x)

            x = GlobalAveragePooling1D()(x)
            x = Dense(dense_units, activation="relu")(x)
            x = Dropout(dropout)(x)
            outputs = Dense(f_steps)(x)

            model = Model(inputs, outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_), loss="mse")
            return model

        # ----------------------------
        # Cache (include eval_h because objective depends on it)
        # ----------------------------
        def best_params_path() -> str:
            fname = f"{self.symbol}_conv1d_best_params_{self.bar_length}_fs{int(future_steps)}_H{int(eval_h)}.json"
            return os.path.join(output_dir, fname)

        def load_best_params():
            p = best_params_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None

        def save_best_params(payload: dict):
            p = best_params_path()
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return p

        # ----------------------------
        # Optuna objective (WFV) - maximize median direction accuracy at eval_h
        # ----------------------------
        def make_objective(train_scaled_df: pd.DataFrame):
            def objective(trial):
                tf.keras.utils.set_random_seed(global_seed + int(trial.number))
                np.random.seed(global_seed + int(trial.number))

                time_step   = trial.suggest_int("time_step", 40, 90)
                filters     = trial.suggest_categorical("filters", [32, 64, 96])
                kernel      = trial.suggest_categorical("kernel", [3, 5])
                dense_units = trial.suggest_categorical("dense_units", [32, 64, 96, 128])
                dropout     = trial.suggest_float("dropout", 0.05, 0.30)
                lr_         = trial.suggest_float("lr", 3e-4, 2e-3, log=True)
                batch_size  = trial.suggest_categorical("batch_size", [64, 128, 256])

                X_all, Y_all, _ = create_supervised_sequences(
                    train_scaled_df, feature_cols, time_step=int(time_step), f_steps=int(future_steps)
                )
                n_seq = int(len(X_all))
                if n_seq < 500:
                    return float("-inf")

                H = int(eval_h)
                if H < 1 or H > int(future_steps):
                    return float("-inf")

                # clamp window/step
                max_window = n_seq - max(1, int(wfv_test_slice)) - 1
                if max_window < max(100, int(wfv_min_samples)):
                    return float("-inf")

                effective_window = int(min(max_window, int(window_size)))
                effective_step = int(min(max(1, int(step_size)), effective_window))

                max_idx = n_seq - effective_window
                if max_idx <= 0:
                    return float("-inf")

                dir_accs = []

                for start_idx in range(0, max_idx, effective_step):
                    end_idx = start_idx + effective_window
                    if end_idx + int(wfv_test_slice) > n_seq:
                        break

                    X_train_w = X_all[start_idx:end_idx]
                    y_train_w = Y_all[start_idx:end_idx]
                    X_test_w  = X_all[end_idx:end_idx + int(wfv_test_slice)]
                    y_test_w  = Y_all[end_idx:end_idx + int(wfv_test_slice)]
                    if len(X_test_w) == 0:
                        break

                    tf.keras.backend.clear_session()
                    model = build_conv1d(
                        time_step=int(time_step),
                        n_features=int(len(feature_cols)),
                        f_steps=int(future_steps),
                        filters=int(filters),
                        kernel=int(kernel),
                        dense_units=int(dense_units),
                        dropout=float(dropout),
                        lr_=float(lr_),
                    )

                    n = int(len(X_train_w))
                    if n >= int(wfv_min_samples):
                        split = int(n * (1.0 - float(wfv_val_split)))
                        split = max(1, min(split, n - 1))
                        X_tr, y_tr = X_train_w[:split], y_train_w[:split]
                        X_val, y_val = X_train_w[split:], y_train_w[split:]

                        es = EarlyStopping(monitor="val_loss", patience=int(wfv_patience), restore_best_weights=True)
                        model.fit(
                            X_tr, y_tr,
                            validation_data=(X_val, y_val),
                            epochs=int(wfv_max_epochs),
                            batch_size=int(batch_size),
                            verbose=0,
                            callbacks=[es],
                        )
                    else:
                        es = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
                        model.fit(
                            X_train_w, y_train_w,
                            epochs=min(8, int(wfv_max_epochs)),
                            batch_size=int(batch_size),
                            verbose=0,
                            callbacks=[es],
                        )

                    pred = model.predict(X_test_w, batch_size=int(batch_size), verbose=0)
                    y_true_H = y_test_w[:, H - 1]
                    y_pred_H = pred[:, H - 1]
                    dir_accs.append(float(np.mean(np.sign(y_true_H) == np.sign(y_pred_H))))

                if not dir_accs:
                    return float("-inf")

                return float(np.median(dir_accs))

            return objective

        # ----------------------------
        # Tune or load cache
        # ----------------------------
        cached = load_best_params() if skip_tuning_if_best_exists else None
        if cached is not None:
            best_params = cached["best_params"]
            best_score = float(cached.get("best_wfv_dir_acc_median", np.nan))
            print(f"\n[{self.symbol}] Loaded cached Conv1D params. best_dir_acc={best_score:.4f}")
        else:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
            study = optuna.create_study(direction="maximize", pruner=pruner)
            study.optimize(make_objective(train_df_scaled), n_trials=int(n_trials))

            best_params = study.best_params
            best_score = float(study.best_value)

            payload = {
                "symbol": self.symbol,
                "bar_length": self.bar_length,
                "future_steps": int(future_steps),
                "eval_h": int(eval_h),
                "best_wfv_dir_acc_median": best_score,
                "best_params": best_params,
            }
            if np.isfinite(best_score) and (0.0 < best_score <= 1.0):
                save_best_params(payload)

            print(f"\n[{self.symbol}] Conv1D Best WFV DirAcc (median, H={eval_h}) = {best_score:.4f}")
            print(f"[{self.symbol}] Conv1D Best params:", best_params)

        # unpack best
        ts = int(best_params["time_step"])
        bs = int(best_params["batch_size"])
        f_ = int(best_params["filters"])
        k_ = int(best_params["kernel"])
        d_ = int(best_params["dense_units"])
        dr_ = float(best_params["dropout"])
        lr_ = float(best_params["lr"])

        # ----------------------------
        # Final train
        # ----------------------------
        tf.keras.utils.set_random_seed(int(global_seed))
        np.random.seed(int(global_seed))
        tf.keras.backend.clear_session()

        X_train_all, Y_train_all, _ = create_supervised_sequences(train_df_scaled, feature_cols, time_step=ts, f_steps=int(future_steps))
        if len(X_train_all) == 0:
            raise ValueError("Not enough training rows to form sequences (reduce time_step/future_steps).")

        model = build_conv1d(ts, int(len(feature_cols)), int(future_steps), f_, k_, d_, dr_, lr_)

        n = int(len(X_train_all))
        split = int(n * 0.8)
        split = max(1, min(split, n - 1))
        X_tr, y_tr = X_train_all[:split], Y_train_all[:split]
        X_val, y_val = X_train_all[split:], Y_train_all[split:]

        es = EarlyStopping(monitor="val_loss", patience=int(wfv_patience), restore_best_weights=True)
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=int(final_epochs),
            batch_size=int(bs),
            verbose=1,
            callbacks=[es],
        )

        # ----------------------------
        # Eval on test
        # ----------------------------
        X_test_all, Y_test_all, _ = create_supervised_sequences(test_df_scaled, feature_cols, time_step=ts, f_steps=int(future_steps))
        if len(X_test_all) > 0:
            Y_test_pred = model.predict(X_test_all, batch_size=int(bs), verbose=0)

            test_rmse = float(np.sqrt(mean_squared_error(Y_test_all.reshape(-1), Y_test_pred.reshape(-1))))
            test_mae  = float(mean_absolute_error(Y_test_all.reshape(-1), Y_test_pred.reshape(-1)))

            H = int(eval_h)
            y_true_H = Y_test_all[:, H - 1]
            y_pred_H = Y_test_pred[:, H - 1]
            dir_acc = float(np.mean(np.sign(y_true_H) == np.sign(y_pred_H)))

            threshold = float(np.percentile(np.abs(y_pred_H), int(conf_percentile)))
            coverage = float(np.mean(np.abs(y_pred_H) >= threshold))

            print(f"\n[{self.symbol}] Conv1D | TEST RMSE(all)={test_rmse:.6f} | MAE(all)={test_mae:.6f}")
            print(f"[{self.symbol}] Conv1D | Direction Acc (H={H})={dir_acc:.4f} | conf p{conf_percentile} coverage={coverage:.2%}")
        else:
            print(f"\n[{self.symbol}] Conv1D | No test sequences produced (test too small).")

        # ----------------------------
        # Future forecast
        # ----------------------------
        anchor_ts = train_df.index[-1] if use_date_split else df.index[-1]

        all_scaled = pd.concat([train_df_scaled, test_df_scaled], axis=0).sort_index()
        all_scaled = all_scaled.loc[:anchor_ts].copy()

        X_anchor, _, _ = create_supervised_sequences(all_scaled, feature_cols, time_step=ts, f_steps=int(future_steps))
        if len(X_anchor) == 0:
            raise ValueError("Could not form anchor sequence (reduce time_step or add more history).")

        last_input = X_anchor[-1].reshape(1, ts, len(feature_cols))
        pred_future_ret = model.predict(last_input, batch_size=1, verbose=0)[0]

        anchor_price = float(df.loc[anchor_ts, "Close"])
        forecast_prices = anchor_price * np.exp(pred_future_ret)

        future_ts = pd.date_range(anchor_ts, periods=int(future_steps) + 1, freq=pandas_freq)[1:]

        forecast_df = pd.DataFrame({
            "timestamp": future_ts,
            "forecast_forward_logret": pred_future_ret,
            "forecast_price": forecast_prices
        })

        out_csv = os.path.join(output_dir, f"{self.symbol}_conv1d_forecast_{self.bar_length}_fs{int(future_steps)}.csv")
        forecast_df.to_csv(out_csv, index=False)
        print(f"\n[{self.symbol}] Saved Conv1D future forecast CSV: {out_csv}")

        # history + forecast plot
        lookback_bars = min(300, len(df))
        hist = df.iloc[-lookback_bars:].copy()

        plt.figure(figsize=(12, 5))
        plt.plot(hist.index, hist["Close"].astype(float), label="Actual Close (history)")
        plt.plot(forecast_df["timestamp"], forecast_df["forecast_price"].astype(float), label="Forecast Close (future)")
        plt.axvline(anchor_ts, linestyle="--", linewidth=1.0)
        plt.title(f"Conv1D | Actual+Forecast | {self.symbol} | {self.bar_length} | steps={future_steps}")
        plt.xlabel("Time")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(output_dir, f"{self.symbol}_conv1d_actual_plus_forecast_{self.bar_length}_fs{int(future_steps)}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[{self.symbol}] Saved combined plot: {out_png}")

        # EMA actual vs pred (1-step reconstruction)
        df_all_scaled = pd.concat([train_df_scaled, test_df_scaled], axis=0).sort_index()
        X_all, _, idx_all = create_supervised_sequences(df_all_scaled, feature_cols, time_step=ts, f_steps=int(future_steps))
        if len(X_all) == 0:
            raise ValueError("Could not form X_all for next-close reconstruction.")

        Y_all_pred = model.predict(X_all, batch_size=int(bs), verbose=0)
        y1_pred = Y_all_pred[:, 0]

        df_base = df.copy()
        close_pred_series = df_base["Close"].astype(float).copy()

        for t, ylog in zip(idx_all, y1_pred):
            loc = df_base.index.get_indexer([t])[0]
            if loc != -1 and (loc + 1) < len(df_base.index):
                t1 = df_base.index[loc + 1]
                close_pred_series.loc[t1] = float(df_base.loc[t, "Close"]) * float(np.exp(float(ylog)))

        df_act = self._add_plot_features_from_binance_ohlcv(df_base)

        df_pred_base = df_base.copy()
        df_pred_base["Close"] = close_pred_series
        df_pred = self._add_plot_features_from_binance_ohlcv(df_pred_base)

        plot_price_with_ema(
            df_actual=df_act,
            df_pred=df_pred,
            pair_name=self.symbol,
            timeframe=self.bar_length,
            output_dir=output_dir,
            suffix="_conv1d_actual_vs_pred",
            last_days=5,
        )

        return forecast_df


    # ============================================================
    # Transformer | Optuna + WFV | Direction objective
    # ============================================================
    def run_future_prediction_transformer_binance_optuna(
        self,
        future_steps: int = 48,
        feature_cols: list = None,
        output_dir: str = "Forecasts",
        use_date_split: bool = False,
        train_end: str = None,
        test_end: str = None,

        # Optuna / WFV
        n_trials: int = 30,
        skip_tuning_if_best_exists: bool = False,
        window_size: int = 3000,
        step_size: int = 750,
        wfv_test_slice: int = 50,
        wfv_val_split: float = 0.2,
        wfv_min_samples: int = 300,
        wfv_max_epochs: int = 8,
        wfv_patience: int = 2,

        # Final training
        final_epochs: int = 20,
        global_seed: int = 42,

        # Objective alignment
        eval_h: int = 24,
        conf_percentile: int = 75,

        # TF / GPU
        forbid_leakage_cols: bool = True,
        tf_memory_growth: bool = True,
        tf_mixed_precision: bool = False,
        enable_eager_debug: bool = False,
    ):
        """
        Transformer future prediction for Binance data using Optuna WFV objective:
        Objective: MAXIMIZE median direction accuracy at horizon eval_h.
        """

        import os, json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        import optuna
        import tensorflow as tf
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, LayerNormalization,
            MultiHeadAttention, GlobalAveragePooling1D
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.callbacks import EarlyStopping

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        from Future_analysis_plots import plot_price_with_ema

        self._tf_setup_gpu(tf_memory_growth=tf_memory_growth, tf_mixed_precision=tf_mixed_precision)

        if enable_eager_debug:
            tf.config.run_functions_eagerly(True)
            tf.data.experimental.enable_debug_mode()

        os.makedirs(output_dir, exist_ok=True)

        def _bar_length_to_pandas_freq(bar_length: str) -> str:
            bl = str(bar_length).strip()
            if bl.endswith("m") and bl[:-1].isdigit():
                return f"{int(bl[:-1])}min"
            if bl.endswith("h") and bl[:-1].isdigit():
                return f"{int(bl[:-1])}H"
            if bl.endswith("d") and bl[:-1].isdigit():
                return f"{int(bl[:-1])}D"
            if bl.endswith("w") and bl[:-1].isdigit():
                return f"{int(bl[:-1])}W"
            if bl == "1M":
                return "MS"
            raise ValueError(f"Unsupported bar_length='{bar_length}' for pandas date_range.")

        pandas_freq = _bar_length_to_pandas_freq(self.bar_length)

        # validate
        if self.data is None or len(self.data) < 500:
            raise ValueError("self.data is empty/too small. Ensure Binance API data was loaded first.")

        df = self.data.copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("self.data must have a DatetimeIndex.")
        if "Close" not in df.columns:
            raise ValueError("Expected 'Close' column in self.data (Binance OHLCV).")

        df = df.sort_index()
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="last")]

        # features
        if feature_cols is None:
            drop_cols = {"Open", "High", "Low", "Close", "Volume", "ml_target", "prediction"}
            if forbid_leakage_cols:
                drop_cols |= {"change_tomorrow", "change_tomorrow_direction"}
            feature_cols = [c for c in df.columns if c not in drop_cols]

        if not feature_cols:
            raise ValueError("feature_cols is empty. Run ML_Strategy() first or pass feature_cols explicitly.")

        # split
        if use_date_split:
            if train_end is None or test_end is None:
                raise ValueError("use_date_split=True requires train_end and test_end.")
            train_end_ts = pd.to_datetime(train_end)
            test_end_ts = pd.to_datetime(test_end)
            train_df = df.loc[:train_end_ts].copy()
            test_df = df.loc[train_end_ts:test_end_ts].copy()
        else:
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

        if len(train_df) < 1000:
            raise ValueError("Not enough training bars after cleaning/splitting.")

        # scale train-only
        feat_scaler = MinMaxScaler()
        train_feat_scaled = feat_scaler.fit_transform(train_df[feature_cols].values)
        test_feat_scaled = feat_scaler.transform(test_df[feature_cols].values)

        train_df_scaled = pd.DataFrame(train_feat_scaled, index=train_df.index, columns=feature_cols)
        test_df_scaled = pd.DataFrame(test_feat_scaled, index=test_df.index, columns=feature_cols)
        train_df_scaled["close"] = train_df["Close"].astype(float).values
        test_df_scaled["close"] = test_df["Close"].astype(float).values

        # sequences
        def create_supervised_sequences(local_df: pd.DataFrame, cols: list, time_step: int, f_steps: int):
            feat_vals = local_df[cols].to_numpy(dtype=np.float32, copy=False)
            close_vals = local_df["close"].to_numpy(dtype=np.float64, copy=False)
            ts_index = local_df.index

            n = len(local_df)
            last_i = n - f_steps - 1
            if last_i <= time_step:
                return (
                    np.empty((0, time_step, len(cols)), dtype=np.float32),
                    np.empty((0, f_steps), dtype=np.float32),
                    pd.Index([]),
                )

            X, Y, idx = [], [], []
            for i in range(time_step, last_i + 1):
                X.append(feat_vals[i - time_step:i])
                c0 = close_vals[i]
                future = close_vals[i + 1:i + 1 + f_steps]
                y_path = np.log(future / c0).astype(np.float32)
                Y.append(y_path)
                idx.append(ts_index[i])

            return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32), pd.Index(idx)

        # transformer builder
        def build_transformer(time_step: int, n_features: int, f_steps: int,
                            head_size: int, num_heads: int, ff_dim: int,
                            dense_units: int, dropout: float, lr_: float):

            def transformer_encoder(inputs):
                x = LayerNormalization(epsilon=1e-6)(inputs)
                x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
                x = Dropout(dropout)(x)
                res = x + inputs

                x = LayerNormalization(epsilon=1e-6)(res)
                x = Dense(ff_dim, activation="relu")(x)
                x = Dropout(dropout)(x)
                x = Dense(inputs.shape[-1])(x)
                return x + res

            inputs = Input(shape=(time_step, n_features))
            x = transformer_encoder(inputs)
            x = GlobalAveragePooling1D()(x)
            x = Dropout(dropout)(x)
            x = Dense(dense_units, activation="relu")(x)
            x = Dropout(dropout)(x)
            outputs = Dense(f_steps)(x)

            model = Model(inputs, outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_), loss="mse")
            return model

        # cache
        def best_params_path() -> str:
            fname = f"{self.symbol}_transformer_best_params_{self.bar_length}_fs{int(future_steps)}_H{int(eval_h)}.json"
            return os.path.join(output_dir, fname)

        def load_best_params():
            p = best_params_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None

        def save_best_params(payload: dict):
            p = best_params_path()
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return p

        # objective (maximize dir-acc at eval_h)
        def make_objective(train_scaled_df: pd.DataFrame):
            def objective(trial):
                tf.keras.utils.set_random_seed(global_seed + int(trial.number))
                np.random.seed(global_seed + int(trial.number))

                time_step   = trial.suggest_int("time_step", 40, 100, step=5)
                head_size   = trial.suggest_categorical("head_size", [32, 64, 128])
                num_heads   = trial.suggest_categorical("num_heads", [2, 4, 8])
                ff_dim      = trial.suggest_categorical("ff_dim", [64, 128, 256])
                dense_units = trial.suggest_categorical("dense_units", [32, 64, 96, 128])
                dropout     = trial.suggest_float("dropout", 0.05, 0.30)
                lr_         = trial.suggest_float("lr", 3e-4, 2e-3, log=True)
                batch_size  = trial.suggest_categorical("batch_size", [64, 128, 256])

                X_all, Y_all, _ = create_supervised_sequences(
                    train_scaled_df, feature_cols, time_step=int(time_step), f_steps=int(future_steps)
                )
                n_seq = int(len(X_all))
                if n_seq < 500:
                    return float("-inf")

                H = int(eval_h)
                if H < 1 or H > int(future_steps):
                    return float("-inf")

                max_window = n_seq - max(1, int(wfv_test_slice)) - 1
                if max_window < max(100, int(wfv_min_samples)):
                    return float("-inf")

                effective_window = int(min(max_window, int(window_size)))
                effective_step = int(min(max(1, int(step_size)), effective_window))
                max_idx = n_seq - effective_window
                if max_idx <= 0:
                    return float("-inf")

                dir_accs = []

                for start_idx in range(0, max_idx, effective_step):
                    end_idx = start_idx + effective_window
                    if end_idx + int(wfv_test_slice) > n_seq:
                        break

                    X_train_w = X_all[start_idx:end_idx]
                    y_train_w = Y_all[start_idx:end_idx]
                    X_test_w  = X_all[end_idx:end_idx + int(wfv_test_slice)]
                    y_test_w  = Y_all[end_idx:end_idx + int(wfv_test_slice)]
                    if len(X_test_w) == 0:
                        break

                    tf.keras.backend.clear_session()
                    model = build_transformer(
                        time_step=int(time_step),
                        n_features=int(len(feature_cols)),
                        f_steps=int(future_steps),
                        head_size=int(head_size),
                        num_heads=int(num_heads),
                        ff_dim=int(ff_dim),
                        dense_units=int(dense_units),
                        dropout=float(dropout),
                        lr_=float(lr_),
                    )

                    n = int(len(X_train_w))
                    if n >= int(wfv_min_samples):
                        split = int(n * (1.0 - float(wfv_val_split)))
                        split = max(1, min(split, n - 1))
                        X_tr, y_tr = X_train_w[:split], y_train_w[:split]
                        X_val, y_val = X_train_w[split:], y_train_w[split:]

                        es = EarlyStopping(monitor="val_loss", patience=int(wfv_patience), restore_best_weights=True)
                        model.fit(
                            X_tr, y_tr,
                            validation_data=(X_val, y_val),
                            epochs=int(wfv_max_epochs),
                            batch_size=int(batch_size),
                            verbose=0,
                            callbacks=[es],
                        )
                    else:
                        es = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
                        model.fit(
                            X_train_w, y_train_w,
                            epochs=min(8, int(wfv_max_epochs)),
                            batch_size=int(batch_size),
                            verbose=0,
                            callbacks=[es],
                        )

                    pred = model.predict(X_test_w, batch_size=int(batch_size), verbose=0)
                    y_true_H = y_test_w[:, H - 1]
                    y_pred_H = pred[:, H - 1]
                    dir_accs.append(float(np.mean(np.sign(y_true_H) == np.sign(y_pred_H))))

                if not dir_accs:
                    return float("-inf")

                return float(np.median(dir_accs))

            return objective

        # tune/cache
        cached = load_best_params() if skip_tuning_if_best_exists else None
        if cached is not None:
            best_params = cached["best_params"]
            best_score = float(cached.get("best_wfv_dir_acc_median", np.nan))
            print(f"\n[{self.symbol}] Loaded cached Transformer params. best_dir_acc={best_score:.4f}")
        else:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
            study = optuna.create_study(direction="maximize", pruner=pruner)
            study.optimize(make_objective(train_df_scaled), n_trials=int(n_trials))

            best_params = study.best_params
            best_score = float(study.best_value)

            payload = {
                "symbol": self.symbol,
                "bar_length": self.bar_length,
                "future_steps": int(future_steps),
                "eval_h": int(eval_h),
                "best_wfv_dir_acc_median": best_score,
                "best_params": best_params,
            }
            if np.isfinite(best_score) and (0.0 < best_score <= 1.0):
                save_best_params(payload)

            print(f"\n[{self.symbol}] Transformer Best WFV DirAcc (median, H={eval_h}) = {best_score:.4f}")
            print(f"[{self.symbol}] Transformer Best params:", best_params)

        # unpack
        ts = int(best_params["time_step"])
        bs = int(best_params["batch_size"])
        head_size = int(best_params["head_size"])
        num_heads = int(best_params["num_heads"])
        ff_dim = int(best_params["ff_dim"])
        dense_units = int(best_params["dense_units"])
        dropout = float(best_params["dropout"])
        lr_ = float(best_params["lr"])

        # final train
        tf.keras.utils.set_random_seed(int(global_seed))
        np.random.seed(int(global_seed))
        tf.keras.backend.clear_session()

        X_train_all, Y_train_all, _ = create_supervised_sequences(train_df_scaled, feature_cols, time_step=ts, f_steps=int(future_steps))
        if len(X_train_all) == 0:
            raise ValueError("Not enough training rows to form sequences (reduce time_step/future_steps).")

        model = build_transformer(ts, int(len(feature_cols)), int(future_steps),
                                head_size, num_heads, ff_dim, dense_units, dropout, lr_)

        n = int(len(X_train_all))
        split = int(n * 0.8)
        split = max(1, min(split, n - 1))
        X_tr, y_tr = X_train_all[:split], Y_train_all[:split]
        X_val, y_val = X_train_all[split:], Y_train_all[split:]

        es = EarlyStopping(monitor="val_loss", patience=int(wfv_patience), restore_best_weights=True)
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=int(final_epochs),
            batch_size=int(bs),
            verbose=1,
            callbacks=[es],
        )

        # eval test
        X_test_all, Y_test_all, _ = create_supervised_sequences(test_df_scaled, feature_cols, time_step=ts, f_steps=int(future_steps))
        if len(X_test_all) > 0:
            Y_test_pred = model.predict(X_test_all, batch_size=int(bs), verbose=0)

            test_rmse = float(np.sqrt(mean_squared_error(Y_test_all.reshape(-1), Y_test_pred.reshape(-1))))
            test_mae  = float(mean_absolute_error(Y_test_all.reshape(-1), Y_test_pred.reshape(-1)))

            H = int(eval_h)
            y_true_H = Y_test_all[:, H - 1]
            y_pred_H = Y_test_pred[:, H - 1]
            dir_acc = float(np.mean(np.sign(y_true_H) == np.sign(y_pred_H)))

            threshold = float(np.percentile(np.abs(y_pred_H), int(conf_percentile)))
            coverage = float(np.mean(np.abs(y_pred_H) >= threshold))

            print(f"\n[{self.symbol}] Transformer | TEST RMSE(all)={test_rmse:.6f} | MAE(all)={test_mae:.6f}")
            print(f"[{self.symbol}] Transformer | Direction Acc (H={H})={dir_acc:.4f} | conf p{conf_percentile} coverage={coverage:.2%}")
        else:
            print(f"\n[{self.symbol}] Transformer | No test sequences produced (test too small).")

        # forecast
        anchor_ts = train_df.index[-1] if use_date_split else df.index[-1]

        all_scaled = pd.concat([train_df_scaled, test_df_scaled], axis=0).sort_index()
        all_scaled = all_scaled.loc[:anchor_ts].copy()

        X_anchor, _, _ = create_supervised_sequences(all_scaled, feature_cols, time_step=ts, f_steps=int(future_steps))
        if len(X_anchor) == 0:
            raise ValueError("Could not form anchor sequence (reduce time_step or add more history).")

        last_input = X_anchor[-1].reshape(1, ts, len(feature_cols))
        pred_future_ret = model.predict(last_input, batch_size=1, verbose=0)[0]

        anchor_price = float(df.loc[anchor_ts, "Close"])
        forecast_prices = anchor_price * np.exp(pred_future_ret)

        future_ts = pd.date_range(anchor_ts, periods=int(future_steps) + 1, freq=pandas_freq)[1:]

        forecast_df = pd.DataFrame({
            "timestamp": future_ts,
            "forecast_forward_logret": pred_future_ret,
            "forecast_price": forecast_prices
        })

        out_csv = os.path.join(output_dir, f"{self.symbol}_transformer_forecast_{self.bar_length}_fs{int(future_steps)}.csv")
        forecast_df.to_csv(out_csv, index=False)
        print(f"\n[{self.symbol}] Saved Transformer future forecast CSV: {out_csv}")

        # history + forecast plot
        lookback_bars = min(300, len(df))
        hist = df.iloc[-lookback_bars:].copy()

        plt.figure(figsize=(12, 5))
        plt.plot(hist.index, hist["Close"].astype(float), label="Actual Close (history)")
        plt.plot(forecast_df["timestamp"], forecast_df["forecast_price"].astype(float), label="Forecast Close (future)")
        plt.axvline(anchor_ts, linestyle="--", linewidth=1.0)
        plt.title(f"Transformer | Actual+Forecast | {self.symbol} | {self.bar_length} | steps={future_steps}")
        plt.xlabel("Time")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(output_dir, f"{self.symbol}_transformer_actual_plus_forecast_{self.bar_length}_fs{int(future_steps)}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[{self.symbol}] Saved combined plot: {out_png}")

        # EMA actual vs pred (1-step reconstruction)
        df_all_scaled = pd.concat([train_df_scaled, test_df_scaled], axis=0).sort_index()
        X_all, _, idx_all = create_supervised_sequences(df_all_scaled, feature_cols, time_step=ts, f_steps=int(future_steps))
        if len(X_all) == 0:
            raise ValueError("Could not form X_all for next-close reconstruction.")

        Y_all_pred = model.predict(X_all, batch_size=int(bs), verbose=0)
        y1_pred = Y_all_pred[:, 0]

        df_base = df.copy()
        close_pred_series = df_base["Close"].astype(float).copy()

        for t, ylog in zip(idx_all, y1_pred):
            loc = df_base.index.get_indexer([t])[0]
            if loc != -1 and (loc + 1) < len(df_base.index):
                t1 = df_base.index[loc + 1]
                close_pred_series.loc[t1] = float(df_base.loc[t, "Close"]) * float(np.exp(float(ylog)))

        df_act = self._add_plot_features_from_binance_ohlcv(df_base)

        df_pred_base = df_base.copy()
        df_pred_base["Close"] = close_pred_series
        df_pred = self._add_plot_features_from_binance_ohlcv(df_pred_base)

        plot_price_with_ema(
            df_actual=df_act,
            df_pred=df_pred,
            pair_name=self.symbol,
            timeframe=self.bar_length,
            output_dir=output_dir,
            suffix="_transformer_actual_vs_pred",
            last_days=5,
        )

        return forecast_df