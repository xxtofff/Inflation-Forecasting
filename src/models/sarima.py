import pandas as pd
import numpy as np

from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

def sarimax_forecast(ts_data, n_forecast, target_col=None, exog_cols=None):
    n_forecast = int(n_forecast)

    if exog_cols is None:
        ts_targ = ts_data
        arima_model = auto_arima(ts_targ, seasonal=True, m=12, error_action="ignore", suppress_warnings=True)
        arima_pred, conf_int = arima_model.predict(n_periods=n_forecast, return_conf_int=True)
    else:
        ts_targ = ts_data[target_col]
        ts_exog = ts_data[exog_cols]

        arima_model = auto_arima(ts_targ, X=ts_exog, seasonal=True, m=12, error_action="ignore", suppress_warnings=True)

        future_exog = pd.DataFrame({exog: auto_arima(ts_data[exog], seasonal=True, m=12, error_action="ignore", suppress_warnings=True).predict(n_periods=n_forecast) for exog in exog_cols})

        arima_pred, conf_int = arima_model.predict(n_periods=n_forecast, X=future_exog, return_conf_int=True)

    return arima_model, arima_pred, conf_int


def sarimax_test(ts_data, n_splits, test_size, target_col=None, exog_cols=None):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    rmse = {}
    naive_rmse = {}

    print(f"Folds: {n_splits}")
    print(f"Test Size: {test_size}")

    for n_split, (train_idx, test_idx) in enumerate(tscv.split(ts_data), start=1):
        ts_data_train = ts_data.iloc[train_idx]
        ts_data_test = ts_data.iloc[test_idx]

        _, arima_pred, _ = sarimax_forecast(ts_data_train, len(ts_data_test), target_col=target_col, exog_cols=exog_cols)

        actual = ts_data_test if exog_cols is None else ts_data_test[target_col]
        actual_std = np.std(actual)

        rmse_split = np.sqrt(mean_squared_error(actual, arima_pred))
        rmse[n_split] = rmse_split

        history = ts_data_train if exog_cols is None else ts_data_train[target_col]
        naive_pred = np.tile(history.iloc[-12:].to_numpy(), int(np.ceil(len(actual) / 12)))[:len(actual)]

        naive_rmse_split = np.sqrt(mean_squared_error(actual, naive_pred))
        naive_rmse[n_split] = naive_rmse_split

        print(f"Fold {n_split}, Test RMSE: {rmse_split:.2f}, Seasonal Naive RMSE: {naive_rmse_split:.2f}, STD: {actual_std:.2f}")

    rmse_values = np.fromiter(rmse.values(), dtype=float)
    naive_rmse_values = np.fromiter(naive_rmse.values(), dtype=float)

    print(f"RMSE Mean: {rmse_values.mean():.2f}   Seasonal Naive Mean: {naive_rmse_values.mean():.2f}")
    print(f"RMSE STD: {rmse_values.std():.2f}   Seasonal Naive STD: {naive_rmse_values.std():.2f}")