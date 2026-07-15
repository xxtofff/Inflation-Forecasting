import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.frequencies import to_offset

def naive_forecast(data, target_col, months_ahead):

    if isinstance(data, pd.Series):
        data = data.to_frame(name = target_col if data.name is None else data.name)

    data = data[target_col]
    freq = data.index.freq or to_offset(pd.infer_freq(data.index))
    future_indices = pd.date_range(start = data.index[-1] + freq, periods = months_ahead, freq = freq)
    future_values = pd.Series([data.iloc[-1] for _ in range(months_ahead)], index = future_indices)
    return future_values

def seasonal_naive_forecast(data, target_col, months_ahead):

    if isinstance(data, pd.Series):
        data = data.to_frame(name = target_col if data.name is None else data.name)

    data = data[target_col]
    if len(data) < 12:
        raise ValueError("At least 12 observations are required for a seasonal naive forecast.")
    freq = data.index.freq or to_offset(pd.infer_freq(data.index))
    future_indices = pd.date_range(start = data.index[-1] + freq, periods = months_ahead, freq = freq)
    future_values = pd.Series([data.iloc[-12 + (i % 12)] for i in range(months_ahead)], index = future_indices)
    return future_values


def eval_metrics(actual, forecast):

        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, forecast)
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        std = np.std(actual)

        results = {"RMSE": rmse, "MAE": mae, "MSE": mse, "MAPE": mape, "STD": std}

        return results

def naive_test(ts_data, n_splits, test_size, target_col=None):

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    results = []

    print(f"Folds: {n_splits}")
    print(f"Test Size: {test_size}")

    for _, (train_idx, test_idx) in enumerate(tscv.split(ts_data), start=1):
        ts_data_train = ts_data.iloc[train_idx]
        ts_data_test = ts_data.iloc[test_idx]

        forecast = naive_forecast(ts_data_train, target_col, test_size)
        actual = ts_data_test[target_col] if isinstance(ts_data_test, pd.DataFrame) else ts_data_test

        eval = eval_metrics(actual, forecast)
        results.append(eval)

    fold = pd.Series([*range(1, n_splits+1)], name = 'Fold')
    results = pd.DataFrame(results)
    
    df = pd.concat([fold, results], axis = 1)
    df = df.set_index('Fold')
    
    return df


def seasonal_naive_test(ts_data, n_splits, test_size, target_col=None):

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    results = []

    print(f"Folds: {n_splits}")
    print(f"Test Size: {test_size}")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(ts_data), start=1):
        ts_data_train = ts_data.iloc[train_idx]
        ts_data_test = ts_data.iloc[test_idx]

        forecast = seasonal_naive_forecast(ts_data_train, target_col, test_size)
        actual = ts_data_test[target_col] if isinstance(ts_data_test, pd.DataFrame) else ts_data_test

        eval = eval_metrics(actual, forecast)
        results.append(eval)

    
    fold = pd.Series([*range(1, n_splits+1)], name = 'Fold')
    results = pd.DataFrame(results)
    
    df = pd.concat([fold, results], axis = 1)
    df = df.set_index('Fold')


    return df