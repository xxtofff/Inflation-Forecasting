import pandas as pd
import numpy as np

from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

def eval_metrics(actual, forecast):

        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, forecast)
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        std = np.std(actual)

        results = {"RMSE": rmse, "MAE": mae, "MSE": mse, "MAPE": mape, "STD": std}

        return results

def sarimax_forecast(ts_data, n_forecast, target_col=None):

    if isinstance(ts_data, pd.Series):
        exog_cols = None
    
    else:
        exog_cols = ts_data.columns.drop(target_col).tolist()
        
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


def sarimax_test(ts_data, n_splits, test_size, target_col=None):
    
    if isinstance(ts_data, pd.Series):
        exog_cols = None
    
    else:
        exog_cols = ts_data.columns.drop(target_col).tolist()

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    results = []

    print(f"Folds: {n_splits}")
    print(f"Test Size: {test_size}")

    for _, (train_idx, test_idx) in enumerate(tscv.split(ts_data), start=1):
        ts_data_train = ts_data.iloc[train_idx]
        ts_data_test = ts_data.iloc[test_idx]

        _, arima_pred, _ = sarimax_forecast(ts_data_train, len(ts_data_test), target_col=target_col)

        actual = ts_data_test if exog_cols is None else ts_data_test[target_col]

        eval = eval_metrics(actual, arima_pred)
        results.append(eval)

    fold = pd.Series([*range(1, n_splits+1)], name = 'Fold')
    results = pd.DataFrame(results)
    
    df = pd.concat([fold, results], axis = 1)
    df = df.set_index('Fold')
    
    return df