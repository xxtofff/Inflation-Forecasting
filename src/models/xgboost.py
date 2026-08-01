import pandas as pd
import numpy as np
import shap

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from statsmodels.tsa.stattools import pacf

from scipy.stats import randint, uniform, loguniform


def sig_lags(data, n_sig_lags):
    pacf_vals = pacf(data)
    lag_vals = sorted(enumerate(pacf_vals[1:], start=1), key=lambda x: abs(x[1]), reverse=True)[:n_sig_lags]

    return sorted(lag for lag, _ in lag_vals)

def eval_metrics(actual, forecast):

        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, forecast)
        r2 = r2_score(actual, forecast)

        results = {"RMSE": rmse, "MAE": mae, "R2": r2}

        return results

def create_date_features(data):
    
    date_columns = [data, ]
    date_columns.append(pd.Series(np.sin(2*np.pi*data.index.month/12), name = 'Month_Sin', index = data.index))
    date_columns.append(pd.Series(np.cos(2*np.pi*data.index.month/12), name = 'Month_Cos', index = data.index))

    return pd.concat(date_columns, axis = 1)

def create_lag_features(data, lag_list, lag_column = None):

    """
    Inputs series/dataframe \\
    Returns original series/dataframe with lag features appended
    """

    if isinstance(data, pd.Series):
        column = data
        lag_column = data.name

    else:
        column = data[lag_column]

    lag_columns = [data,]
    for lag in lag_list:
        lag_columns.append(pd.Series(column.shift(lag), name = '{}_Lag{}'.format(lag_column, lag)))
    
    return pd.concat(lag_columns, axis = 1)

def create_rolling_features(data, window_list, rolling_column = None):

    """
    Inputs series/dataframe \\
    Returns dataframe with rolling features appended to the input
    """

    if isinstance(data, pd.Series):
        column = data
        rolling_column = data.name

    else:
        column = data[rolling_column]

    rolling_columns = [data,]

    for window in window_list:
        rolling_columns.append(pd.Series(column.shift(1).rolling(window).mean(), name = '{}_RollMean{}'.format(rolling_column, window))) #lagged to avoid leakage
        rolling_columns.append(pd.Series(column.shift(1).rolling(window).std(), name = '{}_RollSTD{}'.format(rolling_column, window)))
    
    return pd.concat(rolling_columns, axis = 1)

def make_lagroll_features(data, target_col, window_list, lag_list):

    """
    Inputs series/dataframe and the column to be lagged and rolled \\
    Returns dataframe with consolidated lag and window features.
    """

    if isinstance(data, pd.Series):
        data = data.to_frame(name=target_col if data.name is None else data.name)

    
    for col in data.columns.to_list():
        data = create_lag_features(data=data, lag_list = lag_list, lag_column = col)
        data = create_rolling_features(data=data, window_list = window_list, rolling_column = col)

    data = create_date_features(data)

    return data


def prep_training_data(data, target_col):

    """
    Accepts preprocessed data with full features \\
    Returns the full dataframe (lagged) for forecasting and the forecast features
    """
    data = data.copy()
    data[target_col] = data[target_col].shift(-1)
    data.index = data.index + pd.offsets.MonthBegin(1)

    feats = data.drop(columns=[target_col])
    
    return data.dropna(), feats.iloc[[-1]]


def train_test_split(data, target_col, test_size):

    """
    Accepts preprocessed data
    Returns split 
    """

    feat_cols = data.columns.drop(target_col).to_list()

    data_train = data[:-test_size]
    data_test = data[-test_size:]

    x_train = data_train[feat_cols]
    x_test = data_test[feat_cols]
    y_train = data_train[target_col]
    y_test = data_test[target_col]

    return x_train, y_train, x_test, y_test

def opt_model(n_iter, test_size):

    """
    Inputs number of grid samples and test size \\
    Returns the model with the optimal set of parameters evaluated through time-series cross validation.
    """
    
    param_grid = {
    "n_estimators": randint(50, 1000),
    "learning_rate": loguniform(0.005, 0.2),
    "max_depth": randint(2, 10),
    "min_child_weight": randint(1, 15),
    "subsample": uniform(0.5, 0.5),          # 0.5 - 1.0
    "colsample_bytree": uniform(0.5, 0.5),
    "gamma": uniform(0, 5),
    "reg_alpha": loguniform(1e-5, 10),
    "reg_lambda": loguniform(1e-2, 100)
    }

    tscv = TimeSeriesSplit(n_splits=5, test_size = test_size)
    base_model = XGBRegressor(objective="reg:squarederror", random_state=420)
    search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter = n_iter, scoring="neg_root_mean_squared_error", cv=tscv, random_state=420, n_jobs=-1)
    
    return search

def one_step_forecast(data, target_col, test_size, n_iter):

    """
    Input preprocessed data
    Returns forecast features, optimal model, and one-step-ahead forecast
    
    """

    training, forecast_feat = prep_training_data(data = data, target_col = target_col)

    feat_cols = data.columns.drop(target_col).to_list()
    x_train = training[feat_cols]
    y_train = training[target_col]
    x_test = forecast_feat

    best_model = opt_model(n_iter = n_iter, test_size = test_size)
    best_model.fit(x_train, y_train)
    model = best_model.best_estimator_
    forecast = model.predict(x_test)
    forecast = pd.DataFrame({"{}".format(y_train.name) : forecast}, index = x_test.index)

    return x_test, model, forecast
def important_feats(model, forecast_features, threshold):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(forecast_features)

    importance = (pd.DataFrame({"Feature": forecast_features.columns, "Importance": np.abs(shap_values).mean(axis=0)}).sort_values("Importance", ascending=False))

    feat_thresh = importance['Feature'][(importance['Importance'].cumsum()/importance['Importance'].sum()) < threshold].to_list()

    return feat_thresh

def nexog_recursive_forecast(data, target_col, months_ahead, n_sig_lags, n_iter, threshold, window_list = [3, 6, 9, 12]):

    """
    Takes preprocessed data and desired parameters \\
    Returns n months ahead forecast
    """
    
    if isinstance(data, pd.Series):
        data = data.to_frame()

    if months_ahead < 1:
        raise ValueError("You can't 'foresee' the present and the past, silly.")

    lag_list = sig_lags(data[target_col], n_sig_lags = n_sig_lags)
    full_data = make_lagroll_features(data = data, target_col = target_col, window_list = window_list, lag_list = lag_list)
    forecast_feats, model, forecast = one_step_forecast(full_data, target_col, test_size = months_ahead, n_iter = n_iter)
    imp_feats =  important_feats(model, forecast_feats, threshold = threshold)
    imp_feats.insert(0, target_col)
    forecast_feats, model, forecast = one_step_forecast(full_data[imp_feats], target_col, test_size = months_ahead, n_iter = n_iter)
    new_data = pd.concat([data, forecast], axis = 0)


    for i in range(months_ahead-1):
        full_data = make_lagroll_features(data = new_data, target_col = target_col, window_list = window_list, lag_list = lag_list)
        full_data = full_data[imp_feats]
        _, forecast_feat = prep_training_data(data = full_data, target_col = target_col)
        forecast = model.predict(forecast_feat)
        forecast = pd.DataFrame({"{}".format(target_col) : forecast}, index = forecast_feat.index)
        new_data = pd.concat([new_data, forecast], axis = 0)
    
    return model, new_data[-months_ahead:]

def exog_forecast_feats(data, target_col, months_ahead, n_sig_lags, n_iter, threshold, window_list):
    """
    Inputs preprocessed data (w/ exogenous variables) \\
    Returns dataframe of independent forecasts of exogenous variables 
    """

    exog_cols = data.columns.drop(target_col).to_list()
    exog_pred = pd.DataFrame()

    for col in exog_cols:
        exog_col = data[col].dropna()
        _, forecast = nexog_recursive_forecast(data = exog_col, target_col = col, months_ahead = months_ahead,
                                            n_sig_lags = n_sig_lags, n_iter = n_iter, threshold = threshold,
                                            window_list = window_list)
        exog_pred = pd.concat([exog_pred, forecast], axis = 1)

    return exog_pred

def exog_lags(data, n_sig_lags):

    """
    Input dataframe to find appropriate lags for each column \\
    Returns dictionary of lag arrays for each column
    """

    if isinstance(data, pd.Series):
        data = data.to_frame()

    lag_dict = {}

    for col in data.columns.to_list():
        lag_list = sig_lags(data = data[col], n_sig_lags = n_sig_lags)
        lag_dict[col] = lag_list

    return lag_dict

def lag_roll_exog(data, window_list, lag_dict):

    if isinstance(data, pd.Series):
        data = data.to_frame()

    full_data = pd.DataFrame()

    for col in data.columns.to_list():
        lag_roll = make_lagroll_features(data = data[col], target_col=col, window_list = window_list, lag_list = lag_dict[col])
        full_data = pd.concat([full_data, lag_roll], axis = 1)

    full_data = full_data.loc[: , ~full_data.columns.duplicated()]

    return full_data

def exog_recursive_forecast(data, target_col, months_ahead, n_sig_lags, n_iter, threshold, window_list = [3, 6, 9, 12]):
    
    exog_feats = exog_forecast_feats(data = data, target_col = target_col, months_ahead = months_ahead, 
                                 n_sig_lags = n_sig_lags, n_iter = n_iter, threshold = threshold,
                                 window_list = window_list)
    ex_lags = exog_lags(data = data, n_sig_lags=n_sig_lags)
    
    full_data = lag_roll_exog(data = data, window_list = window_list, lag_dict = ex_lags)

    exog_ffeat, exog_model, exog_forecast = one_step_forecast(data = full_data, target_col = target_col, test_size = months_ahead, n_iter = n_iter)

    imp_feats = important_feats(exog_model, exog_ffeat, threshold)
    imp_feats.insert(0, target_col)
    
    exog_ffeat, exog_model, exog_forecast = one_step_forecast(data = full_data[imp_feats], target_col = target_col, test_size = months_ahead, n_iter = n_iter)

    targ_exog_forecast = pd.concat([exog_forecast, exog_feats.loc[exog_forecast.index]], axis = 1)
    up_data = pd.concat([data, targ_exog_forecast], axis = 0)

    for i in range(months_ahead-1):
        full_data = lag_roll_exog(data = up_data, window_list = window_list, lag_dict = ex_lags)
        full_data = full_data[imp_feats].dropna()

        _, forecast_feat = prep_training_data(data = full_data, target_col = target_col)
        exog_forecast = pd.Series(exog_model.predict(forecast_feat), index = forecast_feat.index, name = target_col)
        targ_exog_forecast = pd.concat([exog_forecast, exog_feats.loc[exog_forecast.index]], axis = 1)
        up_data = pd.concat([up_data, targ_exog_forecast], axis = 0)

    return exog_model, up_data[-months_ahead:]