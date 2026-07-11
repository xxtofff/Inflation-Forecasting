import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from scipy.stats import randint, uniform, loguniform


def create_date_features(data):
    
    date_columns = [data, ]
    date_columns.append(pd.Series(np.sin(2*np.pi*data.index.month/12), name = 'Month_Sin', index = data.index))
    date_columns.append(pd.Series(np.cos(2*np.pi*data.index.month/12), name = 'Month_Cos', index = data.index))

    return pd.concat(date_columns, axis = 1)

def create_lag_features(data, lag_list, lag_column = None):

    if lag_column is None:
        column = data
        lag_column = data.name
    else:
        column = data[lag_column]

    lag_columns = [data,]
    for lag in lag_list:
        lag_columns.append(pd.Series(column.shift(lag), name = '{}_Lag{}'.format(lag_column, lag)))
    
    return pd.concat(lag_columns, axis = 1)

def create_rolling_features(data, window_list, rolling_column = None):

    if rolling_column is None:
        column = data
        rolling_column = data.name
    else:
        column = data[rolling_column]

    
    rolling_columns = [data,]
    for window in window_list:
        rolling_columns.append(pd.Series(column.shift(1).rolling(window).mean(), name = '{}_RollMean{}'.format(rolling_column, window))) #lagged to avoid leakage
        rolling_columns.append(pd.Series(column.shift(1).rolling(window).std(), name = '{}_RollSTD{}'.format(rolling_column, window)))
    
    return pd.concat(rolling_columns, axis = 1)

def train_forecast_split(data, target_col, lag_list=[1, 2, 13, 24], window_list=[3, 6, 12]):

    if isinstance(data, pd.Series):
        data = data.to_frame(name=target_col if data.name is None else data.name)

    exog_cols = data.columns.drop(target_col).to_list()

    date = create_date_features(data=data.dropna())
    lag = create_lag_features(data=date, lag_list=lag_list, lag_column=target_col)
    final = create_rolling_features(data=lag, rolling_column=target_col, window_list=window_list)

    if exog_cols:
        for col in exog_cols:
            final = create_lag_features(data=final, lag_list=lag_list, lag_column=col)
            final = create_rolling_features(data=final, rolling_column=col, window_list=window_list)

    forecast_feat = final[final.columns.drop(target_col).to_list()].iloc[[-1]]

    return final, forecast_feat


def best_params(n_iter):
    param_grid = {
    "n_estimators": randint(200, 2500),
    "learning_rate": loguniform(0.005, 0.2),
    "max_depth": randint(2, 10),
    "min_child_weight": randint(1, 15),
    "subsample": uniform(0.5, 0.5),          # 0.5 - 1.0
    "colsample_bytree": uniform(0.5, 0.5),
    "gamma": uniform(0, 5),
    "reg_alpha": loguniform(1e-5, 10),
    "reg_lambda": loguniform(1e-2, 100)
    }

    tscv = TimeSeriesSplit(n_splits=5, test_size = 24)
    base_model = XGBRegressor(objective="reg:squarederror", random_state=420)
    search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter = n_iter, scoring="neg_root_mean_squared_error", cv=tscv, random_state=420, n_jobs=-1)
    
    return search

def train_test_split(data, n_months, targ_col, feat_cols):
    
    if type(targ_col) == str:
        targ_col = targ_col

    else:
        targ_col = targ_col[0]
    
    train = data[:-n_months]
    test = data[-n_months:]

    train_targ = train[targ_col]
    train_feat = train[feat_cols]
    test_targ = test[targ_col]
    test_feat = test[feat_cols]


    return train_feat, train_targ, test_feat, test_targ

def model_pred(train_feat, train_targ, pred_feat, n_iter):
    model_test = best_params(n_iter)
    model_test.fit(train_feat, train_targ)
    pred_targ = model_test.best_estimator_.predict(pred_feat)
    return pred_targ, model_test

def test_diag(data, targ_col, n_iter):
    feat_cols = [col for col in data.columns if col != targ_col]
    train_feat, train_targ, test_feat, test_targ = train_test_split(data.dropna(), 24, targ_col=targ_col, feat_cols=feat_cols)
    pred_targ, model_test = model_pred(train_feat=train_feat, train_targ=train_targ, pred_feat=test_feat, n_iter=n_iter)
    rmse = np.sqrt(mean_squared_error(test_targ, pred_targ))
    std = np.std(test_targ)
    r2 = r2_score(test_targ, pred_targ)
    cv_rmse = -model_test.best_score_
    rows = len(data)
    print(f"Rows      : {rows}")
    print(f"CV RMSE   : {cv_rmse:.3f}")
    print(f"Test RMSE : {rmse:.3f}")
    print(f"Test STD  : {std:.3f}")
    print(f"R²        : {r2:.3f}")
    print(f"Best Params:\n{model_test.best_params_}")
    return {"cv_rmse": cv_rmse, "test_rmse": rmse, "test_std": std, "r2": r2, "best_params": model_test.best_params_}

def forecast_df(data, targ_col):

    feat_cols = [col for col in data.columns if col != targ_col]

    forecast_feat = data[feat_cols].iloc[[-1]].copy()

    forecast_feat.index = forecast_feat.index + pd.offsets.MonthBegin(1)

    full_feat = data[feat_cols].copy()
    full_feat.index = full_feat.index + pd.offsets.MonthBegin(1)

    full_targ = data[targ_col]

    full_df = pd.concat([full_targ, full_feat], axis=1).dropna()

    return full_df[feat_cols], full_df[targ_col], forecast_feat


def months_ahead_forecast(data, target_col, months_ahead):

    """
    Generates recursive multi-step forecasts for the target variable by first training an XGBoost model on the target and separate models for each exogenous variable, if provided, forecasting exogenous variables at each step, updating the dataset with predicted values, regenerating lag and rolling features, predicting the target for the next period, and repeating the process until the specified forecast horizon is reached. Displays a SHAP waterfall plot explaining the first forecast and returns a DataFrame containing the forecasted observations for the specified number of months ahead.

    args:
        data (pd.Series or pd.DataFrame): Time-indexed target series or DataFrame containing the target and optional exogenous variables.
        target_col (str): Name of the target variable to forecast.
        months_ahead (int): Number of future periods to forecast recursively.

    returns:
        pd.DataFrame: DataFrame containing the forecasted target and exogenous variable values for the specified forecast horizon.
    """

    iter = 1

    if isinstance(data, pd.Series):
        data = data.to_frame(name=target_col)

    exog_cols = data.columns.drop(target_col).to_list()
    exog_models = {}

    final_data, _ = train_forecast_split(data, target_col)
    full_feat, full_targ, forecast_feat = forecast_df(final_data, target_col)
    _, model_test = model_pred(full_feat, full_targ, forecast_feat, n_iter=100)
    model = model_test.best_estimator_
    last_obs = forecast_feat.iloc[[0]]

    for col in exog_cols:
        final_exog, _ = train_forecast_split(data, col)
        full_feat_exog, full_targ_exog, forecast_feat_exog = forecast_df(final_exog, col)
        _, model_test_exog = model_pred(full_feat_exog, full_targ_exog, forecast_feat_exog, n_iter=100)
        exog_models[col] = model_test_exog.best_estimator_

    while iter <= months_ahead:

        for col in exog_cols:
            final_exog, _ = train_forecast_split(data, col)
            _, _, forecast_feat_exog = forecast_df(final_exog, col)
            exog_forecast = exog_models[col].predict(forecast_feat_exog)[0]
            data.loc[forecast_feat_exog.index[0], col] = exog_forecast

        final_data, _ = train_forecast_split(data, target_col)
        _, _, forecast_feat = forecast_df(final_data, target_col)
        model_forecast = model.predict(forecast_feat)
        data.loc[forecast_feat.index[0], target_col] = model_forecast[0]

        iter += 1

    explainer = shap.TreeExplainer(model)

    shap.waterfall_plot(shap.Explanation(values=explainer.shap_values(last_obs)[0], base_values=explainer.expected_value, data=last_obs.iloc[0], feature_names=last_obs.columns))
    plt.show()

    return data[-months_ahead:]