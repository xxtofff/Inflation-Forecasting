# Philippine Inflation Rate Forecasting

A time series forecasting project that analyzes and predicts monthly year-on-year Philippine inflation rate using statistical and machine learning models. The project covers data preparation, exploratory data analysis (EDA), feature engineering, and forecasting using baseline, statistical, and machine learning approaches.

## Project Structure

```
.
├── data
│   ├── processed
│   │   └── combined_macro_data.csv
│   └── raw
│       ├── bsp_cpibase2018_dataset.csv
│       ├── global_dubai_crude.csv
│       ├── psa_unemployment_rate.csv
│       └── usd_to_php.csv
├── notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_naive.ipynb
│   ├── 04_sarima.ipynb
│   └── 05_xgboost.ipynb
├── outputs
│   ├── figures
│   │   ├── YoY_Inf_Rate.png
│   │   ├── YoY_Inf_Rate_Hist.png
│   │   ├── xgboost_forecast.jpg
│   │   └── xgboost_forecast.png
│   └── forecasts
│       ├── full_nexog_forecast.csv
│       ├── naive_forecast.csv
│       ├── red_exog_forecast.csv
│       ├── red_nexog_forecast.csv
│       ├── seasonal_naive_forecast.csv
│       ├── xgboost_exog_forecast.csv
│       ├── xgboost_full_nexog_forecast.csv
│       └── xgboost_red_nexog_forecast.csv
├── src
│   └── models
│       ├── naive.py
│       ├── sarima.py
│       └── xgboost.py
└── README.md
```

## Workflow

1. **Data Preparation**

   * Clean and merge macroeconomic datasets.
   * Generate the processed dataset used for modeling.

2. **Exploratory Data Analysis**

   * Visualize inflation trends.
   * Examine rolling averages and the distribution of inflation.

3. **Forecasting**

   * **Naive and Seasonal Naive** as baseline forecasting models.
   * **SARIMA/SARIMAX** for classical time series forecasting.
   * **XGBoost** for machine learning-based forecasting using engineered macroeconomic features.

4. **Outputs**

   * Generate forecast visualizations.
   * Export forecast results as CSV files for further analysis.

## Data Sources

* Bangko Sentral ng Pilipinas (BSP)
* Philippine Statistics Authority (PSA)
* Dubai Fateh Crude Oil Spot Price Data