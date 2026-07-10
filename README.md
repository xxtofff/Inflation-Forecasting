# Inflation Forecasting Portfolio (*under construction*)

A time series forecasting project that analyzes and predicts Philippine inflation using statistical and machine learning models. The project includes data preparation, exploratory data analysis (EDA), and forecasting using SARIMA/SARIMAX and XGBoost.

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
│   ├── 03_sarima.ipynb
│   └── 04_xgboost.ipynb
├── outputs
│   ├── YoY_Inf_Rate.png
│   └── YoY_Inf_Rate_Hist.png
└── src
    └── sarima.py
```

## Workflow

1. **Data Preparation**
   - Clean and merge macroeconomic datasets.
   - Generate the processed dataset used for modeling.

2. **Exploratory Data Analysis**
   - Visualize inflation trends.
   - Examine rolling averages and distribution of inflation.

3. **Forecasting**
   - **SARIMA/SARIMAX** for classical time series forecasting.
   - **XGBoost** for machine learning-based forecasting using macroeconomic features.

## Data Sources

- Bangko Sentral ng Pilipinas (BSP)
- Philippine Statistics Authority (PSA)