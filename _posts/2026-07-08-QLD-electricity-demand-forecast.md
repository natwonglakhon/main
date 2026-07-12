# Forecasting Queensland Electricity Demand: Why Simple Models Beat Complex Ones at Short Horizons

Electricity demand forecasting is one of the core problems in energy market operations. Grid operators need to know how much electricity will be consumed in the coming hours so they can dispatch the right amount of generation, keep the system stable, and avoid unnecessary costs. Getting it wrong in either direction is expensive: too little and you risk blackouts, too much and you're wasting money on generation that isn't needed.

In this project, I build machine learning models to forecast Queensland's electricity demand at 30-minute resolution. Here, I use publicly available data from AEMO, the Open-Meteo weather API, and Queensland public holiday records. I look at two forecasting horizons: predicting demand for the **next hour** and predicting demand for the **next 24 hours**. Importantly, these are quite different problems, and as we shall see, they call for different modelling approaches.

The full code is available at [Github](https://github.com/natwonglakhon/QLD_Electricity_Forecast).

---

## Table of Contents

- [Data](#data)
- [Part I: 1-Hour Ahead Forecast](#part-i-1-hour-ahead-forecast)
  - [Baseline formulation](#baseline-formulation)
  - [Linear Regression (OLS)](#linear-regression-ols)
  - [Bayesian Linear Regression](#bayesian-linear-regression)
  - [Tree Models (Random Forest and XGBoost)](#tree-models-random-forest-and-xgboost)
  - [1-Hour Forecast: Summary](#1-hour-forecast-summary)
- [Part II: 24-Hour Ahead Forecast](#part-ii-24-hour-ahead-forecast)
  - [Baseline features for 24-hour forecasting](#baseline-features-for-24-hour-forecasting)
  - [Linear Regression (OLS)](#linear-regression-ols-1)
  - [Bayesian Linear Regression](#bayesian-linear-regression-1)
  - [Tree Models](#tree-models)
  - [Part II.IV: Engineered Features](#part-iiiv-engineered-features)
    - [Weather engineering](#weather-engineering)
    - [Calendar engineering](#calendar-engineering)
    - [Demand engineering](#demand-engineering)
    - [Results with engineered features](#results-with-engineered-features)
  - [24-Hour Forecast: Summary](#24-hour-forecast-summary)
  - [A surprising finding: Rockhampton vs Brisbane](#a-surprising-finding-rockhampton-vs-brisbane)
- [Conclusion](#conclusion)


---

## Data

Three data sources are merged into a single flat file for modelling.

**AEMO price and demand data** is pulled directly from the AEMO website for the QLD1 region, chosen from June 2020 to June 2026. The two key variables are `TOTALDEMAND` (MW, the target) and `RRP` (Regional Reference Price, $/MWh).

**Weather data** comes from the Open-Meteo historical archive API at hourly resolution, collected for three locations across Queensland: Brisbane (south), Rockhampton (central), and Townsville (north). Covering three locations reflects the geographic scale of Queensland's grid. The variables are apparent temperature, temperature at 2m, dew point, relative humidity, and rainfall.

*Data Cleaning:* There are some missing data from errors/malfunctioning detectors. There, I replace the null data with *lienar interpolation* as the weather data should be gradually changing via linear relationship.

Note that more than half of Queensland's population lives in the southeast, so the expectation going in was that Brisbane weather would dominate the models. However, I found that the speculation turned out to be wrong, and I'll come back to why.

**Calendar data** is constructed using the `holidays` Python library for QLD public holidays, along with weekday, weekend, and workday flags at daily resolution, then merged into the 30-minute demand data.

The final merged dataset runs from August 2020 to June 2026 at 30-minute intervals, giving around 103,000 rows.

---

## Part I: 1-Hour Ahead Forecast

The goal here is: given data **right now**, what will it be in the **next hour**?

### Baseline formulation

For the baseline model, I use only demand lags as predictors. The model is formulated as:

$$\hat{y}_{t+1} = \beta_0 + \beta_1 D_t + \beta_2 D_{t-1} + \beta_3 D_{t-2}$$

where 
 - $\hat{y}_{t+1}$ is the predicted demand one hour ahead, 
 - $D_t$ is the current demand, and 
 - $D_{t-1}, D_{t-2}$ are demand one and two hours ago. Since the data is at 30-minute resolution, one hour corresponds to two lags.

```python
df["demand_now"] = df["TOTALDEMAND"]
df["demand_1h"]  = df["TOTALDEMAND"].shift(2)
df["demand_2h"]  = df["TOTALDEMAND"].shift(4)
df["target"]     = df["TOTALDEMAND"].shift(-2)
```

The train/test split uses `shuffle=False` throughout this project to preserve chronological order.

### Linear Regression (OLS)

The baseline linear regression already does surprisingly well. Demand has very strong *autoregression* over short horizons, which means knowing what demand was doing a few minutes ago is already highly informative about what it will do next.

![Linear Regression baseline: predicted vs actual](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/LR_predicted_vs_actual.png?raw=true?raw=true)

Adding weather features to the linear model is not found to give an improvement. The reason for this may be because the model can only capture linear relationships with those features, while the features themsalves exhibit non-linear relationships.

![Linear Regression with features: predicted vs actual](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/LR1_predicted_vs_actual.png?raw=true)

### Bayesian Linear Regression

I also fit a Bayesian linear regression using the baseline features. The Bayesian formulation treats the model coefficients as random variables with a *prior* distribution that gets updated during training:

$$\hat{y}_{t+1} = \beta_1 D_t + \beta_2 D_{t-1} + \beta_3 D_{t-2} + \epsilon$$

with $\epsilon \sim \mathcal{N}(0, \sigma)$ and $\vec{\beta} \sim \mathcal{N}(\mu_t, \Sigma_t)$. The prior distribution will be updated using Baye's rule and yielding a *posterior* statisitics. For technical people, I also provide the derivations and how the model works [here](https://natwonglakhon.github.io/main/Bayesian-Inference/).

The main advantage of the Bayesian approach is that it gives both a point estimate and, importantly, a **posterior distribution**. This means the model produces uncertainty estimates alongside its predictions, which can be visualised as a confidence interval.

![Bayesian Linear Regression: predicted vs actual](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/BLR_predicted_vs_actual.png?raw=true)

![Bayesian Linear Regression: prediction with uncertainty envelope](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/BLR_prediction.png?raw=true)

### Tree Models (Random Forest and XGBoost)

For the tree models, I first include the discrete calendar features (day of week, weekend flag, etc.) by one-hot encoding them, giving a richer feature set.

Both Random Forest and XGBoost are tree-based and therefore scale-invariant, so no normalisation is needed. They are fit on the same 80/20 chronological split.

![XGBoost with existing features: predicted vs actual](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/XGB_predicted_vs_actual.png?raw=true)

#### Feature importance

![Feature importance: XGBoost 1-hour model](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/Feature_importance_xgb_1h.png?raw=true)

As expected, `demand_now` dominates heavily. For a 1-hour ahead forecast, the current demand reading is by far the most informative signal.

### 1-Hour Forecast: Summary

![All models: 1-hour forecast comparison](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/all_predicted_1h.png?raw=true)

| Model | MAPE |
|---|---|
| **Best** -- Linear Regression (baseline)| 1.01% |
| Linear Regression (with features) | 1.08% |
| XGBoost (with features) | 1.18% |
| Random Forest (with features) | 1.30% |

The main result here is that for a 1-hour ahead forecast, **the linear models win** thanks to the autoregressive property of the demand. With the linear autoregressive structure in demand, these linear models are thus a very effective predictor. Adding weather features or using tree models does not improve things much because the nonlinear patterns in demand, driven by temperature and time of day, play out over **longer timescales** than one hour.

The baseline demand lags alone provide highly accurate short-term forecasts for this dataset.

---

## Part II: 24-Hour Ahead Forecast

Now, extending the horizon to 24 hours forecast. This makes the problem substantially harder. The autoregressive feature weakens considerably, and we now need to model the full daily demand cycle, weather effects, and calendar patterns explicitly rather than letting recent lags carry the weight.

### Baseline features for 24-hour forecasting

For the 24-hour model, the target is demand 24 hours from now. The baseline demand lags are chosen to reflect the same time of day at different historical distances:

```python
df24["demand_now"] = df24["TOTALDEMAND"]
df24["demand_12h"] = df24["TOTALDEMAND"].shift(24)   # 12 hours ago
df24["demand_24h"] = df24["TOTALDEMAND"].shift(48)   # 24 hours ago (same time yesterday)
df24["demand_1w"]  = df24["TOTALDEMAND"].shift(freq="7D")    # same time last week
df24["demand_1y"]  = df24["TOTALDEMAND"].shift(freq="366D")  # same time last year
df24["target"]     = df24["TOTALDEMAND"].shift(-48)          # 24 hours ahead
```

Using `shift(freq="7D")` and `shift(freq="366D")` to ensures that the lags align to the correct timestamps even when there are gaps or leap years in the data.

### Linear Regression (OLS) with 24-hour features

![Linear Regression: 24-hour forecast](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/Lr_predicted_vs_actual_24.png?raw=true)

### Bayesian Linear Regression

![Bayesian LR: 24-hour forecast](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/Bayes_predicted_vs_actual_24.png?raw=true)

![Bayesian LR: 24-hour forecast with uncertainty envelope](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/Baye_pred_24h.png?raw=true)

### Tree Models

![XGBoost: 24-hour forecast](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/XGboost_predicted_vs_actual_24.png?raw=true)

#### Feature importance (24-hour baseline model)

![Feature importance: XGBoost 24-hour baseline](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/Feature_importance_xgb_24h.png?raw=true)

With baseline and the features, the results across models are:

| Model | MAPE |
|---|---|
| Linear Regression (OLS) | 8.63% |
| Bayesian Linear Regression | 8.69% |
| Random Forest | 7.69% |
| XGBoost | 7.59% |

The tree models now **outperform** the linear models. The nonlinear demand patterns that play out over 24 hours, particularly the relationship between temperature and peak afternoon demand, can be well captured by the tree models.

However, 7-8% MAPE is **not** great for an operational forecasting system. The next step is feature engineering.

---

## Part II.IV: Engineered Features

This is where most of the meaningful improvement comes from. The model is extended to:

$$\hat{y}_{t+24} = \text{Baseline} + \text{Engineered features}$$

### Weather engineering

For each location and temperature variable, I compute:

**Cooling and Heating Degree Days (CDD/HDD)** capture the nonlinear demand response to temperature. Air conditioning load ramps sharply above about 30°C. Below that threshold, it contributes little. CDD and HDD encode this threshold behaviour directly:

```python
features_num_agg[f"CDD_{loc}_{var}"] = np.maximum(features_num_agg[f"{loc}_{var}"] - 30, 0)
features_num_agg[f"HDD_{loc}_{var}"] = np.maximum(30 - features_num_agg[f"{loc}_{var}"], 0)
```

**Rolling mean and standard deviation** over the past 24 hours capture whether conditions have been stable or rapidly changing:

```python
features_num_agg[f"{loc}_{var}_rolling_mean_24h"] = features_num_agg[f"{loc}_{var}"].rolling(48).mean()
features_num_agg[f"{loc}_{var}_rolling_std_24h"]  = features_num_agg[f"{loc}_{var}"].rolling(48).std()
```

**Temperature difference** (apparent vs actual temperature) helps capturing the humidity effect: a 35°C day with high humidity feels hotter and drives more air conditioning than a dry 35°C day.

**Weather lags** at 12h and 24h offsets are included for all weather variables across all three locations. For a 24-hour ahead model, using weather from 24 hours ago as a proxy for conditions at the forecast target time avoids leaking future information into the model.

### Calendar engineering

Beyond the basic weekday/weekend flags, I add:

- **Hour of day**: critical for encoding the daily demand cycle (which turns out to be very *important*)
- **Season flags** (summer, winter, spring, autumn)
- **Christmas/New Year break flag** (Dec 24 to Jan 2): commercial and industrial demand drops significantly during this period even on weekdays
- **Working hours flag** (9am to 5pm): useful for separating commercial from residential demand patterns
- **Month and year**: captures seasonal trends and the gradual year-on-year change in baseline demand

### Demand engineering

**Daily demand range** (max minus min within each calendar day, mapped back to 30-minute resolution) captures how thermally stressed the day was without requiring future information:

```python
daily_max = features_num_agg["demand_now"].resample("D").max()
daily_min = features_num_agg["demand_now"].resample("D").min()
daily_range = daily_max - daily_min

features_num_agg["demand_daily_range"] = features_num_agg.index.normalize().map(daily_range).fillna(0)
```

**Rolling mean and standard deviation** of demand over the past 12 hours encode recent demand momentum.

**30-minute demand change** captures whether demand is currently rising or falling.

### Results with engineered features

![Linear Regression: 24-hour forecast with engineered features](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/LR_predicted_vs_actual_24agg.png?raw=true)

![XGBoost: 24-hour forecast with engineered features](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/XGBoost_predicted_vs_actual_24agg.png?raw=true)

![XGBoost: 24-hour forecast time series](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/XGBoost_predicted_24agg.png?raw=true)

![All models: 24-hour forecast comparison](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/all_predicted_24agg.png?raw=true)

#### Feature importance with engineered features

![Feature importance: XGBoost 24-hour engineered model](https://github.com/natwonglakhon/QLD_Electricity_Forecast/blob/4b577f011216608ff05f7c12562f3cd96439818b/images/Feature_importance_xgb_24h.png?raw=true)

### 24-Hour Forecast: Summary

| Model | MAPE (baseline/features) | MAPE (add engineered features) |
|---|---|---|
| Linear Regression (OLS) | 8.63% | ~8% |
| Bayesian Linear Regression | 8.69% | ~8.7% |
| Random Forest | 7.69% | 4.92% |
| **Best**-- XGBoost | 7.59% | 4.59% |

Feature engineering delivers a significant drop for the tree models, from around 7.5% down to 4.59% for XGBoost (40% drop). The linear models see only a modest improvement because they can't exploit the nonlinear structure in the engineered features the way tree models can.

The most important features in the final XGBoost model are hour of day and demand daily range, which makes physical sense: the hour determines where in the daily demand cycle you are, and the demand range tells you how extreme that day's temperature-driven variation has been.

### A surprising finding: Rockhampton vs Brisbane

One unexpected result is that Rockhampton temperature consistently ranks higher in feature importance than Brisbane temperature, despite Brisbane having more than half of Queensland's population. The initial expectation was that SEQ demand would dominate.

A few things probably explain this. One is the large industrial electricity demand associated with Central Queensland's mining sector. The coal mining operations and associated industrial load in Central Queensland are enormous electricity consumers, and they are highly temperature-sensitive for cooling and ventilation. When Central Queensland runs hot, that industrial load spikes in a way that may be more predictable and more extreme than the diffuse residential air conditioning response in Brisbane. It is also possible that Brisbane temperature is somewhat *collinear* with the demand lag features (hot Brisbane days and high recent demand tend to co-occur), so the tree splits assign the overlapping signal to demand lags and thus leave Rockhampton temperature to pick up the residual variation.

However, this hypothesis would require additional industrial load data to verify, particularly with respect to how mining industry activity interacts with demand patterns.

---

## Conclusion and Discussion

The main findings from this project are:

For **1-hour ahead forecasting**, the simple autoregressive signal in demand is so strong that a linear model on three demand lags achieves 1.01% MAPE. Tree models do not add much. Feature engineering does not help much either. The problem is already well-solved by knowing what demand was doing recently.

For **24-hour ahead forecasting**, the story is completely different. Baseline models sit around 7.5-8.6% MAPE regardless of model type. Careful feature engineering, particularly the cooling/heating degree days, daily demand range, and hour of day, brings XGBoost down to 4.59% MAPE, which is within the industry-standard range for short-term load forecasting.

An important lesson from this project is that increasing model complexity does *not* necessarily improve forecasting performance. For one-hour-ahead forecasting, demand persistence dominates and simple linear models perform exceptionally well. For longer forecasting horizons, however, nonlinear interactions between weather, seasonality, and historical demand become increasingly important, allowing tree-based models with carefully engineered features to substantially outperform linear models.

### What decisions could someone make using these models?
Accurate electricity demand forecasting supports a wide range of operational and planning decisions in power systems. While this project focuses on predictive modelling, the resulting forecasts could be used to inform decisions such as: generation scheduling, demand response programs, and Maintenance planning.

### Future work
The clearest path to further improvement would be replacing lagged weather with the **weather forecasts**. Right now the model uses yesterday's weather as a proxy for tomorrow's weather, which works reasonably well given Queensland's climate patterns, but a genuine 24-hour-ahead temperature forecast would almost certainly push the MAPE below 4%. Moreover, further **optimising** the tree models would also improve the models. I will leave that for future investigation. 

