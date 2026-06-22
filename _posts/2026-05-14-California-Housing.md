# Housing Price Prediction Modelling

In this analysis, I investigate machine learning models for estimating residential property values in California using demographic and geographic information. Accurate price estimation is valuable for real estate platforms, financial institutions, and property investors, where automated valuations support pricing decisions, mortgage assessments, and market analysis.

Three models are compared: Linear Regression (LR), Random Forest (RF), and XGBoost. The analysis begins with data cleaning and preparation before evaluating each model through prediction accuracy and residual analysis. Finally, new geographical and structural features are engineered to examine whether domain knowledge can improve predictive performance.

The analysis begins with loading and cleaning the data. Once ready, the data is split into train and test sets. The data is then scaled for the LR model, while RF and XGBoost use the original unscaled data. All three models are evaluated by the $R^2$ score, and their errors are examined through residual analysis. To enhance the models further, new features are aggregated from the existing ones, and all three models are retrained and re-evaluated.

## [Click here for Repository](https://github.com/natwonglakhon/House_price_modelling)


## Business Objective

A real estate valuation model should both produce accurate predictions and provide consistent estimates across different regions and property types. Large prediction errors can lead to overpriced listings, undervalued assets, or inaccurate lending decisions.

The **objective** of this project is therefore to identify a model that balances predictive accuracy, robustness, and interpretability while investigating which property characteristics contribute most to house prices.


---

## I. Data Cleaning and Preparing

The data is retrieved from [Kaggle](https://www.kaggle.com/datasets/nalisha/california-housing-prices-dataset-clean-and-ml/data), covering California housing census block groups. It contains 20,640 rows and 10 features before filtering, with `median_house_value` as the prediction target.

The first thing to check is the distribution of the target variable. The mean (\\$206,856) is noticeably higher than the median (\\$179,700), indicating a right-skewed distribution with a large spike near \\$500,000. This spike is a known artefact of how the dataset was collected: prices were top-coded at \\$500,001, meaning any house worth more than that was simply recorded as $500,001. This creates an artificial cluster that would bias model training and cause divergence in gradient-based optimisers like SGDRegressor. The data is therefore filtered to keep only values up to the 95th percentile.

![Price Distribution](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/median_house_value_distribution.png?raw=true)

There are also 200 missing values in the `total_bedrooms` column. These are filled with the column median, which is a reasonable choice since median is robust to the skewed distribution of room counts.

After cleaning, the dataset has 19,608 rows with no missing values, and the price distribution is still right-skewed but without the extreme outlier cluster.

![Price Distribution (Filtered)](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/median_house_value_distribution_filtered.png?raw=true)

The `ocean_proximity` column contains five categories: `NEAR BAY`, `<1H OCEAN`, `INLAND`, `NEAR OCEAN`, and `ISLAND`. These are converted to numerical dummy variables using one-hot encoding (`get_dummies` with `drop_first=True`), which drops `<1H OCEAN` (the most frequent category) as the baseline, leaving four binary columns.

### Splitting and Scaling

The data is split 70% training and 30% testing using `train_test_split` with `random_state=42`. `StandardScaler` (Z-score normalisation: mean=0, std=1) is then applied, fitted only on training data to avoid leaking test information into the model. Importantly, scaling is applied to continuous features only. Applying it to near-constant binary dummy variables (such as the very rare `ISLAND` category) produces extreme scaled values that destabilise gradient-based optimisers.

---

## II. Baseline Property Valuation Models

### IIA. Linear Regression

The first model is a Linear Regression trained using Stochastic Gradient Descent (`SGDRegressor`, `max_iter=2000`). SGD is used in place of the ordinary least squares solver for its scalability, though both give similar results on a dataset this size.

The model achieves an $R^2$ score of **0.604**, converging after 21 iterations and 288,226 weight updates. This is a reasonable baseline but unsurprisingly limited, because the relationship between features and house price is non-linear in nature, and a linear model cannot learn this structure regardless of how well it is tuned.

Looking at the model coefficients, `median_income` carries the highest weight (58,957), reflecting its dominant role in determining house prices.

The predicted vs actual plot shows that the model handles lower prices reasonably well, but deviates more at higher price ranges, consistent with the non-linear ceiling.

![Test vs LR Prediction](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/sgd_predicted_vs_actual.png?raw=true)

### IIB. Random Forest

The next model is Random Forest, chosen for its ability to handle non-linear data naturally. Rather than fitting a single global function, it builds many decision trees, each trained on a random subset of the data. Each tree makes an independent prediction, and the final output is the average across all trees.

Within each tree, the data is split at decision nodes, where the splitting feature is randomly chosen from a subset of sqrt(N) features (in this case, sqrt(12) ~ 3 features per split). Each node asks a simple yes/no question based on a threshold, for example "is `median_income` > 3.5?". The process continues until the tree reaches its maximum depth or a node has too few samples to split further. Once a branch can no longer split, it becomes a leaf, and predictions at that leaf are the average price of all training rows that ended up there.

A baseline model with 200 trees achieves an $R^2$ of **0.783**.

The model is then tuned using `RandomizedSearchCV` with 5-fold cross-validation, searching over the number of trees, maximum depth, and minimum samples split. Rather than trying every possible combination (which is what `GridSearchCV` does), `RandomizedSearchCV` samples a fixed number of random combinations and finds a good configuration much faster. The best parameters found are `n_estimators=500`, `max_depth=30`, and `min_samples_split=2`, giving a test $R^2$ of **0.784**, only marginally better than the baseline.

![Test vs RF Prediction](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/RF_predicted_vs_actual.png?raw=true)

#### Feature Importance

Feature importance tells us which features reduce prediction error the most (measured by mean squared error reduction across all splits).

![Feature Importance](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/feature_importance.png?raw=true)

Median income is by far the strongest predictor at 44%, followed by location features. It is worth noting that binary features like `ocean_proximity` may have biased importance scores, since they only take values 0 or 1 and therefore have fewer possible split thresholds. This result aligns with expectations from the housing market. Areas with higher household incomes generally support higher purchasing power, which is reflected in property values. For organisations using automated valuation models, this finding highlights income-related demographics as one of the most informative signals when estimating prices.

### IIC. XGBoost

The third model is XGBoost. Instead of building independent trees in parallel like Random Forest, XGBoost builds trees sequentially, with each new tree correcting the errors of the previous ones, gradually minimising the price error. Given this, it is more powerful than the RF model, at the cost of being more computationally expensive.

With `n_estimators=500`, `max_depth=6`, and `learning_rate=0.05`, the model achieves an $R^2$ of **0.802**, improving on both linear regression and Random Forest.

From a business perspective, this improvement translates into more reliable automated property valuations. A model with smaller prediction errors reduces the risk of systematically overpricing or underpricing properties, making it more suitable for supporting valuation decisions than the simpler Linear Regression baseline.

![Test vs XGBoost Prediction](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/xgb_predicted_vs_actual.png?raw=true)

Interestingly, the feature importance here differs from what Random Forest found. `ocean_proximity_INLAND` emerges as the chief contributor (0.573), overtaking `median_income` (0.171). This suggests XGBoost is uncovering a different view of the data's structure, capturing the price penalty for being inland more explicitly than Random Forest did through its averaged splits.

---

## III. Residual Analysis

To understand where the models are failing, the residuals (actual minus predicted price) are examined for each model.

![Residuals vs Fitted](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/residuals_vs_fitted.png?raw=true)

All three models struggle at extreme prices, both very cheap and very expensive properties. LR produces visibly larger errors across the full price range compared to the two tree-based models.

![Residual Distribution](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/residual_distribution.png?raw=true)

The residual distributions are centred around zero with a roughly normal shape, which is a healthy sign. In terms of magnitude, XGBoost produces the smallest errors overall:

| Model | Mean Residual | Std of Residuals |
|-------|--------------|-----------------|
| Linear Regression | -$920 | $61,136 |
| Tuned Random Forest | -$712 | $45,114 |
| XGBoost | -$178 | $43,235 |

#### Errors by Geography

Plotting the absolute prediction errors on a map reveals a clear geographic pattern: the worst predictions are concentrated around major cities and along the coastline.

![Error Map](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/error_map.png?raw=true)

This makes intuitive sense. Urban and coastal markets are more volatile and harder to predict from census block features alone. Latitude and longitude are in the model, but they do not carry explicit information about distance to a city centre or distance to the coast. This finding directly motivates the feature engineering in the next section. 

From a practical perspective, these are precisely the regions where valuation accuracy is most important because they contain many high-value properties and more active markets. This suggests that additional location-specific features or external datasets could provide the greatest improvement in future iterations of the model.

---

## IV. Improving Valuation Through Feature Engineering

Based on the residual analysis, several new features are engineered to give the models better spatial and structural information.

Four ratio features are derived from the existing columns. Log transformation is applied to reduce skewness, since these are ratios and tend to have long right tails:

- `rooms_per_household` = log(total_rooms / households)
- `bedrooms_per_room` = log(total_bedrooms / total_rooms)
- `population_per_household` = log(population / households)
- `income_x_density` = log(median_income x population / total_rooms)

Three Euclidean distance features are added to San Francisco, Los Angeles, and San Diego, motivated directly by the geographic error map. A distance to the nearest coastline feature is also added using a KD-tree, built from all census block centroids labelled as coastal (`NEAR OCEAN`, `NEAR BAY`, `<1H OCEAN`) in the dataset. For each property, the nearest point in that reference set is found and the degree distance is converted to kilometres.

### IVA. Linear Regression

With the aggregated feature set, Linear Regression achieves an $R^2$ of **0.604**, unchanged from the baseline. This confirms that the ceiling for a linear model on this dataset is around 60%, regardless of additional features: the structure of the problem is fundamentally non-linear.

### IVB. Random Forest

The tuned Random Forest improves to **0.791** with the aggregated features.

The updated feature importance now tells a richer story:

![Feature Importance Random Forest (Aggreated)](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/feature_importance_agg.png?raw=true)

Distance to the coastline jumps to second place at 26.6%, confirming that coastal proximity is a genuinely strong signal for house prices that the raw `ocean_proximity` categories were not fully capturing.

### IVC. XGBoost

XGBoost with the aggregated features achieves an $R^2$ of **0.820**, the **best result** in this analysis. The distance to coastline again ranks second in importance (0.261), reinforcing how much geographic distance contributes to predictive power beyond what raw coordinates provide.
![Feature Importance XGBoost (Aggreated)](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/feature_importance_xgb_agg.png?raw=true)


### IVD. Residual Analysis (Aggregated Features)

![Residual Distribution (Aggregated)](https://github.com/natwonglakhon/House_price_modelling/blob/main/images/residual_distribution_agg.png?raw=true)

| Model | Mean Residual | Std of Residuals |
|-------|--------------|-----------------|
| Linear Regression | +$713 | $61,143 |
| Tuned Random Forest | -$868 | $44,365 |
| XGBoost | -$527 | $41,244 |

An interesting pattern emerges: all models' mean residuals increased in magnitude despite improvements in $R^2$. However, all standard deviations decreased. This reflects a trade-off: the aggregated features make models more reliable on average (smaller spread of errors) while slightly shifting the mean. XGBoost retains the best overall profile with the smallest mean residual (-\\$527) and the tightest standard deviation (\\$41,244).

#### Practical Implications

The final XGBoost model achieves an $R^2$ of 82.0%, making it the strongest candidate for an automated valuation system. The feature engineering results demonstrate that incorporating domain knowledge can provide meaningful gains beyond simply increasing model complexity.

While this model is not intended to replace professional appraisals, it could serve as a first-pass valuation tool for:

 - estimating listing prices on real estate platforms,
 - supporting mortgage pre-assessments,
 - identifying potentially underpriced properties for investment analysis,
 - providing rapid price estimates for large property portfolios.

---

## V. Conclusion and Discussion

In this analysis, three models were built and compared: Linear Regression, Random Forest, and XGBoost, to predict median house prices in California. The data was first cleaned by filtering out extreme values near the $500,000 cap, a known artefact in this dataset, and filling missing values in `total_bedrooms` with the column median.

Linear Regression achieved an $R^2$ score of 60.4%, which serves as a reasonable baseline. Adding aggregated features such as rooms per household and population per household did not improve this score meaningfully, as the underlying relationships between features and house price are non-linear, which a linear model cannot learn regardless of the features provided. Switching to Random Forest and XGBoost raised $R^2$ significantly to 78.4% and 80.2% respectively, as these models can capture non-linear relationships naturally. Adding geographical features such as distances to the nearest coastline and major cities pushed XGBoost further to 82.0%, confirming that location encodes price signals that latitude and longitude alone do not fully capture for any of the three models.

Residual analysis revealed that all three models produce residuals centred around zero with a roughly normal distribution, which is a healthy sign. However, plotting absolute errors geographically showed that the worst predictions are concentrated around major cities and coastal areas, where prices are more volatile and harder to predict from the available features alone. This finding directly motivated the addition of the distance features.

Looking inside the models via feature importance, median income remains the strongest single predictor by a significant margin, while geographical features collectively play a crucial supporting role. This is intuitive: neighbourhood and proximity to economic centres strongly shape property values.

One important trade-off emerged from the residual analysis. Although adding aggregated features improved $R^2$ (i.e., reduced the standard deviation of errors), the mean errors increased slightly across all models. **This reflects a trade-off between bias and variance: a model can be more reliable on average while being slightly less accurate in expectation.** Which side of this trade-off to favour is ultimately a business decision. For example, a bank assessing mortgage applications may prioritise reducing worst-case valuation errors to manage financial risk, while a real estate platform may prefer a model with lower average error to improve overall pricing consistency across thousands of listings. Evaluating machine learning models therefore requires considering both predictive performance and the operational context in which the model will be deployed.
