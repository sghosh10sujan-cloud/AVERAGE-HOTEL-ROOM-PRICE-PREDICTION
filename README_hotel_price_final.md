## Hotel Room Price Prediction: A Regression-based Approach

This project involves data exploration and regression modeling to predict the average price per room for hotel bookings using a tabular dataset of 36K+ records. The objective is to identify the booking and property characteristics that drive room price, and build a model that predicts price accurately.

### Files

* `FINAL_RIDGE_LASSO_INCLUDED_AVERAGE_ROOM_PRICE_PREDICTION_PROJECT_2.ipynb` — Jupyter Notebook containing the entire workflow:

  * Data cleaning
  * Statistical feature selection
  * Multicollinearity screening
  * Model training and evaluation

* `hotel_bookings.csv` — The dataset used for analysis (expected to be in the same directory)

NOTE: The dataset was split into train/test before any feature-selection diagnostics were run, to prevent test-set information from leaking into feature engineering decisions.

---

### Features

* **EDA** using pandas, matplotlib, seaborn to explore booking patterns and price distributions across 36K+ records.
* **Statistical feature selection** using Pearson correlation (numeric features, threshold |r| ≥ 0.05) and chi-square tests (categorical features), computed on the training set only.
* **Multicollinearity screening** using VIF, iteratively dropping high-VIF features (threshold > 10) from the training set.
* **Target-leakage detection**, identifying and removing a price-derived binning column that had leaked into the feature set.
* **Feature scaling** using StandardScaler, fit on train and applied to test.
* **Model training** using regression models:

  * Linear Regression
  * Ridge Regression
  * Lasso Regression
  * Decision Tree
  * Random Forest
  * XGBoost
* **Hyperparameter tuning** via RidgeCV, LassoCV, and cross-validation.
* **Model evaluation** using RMSE and R².

---

### Results

The notebook evaluates six regression models after statistical feature selection and multicollinearity screening. The final model (XGBoost) achieves the best fit with an RMSE of 18.38 and R² of 0.675, outperforming the regularized linear models (Ridge/Lasso RMSE ≈ 24.24, R² ≈ 0.434).

---

📌 Author

Sujan Ghosh
