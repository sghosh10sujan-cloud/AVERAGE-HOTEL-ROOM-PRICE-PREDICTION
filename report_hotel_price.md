Project Report: Hotel Room Price Prediction

Problem Statement
Hotels need to price rooms accurately to stay competitive while maximizing revenue — underpricing leaves money on the table, while overpricing costs bookings to competitors. This project aims to build a regression model that predicts the average price per room for a booking based on booking and property characteristics.

Project Overview
A machine learning regression model was developed to analyze booking-level features and predict average room price. The project included a full diagnostic pipeline — correct train/test separation, statistical feature selection, multicollinearity screening, and target-leakage detection — to ensure the final model's performance is genuine rather than an artifact of a flawed process.

Dataset Summary
The dataset includes 36,000+ hotel booking records with features including:
• no_of_adults, no_of_weekend_nights, no_of_week_nights
• arrival_year, arrival_date
• Room type and booking-related categorical features
• avg_price_per_room (target)

The dataset was split into training and test sets before any feature-selection diagnostics were performed, specifically to prevent test-set information from leaking into feature engineering decisions.

Exploratory Data Analysis (EDA Highlights)
• Pearson correlation on the training set (threshold |r| ≥ 0.05) identified no_of_weekend_nights, no_of_week_nights, and arrival_date as weakly related to price, and they were dropped.
• Chi-square tests confirmed all categorical features had a statistically significant relationship with price and were retained.
• VIF screening on the training set flagged severe multicollinearity in arrival_year (VIF = 298) and no_of_adults (VIF = 17.7); both were removed, bringing the maximum VIF in the final feature set down to roughly 7.
• A price-derived binning column was identified as a target-leakage risk and removed before modeling.

Model Development
Six regression models were trained and compared:
• Linear Regression
• Ridge Regression
• Lasso Regression
• Decision Tree
• Random Forest
• XGBoost

Ridge and Lasso were included specifically to test whether regularization could correct for any residual multicollinearity the VIF step may have missed. Both produced results nearly identical to standard Linear Regression, and Lasso's cross-validated alpha shrank no coefficient to zero — confirming the feature set entering the models was already well-conditioned. StandardScaler was fit on the training set and applied to the test set.

XGBoost delivered the best fit:
• RMSE: 18.38
• R²: 0.675

This outperformed the regularized linear models (Ridge/Lasso RMSE ≈ 24.24, R² ≈ 0.434).

Evaluation Metric
RMSE and R² were used to evaluate model accuracy. RMSE was prioritized as the primary metric since it is in the same unit as price and directly reflects the typical prediction error a hotel would see in practice, while R² provided a normalized measure of how much price variance the model explained.

Challenges
• Guarding against target leakage required tracing back a price-derived column that had been engineered from the target itself.
• Multicollinearity was severe enough (VIF = 298) that a naive linear model would have produced unstable, misleading coefficients if fit without screening.
• Balancing feature selection thoroughness (three independent methods: Pearson, chi-square, VIF) against the risk of removing genuinely useful predictors.

Impact
This tool can:
• Be integrated into a hotel's pricing or revenue management workflow to suggest a data-driven room price
• Help revenue managers benchmark actual pricing against a statistically grounded estimate
• Serve as a template for building leakage-checked, multicollinearity-screened regression models

Tech Stack
• Python
• Scikit-learn
• XGBoost
• Statsmodels (VIF)
• Pandas, NumPy
• Matplotlib, Seaborn

Future Work
• Add SHAP-based explainability to interpret individual price predictions
• Explore feature engineering around seasonality (e.g., cyclic encoding of arrival_date) now that raw arrival_date has been dropped
• Test ensembling XGBoost with Random Forest for a further accuracy gain
• Deploy the model behind a simple interface for interactive price estimation
