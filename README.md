# Task 3 — Linear Regression (House Price Prediction)
**Elevate Labs — AI & ML Internship**

## Objective
Implement and understand simple & multiple linear regression by predicting house prices.

## Tools Used
- Python
- Scikit-learn
- Pandas
- Matplotlib / Seaborn

## Dataset
`USA_Housing.csv` — 5000 rows, 7 columns (Avg. Area Income, Avg. Area House Age, Avg. Area Number of Rooms, Avg. Area Number of Bedrooms, Area Population, Price, Address). No missing values. The `Address` column was dropped since it's free text and not usable as a numeric predictor.

## Project Structure
```
house-price-regression/
├── data/
│   └── USA_Housing.csv
├── plots/
│   ├── 01_actual_vs_predicted.png
│   ├── 02_residuals_distribution.png
│   ├── 03_residuals_vs_predicted.png
│   ├── 04_coefficients_bar.png
│   ├── 05_simple_regression_line.png
│   ├── 06_correlation_heatmap.png
│   └── coefficients.csv
├── regression.py
└── README.md
```

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python regression.py
```
Prints preprocessing steps and evaluation metrics to the console, and saves all charts to `plots/`.

## Approach
1. **Preprocess**: loaded the CSV, checked for missing values (none found), dropped the non-numeric `Address` column.
2. **Split**: 80/20 train-test split (`random_state=42` for reproducibility).
3. **Multiple Linear Regression**: fit `sklearn.linear_model.LinearRegression` on all 5 numeric features to predict `Price`.
4. **Evaluation**: scored on the test set using MAE, MSE, RMSE, and R².
5. **Simple Linear Regression**: refit using only the single strongest predictor, to compare against the multiple-feature model and plot a 2D regression line.
6. **Diagnostics**: residual plots and a correlation heatmap to check regression assumptions.

## Results

**Multiple Linear Regression (all 5 features)**

| Metric | Value |
|---|---|
| MAE | 80,879 |
| MSE | 10,089,009,301 |
| RMSE | 100,444 |
| R² | 0.918 |

The model explains about **91.8%** of the variance in house price on unseen test data — a strong fit.

**Coefficients** (effect on Price, holding other features constant):

| Feature | Coefficient |
|---|---|
| Avg. Area House Age | 164,666 |
| Avg. Area Number of Rooms | 119,624 |
| Avg. Area Number of Bedrooms | 2,440 |
| Avg. Area Income | 21.65 |
| Area Population | 15.27 |

Interpretation: a one-unit increase in `Avg. Area House Age` is associated with a **$164,666** rise in predicted price, all else equal — the single biggest swing factor per unit change, even though income has more day-to-day variation. `Avg. Area Number of Bedrooms` has a comparatively small, near-negligible effect once `Number of Rooms` is already in the model.

**Simple Linear Regression** (single feature: `Avg. Area House Age`)
- R² = 0.216, MAE = 248,024

Using just one feature explains far less of the variance (21.6% vs. 91.8%), showing that price genuinely depends on multiple factors together — this is why the multiple regression model is the better fit.

**Residual diagnostics**: residuals are roughly normally distributed and centered at zero with no strong funnel or curve pattern against predicted values, which supports the linear regression assumptions of linearity, homoscedasticity, and normally distributed errors.

**Multicollinearity check**: the correlation heatmap shows the five predictor features are only weakly correlated with each other, so multicollinearity is not a significant concern for this model.

---
*Submitted as part of the Elevate Labs AI & ML Internship (MSME, Govt. of India).*
