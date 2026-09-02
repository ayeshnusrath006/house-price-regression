"""
Task 3: Linear Regression - House Price Prediction
Elevate Labs AI & ML Internship

Run with: python regression.py
Prints preprocessing/evaluation info to console and saves all plots to plots/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ----------------------------------------------------------------------
# 1. Load and preprocess data
# ----------------------------------------------------------------------
df = pd.read_csv("data/USA_Housing.csv")
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Drop the free-text Address column - not usable as a numeric predictor
df = df.drop(columns=["Address"])

X = df.drop(columns=["Price"])
y = df["Price"]

print("\nFeatures used:", list(X.columns))

# ----------------------------------------------------------------------
# 2. Train-test split
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 3. Fit Multiple Linear Regression model
# ----------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------------------------------------------------
# 4. Evaluate: MAE, MSE, RMSE, R^2
# ----------------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MULTIPLE LINEAR REGRESSION - EVALUATION")
print("=" * 50)
print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R^2  : {r2:.4f}")

# ----------------------------------------------------------------------
# 5. Interpret coefficients
# ----------------------------------------------------------------------
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
coef_df.loc[len(coef_df)] = ["Intercept", model.intercept_]
print("\nCoefficients:\n", coef_df.to_string(index=False))
coef_df.to_csv("plots/coefficients.csv", index=False)

# ----------------------------------------------------------------------
# 6. Plot: Actual vs Predicted (multiple regression)
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression: Actual vs Predicted Price")
plt.tight_layout()
plt.savefig("plots/01_actual_vs_predicted.png")
plt.close()

# ----------------------------------------------------------------------
# 7. Residuals plot
# ----------------------------------------------------------------------
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
sns.histplot(residuals, kde=True, color="salmon")
plt.title("Distribution of Residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig("plots/02_residuals_distribution.png")
plt.close()

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.4, color="darkorange")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig("plots/03_residuals_vs_predicted.png")
plt.close()

# ----------------------------------------------------------------------
# 8. Coefficient bar chart
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plot_df = coef_df[coef_df["Feature"] != "Intercept"]
sns.barplot(data=plot_df, x="Coefficient", y="Feature", palette="viridis")
plt.title("Feature Coefficients (Multiple Linear Regression)")
plt.tight_layout()
plt.savefig("plots/04_coefficients_bar.png")
plt.close()

# ----------------------------------------------------------------------
# 9. Simple Linear Regression (single strongest feature) + regression line
# ----------------------------------------------------------------------
best_feature = plot_df.iloc[0]["Feature"]
print(f"\nStrongest single predictor by |coefficient|: {best_feature}")

X_simple = df[[best_feature]]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)
y_pred_s = simple_model.predict(X_test_s)

r2_simple = r2_score(y_test_s, y_pred_s)
mae_simple = mean_absolute_error(y_test_s, y_pred_s)
print(f"Simple Linear Regression ({best_feature}) -> R^2: {r2_simple:.4f}, MAE: {mae_simple:,.2f}")

plt.figure(figsize=(7, 6))
plt.scatter(X_test_s, y_test_s, alpha=0.4, color="steelblue", label="Actual")
order = np.argsort(X_test_s[best_feature].values)
plt.plot(
    X_test_s[best_feature].values[order],
    y_pred_s[order],
    color="red",
    linewidth=2,
    label="Regression line",
)
plt.xlabel(best_feature)
plt.ylabel("Price")
plt.title(f"Simple Linear Regression: Price vs {best_feature}")
plt.legend()
plt.tight_layout()
plt.savefig("plots/05_simple_regression_line.png")
plt.close()

# ----------------------------------------------------------------------
# 10. Correlation heatmap (for multicollinearity check)
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix (Multicollinearity Check)")
plt.tight_layout()
plt.savefig("plots/06_correlation_heatmap.png")
plt.close()

print("\nAll plots saved to plots/")
