import numpy as np
import pandas as pd
import code

df = pd.read_csv("/Users/frankbogle/Downloads/HousingData.csv", header = 0)
print(df.shape)
print(df.head())

# check number of NaN values in dataframe
print(df.isna().sum())

# replace NaN with mean values for each independent variable
df.fillna(df.mean(), inplace=True)

# check number of NaN values expecting none
print(df.isna().sum())
print(df.shape)
print(df.head())

# capture the feature names
feature_names = df.drop('MEDV', axis=1).columns.tolist()

# Prepare the cleansed data for matrix transposition
X = df.drop('MEDV', axis=1).values
y = df['MEDV'].values.reshape(-1, 1)

# Add intercept into X matrix
X = np.c_[np.ones(X.shape[0]), X]

# Ridge regularization parameter (try tuning this)
alpha = 0.01

# Perform matrix algebra with Ridge regularization
X_transpose = X.T
XtX = X_transpose @ X
ridge_penalty = alpha * np.identity(X.shape[1])

# Regularized normal equation
XtX_inv = np.linalg.inv(XtX + ridge_penalty)
beta = XtX_inv @ (X_transpose @ y)

# Display coefficients
intercept = beta[0][0]
coefficients = beta[1:].flatten()

print(f"\nIntercept: {intercept:.4f}\n")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")

# Make predictions using the model
y_pred = X @ beta

# Show first 10 predictions vs actual values
results = pd.DataFrame({
    'Actual': y.flatten(),
    'Predicted': y_pred.flatten()
})

print("\nPredictions vs Actual Values:")
print(results.head(10))

# calculate R squared error
# Sum of squared residuals (SSR)
ss_res = np.sum((y - y_pred) ** 2)

# Total sum of squares (SST)
ss_tot = np.sum((y - np.mean(y)) ** 2)

# R² calculation
r_squared = 1 - (ss_res / ss_tot)
print(f"R²: {r_squared:.4f}")

# code.interact(local=globals())

