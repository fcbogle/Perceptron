#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:22:54 2024

@author: frankbogle
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import dataset from URL
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header = None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())

# Create dependent & independent variables
X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values
# print(X)
# print(y)

# Apply L1 regularization LASSO
# Split data in training and test sets
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set penalty='l1' and solver='liblinear' for L1 support
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
logreg_l1.fit(X_train, y_train)

# Evaluate the model
y_pred = logreg_l1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# create array of coefficients
# coefficients = np.array([logreg_l1.coef_])
coefficients = logreg_l1.coef_
features = ['Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
# Convert to DataFrame for easier visualization
df = pd.DataFrame(coefficients.T, columns=['Class 1', 'Class 2', 'Class 3'], index=features)

# Plot the coefficients
df.plot(kind='bar', figsize=(10, 6))
plt.title("Logistic Regression Coefficients with L1 Regularization")
plt.ylabel("Coefficient Value")
plt.xlabel("Features")
plt.legend(title="Classes")
plt.tight_layout()
plt.show()

# Compute the absolute coefficients for reduction visualization
abs_coefficients = np.abs(coefficients)
non_zero_counts = (abs_coefficients > 0).sum(axis=1)

# Plot the reduction in importance
plt.figure(figsize=(10, 6))
for i in range(abs_coefficients.shape[0]):
    plt.bar(features, abs_coefficients[i], label=f'Class {i+1}', alpha=0.7)

plt.xticks(rotation=90)
plt.title("Reduction in Feature Importance with L1 Regularization")
plt.ylabel("Absolute Coefficient Value")
plt.xlabel("Features")
plt.legend(title="Classes")
plt.tight_layout()
plt.show()

# Display the count of non-zero coefficients for each class
print("Non-zero coefficient counts per class:")
for i, count in enumerate(non_zero_counts):
    print(f"Class {i+1}: {count} non-zero features")


# Print results
# print(f"Model Accuracy with L1 Regularization: {accuracy:.2f}")
# print("Coefficients with L1 Regularization:")
# print(logreg_l1.coef_)


