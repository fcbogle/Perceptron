#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 07:17:34 2024

@author: frankbogle
"""
from matplotlib import pyplot as plt
from sklearn import pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from ucimlrepo import fetch_ucirepo

#fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id = 17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

print("X of type: ", type(X), " description: ", X.describe())
print("y of type: ", type(y), " description: ", y.describe())

# flatten target variable to 1d array
y = y.values.ravel()

# create training and test datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create pipeline with scaling and cross-validated logistic regression
model_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(class_weight = 'balanced', cv = 5, scoring = 'accuracy',
                         max_iter = 1000, random_state = 42)
)

# Train the model
model_pipeline.fit(x_train, y_train)

# Print target variable class numbers using numpy array syntax
unique_classes, class_counts = np.unique(y_test, return_counts=True)
print("Unique classes:", unique_classes)
print("Class counts:", class_counts)


# use trained model for predictions or evaluations
y_predict = model_pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.2f}")

# create confusion matrix to view results
confusion_m = confusion_matrix(y_test, y_predict)

# Use Seaborn heatmap to plot
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_m, annot = True, fmt = "d", cmap = "Blues", cbar = False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Logistic Regression Confusion Matrix")
plt.show()


