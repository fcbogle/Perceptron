#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 07:17:34 2024

@author: frankbogle
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from WisGDAdaline import WisGDAdaline

#fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id = 17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets
print("y.shape 1", y.shape)

# create smaller dataset
X = X.iloc[0:100, :].values
y = y.iloc[0:100].values.ravel()

# convert to binary format
y = np.where(y == 'B', 0, 1)
print("y.shape 2", y.shape)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

wisAda1 = WisGDAdaline(n_iter=15, eta=0.1).fit(X, y)

ax[0].plot(range(1, len(wisAda1.losses_) + 1), np.log10(wisAda1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title('Adaline - Learning rate 0.1')

wisAda2 = WisGDAdaline(n_iter=15, eta=0.01).fit(X, y)
ax[1].plot(range(1, len(wisAda2.losses_) + 1), wisAda2.losses_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean squared error')
ax[1].set_title('Adaline - Learning rate 0.01')

# plt.savefig('images/02_11.png', dpi=300)
plt.show()
