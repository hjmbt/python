# coding:utf-8

import exercise_numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LinearRegression
import pylab as pl

seaborn.set()

np.random.seed(0)
X = np.random.random(size=(20, 1))
y = 3 * X.squeeze() + 2 + np.random.randn(20)

plt.plot(X.squeeze(), y, 'o')
# plt.show()

model = LinearRegression()
model.fit(X, y)
x_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(x_fit)

plt.plot(X.squeeze(), y, 'o')
plt.plot(x_fit.squeeze(), y_fit)
plt.show()
