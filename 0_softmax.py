"""Softmax."""

import numpy as np
import math

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sum = np.sum(np.exp(x), 0)
    return np.exp(x)/sum


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
# create an array with values from -2 to 6 with step 0.1
x = np.arange(-2.0, 6.0, 0.1)
# add 3 vectors creating 3 row matrix
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()