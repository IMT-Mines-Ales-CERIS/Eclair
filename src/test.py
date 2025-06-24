import numpy as np

from sklearn.utils import shuffle

from Utils import Utils

x = np.array([
    [0.1, 0.11, 0.12, 0.13],
    [0.21, 0.22, 0.23, 0.24],
    [0.7, 0.69, 0.68, 0.67],
    [0.97, 0.96, 0.95, 0.94],
    [0.99, 0.99, 0.99, 0.99],
    [0.01, 0.02, 0.03, 0.04]
])

train, test = Utils.Kfold(x, 2)

print(x[train[0], :]) # Get data according to indices for that split.

X = np.array([[1., 0.], [2., 1.], [0., 0.]])
y = np.array([0, 1, 2])

X_shuffled, y_shuffled = shuffle(X, y, random_state=42) # type: ignore

# print(train)
# print(test)

