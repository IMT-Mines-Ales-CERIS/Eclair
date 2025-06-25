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

train, test = Utils.Kfold(x, 6)

print(x[train[0], :]) # Get data according to indices for that split.

X = np.array([[1., 0.], [2., 1.], [0., 0.]])
y = np.array([0, 1, 2])

X_shuffled, y_shuffled = shuffle(X, y, random_state=42) # type: ignore

print(train)
print(test)

a = np.array([1, 2, 3])
b = a.copy()

b[0] = 999

print(a)  # ➜ [1 2 3]
print(b)  # ➜ [999 2 3]

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import KBinsDiscretizer
# import numpy as np

# Charger les données
data = fetch_california_housing()
X, y_continuous = data.data, data.target  # type: ignore , y = prix moyen
# Discrétiser y en 3 classes équilibrées (0, 1, 2) selon la distribution
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
y = discretizer.fit_transform(y_continuous.reshape(-1, 1)).astype(int).ravel()

print("X shape:", X.shape)      # (20640, 8)
print("y shape:", y.shape)      # (20640,)
print("y classes:", np.unique(y, return_counts=True))
