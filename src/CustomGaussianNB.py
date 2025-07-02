import numpy as np

from sklearn.naive_bayes import GaussianNB

from ICustomClassifier import ICustomClassifier

class CustomNaiveBayesClassifier(ICustomClassifier):

    def PredictProba(self,
        X_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        X_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int,], np.dtype[np.int64]],
        **kwargs
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]: # Shape (n_sample, n_classes).
        """Predict posterior probabilities on X_test based on X_train fitting parameters.
        
        ### Parameters :
            * ``X_train`` - Shape, (n_train_samples, n_features).
            * ``X_test`` - Shape, (n_test_samples, n_features).
            * ``y_train`` - Shape, (n_samples,).

        ### Returns :
            * Probabilities for each class by sample.
        """
        gnb = GaussianNB()
        # The columns correspond to the classes in sorted order, as they appear in the attribute classes_.
        return np.round(gnb.fit(X_train, y_train).predict_proba(X_test), 3).astype(np.float64)
