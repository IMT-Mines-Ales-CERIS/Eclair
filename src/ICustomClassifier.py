import numpy as np

from abc import ABC, abstractmethod

                                                                                                                                 
class ICustomClassifier(ABC):

    @abstractmethod
    def PredictProba(self,
        X_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        X_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int,], np.dtype[np.int64]],
        **kwargs
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]: # Shape (n_sample, n_classes).
        """Get posterior probabilities for each classes.
        """
        pass