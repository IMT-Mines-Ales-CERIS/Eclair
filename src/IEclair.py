import numpy as np

from abc import ABC, abstractmethod
from ICustomClassifier import ICustomClassifier


class IEclair(ABC):

    def __init__(self,
        X_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        X_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int,], np.dtype[np.int64]], # Shape (n_samples,).
        posterior_probabilities: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        custom_classifier: ICustomClassifier,
        minimum_occurrence_nb_per_class: int, # Minimum number of occurrences of a class.
    ):
        """Relabeled X_train data and train a classifier on this new data to obtain masses on all new classes on the test set.

        ### Parameters :
            * ``X_train`` - Training dataset, shape, (n_samples, n_features).
            * ``X_test`` - Test dataset, shape, (n_samples, n_features).
            * ``y_train`` - Numerical labels of the training dataset [0,1,2,...], shape (n_samples,).
            * ``posterior_probabilities`` - Posterior probabilites on the training dataset, shape (n_samples, n_classes).
            * ``custom_classifier`` - Classifier to get masses on new classes on the test set.
            * ``minimum_occurrence_nb_per_class``: Minimum number of occurrences to keep a new class.

        Example of a posterior probabilities on training dataset:

        <table>
            <tr><td></td><td>P(a|x)</td><td>P(b|x)</td><td>P(c|x)</td></tr>
            <tr><td>s_1</td><td>0.9</td><td>0.05</td><td>0.05</td></tr>
            <tr><td>s_2</td><td>0.3</td><td>0.4</td><td>0.3</td></tr>
            <tr><td>s_3</td><td>0.1</td><td>0.2</td><td>0.7</td></tr>
        </table>

        """

        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._posterior_probabilities = posterior_probabilities
        self._custom_classifier = custom_classifier
        self._minimum_occurrence_nb_per_class = minimum_occurrence_nb_per_class
        self._nb_classes = len(np.unique(self._y_train)) # Number of classes in data.


    # ---------------------------------------------------------------------------- #
    #                               Getters & Setters                              #
    # ---------------------------------------------------------------------------- #

    @property
    def X_train(self): return self._X_train

    @property
    def X_test(self): return self._X_test
    
    @property
    def y_train(self): return self._y_train

    @property
    def posterior_probabilities(self): return self._posterior_probabilities

    @property
    def custom_classifier(self): return self._custom_classifier

    @property
    def minimum_occurrence_nb_per_class(self): return self._minimum_occurrence_nb_per_class

    @property
    def nb_classes(self): return self._nb_classes

    @property
    def nb_training_samples(self) -> int:
        """Number of samples in the training dataset.
        """
        return self.X_train.shape[0]
    

    # ---------------------------------------------------------------------------- #
    #                               Abstract methods                               #
    # ---------------------------------------------------------------------------- #
    
    @abstractmethod
    def Relabelling(self, **kwargs) -> np.ndarray[tuple[int,], np.dtype[np.int64]]:
        """Get imprecise labels on training dataset.

        ### Returns :
            * ``list`` - New classes on each sample of the training dataset, (2^nb_classes - 1) possible classes.
        """
        pass
    
    
    # ---------------------------------------------------------------------------- #
    #                                Public methods                                #
    # ---------------------------------------------------------------------------- #

    def Predict(self, **kwargs) -> tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]],
            np.ndarray[tuple[int,], np.dtype[np.int64]]
        ]:
        """Get masses on new classes for each X_test sample.

        ### Returns :
            * ``list`` - Masses for each new classes on test dataset.
            * ``list`` - New classes on each sample of the training dataset, (2^nb_classes - 1) possible classes.
        """
        
        # Get new y labels.
        new_y = self.Relabelling()
        # Get masses on the new y labels.
        masses = self.custom_classifier.PredictProba(self.X_train, self.X_test, new_y)        
        return masses, new_y
