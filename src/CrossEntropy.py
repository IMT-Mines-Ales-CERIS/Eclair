import numpy as np

from scipy.stats import entropy

from ICustomClassifier import ICustomClassifier
from IEclair import IEclair
from Utils import Utils


class CrossEntropy(IEclair):

    def __init__(self,
        X_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        X_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int,], np.dtype[np.int64]], # Shape (n_samples,).
        posterior_probabilities: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        custom_classifier: ICustomClassifier,
        minimum_occurrence_nb_per_class: int, # Minimum number of occurrences of a class.
        threshold1: float, # Maximum entropy not to be exceeded when the chosen label is identical to the real label.
        threshold2: float, # Maximum entropy not to be exceeded when the chosen label is not identical to the real label.
        entropy_base: int # The logarithmic base to use to entropy computation.
    ):
        IEclair.__init__(self, 
            X_train,
            X_test,
            y_train,
            posterior_probabilities,
            custom_classifier,
            minimum_occurrence_nb_per_class
        )
        self._threshold1 = threshold1
        self._threshold2 = threshold2
        self._entropy_base = entropy_base


    # ---------------------------------------------------------------------------- #
    #                               Getters & Setters                              #
    # ---------------------------------------------------------------------------- #
    
    @property
    def threshold1(self): return self._threshold1

    @property
    def threshold2(self): return self._threshold2

    @property
    def entropy_base(self): return self._entropy_base


    # ---------------------------------------------------------------------------- #
    #                                Private methods                               #
    # ---------------------------------------------------------------------------- #

    def _EntropyBasedSubsetReduction(self,
        sample_probabilities: np.ndarray[tuple[int,], np.dtype[np.float64]], # Posterior probabilities for one sample.
        threshold: float # Threshold to apply class grouping.
    ) -> list[int]:
        """Get a subset of classes according to their probabilities (higher probabilities) in order to reduce entropy.

        ### Returns :
            * Set of the labels assigned to the sample.
        """
        y: list[float] = [] # Keep the sum of probabilities equal.
        z: list[float] = [] # Use to get the higher remaining probability.
        for i in range(len(sample_probabilities)):
            z.append(sample_probabilities[i])
            y.append(sample_probabilities[i])
        # Classes to remove
        kept_classes = []
        # Compute entropy of the sample.
        sample_entropy = entropy(y, base = self._entropy_base)
        # Get index of the higher probability.
        max_value_index = np.argmax(sample_probabilities)
        # Keep that class (=index).
        kept_classes.append(max_value_index)
        # Keep the sum of probabilities.
        probability_sum = sample_probabilities[max_value_index]
        # Reset probability of that index in z.
        z[max_value_index] = 0.0
        # Delete that probability to y.
        del y[max_value_index]
        # Add the sum.
        y.append(probability_sum)
        # The entropy has to be lower than the threshold.
        while len(y) > 1 and sample_entropy > threshold:
            # Get higher value index.
            max_value_index = np.argmax(z)
            # Update the sum.
            probability_sum += sample_probabilities[max_value_index]
            # Keep the new higher class according to their probability.
            kept_classes.append(max_value_index)
            # Update y.
            y.remove(sample_probabilities[max_value_index])
            # Delete the old sum.
            y = y[:-1]
            # Add the new sum.
            y.append(probability_sum)
            # Recompute the entropy based on the updated y.
            sample_entropy = entropy(y, base = self._entropy_base)
            # Update z.
            z[max_value_index] = 0.0
        return kept_classes
    

    # ---------------------------------------------------------------------------- #
    #                                Public methods                                #
    # ---------------------------------------------------------------------------- #

    def Relabelling(self, **kwargs) -> np.ndarray[tuple[int,], np.dtype[np.int64]]:
        """Relabelling based on entropy computation.

        ### Returns :
            * New classes on each sample of the training dataset, (2^nb_classes - 1) possible classes.
        """        
        
        new_y: list[int] = [] # New labels: subset in natural numbers.

        for i in range(self.nb_training_samples): # Relabelling of each training sample.
            sample = self.posterior_probabilities[i] # Get posterior probabilities of the current sample.
            # Compute entropy.
            sample_entropy = entropy(sample, base = self._entropy_base)

            # If the label with the highest probability is identical to the real label.
            if np.argmax(sample) == self.y_train[i]: 
                # If the entropy is too high, apply a reduction subset.
                if sample_entropy > self.threshold1:
                    # Return subset of classes according to their probabilities and entropy of the vector.
                    classes_subset = self._EntropyBasedSubsetReduction(sample, self.threshold1)
                    # Check for each class if it is present (True or False for each of them).
                    y_b2d = Utils.BinaryToInteger([k in classes_subset for k in range(self.nb_classes)])
                else:
                    y_b2d = Utils.BinaryToInteger([k == self.y_train[i] for k in range(self.nb_classes)])
            # If the label with the highest probability is not identical to the real label.
            else:
                if sample_entropy > self.threshold2:
                    classes_subset = self._EntropyBasedSubsetReduction(sample, self.threshold2)
                    # Add real label, it could appear two times without risk.
                    classes_subset.append(self.y_train[i])
                    y_b2d = Utils.BinaryToInteger([k in classes_subset for k in range(self.nb_classes)])
                # Overconfidence: he is wrong and he has no doubts.
                else:
                    y_b2d = 2**self.nb_classes - 1 # Ignorance.
            new_y.append(y_b2d)
            
        # Get all new labels and count them.
        distinct_labels, label_count = np.unique(new_y, return_counts=True)
        # Get y labels to modify according to their count (minimal number of occurrences).
        y_to_modify = [distinct_labels[i] for i in range(len(distinct_labels)) if (label_count[i] < self._minimum_occurrence_nb_per_class)]
        # Remove or modify some new labels.
        for i in range(len(new_y)):
            # If the label is in y_to_modify.
            if new_y[i] in y_to_modify:
                # Keep the original class.
                new_y[i] = 2**self.y_train[i]
            
        # Case when there is not much data.
        distinct_labels, label_count = np.unique(new_y, return_counts=True)
        if(label_count[np.argmin(label_count)] < self._minimum_occurrence_nb_per_class):
            # Cancel relabelling and keep the original class.
            print("Warning: Can't apply relabelling")
            new_y = []
            for i in range(self.nb_training_samples):
                new_y.append(2**self.y_train[i])
    
        return np.array(new_y)