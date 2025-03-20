import codecs
import csv
import numpy as np
import os

from abc import ABC, abstractmethod
from multiprocessing import Pool
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from typing import Union

from NaiveBayes import NaiveBayesClassifier
from Utils import Utils
from SetValuedClassification import SetValuedClassification


#  .d8888b.                    888                           .d8888b.  888                            d8b  .d888 d8b                  
# d88P  Y88b                   888                          d88P  Y88b 888                            Y8P d88P"  Y8P                  
# 888    888                   888                          888    888 888                                888                         
# 888        888  888 .d8888b  888888 .d88b.  88888b.d88b.  888        888  8888b.  .d8888b  .d8888b  888 888888 888  .d88b.  888d888 
# 888        888  888 88K      888   d88""88b 888 "888 "88b 888        888     "88b 88K      88K      888 888    888 d8P  Y8b 888P"   
# 888    888 888  888 "Y8888b. 888   888  888 888  888  888 888    888 888 .d888888 "Y8888b. "Y8888b. 888 888    888 88888888 888     
# Y88b  d88P Y88b 888      X88 Y88b. Y88..88P 888  888  888 Y88b  d88P 888 888  888      X88      X88 888 888    888 Y8b.     888     
#  "Y8888P"   "Y88888  88888P'  "Y888 "Y88P"  888  888  888  "Y8888P"  888 "Y888888  88888P'  88888P' 888 888    888  "Y8888  888     
                                                                                                                                    
class ICustomClassifier(ABC):

    @abstractmethod
    def PredictProba(self):
        """Get posterior probabilities.
        """
        pass
    
    # TODO: A voir si on ne met pas aussi PredictProbaKFold (et peut être KFold en method de cette classe ?).


# TODO: hériter de NaiveBayesClassifier et modifier Predict et Fitting.

class CustomNaiveBayesClassifier(NaiveBayesClassifier, ICustomClassifier):

    def __init__(self,
        categorical_features,
        numerical_features 
    ):
        NaiveBayesClassifier.__init__(self, categorical_features, numerical_features)

    # @property TODO: Add property to attribute (understand before).

    def PredictProba(self, 
        X_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        X_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int,], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]: # Shape (n_sample, n_classes).
        """Predict posterior probabilities on X_test based on X_train fitting parameters.
        """
        # Get fitting parameters from training set.
        mu, sigma2, nc, classes, lev, freq, u = self.FitContinuousModel(X_train, y_train, 'gaussian')
        # Predict probabilities on X_test based on fitting parameters.
        *_, posterior_probabilities = self.Predict(
            X_test,
            mu,
            sigma2,
            nc,
            classes,
            lev,
            freq,
            u
        )
        return posterior_probabilities
    
    def PredictProbaKFold(self,
            X: np.ndarray[tuple[int, int], np.dtype[np.float64]], # Dataset, shape (n_samples, n_features).
            y: np.ndarray[tuple[int,], np.dtype[np.float64]], # Labels.
            nb_folds: int,
            test_size: float,
            random_state = Union[int, None] # For reproducible output across multiple function calls.
        ) -> tuple[
            list[np.ndarray[tuple[int, int], np.dtype[np.float64]]], # Posterior probabilities by class and by sample for all splits.
            list[np.ndarray[tuple[int, int], np.dtype[np.float64]]], # Posterior probabilities by class and by sample for all splits.
            list[np.ndarray[tuple[int,], np.dtype[np.int64]]]        # Shape (int,).
        ]:
        """Get posterior probabilities of calibration and test sets for all splits.

        ### Parameters :
            * ``posterior_probabilities`` - Posterior probabilities by class and by sample for all splits.
        """
        def _Classifier(
                X_train_indices: np.ndarray[tuple[int,], np.dtype[np.int64]], # The training set indices for one split.
                X_test_indices:  np.ndarray[tuple[int,], np.dtype[np.int64]]  # The testing set indices for one split.
            ) -> tuple[
                np.ndarray[tuple[int,], np.dtype[np.int64]],       # Classes on the test split for that split.
                np.ndarray[tuple[int, int], np.dtype[np.float64]], # Posterior probabilities by class and by sample for one split.
                np.ndarray[tuple[int, int], np.dtype[np.float64]]  # Posterior probabilities by class and by sample for one split.
            ]:
            """Get posterior probabilities from naive bayes classifier for one split.
            """
            X_train, X_calibration, y_train, y_calibration = train_test_split(
                X[X_train_indices, :], # Get data according to indices for that split.
                y[X_train_indices],    # Get labels according to indices for that split.
                test_size = test_size,
                random_state = random_state
            )
            
            mu, sigma2, nc, classes, lev, freq, u = self.FitContinuousModel(X_train, y_train, 'gaussian')
            
            *_, calibration_probabilities = self.Predict(
                X_calibration,
                mu,
                sigma2,
                nc,
                classes,
                lev,
                freq,
                u
            )
            *_, test_probabilities = self.Predict(
                X[X_test_indices, :],
                mu,
                sigma2,
                nc,
                classes,
                lev,
                freq,
                u
            )

            return test_probabilities, calibration_probabilities, y_calibration 
        
        X_train_kfold_indices, X_test_kfold_indices = Utils.Kfold(X, nb_folds)
        # Parallelization.
        with Pool() as pool:
            results = pool.starmap(
                _Classifier, [
                    (X_train_kfold_indices[k], X_test_kfold_indices[k])
                    for k in range(nb_folds)
                ]
            )
            test_probabilities = list(map(lambda item: item[0], results))
            calibration_probabilities = list(map(lambda item: item[1], results))
            y_calibration = list(map(lambda item: item[2], results))

        return test_probabilities, calibration_probabilities, y_calibration
    









# 8888888888         888          d8b         
# 888                888          Y8P         
# 888                888                      
# 8888888    .d8888b 888  8888b.  888 888d888 
# 888       d88P"    888     "88b 888 888P"   
# 888       888      888 .d888888 888 888     
# 888       Y88b.    888 888  888 888 888     
# 8888888888 "Y8888P 888 "Y888888 888 888     

class IEclair(ABC):

    @abstractmethod
    def Relabelling(self):
        """Get imprecise labels.
        """
        pass
    
    @abstractmethod
    def Classify(self):
        """Classify new samples.
        """
        pass


class Eclair:

    def __init__(self,
        minimum_occurrence_nb_per_class: int, # Minimum number of occurrences of a class.
        y: Union[np.ndarray[tuple[int,], np.dtype[np.int64]], None] = None # Shape (n_samples,).
    ):
        self._minimum_occurrence_nb_per_class = minimum_occurrence_nb_per_class
        self._y = y # Real labels.

    @property
    def minimum_occurrence_nb_per_class(self): return self._minimum_occurrence_nb_per_class

    @minimum_occurrence_nb_per_class.setter
    def minimum_occurrence_nb_per_class(self, value): self._minimum_occurrence_nb_per_class = value

    @property
    def y(self): return self._y

    @y.setter
    def y(self, value): self._y = value

    @property
    def nb_classes(self) -> int:
        """Number of classes in data.
        """
        if self._y is None:
            return 0
        return len(self._y)


    # ---------------------------------------------------------------------------- #
    #                                Public methods                                #
    # ---------------------------------------------------------------------------- #

    def LoadingLabels(self,
        path: str
    ):
        """Loading real labels from a file. One label per line.
        """
        if not os.path.isfile(path):
            return        
        y = []
        with codecs.open(path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                y.append(float(line))
            self._y = np.array(y)










#  .d8888b.                                    8888888888          888                                      
# d88P  Y88b                                   888                 888                                      
# 888    888                                   888                 888                                      
# 888        888d888 .d88b.  .d8888b  .d8888b  8888888    88888b.  888888 888d888 .d88b.  88888b.  888  888 
# 888        888P"  d88""88b 88K      88K      888        888 "88b 888    888P"  d88""88b 888 "88b 888  888 
# 888    888 888    888  888 "Y8888b. "Y8888b. 888        888  888 888    888    888  888 888  888 888  888 
# Y88b  d88P 888    Y88..88P      X88      X88 888        888  888 Y88b.  888    Y88..88P 888 d88P Y88b 888 
#  "Y8888P"  888     "Y88P"   88888P'  88888P' 8888888888 888  888  "Y888 888     "Y88P"  88888P"   "Y88888 
#                                                                                         888           888 
#                                                                                         888      Y8b d88P 
#                                                                                         888       "Y88P"  

class CrossEntropy(Eclair, IEclair):

    def __init__(self,
        minimum_occurrence_nb_per_class: int,
        entropy_base: int, # The logarithmic base to use to entropy computation.
        probabilities: Union[np.ndarray[tuple[int, int], np.dtype[np.float64]], None] = None, # Posterior probabilities, shape (n_samples, n_classes).
        y: Union[np.ndarray[tuple[int,], np.dtype[np.int64]], None] = None # Shape (n_samples,).
    ):
        Eclair.__init__(self, 
            minimum_occurrence_nb_per_class=minimum_occurrence_nb_per_class,
            y = y
        )
        self._entropy_base = entropy_base
        self._probabilities = probabilities # Posterior probabilities.


    # ---------------------------------------------------------------------------- #
    #                               Getters & Setters                              #
    # ---------------------------------------------------------------------------- #
    
    @property
    def probabilities(self): return self._probabilities

    @probabilities.setter
    def probabilities(self, value): self._probabilities = value

    @property
    def entropy_base(self): return self._entropy_base

    @entropy_base.setter
    def entropy_base(self, value): self._entropy_base = value
    
    @property
    def nb_samples(self) -> int:
        """Number of samples in data.
        """
        if self._probabilities is None:
            return 0
        return self._probabilities.shape[0]
    

    # ---------------------------------------------------------------------------- #
    #                                Private methods                               #
    # ---------------------------------------------------------------------------- #

    def _EntropyBasedSubsetReduction(self, 
        sample_probabilities: np.ndarray[tuple[int,], np.dtype[np.float64]], # Posterior probabilities for one sample.
        threshold: float # Threshold to apply class grouping.
    ) -> list[int]:
        """Get a subset of classes according to their probabilities (higher probabilities) in order to reduce entropy.
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
            # Update z.
            z[max_value_index] = 0.0
            # Update y.
            del y[max_value_index]
            # Delete the old sum.
            y = y[:-1]
            # Add the new sum.
            y.append(probability_sum)
            # Recompute the entropy based on the updated y.
            sample_entropy = entropy(y, base = self._entropy_base)
        return kept_classes
        

    # ---------------------------------------------------------------------------- #
    #                                Public methods                                #
    # ---------------------------------------------------------------------------- #

    def LoadingProbabilities(self,
        path: str, # Path of a csv file to load posterior probabilities.
        has_header: bool # Specify if csv file has header or not.
    ):
        """Load posterior probabilities from a csv file, shape (n_samples, n_classes).
        """

        if not os.path.isfile(path):
            return
        
        probabilities = []
        with codecs.open(path, 'r', encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter=',')
            if has_header: 
                next(reader)
            for line in reader:
                probabilities.append(np.array(line, dtype = np.float64))
        self._probabilities = probabilities

    def Relabelling(self,
        threshold1: float, # Maximum entropy not to be exceeded when the chosen label is identical to the real label.
        threshold2: float, # Maximum entropy not to be exceeded when the chosen label is not identical to the real label.
    ) -> list[int]:
        """Relabelling based on entropy computation.
        """
        
        new_y: list[int] = [] # New labels: subset in natural numbers.
        for i in range(self.nb_samples):
            sample = self._probabilities[i]
            # Compute entropy.
            sample_entropy = entropy(sample, base = self._entropy_base)
            # If the chosen label is identical to the real label.
            if np.argmax(sample) == self._y[i]: 
                # If the entropy is too high, apply a reduction subset.
                if sample_entropy > threshold1:
                    # Return subset of classes according to their probabilities and entropy of the vector.
                    classes_subset = self._EntropyBasedSubsetReduction(sample, threshold1)
                    # Check for each class if it is present (True or False for each of them).
                    y_b2d = Utils.BinaryToInteger([k in classes_subset for k in range(self.nb_classes)])
                else:
                    y_b2d = Utils.BinaryToInteger([k in [self._y[i]] for k in range(self.nb_classes)])
            # If the chosen label is not identical to the real label.
            else:
                if sample_entropy > threshold2:
                    classes_subset = self._EntropyBasedSubsetReduction(sample, threshold2)
                    # Add real label, it could appear two times without risk.
                    classes_subset.append(self._y[i])
                    y_b2d = Utils.BinaryToInteger([k in classes_subset for k in range(self.nb_classes)])
                # He is wrong and he has no doubts.
                else:
                    y_b2d = 2**self.nb_classes - 1 # TODO: pourquoi le -1 ? car de mon côté j'ai normalisé les classes de 0 à n-1
            new_y.append(y_b2d)
            
        # Get all labels and count them.
        distinct_labels, label_count = np.unique(new_y, return_counts=True)
        # Get y labels to modify according to their count (minimal number of occurrences).
        y_to_modify = [distinct_labels[i] for i in range(len(distinct_labels)) if (label_count[i] < self._minimum_occurrence_nb_per_class)]
        # Remove or modify some new labels.
        for i in range(len(new_y)):
            # If the label is in y_to_modify.
            if new_y[i] in y_to_modify:
                # Keep the original class.
                new_y[i] = 2**self._y[i] # TODO: la question se pose ici également, c'est normal qu'il n'y est pas de -1 alors pourquoi au dessus ?
            
        # Case when there is not much data.
        distinct_labels, label_count = np.unique(new_y, return_counts=True)
        if(label_count[np.argmin(label_count)] < self._minimum_occurrence_nb_per_class):
            # Cancel relabelling and keep the original class.
            print("Warning: Can't apply relabelling")
            new_y = []
            for i in range(self.nb_samples):
                new_y.append(2**self._y[i])
    
        return new_y
    
    def Classify(self,
        X_train,
        X_calibration,
        y_calibration,
        Classifier: ICustomClassifier,
        threshold_entropy_space: np.ndarray[tuple[int,], np.dtype[np.float64]], # Search optimal threshold values.
        beta: np.ndarray[tuple[int,], np.dtype[np.float64]]
    ):
        """Classify new samples and optimize hyper-parameters.
        """

        def ClassificationOnRelabeledData(
                threshold1,
                threshold2
            ):
            """Predict probabilities on relabelling data.
            """

            # Get new y labels.
            new_y = self.Relabelling(threshold1, threshold2)
            # Get masses on the new y labels.
            masses = Classifier.PredictProba(X_train, X_calibration, new_y)
            return masses, new_y

        # Parallelization.
        with Pool() as pool:
            results = pool.starmap(ClassificationOnRelabeledData, [
                (i, i) for i in threshold_entropy_space
            ])
        
        perf_u65_pc = []
        perf_u65_eclair: list[tuple] = []
        perf_u65_sd = []

        for masses, new_y in results:
            # Evaluation.
            pred_pc_tst = SetValuedClassification.PignisticCriterion(
                masses,
                np.unique(new_y),
                self.nb_classes
            )
            perf_u65_pc.append(accuracy_score(
                y_calibration,
                pred_pc_tst,
                normalize=True,
                sample_weight=None)
            )            
            # Optimize eclair hyper-parameters.
            eclair_beta_u65 = []
            for i in range(len(beta)):
                # u65 eclair.
                pred_eclair = SetValuedClassification.EclairGFBeta(
                    masses,
                    np.unique(new_y),
                    beta[i],
                    self.nb_classes
                )
                _, _, u65_eclair, _, _ = SetValuedClassification.SetValuedClassEvaluation(
                    y_calibration,
                    pred_eclair,
                    self.nb_classes
                )
                eclair_beta_u65.append(u65_eclair)

            j_opt = np.argmax(eclair_beta_u65)
            perf_u65_eclair.append((eclair_beta_u65[j_opt], beta[j_opt]))
            # u65 sd.
            pred_sd_tst = SetValuedClassification.StrongDominance(
                masses,
                np.unique(new_y),
                self.nb_classes
            )
            _, _, u65_sd, _, _ = SetValuedClassification.SetValuedClassEvaluation(
                y_calibration,
                pred_sd_tst,
                self.nb_classes
            )
            perf_u65_sd.append(u65_sd)
        
        
        threshold_entropy_opt_pc = threshold_entropy_space[np.argmax(perf_u65_pc)]

        opt_eclair = np.argmax([perf_u65_eclair[k][0] for k in range(len(threshold_entropy_space))] )
        param_opt_eclair = [threshold_entropy_space[opt_eclair], perf_u65_eclair[opt_eclair][1]]

        threshold_entropy_opt_sd = threshold_entropy_space[np.argmax(perf_u65_sd)]
        
        print(f'\n --- thr_entr_opt is {str([threshold_entropy_opt_pc, threshold_entropy_opt_sd, opt_eclair])}')
            
        return threshold_entropy_opt_pc, threshold_entropy_opt_sd, param_opt_eclair










#  .d8888b.                     .d888                                        888 8888888b.                       888 
# d88P  Y88b                   d88P"                                         888 888   Y88b                      888 
# 888    888                   888                                           888 888    888                      888 
# 888         .d88b.  88888b.  888888 .d88b.  888d888 88888b.d88b.   8888b.  888 888   d88P 888d888 .d88b.   .d88888 
# 888        d88""88b 888 "88b 888   d88""88b 888P"   888 "888 "88b     "88b 888 8888888P"  888P"  d8P  Y8b d88" 888 
# 888    888 888  888 888  888 888   888  888 888     888  888  888 .d888888 888 888        888    88888888 888  888 
# Y88b  d88P Y88..88P 888  888 888   Y88..88P 888     888  888  888 888  888 888 888        888    Y8b.     Y88b 888 
#  "Y8888P"   "Y88P"  888  888 888    "Y88P"  888     888  888  888 "Y888888 888 888        888     "Y8888   "Y88888 
                                                                                                                         
class ConformalPrediction(Eclair, IEclair):

    def __init__(self,
        minimum_occurrence_nb_per_class: int,
        calibration_probabilities: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]], # Probabilities on calibration data from the training part of each fold (each fold has train/test and on the training part we have train/calibration split), shape(n_samples, n_classes).
        calibration_y: list[np.ndarray[tuple[int,], np.dtype[np.int64]]], # y real labels on calibration data for each fold.
        test_probabilities: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]], # Probabilities on test data for each fold, shape(n_samples, n_classes).
        y: Union[np.ndarray[tuple[int,], np.dtype[np.int64]], None] = None # Shape (n_samples,).
    ):
        Eclair.__init__(self, 
            minimum_occurrence_nb_per_class=minimum_occurrence_nb_per_class,
            y = y
        )
        self._calibration_probabilities = calibration_probabilities
        self._calibration_y = calibration_y
        self._test_probabilities = test_probabilities


    # ---------------------------------------------------------------------------- #
    #                               Getters & Setters                              #
    # ---------------------------------------------------------------------------- #

    @property
    def calibration_probabilities(self): return self._calibration_probabilities

    @calibration_probabilities.setter
    def calibration_probabilities(self, value): self._calibration_probabilities = value

    @property
    def calibration_y(self): return self._calibration_y

    @calibration_y.setter
    def calibration_y(self, value): self._calibration_y = value

    @property
    def test_probabilities(self): return self._test_probabilities

    @test_probabilities.setter
    def test_probabilities(self, value): self._test_probabilities = value

    @property
    def nb_folds(self): return len(self._calibration_y)

    
    # ---------------------------------------------------------------------------- #
    #                                Public methods                                #
    # ---------------------------------------------------------------------------- #

    def Relabelling(self,
        alpha: float        
    ):
        """Relabelling based on conformal prediction.
        TODO: ## !! Attention conformal prediction produit des empty set comme prediction: de tels exemples sont exclus de l'apprentissage !! ##
        """
        new_y: list[int] = []
        for k in range(self.nb_folds):
            print('Start measuring non-conformity for calibration')
            # Number of samples in fold k.
            nb_calibration_labels  = len(self._calibration_y[k])
            calibration_scores = []
            for i in range(nb_calibration_labels):
                # Get probability of the real label y in fold k and sample i in the calibration data.
                calibration_probabilities_k_fold = self._calibration_probabilities[k]
                calibration_probabilities_sample = calibration_probabilities_k_fold[i]
                calibration_scores.append(1 - calibration_probabilities_sample[self._calibration_y[k][i]])

            print('Start get adjusted quantile')

            q_level = np.ceil((nb_calibration_labels + 1) * (1 - alpha)) / nb_calibration_labels
            qhat = np.quantile(calibration_scores, q_level, interpolation='higher')
            
            print(f'qhat={str(qhat)}')
        
            print('Start predictions')
            # For each probability, check if the value is >= (1 - qhat).
            prediction_sets: np.ndarray[tuple[int, int], np.dtype[bool]] = self._test_probabilities[k] >= (1 - qhat)
            for i in range(len(prediction_sets)):
                # Convert bool list of each sample in a integer (new label considering subset of labels).
                new_y.append(Utils.BinaryToInteger(prediction_sets[i]))
            print('End predictions')
        
        # Get all labels and count them.
        distinct_labels, label_count = np.unique(new_y, return_counts=True)
        # Get y labels to modify according to their count (minimal number of occurrences).
        y_to_modify = [distinct_labels[i] for i in range(len(distinct_labels)) if (label_count[i] < self._minimum_occurrence_nb_per_class)]

        # Relabelling sample with small label occurence.
        
        original_classes=[]
        relabelled_classes=[]
        
        removed_empty_set = []
        remove_fr=[]

        # Remove or modify some new labels.
        for i in range(len(new_y)):
            # Find element relabelled with an empty set.
            if new_y[i] == 0:
                removed_empty_set.append(i)
            # If the label is in y_to_modify.
            if new_y[i] in y_to_modify:
                # Keep the original class.
                new_y[i] = 2**self._y[i]

            # TODO: pourquoi cette partie ? Pourquoi ne pas juste faire comment la Run d'entropy ?
            # The original class is not in the relabelled set.
            if not self._y[i] in Utils.BinaryToClass(Utils.IntegerToBinary(new_y[i], self.nb_classes), self.nb_classes):
                remove_fr.append(i)
                # Add the original class to the relabelledelling.
                relabelled_classes.append(new_y[i])
                original_classes.append(self._y[i])
        

        ## empty set relabelling

        remove_all = np.concatenate([removed_empty_set, remove_fr]).tolist()
        print(f'Number of empty sets : {str(len(removed_empty_set))}')
        print(f'Number of false relabelling : {str(len(remove_fr))}') # TODO expliquer ce que veux dire false relabelling.

        # TODO: pour moi ici la taille de new_y sera différent de _y car on filtre des éléments ?
        # Donc on perd des labels. Comment conserver le lien entre label et sample ?
        # En fonction renommer new_y en new_y_tmp et mettre new_y ici.

        # nb_nyd=len(partiel_y_train_tmp)
        partiel_y_train = [partiel_y_train_tmp[j] for j in range(nb_nyd) if j not in remove_all ]
        
        
        # Log.
        distinct_original_classes, original_classes_count = np.unique(original_classes, return_counts=True) 
        print(f'Original classes which are not relebelled with it : {str(distinct_original_classes)} with freq. {str(original_classes_count)}')

        distinct_relabelled_classes, relabelled_classes_count = np.unique(relabelled_classes, return_counts=True)                            
        print(f'Relabel instead of true class : {str(distinct_relabelled_classes)} with fres. {str(relabelled_classes_count)}')
        
        distinct_labels, label_count = np.unique(new_y, return_counts=True)
        print(f'levels after changes : {str(distinct_labels)}')
        print(f'freq levels after changes : {str(label_count)}')

        #subsets are represented by their index in the natural order

        # TODO: faudrait harmoniser les sorties entre les Run (juste new_y).
        return remove_all, partiel_y_train


if __name__ == '__main__':
    # e = Eclair()
    # e.LoadingProbabilities('src/probTest.csv', True)
    # e.LoadingLabels('src/yTest.csv')


    # EXEMPLE.
    x = np.array([
        [1, 2, 3, 6],
        [4, 5, 6, 15],
        [7, 8, 9, 24],
        [10, 11, 12, 33],
        [10, 20, 30, 60],
        [40, 50, 60, 150]
    ])
    x_cal = np.array([
        [70, 80, 90, 240],
        [100, 110, 120, 330],
    ])
    y = np.array([0, 1, 2, 0, 0, 1])
    y_cal = np.array([2, 0])


    nbc = CustomNaiveBayesClassifier(2, 3)
    #
    nb_splits = 5
    proba_train, calibration_probabilities, y_calibration = nbc.PredictProbaKFold(x, y, nb_splits, 42)
    posterior_proba=np.concatenate( [proba_train[j] for j in range(nb_splits)])
    # OR
    # Divide by train/test.
    posterior_prob = nbc.PredictProba(x, x_cal, y)
    cross_entrop = CrossEntropy(5, 2, posterior_prob, y)
    cross_entrop.Classify(x, x_cal, y_cal, nbc, np.linspace(0.2, 3, num=50), np.linspace(0, 3, num=50))

    print(Utils.BinaryToInteger([False, True, True]))
    print(Utils.IntegerToBinary(10, 4))
    
