import codecs
import csv
import numpy as np
import os

from abc import ABC, abstractmethod
from multiprocessing import Pool
from scipy.stats import entropy
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from typing import Union

from NaiveBayes import NaiveBayesClassifier


# 888     888 888    d8b 888          
# 888     888 888    Y8P 888          
# 888     888 888        888          
# 888     888 888888 888 888 .d8888b  
# 888     888 888    888 888 88K      
# 888     888 888    888 888 "Y8888b. 
# Y88b. .d88P Y88b.  888 888      X88 
#  "Y88888P"   "Y888 888 888  88888P'                                     

class Utils:

    @staticmethod
    def BinaryToClass (
        binaries: list[bool],
        length: int # TODO: pourquoi ajouter length et ne pas utiliser simplement len(binaries) ?
    ) -> list[int]:
        """Get classes (indexes) with a True value in a boolean list.

        ### Parameters :
            * ``binaries`` - List of boolean.
            * length ?

        ### Return :
            * ``classes`` - List of classes according to indexes of True elements.
        """
        classes = []
        for i in range(length):
            if binaries[i]:
                classes.append(i)
        return classes
    
    @staticmethod
    def BinaryToInteger(
        binaries: list[bool]
    ) -> int:
        """Convert a list of boolean to an integer.
        """
        integer = 0
        for i in range(len(binaries)):
            integer += int(binaries[i]) * 2**i
        return integer
    
    @staticmethod
    def GetBeliefPlausibility(
        focals,
        m,
        nb_classes
    ) -> tuple[
        list[float],
        list[float],
        list[float]
    ]:
        """Get bel, pl, pig from body of evidence.
        """
        pl  = [0.0] * nb_classes
        bel = [0.0] * nb_classes
        pig = [0.0] * nb_classes
        for i in range (len(focals)):
            bF_tmp = Utils.IntegerToBinary(focals[i], nb_classes) # Binary vector of F[i].
            for j in range (nb_classes):
                # TODO: k in [j] ??
                # TODO: Enlever +1.
                bC_tmp = [k in [j+1] for k in range(1,nb_classes+1)]#binary vector of class j
                if bF_tmp[j] == bC_tmp[j]: # j in F[i]
                    pl[j] += m[i]
                    pig[j] += m[i]/(np.sum(bF_tmp))
                    if np.sum(bF_tmp) == 1: # [j] equal to F[i]
                        bel[j] = m[i]
        return bel, pl, pig

    @staticmethod
    def IntegerToBinary(
        integer: int,
        length: int
    ):
        """Convert an integer to a boolean list (presence and absence of each class).
        """
        # Init a list of False values of length "length".
        binaries = [False] * length
        print(length)
        for i in reversed(range(length)):
            binaries[i] = (integer // 2**i ) == 1 # When dividing two numbers using this // operator, the result will always be an integer, ignoring the decimal part of the result.
            integer = integer % 2**i
        return binaries
    
    @staticmethod
    def Kfold(
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        nb_folds: int = 2 # Must be at least 2.
    ) -> tuple[
        list[np.ndarray[tuple[int,], np.dtype[np.int64]]], # The training set indices for all splits.
        list[np.ndarray[tuple[int,], np.dtype[np.int64]]]  # The testing set indices for all splits.
    ]:
        """Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling).
        """
        kf = KFold(n_splits = nb_folds)
        X_train = []
        X_test = []
        for _, (train_index, test_index) in enumerate(kf.split(X)):
            X_train.append(train_index)
            X_test.append(test_index)        
        return X_train, X_test
    
    @staticmethod
    def ProbabilityToBelief(
        p, #TODO: shape(int, ) ? Juste pour un sample.
        selflevels
    ) -> tuple[
        list, # TODO float ou int. Utils.IntegerToBinary(F[i] ... donc int ?
        list[float]
    ]:
        """Get body of evidence (F, M) from masses p.
        """
        focals = [] # Focal elements.
        masses = [] # Masses associated to focal elements.
        for i in range (len(p)):
            if p[i] > 10**(-4):
                focals.append(selflevels[i])
                masses.append(p[i])
        masses = list(masses / np.sum(masses))
        return focals, masses










#  .d8888b.           888    888     888         888                        888 
# d88P  Y88b          888    888     888         888                        888 
# Y88b.               888    888     888         888                        888 
#  "Y888b.    .d88b.  888888 Y88b   d88P 8888b.  888 888  888  .d88b.   .d88888 
#     "Y88b. d8P  Y8b 888     Y88b d88P     "88b 888 888  888 d8P  Y8b d88" 888 
#       "888 88888888 888      Y88o88P  .d888888 888 888  888 88888888 888  888 
# Y88b  d88P Y8b.     Y88b.     Y888P   888  888 888 Y88b 888 Y8b.     Y88b 888 
#  "Y8888P"   "Y8888   "Y888     Y8P    "Y888888 888  "Y88888  "Y8888   "Y88888 
                                                                              
class SetValuedClassification:

    @staticmethod
    def SetValuedClassEvaluation(
        truth: np.ndarray[tuple[int,], np.dtype[np.int64]],
        pred: list[list[int]] # TODO: int ou float ?
    ) -> tuple[
        float,
        float,
        float,
        float,
        float
    ]:
        """
        ### Parameters :
            * ``truth`` - Contains indices of example classes.
            * ``pred`` - Contains subsets of elements from 0 to nb_classes - 1.
        ### Returns :
            * acc
            * u50
            * u65
            * u80
            * acc_imp
        """
        acc = 0.0
        u50 = 0.0
        u65 = 0.0
        u80 = 0.0
        acc_imp = 0.0
        if len(truth) != len(pred):
            print('truth and pred must have the same length')
        else:
            inPred = [0.0] * len(truth)
            z = [0.0] * len(truth)
            z65 = [0.0] * len(truth)
            z80 = [0.0] * len(truth)
            z_acc = [0.0] * len(truth)
            for i in range(len(truth)):
                # TODO: ==[True] ?? Pourquoi passer par un tableau ? Pk pas truth[i] in pred[i] ?
                if len(pred[i]) > 0 and [k in pred[i] for k in [truth[i]]] == [True]:
                    inPred[i] = 1.0
                    z[i] = 1.0/len(pred[i])
                    z65[i] = -0.6*(z[i]**2)+1.6*z[i]
                    z80[i] = -1.2*(z[i]**2)+2.2*z[i]
                    if len(pred[i])==1:
                        z_acc[i] = 1.0
            acc = np.sum(z_acc) / len(truth)
            u50 = np.sum(z) / len(truth)
            u65 = np.sum(z65) / len(truth)
            u80 = np.sum(z80) / len(truth)
            acc_imp = np.sum(inPred) / len(truth)
             
        return acc, u50, u65, u80, acc_imp
    
    @staticmethod
    def PignisticCriterion(
        m_test,
        selflevels,
        nb_classes: int
    ) -> list[list[int]]:
        pred = []
        for i in range(len(m_test)):
            focals, masses = Utils.ProbabilityToBelief(m_test[i], selflevels)
            *_, pig = Utils.GetBeliefPlausibility(focals, masses, nb_classes)
            max_pig = np.max(pig)
            pred.append([j for j in range(len(pig)) if pig[j] >= max_pig])
        return pred

    @staticmethod
    def StrongDominance(
        m_test,
        selflevels,
        nb_classes: int
    ) -> list[list[int]]:
        pred = []
        for i in range (len(m_test)):
            N = [False] * nb_classes # TODO: que veux dire N.
            focals, masses = Utils.ProbabilityToBelief(m_test[i], selflevels)
            bel, pl, _ = Utils.GetBeliefPlausibility(focals, masses, nb_classes)
            for a in range (nb_classes):
                comp=0
                for b in range (nb_classes):
                    if(a!=b):
                        if(  (bel[b] >= pl[a]) ):
                            break
                        else:
                            comp+=1
                if( (comp==(nb_classes-1)) ):
                    N[a]=True
            pred.append(Utils.BinaryToClass(N,nb_classes))
        return pred
    
    @staticmethod
    def EclairGFBeta(
        m_test,
        selflevels,
        beta,
        nb_classes
    ):
        pred=[]
        for i in range (len(m_test)):
            focals, masses = Utils.ProbabilityToBelief(m_test[i], selflevels)
            bel, pl, pig = Utils.GetBeliefPlausibility(focals, masses, nb_classes) # TODO : Aucun des éléments utilisés ?
            gain = []
            for j in range(len(focals)):
                gain_tmp=0.0
                for k in range(len(focals)):
                    # TODO : change A et B (signification).
                    A = Utils.BinaryToClass(Utils.IntegerToBinary(focals[j], nb_classes), nb_classes)
                    B = Utils.BinaryToClass(Utils.IntegerToBinary(focals[k], nb_classes), nb_classes) 
                    inters = list(set( A )  & set( B ) )
                    f_beta = ((1 + beta**2) * len(inters)) / (((beta**2) * len(B)) + len(A))
                    gain_tmp = gain_tmp+f_beta * masses[k]
                gain.append(gain_tmp)
            pred.append(Utils.BinaryToClass(Utils.IntegerToBinary(focals[np.argmax(gain)],nb_classes), nb_classes))
        return pred










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
    def PredictProba(self, X_train, X_test, y_train, **kwargs) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]: # Shape (n_sample, n_classes).
        """Get posterior probabilities for each classes.
        """
        
        pass
    
    # TODO: A voir si on ne met pas aussi PredictProbaKFold (et peut être KFold en method de cette classe ?).


class CustomNaiveBayesClassifier(NaiveBayesClassifier, ICustomClassifier):

    def __init__(self):
        NaiveBayesClassifier.__init__(self)

    # @property TODO: Add property to attribute (understand before).

    def _ClassifierKFold(
            self,
            X,
            y,
            X_train_indices: np.ndarray[tuple[int,], np.dtype[np.int64]], # The training set indices for one split.
            X_test_indices:  np.ndarray[tuple[int,], np.dtype[np.int64]],  # The testing set indices for one split.
            categorical_features,
            numerical_features,
            test_size,
            random_state
        ) -> tuple[
            np.ndarray[tuple[int, int], np.dtype[np.float64]], # Posterior probabilities on calibration part by class and by sample for one split.
            np.ndarray[tuple[int, int], np.dtype[np.float64]], # Posterior probabilities on test part by class and by sample for one split.
            np.ndarray[tuple[int,], np.dtype[np.int64]],       # Classes on the calibration split for that split.
        ]:
            """Get posterior probabilities from naive bayes classifier for one split.

            ### Parameters :
                * ``X`` - Dataset, shape, (n_samples, n_features).
                * ``y`` - Labels, shape, (n_samples,).
                * ``X_train_indices`` - X indexes to select for train.
                * ``X_test_indices`` - X indexes to select for test.
                * ``categorical_features`` - Indexes of categorical features (discrete values).
                * ``numerical_features`` - Indexes of numerical features (continuous values).
                * ``nb_folds``: Number of folds, must be at least 2.
                * ``test_size``: Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
                * ``random_state``: Controls the shuffling applied to the data before applying the train_test_split function.
            
            ### Returns :
                * ``list`` - Posterior probabilities on test sets for one split.
                * ``list`` - Posterior probabilities on calibration sets for one split.
                * ``list`` - Labels for the calibration sets for one split.
            """

            X_train, X_calibration, y_train, y_calibration = train_test_split(
                X[X_train_indices, :], # Get data according to indices for that split.
                y[X_train_indices],    # Get labels according to indices for that split.
                test_size = test_size,
                random_state = random_state
            )
            
            mu, sigma2, nc, classes, lev, freq, u = self.FitContinuousModel(X_train, y_train, categorical_features, numerical_features, 'gaussian')
            
            *_, calibration_probabilities = self.Predict(
                X_calibration,
                categorical_features,
                numerical_features,
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
                categorical_features,
                numerical_features,
                mu,
                sigma2,
                nc,
                classes,
                lev,
                freq,
                u
            )

            return test_probabilities, calibration_probabilities, y_calibration

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
            * ``categorical_features`` - Indexes of categorical features (discrete values).
            * ``numerical_features`` - Indexes of numerical features (continuous values).
        """

        categorical_features: np.ndarray[tuple[int,], np.dtype[np.int64]] = kwargs['categorical_features'] if 'categorical_features' in kwargs else np.array([])
        numerical_features: np.ndarray[tuple[int,], np.dtype[np.int64]] = kwargs['numerical_features'] if 'numerical_features' in kwargs else np.array([])

        # Get fitting parameters from training set.
        mu, sigma2, nc, classes, lev, freq, u = self.FitContinuousModel(X_train, y_train, categorical_features, numerical_features, 'gaussian')
        # Predict probabilities on X_test based on fitting parameters.
        *_, posterior_probabilities = self.Predict(
            X_test,
            categorical_features,
            numerical_features,
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
            X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
            y: np.ndarray[tuple[int,], np.dtype[np.float64]],
            categorical_features: np.ndarray[tuple[int,], np.dtype[np.int64]],
            numerical_features: np.ndarray[tuple[int,], np.dtype[np.int64]],
            nb_folds: int,
            test_size: float,
            random_state: Union[int, None]
        ) -> tuple[
            list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
            list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
            list[np.ndarray[tuple[int,], np.dtype[np.int64]]]
        ]:
        """Get posterior probabilities of calibration and test sets for all splits.

        ### Parameters :
            * ``X`` - Dataset, shape, (n_samples, n_features).
            * ``y`` - Labels, shape, (n_samples,).
            * ``categorical_features`` - Indexes of categorical features (discrete values).
            * ``numerical_features`` - Indexes of numerical features (continuous values).
            * ``nb_folds``: Number of folds, must be at least 2.
            * ``test_size``: Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
            * ``random_state``: Controls the shuffling applied to the data before applying the train_test_split function.
        
        ### Returns :
            * ``list`` - Posterior probabilities on test sets for each split.
            * ``list`` - Posterior probabilities on calibration sets for each split.
            * ``list`` - Labels for the calibration sets for each split.
        """
        
        # For a dataset with 6 entries and a split number of 2 here is the return : ([(TRAIN)array([3, 4, 5]), (TEST)array([0, 1, 2])], [(TRAIN)array([0, 1, 2]), (TEST)array([3, 4, 5])]).
        # It is important to make the kfold without shuffle to keep dataset order in test data.
        X_train_kfold_indices, X_test_kfold_indices = Utils.Kfold(X, nb_folds)
        # Parallelization.
        with Pool() as pool:
            results = pool.starmap(
                self._ClassifierKFold, [
                    (
                        X,
                        y,
                        X_train_kfold_indices[k],
                        X_test_kfold_indices[k],
                        categorical_features,
                        numerical_features,
                        test_size,
                        random_state
                    )
                    for k in range(nb_folds)
                ]
            )

            test_probabilities = list(map(lambda item: item[0], results))
            calibration_probabilities = list(map(lambda item: item[1], results))
            y_calibration = list(map(lambda item: item[2], results))

            # results = [
            #     self._ClassifierKFold(
            #         X,
            #         y,
            #         X_train_kfold_indices[0],
            #         X_test_kfold_indices[0],
            #         categorical_features,
            #         numerical_features,
            #         test_size,
            #         random_state
            #     )
            # ]

            # [0] TODO: [[1.], [1.], [1.]]
            # [1] TODO: pourquoi on obtient un test_probabilites = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
                        
            # print()

            # print(X_train_kfold_indices[0])
            # print(X_test_kfold_indices[0])

            # print(X[X_train_kfold_indices[1], :]) # Get data according to indices for that split.
            # print(y[X_train_kfold_indices[1]])

            # print()

            # print(results)

            # print()

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
    def Relabelling(self, **kwargs) -> list[int]:
        """Get imprecise labels.
        """
        pass
    
    @abstractmethod
    def Classify(self,
        X_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        X_calibration: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_calibration: np.ndarray[tuple[int,], np.dtype[np.int64]],
        Classifier: ICustomClassifier,
        **kwargs
    ):
        """Classify samples with the relabelling process.
        """
        pass


class Eclair:

    def __init__(self,
        minimum_occurrence_nb_per_class: int, # Minimum number of occurrences of a class.
        y: Union[np.ndarray[tuple[int,], np.dtype[np.int64]], None] = None # Shape (n_samples,).
    ):
        self._minimum_occurrence_nb_per_class = minimum_occurrence_nb_per_class
        self._y = y # Real labels.
        self._unique_y = []
        if not self._y is None:
            self._unique_y = np.unique(self._y)

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
        return len(self._unique_y)


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
        probabilities: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None, # Posterior probabilities, shape (n_samples, n_classes).
        y: np.ndarray[tuple[int,], np.dtype[np.int64]] | None = None # Shape (n_samples,).
    ):
        Eclair.__init__(self, 
            minimum_occurrence_nb_per_class=minimum_occurrence_nb_per_class,
            y = y
        )
        self._entropy_base = entropy_base
        self._probabilities: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = probabilities # Posterior probabilities.


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
        return 0 if self._probabilities is None else self._probabilities.shape[0]
    

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
    
    def _ClassificationOnRelabeledData(
        self,
        X_train,
        X_calibration,
        threshold1: float,
        threshold2: float,
        Classifier: ICustomClassifier
    ):
        """Predict probabilities on relabelling data.
        """
        # Get new y labels.
        new_y = self.Relabelling(threshold1 = threshold1, threshold2 = threshold2)
        # Get masses on the new y labels.
        masses = Classifier.PredictProba(X_train, X_calibration, new_y)
        return masses, new_y
        

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
        self._probabilities = np.array(probabilities)

    def Relabelling(self, **kwargs) -> list[int]:
        """Relabelling based on entropy computation.
        """

        if not 'threshold1' in kwargs or not 'threshold2' in kwargs:
            return []

        if self._probabilities is None or self._y is None:
            return []
        
        threshold1: float = kwargs['threshold1'] # Maximum entropy not to be exceeded when the chosen label is identical to the real label.
        threshold2: float = kwargs['threshold2'] # Maximum entropy not to be exceeded when the chosen label is not identical to the real label.
        
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
        **kwargs
    ):
        """Classify new samples and optimize hyper-parameters.
        """

        if not 'threshold_entropy_space' in kwargs or not 'beta' in kwargs:
            return

        threshold_entropy_space: np.ndarray[tuple[int,], np.dtype[np.float64]] = kwargs['threshold_entropy_space'] # Search optimal threshold values.
        beta: np.ndarray[tuple[int,], np.dtype[np.float64]] = kwargs['beta']

        # Parallelization.
        with Pool() as pool:
            results = pool.starmap(
                self._ClassificationOnRelabeledData, [
                (
                    X_train,
                    X_calibration,
                    i,
                    i,
                    Classifier
                ) for i in threshold_entropy_space
            ])
        
        perf_u65_pc: list[np._Float64_co] = []
        perf_u65_eclair: list[tuple] = []
        perf_u65_sd: list[float] = []

        for masses, new_y in results:
            # Evaluation.
            pred_pc_tst = SetValuedClassification.PignisticCriterion(
                masses,
                np.unique(new_y),
                self.nb_classes
            )

            # TODO: prb pred_pc_tst est un tableau de tableau et ça ne va pas fonctionner pour accuracy_score.
            
            print(pred_pc_tst)
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

                # TODO: delete nb classes here, unused on original file.
                _, _, u65_eclair, _, _ = SetValuedClassification.SetValuedClassEvaluation(
                    y_calibration,
                    pred_eclair
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
                pred_sd_tst
            )
            perf_u65_sd.append(u65_sd)
        
        
        threshold_entropy_opt_pc = threshold_entropy_space[np.argmax(np.array(perf_u65_pc))]

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
        
        partiel_y_train = []
        # partiel_y_train = [partiel_y_train_tmp[j] for j in range(nb_nyd) if j not in remove_all ]
        
        
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
        [0.1, 0.11, 0.12, 0.13],
        [0.21, 0.22, 0.23, 0.24],
        [0.7, 0.69, 0.68, 0.67],
        [0.97, 0.96, 0.95, 0.94],
        [0.99, 0.99, 0.99, 0.99],
        [0.01, 0.02, 0.03, 0.04]
    ])
    x_cal = np.array([
        [70, 80, 90, 240],
        [100, 110, 120, 330],
    ])
    y = np.array([1, 1, 2, 2, 0, 0])
    y_cal = np.array([2, 0])

    data = load_iris()

    nbc = CustomNaiveBayesClassifier()
    #
    nb_splits = 2
    proba_train, calibration_probabilities, y_calibration = nbc.PredictProbaKFold(data.data, data.target, np.array([]), np.array([0,1,2,3]), nb_splits, 0.2, 42)
    # Get probabilities of all samples in dataset.
    posterior_proba = np.concatenate([proba_train[j] for j in range(nb_splits)])
    print(posterior_proba)
    # OR
    # Divide by train/test.
    # posterior_proba = nbc.PredictProba(x, x_cal, y)
    cross_entrop = CrossEntropy(5, 2, posterior_proba, data.target)
    threshold_entropy_opt_pc, threshold_entropy_opt_sd, param_opt_eclair = cross_entrop.Classify(x, x_cal, y_cal, nbc, threshold_entropy_space = np.linspace(0.2, 3, num=50), beta = np.linspace(0, 3, num=50))
    print(threshold_entropy_opt_pc)

    # print(Utils.BinaryToInteger([False, True, True]))
    # print(Utils.IntegerToBinary(10, 4))
    
