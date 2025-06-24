import numpy as np

from sklearn.model_selection import KFold


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