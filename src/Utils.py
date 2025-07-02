import numpy as np

from sklearn.model_selection import KFold


class Utils:

    @staticmethod
    def BinaryToClass (
        binaries: list[bool]
    ) -> list[int]:
        """Get classes (indexes) with a True value in a boolean list.

        ### Parameters :
            * ``binaries`` - List of boolean.
            * length ?

        ### Return :
            * ``classes`` - List of classes according to indexes of True elements.
        """
        classes = []
        for i in range(len(binaries)):
            if binaries[i]:
                classes.append(i)
        return classes
    
    @staticmethod
    def BinaryToInteger(
        binaries: list[bool]
    ) -> int:
        """Convert a list of boolean to an integer.

        ### Parameters:
            * ``binaries`` - A boolean vector indicating the presence or absence of a class.

        ### Returns :
            * An integer representing a class or a set of classes.
        """
        integer = 0
        for i in range(len(binaries)):
            integer += int(binaries[i]) * 2**i
        return integer
    
    @staticmethod
    def GetBeliefPlausibility(
        focals: list[int],
        m: list[float],
        nb_classes: int
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
            for j in range(nb_classes):
                bC_tmp = [k == j for k in range(nb_classes)] # Binary vector of class j.
                if bF_tmp[j] == bC_tmp[j]: # j in F[i]
                    pl[j] += m[i]
                    pig[j] += m[i]/(np.sum(bF_tmp))
                    if np.sum(bF_tmp) == 1: # [j] equal to F[i]
                        bel[j] = m[i]
        return bel, pl, pig

    @staticmethod
    def IntegerToBinary(
        integer: int,
        nb_classes: int
    ) -> list[bool]:
        """Convert an integer to a boolean list (presence and absence of each class).

        ### Parameters :
            * ``integer`` - Represents a class or a set of classes.
            * ``nb_classes`` - Number of classes.

        ### Returns :
            * A boolean vector indicating the presence or absence of a class.

        Example: 
        <table>
            <tr><td>integer</td><td>nb_classes</td><td>equation</td><td>booleans</td></tr>
            <tr><td>4</td><td>3</td><td>2^2 = 4</td><td>[False, False, True]</td></tr>
            <tr><td>3</td><td>3</td><td>2^0 + 2^1 = 3</td><td>[True, True, False]</td></tr>
            <tr><td>7</td><td>3</td><td>2^0 + 2^1 + 2^2 = 7</td><td>[True, True, True]</td></tr>
        </table>
        """
        binaries = [False] * nb_classes # Init a list of False values of length "nb_classes".
        for i in reversed(range(nb_classes)):
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
        p: np.ndarray[tuple[int,], np.dtype[np.float64]],
        distinct_partial_classes: np.ndarray[tuple[int,], np.dtype[np.int64]]
    ) -> tuple[
        list[int],
        list[float]
    ]:
        """Get body of evidence (F, M) from masses p.

        ### Parameters :
            * ``p`` - Masses.
            * ``distinct_partial_classes`` - Distinct classes.
        
        ### Returns :
            * Focal elements (ex. A is focal element if m(A) > 0).
            * Masses associated to focal elements.
        """
        focals = []
        masses = []
        for i in range(len(p)):
            if p[i] > 10**(-4):
                focals.append(distinct_partial_classes[i])
                masses.append(p[i])
        masses = list(masses / np.sum(masses))
        return focals, masses    


if __name__ == '__main__':
    print(Utils.IntegerToBinary(3, 3))