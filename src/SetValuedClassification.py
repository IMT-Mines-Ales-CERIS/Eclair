import numpy as np

from Utils import Utils

class SetValuedClassification:

    def __init__(self,
        masses: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        distinct_relabeled_classes: np.ndarray[tuple[int,], np.dtype[np.int64]],
        initial_nb_classes: int
    ):
        """Make a decision on which class to assign.
        """
        self._masses = masses
        self._distinct_relabeled_classes = distinct_relabeled_classes
        self._initial_nb_classes = initial_nb_classes
        self._nb_samples = len(self._masses)

    @property
    def masses(self): return self._masses

    @property
    def distinct_relabeled_classes(self): return self._distinct_relabeled_classes

    @property
    def initial_nb_classes(self): return self._initial_nb_classes
    
    # ---------------------------------------------------------------------------- #
    #                               Decision methods                               #
    # ---------------------------------------------------------------------------- #
    
    def PignisticCriterion(self) -> list[list[int]]:
        pred = []
        for i in range(self._nb_samples):
            focals, masses = Utils.ProbabilityToBelief(self._masses[i], self._distinct_relabeled_classes)
            *_, pig = Utils.GetBeliefPlausibility(focals, masses, self._initial_nb_classes)
            max_pig = np.max(pig)
            pred.append([j for j in range(len(pig)) if pig[j] >= max_pig])
        return pred

    def StrongDominance(self) -> list[list[int]]:
        pred = []
        for i in range (self._nb_samples):
            
            # TODO: m_test correspond aux masses (~probabilités) sur X_train et new_y.
            # pourquoi ProbabilityToBelief redonnent des masses ?
            # être certain du vocabulaire que l'on emploi.

            N = [False] * self._initial_nb_classes # TODO: que veux dire N.
            focals, masses = Utils.ProbabilityToBelief(self._masses[i], self._distinct_relabeled_classes)
            bel, pl, _ = Utils.GetBeliefPlausibility(focals, masses, self._initial_nb_classes)
            for a in range(self._initial_nb_classes):
                comp = 0
                for b in range(self._initial_nb_classes):
                    if a != b:
                        if bel[b] >= pl[a]:
                            break
                        else:
                            comp += 1
                if comp == (self._initial_nb_classes - 1):
                    N[a] = True
            pred.append(Utils.BinaryToClass(N, self._initial_nb_classes))
        return pred
    
    def GFBeta(self,
        beta: float
    ) -> list[list[int]]:
        pred = []
        for i in range (self._nb_samples):
            focals, masses = Utils.ProbabilityToBelief(self._masses[i], self._distinct_relabeled_classes)
            bel, pl, pig = Utils.GetBeliefPlausibility(focals, masses, self._initial_nb_classes) # TODO : Aucun des éléments utilisés ?
            gain = []
            for j in range(len(focals)):
                gain_tmp=0.0
                for k in range(len(focals)):
                    # TODO : change A et B (signification).
                    A = Utils.BinaryToClass(Utils.IntegerToBinary(focals[j], self._initial_nb_classes), self._initial_nb_classes)
                    B = Utils.BinaryToClass(Utils.IntegerToBinary(focals[k], self._initial_nb_classes), self._initial_nb_classes) 
                    inters = list(set( A ) & set( B ))
                    f_beta = ((1 + beta**2) * len(inters)) / (((beta**2) * len(B)) + len(A))
                    gain_tmp = gain_tmp + f_beta * masses[k]
                gain.append(gain_tmp)
            pred.append(Utils.BinaryToClass(Utils.IntegerToBinary(focals[np.argmax(gain)], self._initial_nb_classes), self._initial_nb_classes))
        return pred


