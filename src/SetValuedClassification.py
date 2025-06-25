import numpy as np

from Utils import Utils

class SetValuedClassification:

    # ---------------------------------------------------------------------------- #
    #                               Decision methods                               #
    # ---------------------------------------------------------------------------- #
    
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
        m_test, # masses.
        selflevels,
        nb_classes: int
    ) -> list[list[int]]:
        """Make a decision on which class to assign.
        """
        pred = []
        for i in range (len(m_test)):
            
            # TODO: m_test correspond aux masses (~probabilités) sur X_train et new_y.
            # pourquoi ProbabilityToBelief redonnent des masses ?
            # être certain du vocabulaire que l'on emploi.

            N = [False] * nb_classes # TODO: que veux dire N.
            focals, masses = Utils.ProbabilityToBelief(m_test[i], selflevels)
            bel, pl, _ = Utils.GetBeliefPlausibility(focals, masses, nb_classes)
            for a in range(nb_classes):
                comp = 0
                for b in range(nb_classes):
                    if a != b:
                        if bel[b] >= pl[a]:
                            break
                        else:
                            comp += 1
                if comp == (nb_classes - 1):
                    N[a] = True
            pred.append(Utils.BinaryToClass(N, nb_classes))
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


