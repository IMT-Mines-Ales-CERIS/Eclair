import numpy as np

from Utils import Utils

class SetValuedClassification:

    @staticmethod
    def SetValuedClassEvaluation(
        truth: list[int],
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
    ) -> list[int]:
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