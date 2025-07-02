import numpy as np

class SetValuedClassEvaluation:

    @staticmethod
    def SetValuedClassEvaluation(
        truth: np.ndarray[tuple[int,], np.dtype[np.int64]],
        pred: list[list[int]]
    ) -> tuple[
        float,
        float,
        float,
        float,
        float
    ]:
        """
        ### Parameters :
            * ``truth`` - Real labels associated with the test game.
            * ``pred`` - Contains subsets of elements from 0 to nb_classes - 1.
        ### Returns :
            * accuracy.
            * accuracy on imprecise predictions.
            * u50, measure the accuracy of the classifier restricted to samples for which the model's predictive uncertainty is lower than a predefined threshold (<= 0.50).
            * u65, threshold <= 0.65.
            * u80, threshold <= 0.80.
        """
        acc = 0.0
        acc_imp = 0.0
        u50 = 0.0
        u65 = 0.0
        u80 = 0.0
        if len(truth) != len(pred):
            print('truth and pred must have the same length')
        else:
            nb_samples = len(truth)
            inPred = [0.0] * nb_samples
            z = [0.0] * nb_samples
            z65 = [0.0] * nb_samples
            z80 = [0.0] * nb_samples
            z_acc = [0.0] * nb_samples
            for i in range(nb_samples):
                if len(pred[i]) > 0 and truth[i] in pred[i]:
                    inPred[i] = 1.0
                    z[i] = 1.0 / len(pred[i])
                    z65[i] = -0.6 * (z[i]**2) +1.6 * z[i]
                    z80[i] = -1.2 * (z[i]**2) +2.2 * z[i]
                    if len(pred[i]) == 1:
                        z_acc[i] = 1.0
            acc = np.sum(z_acc) / nb_samples
            u50 = np.sum(z) / nb_samples
            u65 = np.sum(z65) / nb_samples
            u80 = np.sum(z80) / nb_samples
            acc_imp = np.sum(inPred) / nb_samples
             
        return acc, acc_imp, u50, u65, u80