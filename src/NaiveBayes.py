import numpy as np

class NaiveBayesClassifier:

    def __init__(self,
        categorical_features,
        numerical_features
    ):
        self._categorical_features = categorical_features
        self._numerical_features = numerical_features

        # TODO: A CONFIRMER SI CAT_FEAT et NUM_FEAT dependent ou non du jeu de données (sinon mettre en argument ?).

    # ---------------------------------------------------------------------------- #
    #                                Private methods                               #
    # ---------------------------------------------------------------------------- #

    def _Gaussian(self,
        x: float,
        mu: float,
        sig: float
    ) -> float:
        if(sig > 10**-10):
            return 1. / (np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
        else :
            sig = 10**-10
            return 1. / (np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

    def _Softmax(self,
        v: list[float]
    ): #produit des nan quand les valeurs sont trop grandes
        proba = []
        for i in range(len(v)):
            proba.append(np.exp(v[i]) / (np.sum(np.exp(v))))
        return proba


    # ---------------------------------------------------------------------------- #
    #                                Public methods                                #
    # ---------------------------------------------------------------------------- #

    def FitContinuousModel(self,
        x_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int, int], np.dtype[np.int64]],
        probability_type: str = 'gaussian'
    ):
        """
        ### Parameters :
            * ``x_train`` - Shape, (n_samples, n_features).
            * ``y_train`` - Shape, (n_samples,).
            * ``probability_type`` - Type of probabilities, default "*gaussian*".
        """
        nb_samples = len(y_train)
        distinct_classes, class_count = np.unique(y_train, return_counts=True)
        nb_classes = len(distinct_classes)
        # Numerical features.
        if probability_type == 'gaussian':
            mu = [
                [
                    np.mean([
                        x_train[i, self._numerical_features[j]] for i in range(nb_samples) if (y_train[i] == distinct_classes[k])
                    ]) for j in range(len(self._numerical_features))
                ] for k in range(nb_classes)
            ]
            sigma2 = [
                [
                    np.std([
                        x_train[i,self._numerical_features[j]] for i in range(nb_samples) if (y_train[i] == distinct_classes[k])
                    ]) for j in range(len(self._numerical_features))
                ] for k in range(nb_classes)]
        # Categorical features 
        lev = []
        freq = []
        for ic in range(nb_classes):
            lev_ic: list[np.ndarray[tuple[int,], np.dtype[np.float64]]] = []
            freq_ic: list[list[float]] = []
            for cat_f in range(len(self._categorical_features)):
                lev_tmp, counts_tmp = np.unique([
                    x_train[k, self._categorical_features[cat_f]]  for k in range(nb_samples) if (y_train[k]==distinct_classes[ic])
                ], return_counts=True)
                lev_ic.append(lev_tmp)
                freq_ic.append([counts_tmp[t] / np.sum(counts_tmp) for t in range(len(counts_tmp))])
            lev.append(lev_ic)
            freq.append(freq_ic)

        class_rate:list[float] = []# number of occurence of each class
        for ic in range(nb_classes):
            class_rate.append(class_count[ic] / nb_samples)

        utilities = np.identity(len(distinct_classes))

        return mu, sigma2, class_rate, distinct_classes, lev, freq, utilities
    
    def Predict(self,
        x_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        mu,
        sigma2,
        nc,
        classes,
        lev,
        freq,
        utilities
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]: # Shape (n_sample, n_classes).
        """
        ### Parameters :
        """
        nb_classes = len(classes)
        nb_test_samples = len(x_test)
        preds_proba_sm = np.zeros((nb_test_samples, nb_classes))
        preds_proba = np.zeros((nb_test_samples, nb_classes))
        preds_bo = []
        preds_eu = []
        nba = len(self._numerical_features) + len(self._categorical_features)
        for ir in range(nb_test_samples):
            z=[]
            for ia in range(nba):
                z.append(x_test[ir][ia])
            pic = []
            max_pic = 0
            for ic in range(nb_classes):
                pic_tmp = nc[ic]
                for i_nf in range(len(self._numerical_features)):
                    pic_tmp *= self._Gaussian(z[self._numerical_features[i_nf]], mu[ic][i_nf], sigma2[ic][i_nf])
                for i_cf in range(len(self._categorical_features)):
                    idx = np.where(np.array(lev[ic][i_cf]) == z[self._categorical_features[i_cf]])
                    if(len(idx[0])==0):
                        pic_tmp *= 10**-10
                    else :
                        pic_tmp *= freq[ic][i_cf][idx[0][0]]                    
                if max_pic < pic_tmp:
                    max_pic = pic_tmp  
                pic.append(pic_tmp)
            preds_bo.append(classes[np.argmax(pic)])
            preds_proba[ir] = pic
            #softmax produit des nan quand les valeurs sont trop grandes
            preds_proba_sm[ir] = self._Softmax([pic[j]-max_pic for j in range(nb_classes)])
            eu = []
            for ic in range(nb_classes):
                eu.append(np.sum([utilities[t][ic]*preds_proba[ir][t] for t in  range(nb_classes)]))
            preds_eu.append( classes[np.argmax( eu )] )
        return preds_bo, preds_eu, preds_proba, preds_proba_sm