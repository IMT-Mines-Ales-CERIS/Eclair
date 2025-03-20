import keras
import numpy as np

from multiprocessing import Pool
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from typing import Union

#

from NaiveBayes import nbc


# 8888888888         888          d8b         
# 888                888          Y8P         
# 888                888                      
# 8888888    .d8888b 888  8888b.  888 888d888 
# 888       d88P"    888     "88b 888 888P"   
# 888       888      888 .d888888 888 888     
# 888       Y88b.    888 888  888 888 888     
# 8888888888 "Y8888P 888 "Y888888 888 888     
                                            
class Eclair:

    # TODO: finalement Eclair n'est pas là pour calculer des proba à posteriori mais pour faire le relabelling ?
    # -> Séparer la partie classifier et ajouter des attributs pour tenir compte des proba à posteriori ?

    # TODO: add ndtype to input data.

    def __init__(self,
            X: np.ndarray[tuple[int, int]], # Shape (n_samples, n_features).
            y: np.ndarray[tuple[int,], np.dtype[np.int64]] # Shape (n_samples,).
        ):

        self._X = X # Training set
        self._y = y # Target values.


    def _Kfold(self,
            nb_splits: int # Must be at least 2.
        ) -> tuple[
            list[np.ndarray[tuple[int,], np.dtype[np.int64]]], # The training set indices for all splits.
            list[np.ndarray[tuple[int,], np.dtype[np.int64]]]  # The testing set indices for all splits.
        ]:
        """Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling).
        """

        kf = KFold(n_splits = nb_splits)
        X_train = []
        X_test = []

        for _, (train_index, test_index) in enumerate(kf.split(self._X)):
            X_train.append(train_index)
            X_test.append(test_index)
        
        return X_train, X_test
    

    def NaiveBayesClassifier(self,
            cat_feat,
            num_feat,
            nb_splits: int,
            test_size: float,
            random_state = Union[int, None] # For reproducible output across multiple function calls.
        ) -> tuple[
            list[np.ndarray[tuple[int,], np.dtype[np.int64]]],       # Shape (int,).
            list[np.ndarray[tuple[int, int], np.dtype[np.float64]]], # Posterior probabilities by class and by sample for all splits.
            list[np.ndarray[tuple[int, int], np.dtype[np.float64]]]  # Posterior probabilities by class and by sample for all splits.
        ]:
        """Get posterior probabilities from naive bayes classifier for all splits.
        """

        def _Classifier(
                X_train_indices: np.ndarray[tuple[int,], np.dtype[np.int64]], # The training set indices for one split.
                X_test_indices:  np.ndarray[tuple[int,], np.dtype[np.int64]]  # The testing set indices for one split.
            ) -> tuple[
                np.ndarray[tuple[int,], np.dtype[np.int64]], # Classes on the test split for that split.
                np.ndarray[tuple[int, int], np.dtype[np.float64]], # Posterior probabilities by class and by sample for one split.
                np.ndarray[tuple[int, int], np.dtype[np.float64]]  # Posterior probabilities by class and by sample for one split.
            ]:
            """Get posterior probabilities from naive bayes classifier for one split.
            """

            X_train, X_cal, y_train, y_cal = train_test_split(
                self._X[X_train_indices, :], # Get data according to indices for that split.
                self._y[X_train_indices],    # Get labels according to indices for that split.
                test_size = test_size,
                random_state = random_state
            )

            mu, sigma2, nc, classes, lev, freq = nbc.fit_continuous_model(
                X_train,
                y_train,
                cat_feat,
                num_feat,
                'gaussian'
            )

            u = np.identity(len(classes))

            *_, preds_proba_sm_cal = nbc.predict(
                X_cal,
                cat_feat,
                num_feat,
                mu,
                sigma2,
                nc,
                classes,
                lev,
                freq,
                u
            )
            *_, preds_proba_sm = nbc.predict(
                self._X[X_test_indices, :],
                cat_feat,
                num_feat,
                mu,
                sigma2,
                nc,
                classes,
                lev,
                freq,
                u
            )

            return y_cal, preds_proba_sm_cal, preds_proba_sm
        
        X_train_kfold_indices, X_test_kfold_indices = self._Kfold(nb_splits, random_state)
        # Parallelization.
        with Pool() as pool:
            async_results = [pool.apply_async(
                _Classifier,
                args=(
                    X_train_kfold_indices[k],
                    X_test_kfold_indices[k],
                ))
                for k in range(nb_splits)
            ]
            results = [res.get() for res in async_results]

        y_cal = list(map(lambda item: item[0], results))
        cpc_proba_cal = list(map(lambda item: item[1], results))
        cpc_proba_test = list(map(lambda item: item[2], results))

        return y_cal, cpc_proba_cal, cpc_proba_test


    def DeepNeuralNetworkClassifier(self,
            epoch: int,
            batch_size: int,
            nb_classes: int,
            image_width: int,
            image_height: int,
            nb_splits: int,
            test_size: float,
            random_state = Union[int, None]
        ) -> tuple[
            list[np.ndarray[tuple[int,], np.dtype[np.int64]]],       # Shape (int,).
            list[np.ndarray[tuple[int, int], np.dtype[np.float64]]], # Posterior probabilities by class and by sample for all splits.
            list[np.ndarray[tuple[int, int], np.dtype[np.float64]]]  # Posterior probabilities by class and by sample for all splits.
        ]:
        """Get posterior probabilities from convolutional neural network classifier for all splits.
        """

        def _Classifier(
                X_train_indices: np.ndarray[tuple[int,], np.dtype[np.int64]], # The training set indices for one split.
                X_test_indices:  np.ndarray[tuple[int,], np.dtype[np.int64]],  # The testing set indices for one split.
                model_cpc: keras.Model
            ) -> tuple[
                np.ndarray[tuple[int,], np.dtype[np.int64]], # Classes on the test split for that split.
                np.ndarray[tuple[int, int], np.dtype[np.float64]], # Posterior probabilities by class and by sample for one split.
                np.ndarray[tuple[int, int], np.dtype[np.float64]]  # Posterior probabilities by class and by sample for one split.
                ]:
            """Get posterior probabilities from convolutional neural network classifier for one split.
            """

            X_train, X_cal, y_train, y_cal = train_test_split(
                self._X[X_train_indices, :], # Get data according to indices for that split.
                self._y[X_train_indices],    # Get labels according to indices for that split.
                test_size = test_size,
                random_state = random_state
            )

            model_cpc.compile(
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer = 'adam',
                metrics = ['accuracy']
            )

            # Training.
            model_cpc.fit(
                X_train,
                y_train,
                batch_size = batch_size,
                epochs = epoch,
                verbose = 1
            )
            # Feedforward.
            probability_model_cpc = keras.Sequential([model_cpc, keras.layers.Softmax()])

            # We get posterior results.
            # Is it softmax results on nb_classes neurons.
            return y_cal, probability_model_cpc(X_cal), probability_model_cpc(self._X[X_test_indices, :])
        
    
        model_cpc = keras.Sequential()
        # Add convolution 2D
        model_cpc.add(keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation='relu',
            kernel_initializer='he_normal',
            input_shape=(image_width, image_height, 1))
        )
        model_cpc.add(keras.layers.MaxPooling2D((2, 2)))
        # Add dropouts to the model.
        model_cpc.add(keras.layers.Dropout(0.25))
        model_cpc.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model_cpc.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # Add dropouts to the model.
        model_cpc.add(keras.layers.Dropout(0.25))
        model_cpc.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        # Add dropouts to the model.
        model_cpc.add(keras.layers.Dropout(0.25))
        model_cpc.add(keras.layers.Flatten())
        model_cpc.add(keras.layers.Dense(128, activation='relu'))
        # Add dropouts to the model.
        model_cpc.add(keras.layers.Dropout(0.25))
        model_cpc.add(keras.layers.Dense(nb_classes))
        # Display resume.
        print(model_cpc.summary())

        X_train_kfold_indices, X_test_kfold_indices = self._Kfold(nb_splits, random_state)
        # Parallelization.
        with Pool() as pool:
            async_results = [pool.apply_async(
                _Classifier,
                args=(
                    X_train_kfold_indices[k],
                    X_test_kfold_indices[k],
                    keras.models.clone_model(model_cpc),
                ))
                for k in range(nb_splits)
            ]
            results = [res.get() for res in async_results]

        y_cal = list(map(lambda item: item[0], results))
        cpc_proba_cal = list(map(lambda item: item[1], results))
        cpc_proba_test = list(map(lambda item: item[2], results))

        return y_cal, cpc_proba_cal, cpc_proba_test
    
    
    # TODO: si Eclair uniquement pour le rebelling alors posterior et file_posterior seront attributs.
    # TODO: est ce que c'est la list ?


if __name__ == '__main__':
    x = np.array([
        [1, 2, 3, 6],
        [4, 5, 6, 15],
        [7, 8, 9, 24],
        [10, 11, 12, 33],
        [10, 20, 30, 60],
        [40, 50, 60, 150],
        [70, 80, 90, 240],
        [100, 110, 120, 330],
    ])
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    print(x.dtype)
    print(type(x))

    e = Eclair(x, y)

    print(e._Kfold(2)[0][0].shape)
    print(e._Kfold(2)[0][0].dtype)
    
    # print(e.trainModelkFolds_nbc(x, y, [], [], 2, 42))

    def test(a,b,c,d):
        return a+1, [b+1, b+5], c+1, d+1
    

    X_train, X_cal, y_train, y_cal_tmp = train_test_split(
        x, # Get data according to indices for that split.
        y,    # Get labels according to indices for that split.
    )

    print(X_cal),
    print(y_cal_tmp)
    print(y_cal_tmp.shape)
    print(y_cal_tmp.dtype)