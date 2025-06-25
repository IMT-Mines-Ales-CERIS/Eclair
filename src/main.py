import numpy as np
import os
import time

from multiprocessing import Pool
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from CrossEntropy import CrossEntropy
from CustomGaussianNB import CustomNaiveBayesClassifier
from ICustomClassifier import ICustomClassifier
from SetValuedClassification import SetValuedClassification
from Utils import Utils

def PredictProbaKFold(
        X: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y: np.ndarray[tuple[int,], np.dtype[np.int64]],
        custom_classifier: ICustomClassifier,
        nb_folds: int
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get posterior probabilities of the training dataset by sampling with kfold.

    ### Parameters :
        * ``X`` - Dataset, shape, (n_samples, n_features).
        * ``y`` - Labels, shape, (n_samples,).
        * ``custom_classifier`` - Classifier used to predict posterior probabilities.
        * ``nb_folds`` - Number of folds, must be at least 2.

    ### Returns :
        * Posterior probabilities on training dataset.
    """
    
    # For a dataset with 6 entries and a split number of 2 here is the return : ([(TRAIN)array([3, 4, 5]), (TEST)array([0, 1, 2])], [(TRAIN)array([0, 1, 2]), (TEST)array([3, 4, 5])]).
    # It is important to make the kfold without shuffle to keep dataset order in test data (0, 1, 2, 3, 4, 5).
    X_train_kfold_indices, X_test_kfold_indices = Utils.Kfold(X, nb_folds)
    
    # Parallelization.
    # If processes argument is None then the number returned by os.process_cpu_count() is used.
    
    # print(f'Processes used {os.process_cpu_count()}\n')
    # with Pool() as pool:
    #     results = pool.starmap(
    #         custom_classifier.PredictProba, [
    #             (
    #                 X[X_train_kfold_indices[k], :],
    #                 X[X_test_kfold_indices[k], :],
    #                 y[X_train_kfold_indices[k]]
    #             )
    #             for k in range(nb_folds)
    #         ]
    #     )

    # OR
    
    results = [
        custom_classifier.PredictProba(
            X[X_train_kfold_indices[k], :],
            X[X_test_kfold_indices[k], :],
            y[X_train_kfold_indices[k]]
        ) for k in range(nb_folds)
    ]

    return np.concatenate(results)

        

if __name__ == '__main__':
    start = time.time()

    gnb = CustomNaiveBayesClassifier()
    X, y = load_iris(return_X_y = True)

    # Shuffle the data because the labels are initially ordered.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, shuffle = True, random_state = 42
    )
    # Get posterior probabilities on the training set for each class.
    # If nb_folds = nb_samples, it's like a leave-one-out.
    nb_folds = 20
    posterior_probabilities = PredictProbaKFold(np.array(X_train), np.array(y_train), gnb, nb_folds)
    cross_entropy = CrossEntropy(X_train, X_test, y_train, posterior_probabilities, gnb, 2, 0.6, 0.6, 2)
    masses, new_y = cross_entropy.Predict()

    print(f'Distinct new labels: {np.unique(new_y)}\n')

    pred_sd_set = SetValuedClassification.StrongDominance(
        masses,
        np.unique(new_y),
        len(np.unique(y))
    )

    pred_sd = []
    for i, value in enumerate(pred_sd_set):
        if len(value) > 1:
            pred_sd.append(len(np.unique(y)) + i)
        else:
            pred_sd.append(value[0])

    # format = "{:<10}{:^8}{:>5}"
    # print(format.format('pred_set', 'pred', 'y_test'))
    # for a, b, c in zip(pred_sd_set, pred_sd, y_test):
    #     print(format.format(str(a), str(b), str(c)))

    print(f'\nAccuracy: {accuracy_score(y_test, pred_sd)}')

    print(f'\nDuration: {np.round(time.time() - start, 2)}s')