import numpy as np
import time

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from CrossEntropy import CrossEntropy
from CustomGaussianNB import CustomNaiveBayesClassifier
from ICustomClassifier import ICustomClassifier
from SetValuedClassEvaluation import SetValuedClassEvaluation
from SetValuedClassification import SetValuedClassification
from Utils import Utils


def ProcessLabelSets(
        label_sets: list[list[int]],
        initial_nb_classes: int
    ) -> list[int]:
    """Convert [[0,1], [0], [1]] to [2, 0, 1], if there is 2 classes (0 and 1).
    """
    labels = []
    for i, value in enumerate(label_sets):
        # Assign a wrong labels when the set contains more than 1 class.
        labels.append(initial_nb_classes + i if len(value) > 1 else value[0])
    return labels


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
    
    results = [
        custom_classifier.PredictProba(
            X[X_train_kfold_indices[k], :],
            X[X_test_kfold_indices[k], :],
            y[X_train_kfold_indices[k]]
        ) for k in range(nb_folds)
    ]

    return np.concatenate(results)


def BasicExample(X, y):
    gnb = CustomNaiveBayesClassifier()

    # Shuffle the data because the labels are initially ordered.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, shuffle = True, random_state = 42
    )
    # Get posterior probabilities on the training set for each class.
    # If nb_folds = nb_samples, it's like a leave-one-out.
    nb_folds = 20
    posterior_probabilities = PredictProbaKFold(np.array(X_train), np.array(y_train), gnb, nb_folds)
    cross_entropy = CrossEntropy(X_train, X_test, y_train, posterior_probabilities, gnb, 2, 1.0, 1.0, 2)
    masses, new_y = cross_entropy.Predict()

    print(f'Distinct new classes: {np.unique(new_y)}')

    decision = SetValuedClassification(
        masses,
        np.unique(new_y),
        len(np.unique(y))
    )

    pred_set = decision.StrongDominance()

    return SetValuedClassEvaluation.SetValuedClassEvaluation(y_test, pred_set)


def BasicExampleOptimization(X, y):
    gnb = CustomNaiveBayesClassifier()

    # Shuffle the data because the labels are initially ordered.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, shuffle = True, random_state = 42
    )
    # Get posterior probabilities on the training set for each class.
    # If nb_folds = nb_samples, it's like a leave-one-out.
    nb_folds = 20
    posterior_probabilities = PredictProbaKFold(np.array(X_train), np.array(y_train), gnb, nb_folds)

    best_params = []
    best_accuracy = 0
    for threshold in np.linspace(0, 3, num=20):
        cross_entropy = CrossEntropy(X_train, X_test, y_train, posterior_probabilities, gnb, 2, threshold, threshold, 2)
        masses, new_y = cross_entropy.Predict()

        decision = SetValuedClassification(
            masses,
            np.unique(new_y),
            len(np.unique(y))
        )

        for beta in np.linspace(0.2, 3, num=20):
            pred_set = decision.GFBeta(beta)
            pred = ProcessLabelSets(pred_set, len(np.unique(y)))

            accuracy = accuracy_score(y_test, pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = [(threshold, beta)]
            elif accuracy == best_accuracy:
                best_params.append((threshold, beta))
    
    print(best_params)
    return best_accuracy


def LeaveOneOut(X, y):
    gnb = CustomNaiveBayesClassifier()
    accuracy = []
    nb_folds = len(X) # Leave one out.
    X_train_kfold_indices, X_test_kfold_indices = Utils.Kfold(np.array(X), nb_folds)
    for k in range(nb_folds):
        X_train = X[X_train_kfold_indices[k], :]
        X_test = X[X_test_kfold_indices[k], :]
        y_train = y[X_train_kfold_indices[k]]
        y_test = y[X_test_kfold_indices[k]]

        posterior_probabilities = PredictProbaKFold(np.array(X_train), np.array(y_train), gnb, len(X_train))
        cross_entropy = CrossEntropy(X_train, X_test, y_train, posterior_probabilities, gnb, 2, 0.936, 0.936, 2)
        masses, new_y = cross_entropy.Predict()

        decision = SetValuedClassification(
            masses,
            np.unique(new_y),
            len(np.unique(y))
        )

        pred_set = decision.GFBeta(0.08)
        pred = ProcessLabelSets(pred_set, len(np.unique(y)))

        accuracy.append(accuracy_score(y_test, pred)) # 0 or 1, only one test sample (leave one out).
    return np.mean(accuracy)


def OptimizeLeaveOneOut(X, y):
    """Prediction on the entire dataset with optimization of cross entropy threshold and CF beta parameters.
    """
    gnb = CustomNaiveBayesClassifier()
    all_accuracies = []
    nb_folds = len(X) # Leave one out.
    X_train_kfold_indices, X_test_kfold_indices = Utils.Kfold(np.array(X), nb_folds)
    for k in range(nb_folds):
        X_train = X[X_train_kfold_indices[k], :]
        X_test = X[X_test_kfold_indices[k], :]
        y_train = y[X_train_kfold_indices[k]]
        y_test = y[X_test_kfold_indices[k]]
        # Leave one out on training dataset to get posterior probabilities on the set.
        posterior_probabilities = PredictProbaKFold(np.array(X_train), np.array(y_train), gnb, len(X_train))
        
        best_accuracy = 0
        for threshold in np.linspace(0, 3, num=20):
            cross_entropy = CrossEntropy(X_train, X_test, y_train, posterior_probabilities, gnb, 2, threshold, threshold, 2)
            masses, new_y = cross_entropy.Predict()

            decision = SetValuedClassification(
                masses,
                np.unique(new_y),
                len(np.unique(y))
            )

            for beta in np.linspace(0, 3, num=20):
                pred_set = decision.GFBeta(beta)
                pred = ProcessLabelSets(pred_set, len(np.unique(y)))

                accuracy = accuracy_score(y_test, pred) # 0 or 1, only one test sample (leave one out).
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    if best_accuracy == 1.0: # Break if best_accuracy == 1.
                        break

        all_accuracies.append(best_accuracy)

    return np.mean(all_accuracies) # 0.96 in 78.88s (pc labo).

if __name__ == '__main__':
    start = time.time()

    X, y = load_iris(return_X_y = True)

    accuracy = BasicExample(X, y)
    # accuracy = LeaveOneOut(X, y)
    # accuracy = OptimizeLeaveOneOut(X, y)

    print(f'\nEvaluation: {accuracy}')

    print(f'\nDuration: {np.round(time.time() - start, 2)}s')