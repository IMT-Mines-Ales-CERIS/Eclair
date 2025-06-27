# Cautious Classification Framework

This project provides a modular and extensible code based on the paper [Cautious classification based on belief functions theory and imprecise relabelling](https://imt-mines-ales.hal.science/hal-03472031v1/file/cautious-classification.pdf). It allows users to plug in their **own relabelling logic**, offers the flexibility to use different classifiers, and provides the opportunity to experiment with a relabeling strategy based on cross-entropy.

## Requirements

* Python ≥ 3.10

```
pip install -r requirements.txt
```

## Code Architecture

The project is structured around two abstract methods that users must implement:

### `IEclair.py`

This class defines the contract for any relabelling method.

```python
class IEclair(ABC):
    @abstractmethod
    def Relabelling(self, **kwargs) -> np.ndarray[tuple[int,], np.dtype[np.int64]]:
        """Compute imprecise labels based on a cautious strategy."""
        pass
````

You must implement this method in your own relabelling class (ex. *CrossEntropy.py* class).

---

### `ICustomClassification.py`

This class defines the signature of a method capable of producing **posterior probabilities** on demand.

```python
class ICustomClassification(ABC):
    @abstractmethod
    def PredictProba(self,
        X_train: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        X_test: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[int,], np.dtype[np.int64]],
        **kwargs
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        pass
```

You must implement this abstract method in your classifier class (ex. *CustomGaussianNB.py* class).

<div align=center>
    <img src="images/eclair.png" alt="Architecture">
</div>

## Relabelling based on Cross-Entropy

The current implementation includes a default relabelling strategy based on **cross-entropy**, in line with the original paper.

```python
gnb = CustomNaiveBayesClassifier()
X, y = load_iris(return_X_y = True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, shuffle = True, random_state = 42
)
# Get posterior probabilities on the training set for each class.
# If nb_folds = nb_samples, it's like a leave-one-out.
nb_folds = 20
posterior_probabilities = PredictProbaKFold(np.array(X_train), np.array(y_train), gnb, nb_folds)
cross_entropy = CrossEntropy(X_train, X_test, y_train, posterior_probabilities, gnb, 2, 0.6, 0.6, 2)
masses, new_y = cross_entropy.Predict()
```

`Predict` method in `IEclair` is responsible for performing the entire cautious classification pipeline, including:

* Performing cautious relabelling on the training data,
* Fitting a model on the training set with new labels,
* Predicting posterior probabilities on the test set, based on the relabelled labels.

## Acknowledgments

I would like to thank [Pierre-Antoine Jean](https://github.com/PAJEAN) for taking over and restructuring the initial code to make it available to the community.

## Citation

```latex
@article{imoussaten2022cautious,
  title={Cautious classification based on belief functions theory and imprecise relabelling},
  author={Imoussaten, Abdelhak and Jacquin, Lucie},
  journal={International Journal of Approximate Reasoning},
  volume={142},
  pages={130--146},
  year={2022},
  publisher={Elsevier}
}
```