import numpy as np


class Perceptron:
    """
    Implementation of a Perceptron classifier   

    For an explanation about how perceptron classifiers work:
    https://en.wikipedia.org/wiki/Perceptron

    Attributes
    ----------
    weights : ndarray
        1d array containing the Perceptron Model weights as float data type
    errors : List[int]
        list with the number of misclassifications per training iteration
    _fitted : Boolean
        indicates whether the model was already fitted

    """

    def __init__(self, lr=0.01, n_iter=50, random_state=42):
        """
        Initializes the model's attributes

        Parameters
        ----------
        lr : float
            Learning rate (between 0.0 and 1.0)
        n_item : int
            Number of iterations over the training set
        random_state : int
            Random seed for weight initialization
        """

        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state
        self.weights = None
        self.errors = []
        self._fitted = False
    
    def _build_ndarrays(self, X=None, y=None):
        """
        Build nd arrays from array-like objects.

        Parameters
        ---------
        X : {array-like object}, shape = [n_examples, n_features]
            Training feature vectors, where n_examples is the number of 
            examples in the training set and n_features is the number
            of features in each feature vector
        y : {array-like object}, shape = [n_examples]
            Training target values
        
        Returns
        -------
        X_arr : np.ndarray, if a valid array like object is provided for X,
                else None
            ndarray representing the training feature vectors
        y_array : np.ndarray, if a valid array like object is provided for y,
                    else None
            ndarray representing the training targets

        """
        n = len(X)

        X_arr = np.hstack((
                np.ones(shape=(n, 1)),
                np.array(X)
        )) if X is not None else None
        
        y_arr = np.array(y) if y is not None else None

        return X_arr, y_arr


    def fit(self, X, y):
        """
        Fit training data

        Parameters
        ---------
        X : {array-like object}, shape = [n_examples, n_features]
            Training feature vectors, where n_examples is the number of 
            examples in the training set and n_features is the number
            of features in each feature vector
        y : {array-like object}, shape = [n_examples]
            Training target values
        
        Returns
        -------
        self : Perceptron
        """
        # Sets the fitted flag to False
        self._fitted = False

        # Converts the input to ndarrays
        X_arr, y_arr = self._build_ndarrays(X, y)

        # Initializes the model weights with small random numbers
        self.weights = np.random.RandomState(self.random_state).normal(
            loc=0.0,
            scale=0.01,
            size=(X_arr.shape[1])
        )

        # Fit the model weights to the data by updating the model's weights
        # for every example
        for _ in range(self.n_iter):
            errors = 0
            for x_i, target in zip(X_arr, y_arr):
                y_pred = self.predict(x_i)
                self.weights += self.lr * (target - y_pred) * x_i
                errors += int(y_pred != target)
                print(x_i)
                print(target)
                print(y_pred)
            self.errors.append(errors)

        self._fitted = True

    def predict(self, X):
        """
        Predicts the class labels for input based on the model weights

        Parameters
        ----------

        X : {array-like object}
            Input data to make predictions on


        Returns
        -------
        y_pred : ndarray
            ndarray with the predicted classes
        """
        X_arr = X
        if self._fitted:
            X_arr, _ = self._build_ndarrays(X=X)
        y_pred = np.where(np.dot(X_arr, self.weights) >= 0.0, 1, -1)
        return y_pred
