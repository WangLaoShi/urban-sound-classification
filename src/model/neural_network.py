import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.utils import class_weight
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from ..data import Dataset
from keras.callbacks import EarlyStopping
import numpy as np
from numpy.random import seed
seed(1)
tensorflow.random.set_seed(1)

import warnings  
warnings.filterwarnings("ignore",category=FutureWarning)

class NeuralNetwork:
    
    @staticmethod
    def create_model(neurons=(144, 70, 40, 30, 10),
                     learning_rate=0.01,
                     momentum=0.0,
                     optimizer="sgd",
                     loss=keras.losses.SparseCategoricalCrossentropy(),
                     metrics=["accuracy"]):
        """
        Neural network creation function

        Args:
            neurons (tuple, optional): Number of neurons per layer. Defaults to (144, 70, 40, 30, 10).
            learning_rate (float, optional): Optimizer learning rate. Defaults to 0.01.
            momentum (float, optional): Optimizer momentum. Defaults to 0.0.
            loss ([type], optional): Loss function to use. Defaults to keras.losses.SparseCategoricalCrossentropy().
            metrics (list, optional): Metrics to compile. Defaults to ["accuracy"].

        Returns:
            Keras model: Compiled keras neural network
        """
        # Initialize a list to hold the layers of the model
        ll=[layers.Dense(units=neurons[0], activation='relu')]
        # Add hidden layers to the model
        for n in neurons[1:-1]:
            ll.append(layers.Dense(n, activation='relu'))
        # Add the output layer with softmax activation
        ll.append(layers.Dense(neurons[-1], activation='softmax'))

        # Create a Sequential model with the defined layers
        model = keras.Sequential(ll)

        # Choose the optimizer based on the input argument
        if optimizer == "sgd":
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer == "adam":
            opt = keras.optimizers.Adam()

        # Compile the model with the specified optimizer, loss function, and metrics
        model.compile(optimizer=opt,
                      loss=loss, 
                      metrics=metrics) 
        return model
    
    @staticmethod
    def optimize_model(method="grid", 
                       param_grid={}, 
                       dataset_path="../data/processed/extended/train_scaled_extended.csv", 
                       iterations=10):
        """
        Optimize a model by performing random or grid search

        Args:
            method (str, optional): Random or grid search. Defaults to "grid".
            param_grid (dict, optional): Parameters grid. Defaults to {}.
            dataset_path (str, optional): Path to the dataset to load. Defaults to "../data/processed/extended/train_scaled_extended.csv".
            iterations (int, optional): Number of iteration for the random search. Defaults to 10.

        Returns:
            Scikitlearn search results: results of the parameter optimization
        """

        # Wrap the Keras model for use with scikit-learn
        model = KerasClassifier(build_fn=NeuralNetwork.create_model, verbose=0)

        # Load the dataset
        d = Dataset(dataset_path, test_size=0)
        x_train, y_train = d.get_splits()

        # Choose the search method (grid or random) and set up the search object
        if method == "grid":
            search = GridSearchCV(estimator=model, 
                                  param_grid=param_grid, 
                                  n_jobs=-1, 
                                  scoring="accuracy",
                                  error_score="raise", 
                                  verbose=2, 
                                  cv=StratifiedKFold(n_splits=5))
            
        elif method == "random":
            search = RandomizedSearchCV(estimator=model, 
                                        param_distributions=param_grid, 
                                        n_iter=iterations, 
                                        scoring='accuracy', 
                                        n_jobs=-1, 
                                        error_score="raise", 
                                        verbose=2, 
                                        random_state=1, 
                                        cv=StratifiedKFold(n_splits=5))

        # Set up early stopping to prevent overfitting
        stopper = EarlyStopping(monitor='accuracy', patience=3, verbose=0)
        # Define fit parameters including early stopping
        fit_params = dict(callbacks=[stopper])
        # Compute class weights to handle imbalanced datasets
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        weights_dict = dict(zip(np.unique(y_train), class_weights))
        # Perform the search to find the best hyperparameters
        results = search.fit(x_train, y_train, class_weight=weights_dict, **fit_params)
        return results
